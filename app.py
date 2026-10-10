"""f1-adapter: reshapes f1-dash's /api/current into the ESPN-service contract
that nande-rpc's Live Sports picker and refetchEspnEvent expect.

Only the Formula 1 slice of that contract is implemented:

    GET /sports                                       -> {"sports": [...]}
    GET /sports/f1/categories                         -> {"categories": []}
    GET /sports/f1/live | /today                      -> {"events": [event]}
    GET /sports/f1/categories/{id}/competitions       -> {"competitions": [...]}
    GET /sports/f1/competitions/{id}/events           -> {"events": [event]}
    GET /events/f1/{competition_id}/{event_id}        -> {"event": event}   (polled every 1s)

Extras: /f1/snapshot (same event, flat), /f1/raw (upstream state, for debugging),
/healthz (no auth).

The event is always id "current": picking it once makes presence follow whichever
session is live, instead of pinning to one session.
"""
from __future__ import annotations

import asyncio
import hmac
import os
import re
import time
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from typing import Any, Callable
from urllib.parse import urlparse

import httpx
from fastapi import APIRouter, Depends, FastAPI, Header, HTTPException

# ---------------------------------------------------------------- config
F1DASH_URL = os.getenv("F1DASH_URL", "http://localhost:4000/api/current")
API_KEY = os.getenv("ADAPTER_API_KEY", "")
ARTWORK = os.getenv(
    "F1_ARTWORK_URL",
    "https://i.postimg.cc/pL72HRzS/418a6e796c6bbd9eb76a840cf6708747.png",
)
CACHE_TTL_S = float(os.getenv("CACHE_TTL_S", "0.5"))       # the script polls at 1s
STALE_AFTER_S = float(os.getenv("STALE_AFTER_S", "20"))    # serve last good state this long if f1-dash drops
COLD_START_TIMEOUT_S = float(os.getenv("COLD_START_TIMEOUT_S", "40"))  # Render free-tier wake-up
WARM_TIMEOUT_S = float(os.getenv("WARM_TIMEOUT_S", "5"))
HEARTBEAT_MAX_AGE_S = float(os.getenv("HEARTBEAT_MAX_AGE_S", "45"))
SHOW_RACE_CLOCK = os.getenv("SHOW_RACE_CLOCK", "0") == "1"  # race/sprint show laps, not the 2h limit

COMPETITION_ID = "f1"
EVENT_ID = "current"

# F1 SessionStatus -> ESPN-style state ("in" / "pre" / "post")
STATE_BY_STATUS = {
    "Started": "in",
    "Aborted": "in",  # red-flagged, session still running
    "Finished": "post",
    "Finalised": "post",
    "Ends": "post",
}
STATUS_TEXT = {"in": "Live", "pre": "Upcoming", "post": "Finished"}

# F1 TrackStatus.Status codes ("1" = all clear, so it is omitted)
TRACK_FLAGS = {
    "2": "Yellow Flag",
    "4": "Safety Car",
    "5": "Red Flag",
    "6": "Virtual Safety Car",
    "7": "VSC Ending",
}

QUALI_KINDS = ("qualifying", "sprint_quali")
RACE_KINDS = ("race", "sprint")


# ---------------------------------------------------------------- helpers
def pick(d: Any, *names: str, default: Any = None) -> Any:
    """Case-insensitive dict lookup, so TimingData/timingData/timingdata all work."""
    if not isinstance(d, dict):
        return default
    lowered = {str(k).lower(): v for k, v in d.items()}
    for name in names:
        value = lowered.get(name.lower())
        if value is not None:
            return value
    return default


_TOPICS = ("timingdata", "sessioninfo", "sessionstatus", "driverlist")


def unwrap_state(raw: Any) -> dict:
    """Return the dict that actually holds the F1 topics, tolerating a wrapper object."""
    if not isinstance(raw, dict):
        return {}

    def looks_like_state(d: Any) -> bool:
        return isinstance(d, dict) and any(str(k).lower() in _TOPICS for k in d)

    if looks_like_state(raw):
        return raw
    for key in ("data", "state", "current"):
        inner = pick(raw, key)
        if looks_like_state(inner):
            return inner
    return raw


def parse_ts(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    s = value.replace("Z", "+00:00")
    # F1 sends 3..7 fractional digits; fromisoformat on older Pythons wants exactly 3 or 6.
    s = re.sub(r"\.(\d+)", lambda m: "." + m.group(1)[:6].ljust(6, "0"), s)
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def parse_hms(value: Any) -> float | None:
    """'00:12:34' -> 754.0; '-05:00:00' -> -18000.0."""
    if not isinstance(value, str) or not value.strip():
        return None
    s = value.strip()
    sign = -1 if s.startswith("-") else 1
    s = s.lstrip("+-")
    try:
        h, m, sec = s.split(":")
        return sign * (int(h) * 3600 + int(m) * 60 + float(sec))
    except ValueError:
        return None


def fmt_clock(seconds: int) -> str:
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def session_kind(info: Any) -> str:
    name = str(pick(info, "Name", default="")).lower()
    typ = str(pick(info, "Type", default="")).lower()
    # Check Name first: a Sprint has Type "Race" and Sprint Qualifying has Type "Qualifying".
    if "sprint" in name and ("qualifying" in name or "shootout" in name):
        return "sprint_quali"
    if "sprint" in name:
        return "sprint"
    if "qualifying" in name or typ == "qualifying":
        return "qualifying"
    if name == "race" or typ == "race":
        return "race"
    if "practice" in name or typ == "practice":
        return "practice"
    return "other"


def session_part(timing: Any) -> int | None:
    try:
        return int(pick(timing, "SessionPart"))
    except (TypeError, ValueError):
        return None


def session_start_utc(info: Any) -> str | None:
    """SessionInfo.StartDate is circuit-local time; GmtOffset is its offset from UTC."""
    raw = pick(info, "StartDate")
    if not isinstance(raw, str):
        return None
    local = parse_ts(raw)
    if local is None:
        return None
    if not re.search(r"(Z|[+-]\d\d:\d\d)$", raw):  # naive -> apply the circuit offset
        offset = parse_hms(pick(info, "GmtOffset")) or 0
        local = local - timedelta(seconds=offset)
    return local.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def find_leader(timing: Any) -> tuple[str | None, dict | None]:
    lines = pick(timing, "Lines")
    if not isinstance(lines, dict):
        return None, None
    best: tuple[int, str, dict] | None = None
    for key, line in lines.items():
        if not isinstance(line, dict):
            continue
        try:
            position = int(pick(line, "Position"))
        except (TypeError, ValueError):
            continue
        if position > 0 and (best is None or position < best[0]):
            best = (position, str(pick(line, "RacingNumber", default=key)), line)
    return (best[1], best[2]) if best else (None, None)


def clock_remaining(state: dict, now: datetime) -> int | None:
    clock = pick(state, "ExtrapolatedClock")
    base = parse_hms(pick(clock, "Remaining"))
    if base is None:
        return None
    if pick(clock, "Extrapolating") is True:  # running clock: Remaining was valid at Utc
        at = parse_ts(pick(clock, "Utc"))
        if at:
            base -= (now - at).total_seconds()
    return max(0, int(base))


# ---------------------------------------------------------------- core mapping
def build_event(state: dict, now: datetime) -> dict:
    info = pick(state, "SessionInfo", default={})
    timing = pick(state, "TimingData", default={})
    kind = session_kind(info)
    status_raw = pick(pick(state, "SessionStatus"), "Status")
    st = STATE_BY_STATUS.get(status_raw, "pre")

    heartbeat = parse_ts(pick(pick(state, "Heartbeat"), "Utc"))
    feed_stale = heartbeat is not None and (now - heartbeat).total_seconds() > HEARTBEAT_MAX_AGE_S
    live = st != "pre" and not feed_stale  # don't show leftovers before start or after the feed dies

    start_local = pick(info, "StartDate")
    year = start_local[:4] if isinstance(start_local, str) and start_local[:4].isdigit() else None
    gp = pick(pick(info, "Meeting"), "Name") or "Formula 1"
    name = f"{gp} {year}" if year else gp
    session_name = pick(info, "Name")

    part = session_part(timing)
    part_label = f"{'SQ' if kind == 'sprint_quali' else 'Q'}{part}" if kind in QUALI_KINDS and part else None

    # Leader: race/sprint show their last completed lap; everything else shows best lap.
    leader_name = leader_team = leader_time = None
    if live:
        number, line = find_leader(timing)
        if line is not None:
            driver = pick(pick(state, "DriverList"), number or "")
            leader_name = (pick(driver, "FullName") or pick(driver, "BroadcastName")
                           or pick(driver, "Tla"))
            leader_team = pick(driver, "TeamName")
            key = "LastLapTime" if kind in RACE_KINDS else "BestLapTime"
            leader_time = pick(pick(line, key), "Value") or None

    laps = total_laps = None
    if live and kind in RACE_KINDS:
        count = pick(state, "LapCount")
        laps, total_laps = pick(count, "CurrentLap"), pick(count, "TotalLaps")

    remaining_s = None
    if st == "in" and not feed_stale and (kind not in RACE_KINDS or SHOW_RACE_CLOCK):
        remaining_s = clock_remaining(state, now)

    track_status = None
    if live:
        code = str(pick(pick(state, "TrackStatus"), "Status", default=""))
        track_status = TRACK_FLAGS.get(code) or ("Chequered Flag" if status_raw == "Finished" else None)

    # Ends the pick in the script, so be conservative: only once results are final, and
    # not between qualifying parts (Q1/Q2 can be "Finalised"-looking before Q3 starts).
    completed = status_raw in ("Finalised", "Ends")
    if completed and kind in QUALI_KINDS and part not in (None, 3):
        completed = False

    status_text = (f"{session_name} ({STATUS_TEXT[st]})" if session_name else "Waiting for session")
    return {
        "event_id": EVENT_ID,
        "competition_id": COMPETITION_ID,
        "sport": "f1",
        "name": name,
        "short_name": name,
        "competition": f"Formula 1 {year}" if year else "Formula 1",
        "artwork": ARTWORK,
        "start": session_start_utc(info),
        "status": status_text,
        "state": st,
        "completed": completed,
        "f1_session": session_name,
        "f1_part": part_label,
        "f1_track_status": track_status,
        "f1_time_remaining": fmt_clock(remaining_s) if remaining_s is not None else None,
        "f1_time_remaining_seconds": remaining_s,
        "f1_leader_name": leader_name,
        "f1_leader_team": leader_team,
        "f1_leader_laps": laps,
        "f1_total_laps": total_laps,
        "f1_leader_lap_time": leader_time,
    }


# ---------------------------------------------------------------- upstream
def describe_error(exc: Exception) -> str:
    """Short, safe description of why the f1-dash call failed (shown in the 503 body)."""
    path = urlparse(F1DASH_URL).path or "/"
    if isinstance(exc, httpx.HTTPStatusError):
        return f"HTTP {exc.response.status_code} from {path}"
    if isinstance(exc, ValueError):
        return f"non-JSON response from {path}"
    return f"{type(exc).__name__} calling {path}"


class Upstream:
    """Fetch-on-demand with a short cache; keeps serving the last good state briefly on errors."""

    def __init__(self, client: httpx.AsyncClient):
        self.client = client
        self.lock = asyncio.Lock()
        self.state: dict | None = None
        self.fetched_at = 0.0

    def _fresh(self) -> bool:
        return self.state is not None and time.monotonic() - self.fetched_at < CACHE_TTL_S

    async def get(self) -> dict:
        if self._fresh():
            return self.state  # type: ignore[return-value]
        async with self.lock:
            if self._fresh():
                return self.state  # type: ignore[return-value]
            timeout = COLD_START_TIMEOUT_S if self.state is None else WARM_TIMEOUT_S
            try:
                resp = await self.client.get(F1DASH_URL, timeout=timeout)
                resp.raise_for_status()
                self.state = unwrap_state(resp.json())
                self.fetched_at = time.monotonic()
            except (httpx.HTTPError, httpx.InvalidURL, ValueError) as exc:
                too_old = time.monotonic() - self.fetched_at > STALE_AFTER_S
                if self.state is None or too_old:
                    raise HTTPException(503, detail=f"f1-dash unavailable ({describe_error(exc)})")
            return self.state  # type: ignore[return-value]


# ---------------------------------------------------------------- app
def create_app(transport: httpx.AsyncBaseTransport | None = None,
               clock: Callable[[], datetime] | None = None) -> FastAPI:
    now_fn = clock or (lambda: datetime.now(timezone.utc))

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        async with httpx.AsyncClient(transport=transport, follow_redirects=True) as client:
            app.state.upstream = Upstream(client)
            yield

    app = FastAPI(title="f1-adapter", lifespan=lifespan)

    async def require_key(x_api_key: str = Header(default="")):
        if API_KEY and not hmac.compare_digest(x_api_key, API_KEY):
            raise HTTPException(401, detail="bad api key")

    api = APIRouter(dependencies=[Depends(require_key)])

    async def current_event() -> dict:
        return build_event(await app.state.upstream.get(), now_fn())

    @app.get("/healthz")
    async def healthz():
        return {"ok": True}

    @api.get("/sports")
    async def sports():
        return {"sports": [{"id": "f1", "label": "Formula 1", "provider": "espn"}]}

    @api.get("/sports/f1/categories")
    async def categories():
        return {"categories": []}  # the script adds "Live now" / "Today" itself

    @api.get("/sports/f1/live")
    async def live():
        ev = await current_event()
        return {"events": [ev] if ev["state"] == "in" else []}

    @api.get("/sports/f1/today")
    async def today():
        return {"events": [await current_event()]}

    @api.get("/sports/f1/categories/{category_id}/competitions")
    async def competitions(category_id: str):
        return {"competitions": [{"id": COMPETITION_ID, "label": "Formula 1"}]}

    @api.get("/sports/f1/competitions/{competition_id}/events")
    async def competition_events(competition_id: str):
        if competition_id != COMPETITION_ID:
            raise HTTPException(404, detail="unknown competition")
        return {"events": [await current_event()]}

    @api.get("/events/f1/{competition_id}/{event_id}")
    async def event(competition_id: str, event_id: str):
        if competition_id != COMPETITION_ID or event_id != EVENT_ID:
            raise HTTPException(404, detail="unknown event")
        return {"event": await current_event()}

    @api.get("/f1/snapshot")
    async def snapshot():
        return await current_event()

    @api.get("/f1/raw")
    async def raw():
        return await app.state.upstream.get()

    app.include_router(api)
    return app


app = create_app()
