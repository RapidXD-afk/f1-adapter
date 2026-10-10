from datetime import datetime, timedelta, timezone

import httpx
import pytest
from fastapi.testclient import TestClient

import app as adapter

NOW = datetime(2026, 7, 5, 13, 30, tzinfo=timezone.utc)


def iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"  # F1 style: 3 fractional digits


DRIVERS = {
    "1": {"RacingNumber": "1", "FullName": "Max VERSTAPPEN", "BroadcastName": "M VERSTAPPEN",
          "Tla": "VER", "TeamName": "Red Bull Racing"},
    "4": {"RacingNumber": "4", "FullName": "Lando NORRIS", "BroadcastName": "L NORRIS",
          "Tla": "NOR", "TeamName": "McLaren"},
}


def make_state(*, session="Race", typ="Race", status="Started", track="1", part=None,
               laps=(31, 57), clock=("00:10:00", True, 30), hb_age=1,
               start="2026-07-05T15:00:00", offset="02:00:00") -> dict:
    """Norris (#4) leads on track position; Verstappen (#1) is P2 -> proves we pick by Position."""
    timing = {"Lines": {
        "1": {"Position": "2", "RacingNumber": "1",
              "LastLapTime": {"Value": "1:33.100"}, "BestLapTime": {"Value": "1:32.000"}},
        "4": {"Position": "1", "RacingNumber": "4",
              "LastLapTime": {"Value": "1:32.456"}, "BestLapTime": {"Value": "1:31.900"}},
    }}
    if part is not None:
        timing["SessionPart"] = part
    state = {
        "Heartbeat": {"Utc": iso(NOW - timedelta(seconds=hb_age))},
        "SessionInfo": {"Meeting": {"Name": "Austrian Grand Prix"}, "Name": session, "Type": typ,
                        "StartDate": start, "GmtOffset": offset},
        "SessionStatus": {"Status": status},
        "TrackStatus": {"Status": track},
        "DriverList": DRIVERS,
        "TimingData": timing,
    }
    if laps:
        state["LapCount"] = {"CurrentLap": laps[0], "TotalLaps": laps[1]}
    if clock:
        remaining, extrapolating, age = clock
        state["ExtrapolatedClock"] = {"Utc": iso(NOW - timedelta(seconds=age)),
                                      "Remaining": remaining, "Extrapolating": extrapolating}
    return state


# ------------------------------------------------------------ mapping
def test_race_live():
    ev = adapter.build_event(make_state(), NOW)
    assert ev["state"] == "in" and ev["completed"] is False
    assert ev["f1_session"] == "Race" and ev["f1_part"] is None
    assert ev["f1_leader_name"] == "Lando NORRIS" and ev["f1_leader_team"] == "McLaren"
    assert (ev["f1_leader_laps"], ev["f1_total_laps"]) == (31, 57)
    assert ev["f1_leader_lap_time"] == "1:32.456"          # last lap in a race
    assert ev["f1_time_remaining"] is None                 # race shows laps, not the 2h limit
    assert ev["f1_track_status"] is None
    assert ev["name"] == "Austrian Grand Prix 2026" and ev["competition"] == "Formula 1 2026"
    assert ev["sport"] == "f1" and ev["event_id"] == "current"


def test_race_clock_opt_in(monkeypatch):
    monkeypatch.setattr(adapter, "SHOW_RACE_CLOCK", True)
    ev = adapter.build_event(make_state(), NOW)
    assert ev["f1_time_remaining"] == "09:30"


@pytest.mark.parametrize("code,expected", [
    ("1", None), ("2", "Yellow Flag"), ("4", "Safety Car"), ("5", "Red Flag"),
    ("6", "Virtual Safety Car"), ("7", "VSC Ending"),
])
def test_track_status_codes(code, expected):
    assert adapter.build_event(make_state(track=code), NOW)["f1_track_status"] == expected


def test_qualifying_part_clock_and_best_lap():
    ev = adapter.build_event(make_state(session="Qualifying", typ="Qualifying", part=2, laps=None), NOW)
    assert ev["f1_part"] == "Q2"
    assert ev["f1_time_remaining"] == "09:30" and ev["f1_time_remaining_seconds"] == 570
    assert ev["f1_leader_lap_time"] == "1:31.900"          # best lap outside races
    assert ev["f1_leader_laps"] is None and ev["f1_total_laps"] is None


def test_sprint_qualifying_is_sq():
    ev = adapter.build_event(make_state(session="Sprint Qualifying", typ="Qualifying", part=1, laps=None), NOW)
    assert ev["f1_part"] == "SQ1"


def test_sprint_is_race_like_despite_type_race():
    ev = adapter.build_event(make_state(session="Sprint", typ="Race"), NOW)
    assert ev["f1_leader_lap_time"] == "1:32.456" and ev["f1_leader_laps"] == 31
    assert ev["f1_time_remaining"] is None


def test_practice_shows_clock_and_session_name():
    ev = adapter.build_event(make_state(session="Practice 1", typ="Practice", laps=None), NOW)
    assert ev["f1_session"] == "Practice 1" and ev["f1_part"] is None
    assert ev["f1_time_remaining"] == "09:30"


def test_paused_clock_is_not_extrapolated():
    state = make_state(session="Qualifying", typ="Qualifying", part=3, laps=None,
                       clock=("00:08:00", False, 120))
    assert adapter.build_event(state, NOW)["f1_time_remaining"] == "08:00"


def test_clock_over_an_hour_uses_h_mm_ss():
    state = make_state(session="Practice 2", typ="Practice", laps=None, clock=("01:05:09", False, 0))
    assert adapter.build_event(state, NOW)["f1_time_remaining"] == "1:05:09"


def test_empty_state_is_pre_with_nothing_leaking():
    ev = adapter.build_event({}, NOW)
    assert ev["state"] == "pre" and ev["completed"] is False
    assert ev["name"] == "Formula 1" and ev["status"] == "Waiting for session"
    assert all(ev[k] is None for k in ev if k.startswith("f1_"))


def test_before_start_clears_live_fields():
    ev = adapter.build_event(make_state(status="Inactive"), NOW)
    assert ev["state"] == "pre"
    assert ev["f1_leader_name"] is None and ev["f1_track_status"] is None and ev["f1_leader_laps"] is None


def test_chequered_flag_keeps_winner():
    ev = adapter.build_event(make_state(status="Finished"), NOW)
    assert ev["state"] == "post" and ev["f1_track_status"] == "Chequered Flag"
    assert ev["f1_leader_name"] == "Lando NORRIS" and ev["completed"] is False


def test_completed_only_when_final():
    assert adapter.build_event(make_state(status="Finalised"), NOW)["completed"] is True
    q2 = make_state(session="Qualifying", typ="Qualifying", status="Finalised", part=2, laps=None)
    q3 = make_state(session="Qualifying", typ="Qualifying", status="Finalised", part=3, laps=None)
    assert adapter.build_event(q2, NOW)["completed"] is False
    assert adapter.build_event(q3, NOW)["completed"] is True


def test_stale_heartbeat_drops_live_fields_but_keeps_session():
    ev = adapter.build_event(make_state(track="5", hb_age=300), NOW)
    assert ev["f1_leader_name"] is None and ev["f1_track_status"] is None
    assert ev["f1_time_remaining"] is None and ev["f1_leader_laps"] is None
    assert ev["f1_session"] == "Race" and ev["state"] == "in"


@pytest.mark.parametrize("start,offset,expected", [
    ("2026-07-05T15:00:00", "02:00:00", "2026-07-05T13:00:00Z"),
    ("2026-05-03T16:00:00", "-05:00:00", "2026-05-03T21:00:00Z"),
    ("2026-07-05T13:00:00Z", "02:00:00", "2026-07-05T13:00:00Z"),  # already has a zone
])
def test_start_converted_to_utc(start, offset, expected):
    assert adapter.build_event(make_state(start=start, offset=offset), NOW)["start"] == expected


def _lower(obj):
    if isinstance(obj, dict):
        return {str(k).lower(): _lower(v) for k, v in obj.items()}
    return obj


def test_key_case_and_wrapper_tolerated():
    base = adapter.build_event(make_state(), NOW)
    assert adapter.build_event(adapter.unwrap_state({"data": _lower(make_state())}), NOW) == base


# ------------------------------------------------------------ HTTP layer
def make_client(handler):
    app = adapter.create_app(transport=httpx.MockTransport(handler), clock=lambda: NOW)
    return TestClient(app)


def serving(state):
    return lambda request: httpx.Response(200, json=state)


def test_picker_endpoints_match_script_contract():
    with make_client(serving(make_state())) as c:
        sports = c.get("/sports").json()["sports"]
        assert sports == [{"id": "f1", "label": "Formula 1", "provider": "espn"}]
        assert c.get("/sports/f1/categories").json() == {"categories": []}
        assert len(c.get("/sports/f1/live").json()["events"]) == 1
        assert len(c.get("/sports/f1/today").json()["events"]) == 1
        assert len(c.get("/sports/f1/competitions/f1/events").json()["events"]) == 1
        assert c.get("/sports/f1/categories/x/competitions").json()["competitions"][0]["id"] == "f1"
        ev = c.get("/events/f1/f1/current").json()["event"]
        assert ev["event_id"] == "current" and ev["competition_id"] == "f1" and ev["sport"] == "f1"
        assert c.get("/f1/snapshot").json() == ev


def test_live_list_empty_when_not_in_session_but_today_still_lists():
    with make_client(serving(make_state(status="Inactive"))) as c:
        assert c.get("/sports/f1/live").json() == {"events": []}
        assert len(c.get("/sports/f1/today").json()["events"]) == 1


def test_unknown_routes_404():
    with make_client(serving(make_state())) as c:
        assert c.get("/events/f1/f1/other").status_code == 404
        assert c.get("/events/f1/nope/current").status_code == 404
        assert c.get("/sports/football/live").status_code == 404


def test_api_key_enforced(monkeypatch):
    monkeypatch.setattr(adapter, "API_KEY", "secret")
    with make_client(serving(make_state())) as c:
        assert c.get("/healthz").status_code == 200
        assert c.get("/sports").status_code == 401
        assert c.get("/sports", headers={"X-API-Key": "wrong"}).status_code == 401
        assert c.get("/sports", headers={"X-API-Key": "secret"}).status_code == 200


def test_cache_collapses_rapid_requests():
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(200, json=make_state())

    with make_client(handler) as c:
        for _ in range(5):
            c.get("/events/f1/f1/current")
    assert len(calls) == 1


def failing_after_first(state):
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["n"] > 1:
            raise httpx.ConnectError("boom", request=request)
        return httpx.Response(200, json=state)

    return handler


def test_upstream_down_at_start_is_503_with_detail():
    def handler(request):
        raise httpx.ConnectError("boom", request=request)

    with make_client(handler) as c:
        r = c.get("/events/f1/f1/current")
        assert r.status_code == 503 and "unavailable" in r.json()["detail"]


def test_upstream_blip_serves_last_good_then_503(monkeypatch):
    monkeypatch.setattr(adapter, "CACHE_TTL_S", 0)
    with make_client(failing_after_first(make_state())) as c:
        assert c.get("/events/f1/f1/current").status_code == 200
        assert c.get("/events/f1/f1/current").status_code == 200   # within STALE_AFTER_S
        monkeypatch.setattr(adapter, "STALE_AFTER_S", -1)
        assert c.get("/events/f1/f1/current").status_code == 503   # too old now
