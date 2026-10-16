"""Region rotation: calendar, region detection, send gating, follow-up review."""

import os
import tempfile
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
import pytest_asyncio

from harvey import research, schedule
from harvey.agents.followups import FollowupReviewer
from harvey.agents.sender import FOLLOWUP_REVIEW_PREFIX
from harvey.config import ScheduleConfig
from harvey.models.prospect import Prospect
from harvey.state import StateManager
from tests.test_outbox_native import Cfg, FakeProvider, make_sender

MON_930_NY = datetime(2026, 10, 19, 13, 30, tzinfo=timezone.utc)
TUE_930_LONDON = datetime(2026, 10, 20, 8, 30, tzinfo=timezone.utc)
SAT_NOON = datetime(2026, 10, 24, 12, 0, tzinfo=timezone.utc)


@pytest_asyncio.fixture
async def state():
    with tempfile.TemporaryDirectory() as tmpdir:
        sm = StateManager(os.path.join(tmpdir, "test.db"))
        await sm.init_db()
        yield sm


@pytest.fixture
def at(monkeypatch):
    def _set(moment):
        monkeypatch.setattr(schedule, "_now", lambda now=None: now or moment)
    return _set


def _sched(**kw):
    return ScheduleConfig(enabled=True, **kw)


# ── Calendar ───────────────────────────────────────────────────────

def test_rotation_follows_weekdays_in_each_regions_timezone():
    s = _sched()
    assert schedule.todays_regions(s, MON_930_NY) == ["us"]
    assert schedule.is_sending_now(s, "us", MON_930_NY)
    assert not schedule.is_sending_now(s, "uk", MON_930_NY)
    assert schedule.is_sending_now(s, "uk", TUE_930_LONDON)
    assert not schedule.is_sending_now(s, "us", TUE_930_LONDON)
    assert schedule.todays_regions(s, SAT_NOON) == []
    assert schedule.is_rest_day(s, "America/New_York", SAT_NOON)
    assert schedule.next_region(s, SAT_NOON) == "us"  # Monday


def test_window_closes():
    s = _sched()
    assert not schedule.is_sending_now(s, "us", MON_930_NY + timedelta(hours=3))
    assert schedule.todays_regions(s, MON_930_NY + timedelta(hours=3)) == []


def test_region_detection():
    s = _sched()
    assert schedule.classify_text("Acme Ltd, London SW1A 2AA, +44 20 7946 0000", s) == "uk"
    assert schedule.classify_text("Muster GmbH, Berlin, +49 30 1234", s) == "europe"
    assert schedule.classify_text("Acme Inc., Austin, TX 78701", s) == "us"
    assert schedule.classify_text("We build apps for clients worldwide", s) == ""
    assert [schedule.classify_domain(d) for d in ("a.co.uk", "a.de", "a.io")] == ["uk", "europe", ""]


@pytest.mark.asyncio
async def test_region_is_cached_and_unknown_waits_for_a_website_check(state):
    s = _sched()
    p = Prospect(first_name="A", last_name="B", title="CTO", company="Acme", email="a@acme.io")
    assert await schedule.region_of(state, p, None, s, fetch=False) == ""
    assert await state.get_setting("region:Acme") == ""  # not cached as unknown yet
    await state.set_setting("region:Acme", "uk")
    assert await schedule.region_of(state, p, None, s) == "uk"


# ── Sender gating ──────────────────────────────────────────────────

async def _queue(state, region, step=1, status="approved"):
    p = Prospect(first_name="Jo", last_name=region, title="CTO", company=f"Co-{region}",
                 email=f"jo@{region}-co.com", email_status="verified")
    pid = await state.add_prospect(p)
    await state.set_setting(f"region:Co-{region}", region)
    past = (datetime.now(timezone.utc) - timedelta(hours=1)).replace(tzinfo=None).isoformat()
    item = await state.add_outbox_item(
        prospect_id=pid, campaign_id=f"c-{region}-{step}", step=step,
        to_email=p.email, subject="hello there", body="A short plain note. Worth a call?",
        send_at=past, status=status)
    return pid, item


def _rotating_sender(state, provider, **kw):
    sender = make_sender(state, provider, require_approval=False)
    sender.config.schedule = _sched(**kw)
    return sender


@pytest.mark.asyncio
async def test_sender_only_sends_todays_region_in_its_window(state, at):
    provider = FakeProvider()
    sender = _rotating_sender(state, provider)
    await _queue(state, "us")
    await _queue(state, "uk")

    at(MON_930_NY)
    await sender._drain_due()
    assert [m["to"] for m in provider.sent] == ["jo@us-co.com"]

    at(TUE_930_LONDON)
    await sender._drain_due()
    assert [m["to"] for m in provider.sent] == ["jo@us-co.com", "jo@uk-co.com"]


@pytest.mark.asyncio
async def test_followups_need_a_review_and_respect_caps(state, at):
    provider = FakeProvider()
    sender = _rotating_sender(state, provider, max_daily_new=1)
    _, fu = await _queue(state, "us", step=2)
    await _queue(state, "us")
    pid3, _ = await _queue(state, "us", step=1)  # second new email: over the cap

    at(MON_930_NY)
    await sender._drain_due()
    assert len(provider.sent) == 1  # one new email; follow-up unreviewed

    await state.set_setting(FOLLOWUP_REVIEW_PREFIX + fu, "send:7")
    await sender._drain_due()
    assert len(provider.sent) == 2 and provider.sent[-1]["to"] == "jo@us-co.com"


# ── Follow-up reviewer ─────────────────────────────────────────────

class JsonBrain:
    def __init__(self, result):
        self.result = result

    async def think_json(self, *a, **kw):
        return self.result


async def _sent_then_followups(state):
    pid, first = await _queue(state, "us", step=1)
    await state.update_outbox_item(first, status="sent", sent_at=datetime.now().isoformat())
    _, fu2 = await _queue(state, "us", step=2, status="pending_review")
    await state.update_outbox_item(fu2, body="old body")
    return pid, fu2


def _reviewer(state, brain):
    cfg = Cfg()
    cfg.schedule = _sched()
    return FollowupReviewer(brain, state, cfg, SimpleNamespace(serper_api_key=""))


@pytest.mark.asyncio
async def test_reviewer_cancels_low_chance(state, monkeypatch):
    async def no_research(*a, **kw):
        return ""
    monkeypatch.setattr(research, "research_prospect", no_research)
    pid, fu = await _sent_then_followups(state)
    # Same prospect id across _queue calls (dedup by email), so items group.
    await _reviewer(state, JsonBrain({"chance": 2, "worth_following_up": False,
                                      "reason": "no need"})).run()
    item = await state.get_outbox_item(fu)
    assert item["status"] == "cancelled"
    assert (await state.get_setting(FOLLOWUP_REVIEW_PREFIX + fu)).startswith("skip")


@pytest.mark.asyncio
async def test_reviewer_approves_and_rewrites(state, monkeypatch):
    async def no_research(*a, **kw):
        return "[https://x] They are hiring three backend engineers."
    monkeypatch.setattr(research, "research_prospect", no_research)
    pid, fu = await _sent_then_followups(state)
    await _reviewer(state, JsonBrain({
        "chance": 7, "worth_following_up": True, "reason": "hiring",
        "emails": [{"step": 2, "subject": "the backend roles", "body": "New short note. Open to it?"}],
    })).run()
    item = await state.get_outbox_item(fu)
    assert item["status"] == "pending_review" and item["body"] == "New short note. Open to it?"
    assert await state.get_setting(FOLLOWUP_REVIEW_PREFIX + fu) == "send:7"
