"""Warmup ramp, deliverability checks, deep-research writer and the
Windows Claude CLI resolution."""

from datetime import date
from types import SimpleNamespace

import pytest

import harvey.brain as brain_mod
from harvey import deliverability as d
from harvey.agents import writer as writer_mod
from harvey.config import EmailChannelConfig, WarmupConfig


# ── Warmup ramp ────────────────────────────────────────────────────

def test_no_start_date_means_no_ramp():
    assert WarmupConfig().daily_cap(30) == 30
    assert EmailChannelConfig(max_daily_sends=12).effective_daily_sends() == 12


def test_ramp_holds_then_climbs_to_ceiling():
    w = WarmupConfig(start_date="2026-10-01", min_days=14, start_sends=5, daily_increase=3)
    cap = lambda day: w.daily_cap(30, date(2026, 10, day))
    assert [cap(1), cap(14)] == [0, 0]          # warming only
    assert [cap(15), cap(16), cap(20)] == [5, 8, 20]
    assert cap(31) == 30                         # never above max_daily_sends


def test_bad_start_date_rejected():
    with pytest.raises(ValueError):
        WarmupConfig(start_date="10/01/2026")


# ── Deliverability checks (DNS stubbed) ───────────────────────────

def _txt(records):
    return lambda name: records.get(name, [])


def test_spf_checks(monkeypatch):
    monkeypatch.setattr(d, "_txt", _txt({"a.com": ["v=spf1 include:_spf.google.com ~all"]}))
    assert d.check_spf("a.com").status == d.OK
    monkeypatch.setattr(d, "_txt", _txt({"a.com": ["v=spf1 +all"]}))
    assert d.check_spf("a.com").status == d.FAIL
    monkeypatch.setattr(d, "_txt", _txt({"a.com": ["v=spf1 -all", "v=spf1 ~all"]}))
    assert d.check_spf("a.com").status == d.FAIL
    monkeypatch.setattr(d, "_txt", _txt({}))
    assert d.check_spf("a.com").status == d.FAIL


def test_dmarc_and_dkim(monkeypatch):
    monkeypatch.setattr(d, "_txt", _txt({
        "_dmarc.a.com": ["v=DMARC1; p=none; rua=mailto:x@a.com"],
        "google._domainkey.a.com": ["v=DKIM1; k=rsa; p=MIIBIjANBg"],
        "s1._domainkey.b.com": ["v=DKIM1; p="],  # revoked key
    }))
    assert d.check_dmarc("a.com").status == d.WARN
    assert d.check_dmarc("b.com").status == d.FAIL
    assert d.check_dkim("a.com").status == d.OK
    assert d.check_dkim("b.com", "s1").status == d.FAIL


def test_blocklist_refusal_is_not_a_listing(monkeypatch):
    answers = {"a.com.dbl.spamhaus.org": ["127.255.255.254"],
               "a.com.multi.surbl.org": ["127.0.0.2"]}
    monkeypatch.setattr(d, "_a", lambda name: answers.get(name, []))
    by_zone = {c.name.split()[-1]: c.status for c in d.check_domain_blocklists("a.com")}
    assert by_zone == {"dbl.spamhaus.org": d.INFO, "multi.surbl.org": d.FAIL,
                       "multi.uribl.com": d.OK}


def test_warmup_check_reports_progress():
    w = WarmupConfig(start_date="2026-10-01", min_days=14)
    assert d.check_warmup(w, 30, date(2026, 10, 5)).status == d.FAIL
    assert d.check_warmup(w, 30, date(2026, 10, 20)).status == d.OK
    assert d.check_warmup(WarmupConfig(), 30).status == d.WARN


# ── Deep research writer ───────────────────────────────────────────

class FakeState:
    def __init__(self):
        self.settings, self.statuses, self.campaigns = {}, {}, []

    async def get_setting(self, key, default=""):
        return self.settings.get(key, default)

    async def increment_setting(self, key, by=1):
        self.settings[key] = str(int(self.settings.get(key, "0")) + by)
        return int(self.settings[key])

    async def update_prospect_status(self, pid, status):
        self.statuses[pid] = status

    async def add_campaign(self, campaign):
        self.campaigns.append(campaign)
        return f"c{len(self.campaigns)}"

    async def log_action(self, **kwargs):
        pass


def _prospect(pid, score):
    return SimpleNamespace(id=pid, score=score, email=f"{pid}@x.com", company=pid,
                           full_name=lambda: pid.title())


def _writer(state, max_daily_sends):
    config = SimpleNamespace(channels=SimpleNamespace(email=SimpleNamespace(
        max_daily_sends=max_daily_sends, provider="smtp", deep_research=True)))
    return writer_mod.Writer(brain=None, state=state, config=config)


@pytest.mark.asyncio
async def test_deep_mode_quota_and_no_hook(monkeypatch):
    state = FakeState()
    w = _writer(state, max_daily_sends=6)  # 6 sends / 3 emails = 2 people a day
    step = writer_mod.EmailStep(step=1, subject="s", body="b", delay_days=0)

    async def fake_research(prospect):
        if prospect.id == "nohook":
            return [], "no specific hook"
        return [step], ""

    monkeypatch.setattr(w, "_write_researched_sequence", fake_research)
    prospects = [_prospect("a", 50), _prospect("nohook", 90), _prospect("b", 70),
                 _prospect("c", 10)]
    await w._run_deep(prospects)

    assert state.statuses == {"nohook": "no_hook", "b": "queued", "a": "queued"}
    assert [c.prospect_ids for c in state.campaigns] == [["b"], ["a"]]

    await w._run_deep([_prospect("c", 10)])  # quota used: nothing more today
    assert "c" not in state.statuses


# ── Claude CLI resolution ──────────────────────────────────────────

def test_windows_npm_shim_resolves_to_native_exe(monkeypatch, tmp_path):
    shim = tmp_path / "claude.cmd"
    shim.write_text("")
    native = tmp_path / "node_modules" / "@anthropic-ai" / "claude-code" / "bin" / "claude.exe"
    native.parent.mkdir(parents=True)
    native.write_text("")
    monkeypatch.setattr(brain_mod.sys, "platform", "win32")
    monkeypatch.setattr(brain_mod.shutil, "which", lambda name: str(shim))
    assert brain_mod._claude_executable() == str(native)

    native.unlink()  # never fall back to running prompts through cmd.exe
    assert brain_mod._claude_executable() == "claude"


def test_non_windows_uses_path_lookup(monkeypatch):
    monkeypatch.setattr(brain_mod.sys, "platform", "linux")
    monkeypatch.setattr(brain_mod.shutil, "which", lambda name: "/usr/local/bin/claude")
    assert brain_mod._claude_executable() == "/usr/local/bin/claude"
