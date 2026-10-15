"""Tellwing gets only people who said yes; cold email carries the footer;
quoted text never decides intent."""

import json

import httpx
import pytest

from harvey.gate import pre_send_check
from harvey.integrations.mail_provider import InboundMessage, strip_quoted_reply
from harvey.integrations.tellwing import TellwingClient
from harvey.models.prospect import Prospect
from tests.test_outbox_native import (  # noqa: F401  (state is a fixture)
    Cfg,
    Env,
    FakeProvider,
    make_sender,
    seed_campaign,
    seed_prospect,
    state,
)

ADDRESS = Cfg.compliance.postal_address
LANDING = "https://tellwing.site/acme/websites"
QUESTION = "Mind if I send you a case study now and then?"

QUOTED_ORIGINAL = (
    "\n\nOn Tue, Oct 14, 2026 at 3:02 PM Harvey <harvey@x.co> wrote:\n"
    "> Hi Jane, noticed your booking page sends people to a phone number.\n"
    f"> {ADDRESS}\n"
    '> Not interested? Reply "no thanks" and I won\'t email again.\n'
)


class TwCfg(Cfg):
    class tellwing:
        enabled = True
        list_id = 7
        landing_url = LANDING
        permission_question = QUESTION


class TwEnv(Env):
    tellwing_api_key = "tw_live_test"


class TwBrain:
    paused_reason = ""

    def __init__(self, intent="interested", consent=None):
        self.intent = intent
        self.consent = consent
        self.consent_calls = 0

    def load_skills_for_agent(self, *_):
        return ""

    def load_prompt(self, *_a, **_k):
        return ""

    async def think(self, prompt, session_id=None, agent="", task="", **_):
        if task == "classify_intent":
            return self.intent
        return "Happy to. Does Tuesday at 10 work for a quick call?"

    async def think_json(self, prompt, session_id=None, agent="", task=""):
        self.consent_calls += 1
        return self.consent


class TellwingRecorder:
    """httpx.MockTransport that records requests and replays canned answers."""

    def __init__(self, *responses):
        self.requests = []
        self.responses = list(responses) or [(201, {"data": {"id": "sub_1"}})]

    def __call__(self, request):
        self.requests.append(request)
        status, body = self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]
        return httpx.Response(status, json=body)


def make_tw_handler(state, provider, brain, recorder):
    from harvey.agents.handler import Handler
    handler = Handler(brain=brain, state=state, config=TwCfg(), env=TwEnv())
    handler.provider = provider
    handler.tellwing = TellwingClient(
        "tw_live_test", 7, transport=httpx.MockTransport(recorder)
    )
    return handler


def inbound(n, body, email="jane@acme.com"):
    return InboundMessage(
        provider_id=f"r{n}", from_email=email, subject="Re: your booking page",
        body=body, message_id=f"<r{n}@acme.com>",
    )


async def reply_items(state):
    return [i for i in await state.get_outbox() if i["kind"] == "reply"]


# ── quoted text ──


def test_strip_quoted_reply_cuts_at_the_quote():
    assert strip_quoted_reply("Sounds good." + QUOTED_ORIGINAL) == "Sounds good."
    assert strip_quoted_reply("sure\r\n\r\nFrom: Harvey\r\nSent: Tue\r\nno thanks") == "sure"
    assert strip_quoted_reply("On second thought, yes.\nThanks") == "On second thought, yes.\nThanks"


@pytest.mark.asyncio
async def test_quoted_footer_is_not_an_opt_out(state):
    pid = await seed_prospect(state, status="contacted")
    provider = FakeProvider()
    provider.inbound = [inbound(1, "Sounds interesting, tell me more." + QUOTED_ORIGINAL)]
    handler = make_tw_handler(state, provider, TwBrain("interested"), TellwingRecorder())

    assert await handler.run() is True

    assert (await state.get_prospect(pid)).status == "replied"


@pytest.mark.asyncio
async def test_no_thanks_reply_is_an_opt_out(state):
    pid = await seed_prospect(state, status="contacted")
    provider = FakeProvider()
    provider.inbound = [inbound(1, "No thanks." + QUOTED_ORIGINAL)]
    handler = make_tw_handler(state, provider, TwBrain("interested"), TellwingRecorder())

    await handler.run()

    assert (await state.get_prospect(pid)).status == "opted_out"
    assert await reply_items(state) == []


# ── the permission question and the yes ──


@pytest.mark.asyncio
async def test_interested_reply_asks_permission_once(state):
    pid = await seed_prospect(state, status="contacted")
    provider = FakeProvider()
    provider.inbound = [inbound(1, "Tell me more." + QUOTED_ORIGINAL)]
    handler = make_tw_handler(state, provider, TwBrain("interested"), TellwingRecorder())

    await handler.run()

    [reply] = await reply_items(state)
    assert reply["body"].endswith(QUESTION)
    consent = await state.get_tellwing_consent(pid)
    assert consent["status"] == "asked"
    assert consent["ask_outbox_id"] == reply["id"]

    # A second interested reply never asks again
    provider.inbound = [inbound(2, "Also, what do you charge?")]
    await handler.run()
    second = [r for r in await reply_items(state) if r["id"] != reply["id"]]
    assert second and QUESTION not in second[0]["body"]


@pytest.mark.asyncio
async def test_guessed_address_is_never_asked(state):
    pid = await seed_prospect(state, status="contacted", estatus="guess")
    provider = FakeProvider()
    provider.inbound = [inbound(1, "Tell me more.")]
    handler = make_tw_handler(state, provider, TwBrain("interested"), TellwingRecorder())

    await handler.run()

    assert await state.get_tellwing_consent(pid) is None
    [reply] = await reply_items(state)
    assert QUESTION not in reply["body"]


async def ask_and_send(state, handler, provider):
    """Run an interested reply through, then mark the question as sent."""
    provider.inbound = [inbound(1, "Tell me more.")]
    await handler.run()
    [reply] = await reply_items(state)
    await state.update_outbox_item(reply["id"], status="sent")


@pytest.mark.asyncio
async def test_yes_adds_them_to_tellwing_with_their_words(state):
    pid = await seed_prospect(state, status="contacted")
    provider = FakeProvider()
    recorder = TellwingRecorder()
    brain = TwBrain("interested", consent={
        "answer": "yes", "quote": "Yes, send me the case studies.",
    })
    handler = make_tw_handler(state, provider, brain, recorder)
    await ask_and_send(state, handler, provider)

    provider.inbound = [inbound(2, "Yes, send me the case studies. Tuesday works." + QUOTED_ORIGINAL)]
    await handler.run()

    consent = await state.get_tellwing_consent(pid)
    assert consent["status"] == "subscribed"
    assert consent["subscriber_id"] == "sub_1"

    [req] = recorder.requests
    assert req.method == "POST"
    assert str(req.url) == "https://api.tellwing.com/v1/lists/7/subscribers"
    assert req.headers["Authorization"] == "Bearer tw_live_test"
    assert req.headers["Idempotency-Key"] == consent["idempotency_key"]
    payload = json.loads(req.content)
    assert payload["email"] == "jane@acme.com"
    assert payload["name"] == "Jane"
    assert payload["consent_source"] == "api_other"
    assert '"Yes, send me the case studies."' in payload["consent_text"]
    assert QUESTION in payload["consent_text"]
    assert "resubscribe" not in payload


@pytest.mark.asyncio
async def test_invented_quote_falls_back_to_their_reply(state):
    pid = await seed_prospect(state, status="contacted")
    provider = FakeProvider()
    brain = TwBrain("interested", consent={"answer": "yes", "quote": "Absolutely, sign me up!"})
    handler = make_tw_handler(state, provider, brain, TellwingRecorder())
    await ask_and_send(state, handler, provider)

    provider.inbound = [inbound(2, "sure, go ahead")]
    await handler.run()

    consent = await state.get_tellwing_consent(pid)
    assert '"sure, go ahead"' in consent["consent_text"]
    assert "sign me up" not in consent["consent_text"]


@pytest.mark.asyncio
async def test_unsent_question_cannot_be_answered(state):
    pid = await seed_prospect(state, status="contacted")
    provider = FakeProvider()
    recorder = TellwingRecorder()
    brain = TwBrain("interested", consent={"answer": "yes", "quote": "yes"})
    handler = make_tw_handler(state, provider, brain, recorder)
    provider.inbound = [inbound(1, "Tell me more.")]
    await handler.run()  # question queued, still pending your approval

    provider.inbound = [inbound(2, "yes")]
    await handler.run()

    assert brain.consent_calls == 0
    assert (await state.get_tellwing_consent(pid))["status"] == "asked"
    assert recorder.requests == []


@pytest.mark.asyncio
async def test_no_to_the_question_is_final(state):
    pid = await seed_prospect(state, status="contacted")
    provider = FakeProvider()
    recorder = TellwingRecorder()
    handler = make_tw_handler(state, provider, TwBrain("interested", consent={"answer": "no"}), recorder)
    await ask_and_send(state, handler, provider)

    provider.inbound = [inbound(2, "Nah, just the call please.")]
    await handler.run()

    assert (await state.get_tellwing_consent(pid))["status"] == "declined"
    assert recorder.requests == []


@pytest.mark.asyncio
async def test_opt_out_withdraws_a_pending_yes(state):
    pid = await seed_prospect(state, status="contacted")
    await state.add_tellwing_ask(pid, "item1", QUESTION)
    await state.update_tellwing_consent(
        pid, status="granted", consent_text="yes", consent_at="2026-10-14T15:00:00+00:00",
        idempotency_key="k1",
    )
    provider = FakeProvider()
    recorder = TellwingRecorder()
    provider.inbound = [inbound(1, "Actually, please stop emailing me.")]
    handler = make_tw_handler(state, provider, TwBrain("interested"), recorder)

    await handler.run()

    assert (await state.get_tellwing_consent(pid))["status"] == "declined"
    assert recorder.requests == []


@pytest.mark.asyncio
async def test_tellwing_refusal_and_retry(state):
    unsub = await seed_prospect(state, email="unsub@acme.com", status="replied")
    flaky = await seed_prospect(state, email="flaky@acme.com", status="replied")
    for pid, key in ((unsub, "k-unsub"), (flaky, "k-flaky")):
        await state.add_tellwing_ask(pid, "item", QUESTION)
        await state.update_tellwing_consent(
            pid, status="granted", consent_text="yes",
            consent_at="2026-10-14T15:00:00+00:00", idempotency_key=key,
        )
    recorder = TellwingRecorder(
        (422, {"error": {"code": "recipient_unsubscribed", "message": "unsubscribed"}}),
        (503, {"error": {"code": "internal_error", "message": "try later"}}),
    )
    handler = make_tw_handler(state, FakeProvider(), TwBrain(), recorder)

    await handler._sync_tellwing()

    assert (await state.get_tellwing_consent(unsub))["status"] == "blocked"
    retry = await state.get_tellwing_consent(flaky)
    assert retry["status"] == "granted" and retry["attempts"] == 1


# ── cold email footer, landing link, gate ──


def _prospect():
    return Prospect(first_name="J", last_name="D", title="VP",
                    email="jane@acme.com", email_status="verified")


def test_gate_requires_postal_address_on_cold_email():
    body = f"Saw your booking page.\n\n{ADDRESS}"
    assert pre_send_check("jane@acme.com", "booking", body, prospect=_prospect(),
                          postal_address=ADDRESS)
    missing = pre_send_check("jane@acme.com", "booking", "Saw it.", prospect=_prospect(),
                             postal_address=ADDRESS)
    assert any("postal address" in r for r in missing.reasons)
    unset = pre_send_check("jane@acme.com", "booking", "Saw it.", prospect=_prospect(),
                           postal_address="")
    assert any("compliance.postal_address" in r for r in unset.reasons)
    # Replies to someone who wrote to us aren't cold email
    assert pre_send_check("jane@acme.com", "Re: booking", "Tuesday works.",
                          prospect=_prospect(), kind="reply", postal_address=None)


def test_gate_allows_only_the_landing_page_link():
    ok = f"Here's what I mean: {LANDING}.\n\n{ADDRESS}"
    assert pre_send_check("jane@acme.com", "booking", ok, prospect=_prospect(),
                          postal_address=ADDRESS, landing_url=LANDING)
    bad = f"Here's what I mean: https://tellwing.site/acme/websites.evil.com\n\n{ADDRESS}"
    result = pre_send_check("jane@acme.com", "booking", bad, prospect=_prospect(),
                            postal_address=ADDRESS, landing_url=LANDING)
    assert any("landing page" in r for r in result.reasons)


@pytest.mark.asyncio
async def test_sent_cold_email_carries_footer_and_landing_link(state):
    pid = await seed_prospect(state)
    campaign = await seed_campaign(state, [pid])
    campaign.sequence[0].body = "See {{landing_url}} for {{company}}."
    await state.update_campaign(campaign.id, sequence_json=campaign.sequence_json())
    provider = FakeProvider()
    sender = make_sender(state, provider, require_approval=False)
    sender.config = TwCfg()

    await sender._run_native()

    [sent] = provider.sent
    assert sent["body"] == (
        f"See {LANDING} for Acme.\n\n{ADDRESS}\n"
        'Not interested? Reply "no thanks" and I won\'t email again.'
    )
    [row] = await state.get_outbox(status="sent")
    assert row["body"] == sent["body"]  # the record matches what left


@pytest.mark.asyncio
async def test_cold_email_waits_for_a_postal_address(state):
    pid = await seed_prospect(state)
    await seed_campaign(state, [pid])
    provider = FakeProvider()
    sender = make_sender(state, provider, require_approval=False)

    class NoAddress(Cfg):
        class compliance:
            postal_address = ""
            opt_out_line = ""
    sender.config = NoAddress()

    await sender._run_native()

    assert provider.sent == []
    assert len(await state.get_outbox(status="approved")) == 2  # held, not failed
