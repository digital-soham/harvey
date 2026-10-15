"""Tellwing — where prospects go once they've said yes, never before.

Cold email is never sent through Tellwing (its terms forbid scraped,
guessed or bought addresses, and its sending account is shared by every
Tellwing customer). Harvey only calls one endpoint, to add someone who
replied yes to the permission question, with the exact words they used:

    POST https://api.tellwing.com/v1/lists/{id}/subscribers

Docs: https://tellwing.com/docs. ``resubscribe`` is never sent, so anyone
who already unsubscribed from Tellwing stays unsubscribed.
"""

import logging
from dataclasses import dataclass

import httpx

logger = logging.getLogger("harvey.tellwing")

API_BASE = "https://api.tellwing.com/v1"

# The recipient refuses mail: never retry, never resubscribe.
BLOCKED_CODES = {"recipient_unsubscribed", "recipient_suppressed"}


@dataclass
class TellwingResult:
    # subscribed | existing | blocked | retry | error
    outcome: str
    code: str = ""
    message: str = ""
    subscriber_id: str = ""


class TellwingClient:
    def __init__(self, api_key: str, list_id: int, transport=None):
        self.api_key = api_key
        self.list_id = int(list_id or 0)
        self._transport = transport  # httpx.MockTransport in tests

    def is_configured(self) -> bool:
        return bool(self.api_key and self.list_id > 0)

    async def add_subscriber(
        self,
        *,
        email: str,
        consent_text: str,
        consent_timestamp: str,
        idempotency_key: str,
        name: str = "",
        last_name: str = "",
    ) -> TellwingResult:
        payload = {
            "email": email,
            "consent_source": "api_other",
            "consent_text": consent_text,
            "consent_timestamp": consent_timestamp,
        }
        if name:
            payload["name"] = name
        if last_name:
            payload["last_name"] = last_name
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Idempotency-Key": idempotency_key,
        }
        url = f"{API_BASE}/lists/{self.list_id}/subscribers"
        try:
            async with httpx.AsyncClient(
                timeout=20, transport=self._transport
            ) as client:
                resp = await client.post(url, json=payload, headers=headers)
        except httpx.HTTPError as e:
            return TellwingResult("retry", message=f"network error: {e}"[:300])

        try:
            data = resp.json()
        except ValueError:
            data = {}
        if resp.status_code in (200, 201):
            sid = ""
            if isinstance(data, dict):
                sid = str((data.get("data") or data).get("id") or "")
            # 200: already in Tellwing; left as it was (never resubscribed).
            outcome = "subscribed" if resp.status_code == 201 else "existing"
            return TellwingResult(outcome, subscriber_id=sid)

        err = data.get("error", {}) if isinstance(data, dict) else {}
        code = str(err.get("code") or "")
        message = str(err.get("message") or resp.text[:200])
        if code in BLOCKED_CODES:
            return TellwingResult("blocked", code=code, message=message)
        if resp.status_code == 429 or resp.status_code >= 500:
            return TellwingResult("retry", code=code or str(resp.status_code), message=message)
        # unauthorized, forbidden_scope, plan_required, not_found, ... —
        # a setup problem: kept for retry once it's fixed, and logged loudly.
        return TellwingResult("error", code=code or str(resp.status_code), message=message)
