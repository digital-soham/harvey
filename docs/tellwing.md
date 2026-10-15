# Cold email footer, and handing yes-replies to Tellwing

Two related features:

1. **The compliance footer.** Every cold (sequence) email carries a postal
   address and an opt-out line.
2. **The Tellwing handoff.** A prospect who explicitly says yes to occasional
   email is added to a [Tellwing](https://tellwing.com) list, with their own
   words recorded as consent.

## The footer (CAN-SPAM)

```yaml
compliance:
  postal_address: "123 Main St, Austin, TX 78701"
  opt_out_line: 'Not interested? Reply "no thanks" and I won''t email again.'
```

- **When it's added.** The footer is appended at send time to sequence
  emails only. Replies to someone who wrote to you don't get it.
- **Without an address, nothing goes out.** If `postal_address` is empty,
  cold email is **held** in the Outbox, not failed. It goes out once the
  address is set.
- **The gate enforces it.** The deterministic pre-send gate rejects a
  sequence email whose final body doesn't contain the address.
- **What counts as an opt-out.** A reply of "no thanks" or "no thank you"
  is an immediate, permanent opt-out, alongside "unsubscribe", "remove me"
  and the rest.

### Quoted text never decides intent

Before a reply is classified, Harvey cuts the quoted original out of it
(`On … wrote:`, Outlook's `From:/Sent:` block, `>` lines). Otherwise every
reply would quote the footer's "no thanks" and read as an opt-out.

## The Tellwing handoff

Tellwing never sends cold email. Its terms forbid scraped, guessed or bought
addresses, and its sending infrastructure is shared. Cold email goes out from
your own mailbox. Tellwing only gets people who said yes.

```yaml
tellwing:
  enabled: true
  list_id: 12                                    # numeric Tellwing list id
  landing_url: "https://tellwing.site/you/offer" # optional
  permission_question: "Separately, mind if I send you a short case study now and then? A simple yes or no is fine."
```

`.env`: `TELLWING_API_KEY=tw_live_…`. Give the key **only** the
`subscribers:write` scope. The API needs Tellwing's Growth plan.

### The flow

1. **Landing page.** If `landing_url` is set, it's the only link a cold
   email may carry. The writer types `{{landing_url}}` and Harvey fills it
   in. The gate blocks any other URL. Sign-ups there are Tellwing's own
   `website_form` consent.
2. **The question is asked once.** When a reply is classified as
   *interested*, the permission question is appended to Harvey's answer.
   Only once per person, and only to a verified address.
3. **The answer only counts if the question was sent.** A later reply is
   checked for a yes only if the outbox item that asked it was actually
   **sent**, with the question still in its body. If you rejected or edited
   it in the Outbox, there's nothing to answer.
4. **A yes needs their own words.** The model must return the exact sentence
   where they agree. If that sentence isn't in the reply verbatim, Harvey
   stores the reply itself instead. Invented words are never stored.
5. **Sync.** Each cycle, granted consents are sent to:

   ```
   POST https://api.tellwing.com/v1/lists/{list_id}/subscribers
   ```

   - The body has `consent_source: "api_other"`, the consent timestamp, and
     `consent_text` with the question and their verbatim answer.
   - Each person keeps one `Idempotency-Key`, so retries never duplicate.
   - `resubscribe` is never sent, so anyone who unsubscribed from Tellwing
     stays unsubscribed.

### Who is never added

- people who didn't reply
- people who said no to the question, or weren't interested
- people who opted out, at any point (an opt-out withdraws a pending yes)
- addresses that aren't `verified` or `risky` (never a guess)

If someone opts out **after** joining the Tellwing list, Harvey doesn't
delete them from Tellwing. It logs an escalation so you can unsubscribe
them there.

### Consent states

Stored in the `tellwing_consent` table (migration v12), one row per prospect:

| Status | Meaning |
|---|---|
| `asked` | the question was queued |
| `granted` | they said yes; waiting to sync |
| `subscribed` | added to Tellwing (or already there, left as it was) |
| `declined` | said no, not interested, or opted out. Final. |
| `blocked` | Tellwing refused (`recipient_unsubscribed` / `recipient_suppressed`), or the address isn't deliverable |

How sync failures are handled:

- **Network errors, 429 and 5xx** keep the row `granted` and retry next
  cycle (`attempts`, `last_error`).
- **Setup errors** (`unauthorized`, `forbidden_scope`, `plan_required`,
  `not_found`) are logged loudly and retried once fixed.

## Files

| File | What it is |
|---|---|
| `harvey/integrations/tellwing.py` | the one-endpoint Tellwing client |
| `harvey/agents/handler.py` | asking, reading the answer, sync |
| `harvey/agents/sender.py` | footer and `{{landing_url}}` at send time |
| `harvey/gate.py` | postal address and landing-link checks |
| `harvey/integrations/mail_provider.py` | `strip_quoted_reply` |
| `tests/test_tellwing.py` | the whole flow, Tellwing faked with `httpx.MockTransport` |
