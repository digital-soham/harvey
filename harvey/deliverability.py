"""Deliverability readiness - is this sending domain safe to cold email from?

Everything here is free and read-only: public DNS, RDAP, DNS blocklists, and
Harvey's own send history. None of it can measure inbox reputation directly
(only mailbox providers know that), so the report also points at the two
things that can: Google Postmaster Tools and an inbox-placement test.
"""

from dataclasses import dataclass
from datetime import date, datetime

import dns.resolver
import httpx

OK, WARN, FAIL, INFO = "ok", "warn", "fail", "info"

# Selectors used by the common mailbox providers, tried in order when the
# operator hasn't named one (warmup.dkim_selector).
COMMON_DKIM_SELECTORS = (
    "google", "default", "selector1", "selector2", "k1", "k2", "s1", "s2",
    "mail", "dkim", "fm1", "fm2", "fm3", "zoho", "zmail", "mxvault",
    "protonmail", "protonmail2", "mandrill", "smtp", "sig1", "everlytickey1",
)

# (zone, how a "query refused" answer looks) - public resolvers like 8.8.8.8
# are blocked by these lists and get a sentinel answer instead of a verdict.
DOMAIN_BLOCKLISTS = (
    ("dbl.spamhaus.org", lambda ip: ip.startswith("127.255.255.")),
    ("multi.surbl.org", lambda ip: ip == "127.0.0.1"),
    ("multi.uribl.com", lambda ip: ip == "127.0.0.1"),
)
IP_BLOCKLISTS = (
    ("zen.spamhaus.org", lambda ip: ip.startswith("127.255.255.")),
    ("bl.spamcop.net", lambda ip: False),
)

MIN_DOMAIN_AGE_DAYS = 30
GOOD_DOMAIN_AGE_DAYS = 90
MAX_BOUNCE_RATE = 0.03
MIN_SENDS_FOR_RATES = 20


@dataclass
class Check:
    name: str
    status: str
    detail: str


def _txt(name: str) -> list[str]:
    try:
        answers = dns.resolver.resolve(name, "TXT", lifetime=8)
    except (dns.resolver.NXDOMAIN, dns.resolver.NoAnswer, dns.resolver.NoNameservers):
        return []
    return [b"".join(r.strings).decode("utf-8", "replace") for r in answers]


def _a(name: str) -> list[str]:
    try:
        return [r.address for r in dns.resolver.resolve(name, "A", lifetime=8)]
    except (dns.resolver.NXDOMAIN, dns.resolver.NoAnswer, dns.resolver.NoNameservers):
        return []


def check_mx(domain: str) -> Check:
    try:
        mx = sorted(dns.resolver.resolve(domain, "MX", lifetime=8), key=lambda r: r.preference)
    except (dns.resolver.NXDOMAIN, dns.resolver.NoAnswer, dns.resolver.NoNameservers):
        return Check("MX", FAIL, "No MX record - replies and bounces have nowhere to go.")
    return Check("MX", OK, ", ".join(str(r.exchange).rstrip(".") for r in mx[:3]))


def check_spf(domain: str) -> Check:
    records = [t for t in _txt(domain) if t.lower().startswith("v=spf1")]
    if not records:
        return Check("SPF", FAIL, "No SPF record. Add the TXT record your mail provider gives you.")
    if len(records) > 1:
        return Check("SPF", FAIL, f"{len(records)} SPF records - receivers treat that as no SPF. Merge them.")
    spf = records[0]
    terms = spf.lower().split()
    if "+all" in terms or "all" in terms:
        return Check("SPF", FAIL, f"'+all' lets anyone send as you: {spf}")
    lookups = sum(t.lstrip("+-~?").split(":")[0].split("=")[0] in
                  ("include", "a", "mx", "ptr", "exists", "redirect") for t in terms)
    if lookups > 10:
        return Check("SPF", FAIL, f"{lookups} DNS lookups (limit is 10) - SPF will permerror.")
    if "?all" in terms:
        return Check("SPF", WARN, f"'?all' is neutral; use '~all' or '-all': {spf}")
    return Check("SPF", OK, spf)


def check_dmarc(domain: str) -> Check:
    records = [t for t in _txt(f"_dmarc.{domain}") if t.lower().startswith("v=dmarc1")]
    if not records:
        return Check("DMARC", FAIL, "No DMARC record. Gmail and Yahoo require one for bulk senders. "
                     f"Start with TXT _dmarc.{domain} = \"v=DMARC1; p=none; rua=mailto:you@{domain}\"")
    tags = dict(
        part.strip().split("=", 1) for part in records[0].split(";") if "=" in part
    )
    policy = tags.get("p", "").strip().lower()
    if policy == "none":
        return Check("DMARC", WARN, f"p=none (monitoring only) - fine while warming; "
                     f"move to quarantine once reports look clean. {records[0]}")
    return Check("DMARC", OK, records[0])


def check_dkim(domain: str, selector: str = "") -> Check:
    selectors = (selector,) if selector else COMMON_DKIM_SELECTORS
    for sel in selectors:
        for record in _txt(f"{sel}._domainkey.{domain}"):
            if "p=" in record.replace(" ", "") and "p=;" not in record.replace(" ", "") + ";":
                return Check("DKIM", OK, f"selector '{sel}' found")
    if selector:
        return Check("DKIM", FAIL, f"No DKIM key at {selector}._domainkey.{domain}.")
    return Check("DKIM", WARN, "No key at any common selector. If your provider uses a custom one, "
                 "set channels.email.warmup.dkim_selector; otherwise turn DKIM on in your provider.")


def check_domain_age(domain: str) -> Check:
    try:
        r = httpx.get(f"https://rdap.org/domain/{domain}", timeout=15, follow_redirects=True)
        r.raise_for_status()
        events = r.json().get("events", [])
    except Exception as e:
        return Check("Domain age", INFO, f"Couldn't look up registration date ({type(e).__name__}).")
    registered = next((e.get("eventDate") for e in events if e.get("eventAction") == "registration"), None)
    if not registered:
        return Check("Domain age", INFO, "Registry didn't publish a registration date.")
    age = (date.today() - datetime.fromisoformat(registered.replace("Z", "+00:00")).date()).days
    status = FAIL if age < MIN_DOMAIN_AGE_DAYS else WARN if age < GOOD_DOMAIN_AGE_DAYS else OK
    note = {FAIL: " - brand-new domains are distrusted; warm only, no cold email yet.",
            WARN: " - still young; keep volume low.", OK: ""}[status]
    return Check("Domain age", status, f"{age} days (registered {registered[:10]}){note}")


def _blocklist_checks(query: str, label: str, lists) -> list[Check]:
    checks = []
    for zone, refused in lists:
        hits = _a(f"{query}.{zone}")
        if not hits:
            checks.append(Check(f"{label} {zone}", OK, "not listed"))
        elif any(refused(ip) for ip in hits):
            checks.append(Check(f"{label} {zone}", INFO,
                                "list refused the query (common via public DNS like 8.8.8.8) - "
                                "check manually at https://check.spamhaus.org"))
        else:
            checks.append(Check(f"{label} {zone}", FAIL, f"LISTED ({', '.join(hits)})"))
    return checks


def check_domain_blocklists(domain: str) -> list[Check]:
    return _blocklist_checks(domain, "Blocklist", DOMAIN_BLOCKLISTS)


def check_ip_blocklists(smtp_host: str) -> list[Check]:
    ips = _a(smtp_host)
    if not ips:
        return [Check("Sending IP", INFO, f"Couldn't resolve {smtp_host}.")]
    reversed_ip = ".".join(reversed(ips[0].split(".")))
    return _blocklist_checks(reversed_ip, f"IP {ips[0]}", IP_BLOCKLISTS)


def check_warmup(warmup, ceiling: int, today: date | None = None) -> Check:
    days = warmup.days_in(today)
    if days is None:
        return Check("Warmup", WARN, "No warmup start date set, so sends aren't ramped. Once your warmup "
                     "service is on, set channels.email.warmup.start_date: YYYY-MM-DD.")
    if days < 0:
        return Check("Warmup", WARN, f"Starts {warmup.start_date} (in {-days} days).")
    if days < warmup.min_days:
        return Check("Warmup", FAIL, f"Day {days} of {warmup.min_days} - warming only, no cold email "
                     f"until {warmup.min_days - days} more day(s) pass.")
    cap = warmup.daily_cap(ceiling, today)
    full_on = warmup.min_days + max(0, -(-(ceiling - warmup.start_sends) // max(warmup.daily_increase, 1)))
    ramp = "full volume" if cap >= ceiling else f"reaches {ceiling}/day on day {full_on}"
    return Check("Warmup", OK, f"Day {days}: today's cap is {cap}/day ({ramp}). "
                 "Keep the warmup service running alongside Harvey.")


def check_send_history(total_sent: int, bounces: int) -> Check:
    if total_sent < MIN_SENDS_FOR_RATES:
        return Check("Bounce rate", INFO, f"{bounces} bounces / {total_sent} sent - too few sends to judge yet.")
    rate = bounces / total_sent
    status = OK if rate <= MAX_BOUNCE_RATE else FAIL
    return Check("Bounce rate", status, f"{rate:.1%} ({bounces}/{total_sent}); keep it under {MAX_BOUNCE_RATE:.0%}.")
