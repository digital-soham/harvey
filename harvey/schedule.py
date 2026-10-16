"""Region rotation — which market Harvey works on which day, and when to send.

With ``schedule.enabled``, each weekday belongs to one region (e.g. Mon US,
Tue UK, Wed Europe, Thu US, Fri UK). On a region's day Harvey researches and
emails that region's best prospects, and sends only inside the recipient's
local morning window. Days with no region (weekends by default) send
nothing; the follow-up reviewer uses them to decide which follow-ups are
worth sending.

A prospect's region comes from, in order: a cached verdict, the company's
recorded location, its domain's country TLD, then a free keyword scan of its
homepage and contact page (phone codes, currency, addresses, "GmbH"/"Ltd").
"""

import logging
import re
from datetime import datetime, time, timedelta, timezone

import pytz

logger = logging.getLogger("harvey.schedule")

WEEKDAYS = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")
REGION_CACHE_PREFIX = "region:"

EU_TLDS = {"de", "fr", "nl", "es", "it", "ie", "se", "dk", "fi", "no", "be", "at", "ch",
           "pl", "pt", "cz", "eu", "lu", "ee", "lv", "lt", "gr", "hu", "ro", "sk", "si",
           "hr", "bg", "is"}

# Lowercase patterns per region. Weighted by count of distinct matches.
REGION_PATTERNS = {
    "uk": [r"united kingdom", r"\bengland\b", r"\bscotland\b", r"\bwales\b", r"\blondon\b",
           r"\bmanchester\b", r"\bbirmingham\b", r"\bedinburgh\b", r"\bbristol\b", r"\bleeds\b",
           r"\+44", r"£", r"\bltd\b", r"companies house", r"\b[a-z]{1,2}\d{1,2}[a-z]? \d[a-z]{2}\b"],
    "europe": [r"\bgermany\b", r"\bdeutschland\b", r"\bfrance\b", r"\bnetherlands\b",
               r"\bspain\b", r"\bitaly\b", r"\bireland\b", r"\bsweden\b", r"\bdenmark\b",
               r"\bfinland\b", r"\bnorway\b", r"\bbelgium\b", r"\baustria\b", r"\bswitzerland\b",
               r"\bpoland\b", r"\bportugal\b", r"\bberlin\b", r"\bmunich\b", r"\bparis\b",
               r"\bamsterdam\b", r"\bdublin\b", r"\bstockholm\b", r"\bcopenhagen\b",
               r"\bmadrid\b", r"\bbarcelona\b", r"\bmilan\b", r"\bzurich\b", r"\bvienna\b",
               r"\bgmbh\b", r"\bb\.v\.", r"\bs\.a\.s\b", r"\bab\b \(publ\)", r"€",
               r"\+49", r"\+33", r"\+31", r"\+34", r"\+39", r"\+353", r"\+46", r"\+45"],
    "us": [r"united states", r"\busa\b", r"\bnew york\b", r"\bsan francisco\b",
           r"\blos angeles\b", r"\bchicago\b", r"\baustin\b", r"\bboston\b", r"\bseattle\b",
           r"\bmiami\b", r"\bdenver\b", r"\batlanta\b", r"\bdelaware\b", r"\bcalifornia\b",
           r"\btexas\b", r"\bllc\b", r"\binc\.", r"\b[A-Z]{2} \d{5}(?:-\d{4})?\b",
           r"\+1[ \-(]", r"\(\d{3}\) \d{3}-\d{4}"],
}
_COMPILED = {r: [re.compile(p, re.I if not p.startswith(r"\b[A-Z]") else 0) for p in ps]
             for r, ps in REGION_PATTERNS.items()}


# ── Calendar ───────────────────────────────────────────────────────

def _now(now: datetime | None) -> datetime:
    return now or datetime.now(timezone.utc)


def _local(region_cfg, now: datetime | None) -> datetime:
    return _now(now).astimezone(pytz.timezone(region_cfg.timezone))


def _window(schedule) -> tuple[time, time]:
    return time.fromisoformat(schedule.send_window_start), time.fromisoformat(schedule.send_window_end)


def is_region_day(schedule, region: str, now: datetime | None = None) -> bool:
    """True when it's this region's weekday in the region's own timezone."""
    cfg = schedule.regions.get(region)
    if not cfg:
        return False
    return schedule.weekdays.get(WEEKDAYS[_local(cfg, now).weekday()]) == region


def is_sending_now(schedule, region: str, now: datetime | None = None) -> bool:
    """Region's day AND inside its local send window."""
    if not is_region_day(schedule, region, now):
        return False
    start, end = _window(schedule)
    return start <= _local(schedule.regions[region], now).time() <= end


def todays_regions(schedule, now: datetime | None = None) -> list[str]:
    """Regions whose day it is and whose send window hasn't closed yet."""
    _, end = _window(schedule)
    return [r for r in schedule.regions
            if is_region_day(schedule, r, now) and _local(schedule.regions[r], now).time() <= end]


def next_region(schedule, now: datetime | None = None) -> str:
    """The region whose next send window opens soonest (or is open now)."""
    start, end = _window(schedule)
    now = _now(now)
    for offset in range(0, 8):
        best = None
        for region, cfg in schedule.regions.items():
            local = _local(cfg, now) + timedelta(days=offset)
            if schedule.weekdays.get(WEEKDAYS[local.weekday()]) != region:
                continue
            if offset == 0 and local.time() > end:
                continue
            opens = local.replace(hour=start.hour, minute=start.minute, second=0, microsecond=0)
            if best is None or opens.astimezone(timezone.utc) < best[0]:
                best = (opens.astimezone(timezone.utc), region)
        if best:
            return best[1]
    return next(iter(schedule.regions), "")


def is_rest_day(schedule, tz_name: str, now: datetime | None = None) -> bool:
    """No region is assigned to today (in the operator's timezone)."""
    day = WEEKDAYS[_now(now).astimezone(pytz.timezone(tz_name)).weekday()]
    return day not in schedule.weekdays


# ── Region detection ───────────────────────────────────────────────

def classify_text(text: str, schedule=None) -> str:
    """Region with the most distinct matching clues, or '' if unclear."""
    if not text:
        return ""
    scores = {r: sum(1 for p in pats if p.search(text)) for r, pats in _COMPILED.items()}
    if schedule is not None:
        for region, cfg in schedule.regions.items():
            scores[region] = scores.get(region, 0) + sum(
                1 for g in cfg.geography if g and g.lower() in text.lower())
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    if not ranked or ranked[0][1] == 0 or (len(ranked) > 1 and ranked[0][1] == ranked[1][1]):
        return ""
    return ranked[0][0]


def classify_domain(domain: str) -> str:
    tld = (domain or "").lower().rstrip(".").rsplit(".", 1)[-1]
    if tld == "uk":
        return "uk"
    if tld in EU_TLDS:
        return "europe"
    if tld == "us":
        return "us"
    return ""


async def _site_text(website: str) -> str:
    from harvey import research

    async def grab(url):
        pages = await research.read_website(url)
        return " ".join(text for _, text in pages[:1])

    if not website:
        return ""
    base = website if website.startswith("http") else f"https://{website}"
    text = await grab(base)
    for path in ("/contact", "/contact-us", "/about"):
        extra = await grab(base.rstrip("/") + path)
        if extra:
            return text + " " + extra
    return text


async def ready_by_region(state, schedule) -> dict[str, int]:
    """New prospects with a deliverable email, counted per region."""
    counts: dict[str, int] = {}
    for p in await state.get_prospects_by_status("new"):
        if not p.email or (p.email_status or "guess") not in ("verified", "risky"):
            continue
        company = await state.get_company(p.company_id) if p.company_id else None
        region = await region_of(state, p, company, schedule)
        counts[region] = counts.get(region, 0) + 1
    return counts


def written_key(regions: list[str], now: datetime | None = None) -> str:
    """Settings key counting researched sequences written for today's region."""
    return f"deep_written:{_now(now):%Y-%m-%d}:{'+'.join(regions)}"


async def region_of(state, prospect, company, schedule, fetch: bool = True) -> str:
    """'us' / 'uk' / 'europe' (any configured region) or '' when unknown."""
    key_id = (company.id if company else "") or (prospect.company or prospect.email)
    cache_key = REGION_CACHE_PREFIX + key_id
    cached = await state.get_setting(cache_key, "")
    if cached:
        return "" if cached == "?" else cached

    region = classify_text(company.location, schedule) if company and company.location else ""
    domain = (company.domain if company else "") or prospect.email.rpartition("@")[2]
    region = region or classify_domain(domain)
    if not region and fetch:
        website = (company.website or company.domain) if company else domain
        try:
            region = classify_text(await _site_text(website), schedule)
        except Exception as e:
            logger.debug(f"Region scan failed for {website}: {e}")
    if region and region not in schedule.regions:
        region = ""
    if region or fetch:  # don't cache "unknown" before the website was checked
        await state.set_setting(cache_key, region or "?")
    return region
