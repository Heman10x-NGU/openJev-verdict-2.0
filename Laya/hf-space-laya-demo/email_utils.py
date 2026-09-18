"""Email helpers for RL Agent: clean raw emails into a compact state and a ready-made set of email questions.

Jev-style models lose accuracy on long, noisy state, and RL Agent reads at most max_len (512) tokens,
so strip quoted replies, signatures and disclaimers in code before asking questions.
"""
import re

_QUOTE_HEADERS = [
    re.compile(r"^\s*On .{0,300}wrote:\s*$", re.I),
    re.compile(r"^\s*-{2,}\s*(Original|Forwarded) Message\s*-{2,}", re.I),
    re.compile(r"^\s*_{8,}\s*$"),
    re.compile(r"^\s*From:\s.+$", re.I),
]
_SIGNATURE_MARKERS = [
    re.compile(r"^\s*--\s*$"),
    re.compile(r"^\s*(best|kind|warm|many thanks|thanks|thank you|regards|cheers|sincerely)[\w ,!.]*$", re.I),
    re.compile(r"^\s*sent from my (iphone|android|mobile|ipad)", re.I),
]
_DISCLAIMER = re.compile(r"(confidential|intended (solely )?for the (use of the )?(named )?(addressee|recipient)|"
                         r"if you (have )?received this (e-?mail|message) in error)", re.I)


def clean_email_body(body, max_chars=3000):
    """Remove quoted history, signature and legal disclaimer; collapse whitespace; truncate."""
    text = (body or "").replace("\r\n", "\n").replace("\r", "\n").replace("\\n", "\n")
    lines = []
    for line in text.split("\n"):
        if any(p.match(line) for p in _QUOTE_HEADERS) and lines:
            break  # everything below is the previous thread
        if line.lstrip().startswith(">"):
            continue
        lines.append(line.rstrip())
    # a sign-off only counts near the end (last 40%, or last 8 lines of a short email) and must be a short line
    cut = len(lines)
    for i in range(max(1, min(int(len(lines) * 0.6), len(lines) - 8)), len(lines)):
        if len(lines[i].strip()) <= 40 and any(p.match(lines[i]) for p in _SIGNATURE_MARKERS):
            cut = i
            break
    lines = lines[:cut]
    paragraphs = [p for p in re.split(r"\n\s*\n", "\n".join(lines)) if not _DISCLAIMER.search(p)]
    text = re.sub(r"[ \t]+", " ", "\n\n".join(p.strip() for p in paragraphs if p.strip()))
    return text[:max_chars]


def email_state(subject, body, sender=None, clean=True, **extra):
    """Build the state dict the email questions refer to (`subject`, `body`, optional `from`)."""
    state = {"subject": (subject or "").strip(), "body": clean_email_body(body) if clean else (body or "")}
    if sender:
        state["from"] = sender
    state.update({k: v for k, v in extra.items() if v is not None})
    return state


def email_questions(categories=None):
    """A default fan-out of email questions. `categories` = {key: description} for your own routing labels."""
    categories = categories or {
        "billing": "invoices, payments, refunds", "technical": "bugs, outages, integrations",
        "sales": "pricing, demos, new purchases", "account": "login, access, profile changes",
        "hr": "hiring, leave, payroll", "other": "none of the above",
    }
    return {
        "category": {"type": "choice", "instructions": "Which team should handle the email in `body`?", "criteria": categories},
        "is_spam": {"type": "noul", "instructions": "Is this email unsolicited spam or bulk marketing?"},
        "is_phishing": {"type": "noul", "instructions": "Is this email a phishing or scam attempt to steal money, credentials, or personal data?",
                        "criteria": {"true": "phishing, scam, or fraud", "false": "a legitimate email"}},
        "urgency": {"type": "score", "instructions": "How urgent is the issue described in `body`?",
                    "criteria": ["no time pressure", "needs attention soon", "blocking issue or hard deadline"]},
        "needs_reply": {"type": "noul", "instructions": "Does the sender expect a reply?"},
        "sentiment": {"type": "score", "instructions": "What is the sender's tone in `body`?",
                      "criteria": ["angry or very negative", "negative", "neutral", "positive"]},
    }
