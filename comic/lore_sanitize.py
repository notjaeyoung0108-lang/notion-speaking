"""Runtime lore sanitizers for GPT prompt injection."""
from __future__ import annotations

import re


_ARCHETYPE_ROLE_FUNCTIONS = {
    "Rachel": "adaptive newcomer / approval-seeking protagonist",
    "Chandler": "deadpan deflector who helps quietly",
    "Monica": "control-as-care team lead",
    "Ross": "detail-obsessed approval seeker",
    "Phoebe": "unpredictable outsider friend",
}


def sanitize_character_bible(text: str) -> str:
    """Remove external sitcom archetype names from runtime character lore.

    Human-facing lore may keep archetype labels, but GPT prompts should receive
    local role functions instead of Friends character names.
    """
    out = text or ""
    for archetype, role_function in _ARCHETYPE_ROLE_FUNCTIONS.items():
        out = re.sub(
            rf"\barchetype\s+\*\*{re.escape(archetype)}\*\*",
            f"role_function: {role_function}",
            out,
            flags=re.I,
        )
        out = re.sub(
            rf"\barchetype\s+{re.escape(archetype)}\b",
            f"role_function: {role_function}",
            out,
            flags=re.I,
        )
    out = re.sub(r"\bRoss\s+sulk\b", "approval-seeking sulk", out, flags=re.I)
    for archetype in _ARCHETYPE_ROLE_FUNCTIONS:
        out = re.sub(rf"\b{re.escape(archetype)}\b", "", out)
    out = re.sub(r"\s{2,}", " ", out)
    out = re.sub(r" +\n", "\n", out)
    return out.strip()
