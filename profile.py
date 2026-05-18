"""
Loads ./profile.json and turns it into context strings that the LLM prompts
can embed. Falls back to a clearly-empty placeholder if profile.json hasn't
been set up yet, so the dashboard can still start.
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROFILE_PATH = HERE / "profile.json"
EXAMPLE_PATH = HERE / "profile.example.json"


def _strip_comment_keys(d: dict) -> dict:
    """Filter out documentation-style keys like `_comment`, `_note`, etc."""
    return {k: v for k, v in d.items() if not k.startswith("_")}


def load_profile() -> dict:
    """Read profile.json. If missing, fall back to profile.example.json with
    every field replaced by a clear `<NOT SET>` placeholder."""
    if PROFILE_PATH.exists():
        try:
            return _strip_comment_keys(json.loads(PROFILE_PATH.read_text()))
        except Exception as e:
            print(f"WARN: profile.json exists but is invalid JSON: {e}")
    # Fallback: example structure with placeholders so the dashboard still works
    if EXAMPLE_PATH.exists():
        ex = _strip_comment_keys(json.loads(EXAMPLE_PATH.read_text()))
        return {k: ("<NOT SET — fill profile.json>" if isinstance(v, str) else v) for k, v in ex.items()}
    return {}


def bio_block(profile: dict | None = None) -> str:
    """Build a compact bio paragraph for system prompts. Includes only
    fields that are set."""
    if profile is None:
        profile = load_profile()
    lines: list[str] = []
    name = profile.get("name", "").strip()
    degree = profile.get("degree", "").strip()
    school = profile.get("school", "").strip()
    grad = profile.get("grad_date", "").strip()
    gpa = profile.get("gpa", "").strip()
    undergrad = profile.get("undergrad", "").strip()
    email = profile.get("email", "").strip()
    github = profile.get("github", "").strip()
    location = profile.get("location", "").strip()
    visa = profile.get("visa_status", "").strip()
    stack = profile.get("stack", "").strip()

    if name:
        lines.append(f"You write applications for {name}.")
    parts = []
    if degree:
        parts.append(degree)
    if school:
        parts.append(f"at {school}")
    if grad:
        parts.append(f"(graduating {grad})")
    if gpa:
        parts.append(f"GPA {gpa}")
    if parts:
        lines.append("- " + ", ".join(parts))
    if undergrad:
        lines.append(f"- Undergrad: {undergrad}")
    if visa or location:
        loc_visa = ", ".join(filter(None, [visa, f"based in {location}" if location else ""]))
        lines.append(f"- {loc_visa}")
    if email or github:
        contact = ", ".join(filter(None, [email, f"github.com/{github}" if github else ""]))
        lines.append(f"- {contact}")
    if stack:
        lines.append(f"- Headline stack: {stack}")
    return "\n".join(lines).strip()


def voice_hint(profile: dict | None = None) -> str:
    if profile is None:
        profile = load_profile()
    return profile.get("voice", "competent, specific, low-fluff").strip()


def forbidden_topics_block(profile: dict | None = None) -> str:
    """Build the 'NEVER mention these topics' block for the writer prompt."""
    if profile is None:
        profile = load_profile()
    topics = profile.get("forbidden_topics") or []
    # Always exclude this tool itself
    builtins = [
        "the wellfound-apply-helper itself",
        "this auto-fill tool",
        "job-application bot",
        "auto-apply",
        "bookmarklet that filled this in",
    ]
    all_topics = list(dict.fromkeys(builtins + list(topics)))
    return ", ".join(all_topics)


if __name__ == "__main__":
    p = load_profile()
    print("=== bio block ===")
    print(bio_block(p))
    print()
    print("=== voice ===")
    print(voice_hint(p))
    print()
    print("=== forbidden ===")
    print(forbidden_topics_block(p))
