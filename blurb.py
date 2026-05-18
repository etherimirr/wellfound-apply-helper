"""
LLM-grounded blurb generator. Two LLM calls per blurb:

  Step 1 — PICKER (gpt-4o-mini, JSON-mode):
    Given the JD + the user's full project pool, pick 2-3 project_ids that
    best fit. Picker reads project headers, tags, and gates.

  Step 2 — WRITER (gpt-4o):
    Given the picked project KBs + JD + user's profile, write a FIT-PITCH
    blurb ("What interests you about working for this company?") — opening
    hook, project bridge paragraph(s), brief closing.

  Step 3 — CRITIC / SCRUB:
    Regex-based last-mile pass that removes em-dashes, AI-cliches, and any
    forbidden topics declared in profile.json.

All "who is this for" data flows from `profile.json` and `projects/`. There
is no hardcoded bio or project list anywhere in this file.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from llm_utils import openai_client, llm_call, parse_json_from_response
from profile import load_profile, bio_block, voice_hint, forbidden_topics_block
from project_pool import (
    load_full_pool,
    load_project_kb,
    format_pool_for_prompt,
)


# ---------- AI-tell scrubbing -----------------------------------------------

_SCRUB_PAIRS: list[tuple[str, str]] = [
    # Em / en dashes — classic LLM tell
    ("—", ", "), ("–", ", "), ("―", ", "), ("‒", ", "),
    # Arrows, bullets, smart quotes
    ("→", " then "), ("⟶", " then "), ("⇒", " then "), ("➜", " then "),
    ("←", " from "), ("⟵", " from "), ("↔", " and "),
    ("…", "..."),
    ("•", ""), ("●", ""), ("◦", ""), ("▪", ""), ("✓", ""), ("✗", ""),
    ("“", '"'), ("”", '"'), ("‘", "'"), ("’", "'"),
    # Phrases LLMs reach for that read like cover-letter boilerplate
    (r"\bI'm passionate about\b", "I work on"),
    (r"\bI'd love to\b", "I want to"),
    (r"\bI would love to\b", "I want to"),
    (r"\bexactly what I\b", "what I"),
    (r"\bthrilled to\b", "interested in"),
    (r"\bsynergy\b", "overlap"),
    (r"\bleverag\w*\b", "use"),
    (r"\bdelv\w*\b", "look at"),
    (r"\butiliz\w*\b", "use"),
    (r"\bharness(?:es|ed|ing)?\b", "use"),
    (r"\brealm\b", "area"),
    (r"\bcutting-edge\b", "modern"),
    (r"\btransformative\b", "significant"),
    (r"\bembark on\b", "start"),
    (r"\bever-evolving\b", "changing"),
    (r"\bparticularly compelling\b", "interesting"),
    (r"\baligns perfectly\b", "matches"),
    (r"\bdeep understanding\b", "working knowledge"),
    (r"\bseamless transition\b", "quick ramp-up"),
    (r"\bperfectly\b", "well"),
    # Closing-template tells
    (r"\bLet'?s connect to discuss\b[^.]*", "I'd like to learn more"),
    (r"\bmake a meaningful impact\b", "ship real work"),
    (r"\bhoned my (skills|ability|abilities|expertise) (in|to)\b", r"taught me \2"),
    (r"\bhoned my (skills|ability|abilities|expertise)\b", "taught me"),
    (r"\bhoned\b", "developed"),
    (r"\brobust(\s+and\s+\w+)?\s+", ""),
    (r"\brobust\b", ""),
    (r"\bscalable,?\s*efficient\b", "scalable"),
    (r"\binnovative\s+approach(es)?\b", "approach"),
    (r"\bmodern\s+solutions\b", "tooling"),
    (r"\buser[- ]friendly\b", ""),
    (r"\bparamount\b", "central"),
    (r"\bintricacies\b", "details"),
    (r"\bvaluable insights\b", "insights"),
    (r"\bcohesive\b", "unified"),
    # Cleanup
    (r"\s{2,}", " "),
]


def scrub_ai_tells(text: str) -> str:
    out = text
    for pat, rep in _SCRUB_PAIRS:
        if pat.startswith(r"\b"):
            out = re.sub(pat, rep, out, flags=re.IGNORECASE)
        else:
            out = out.replace(pat, rep)
    out = re.sub(r"[ \t]+", " ", out)
    out = re.sub(r" ,", ",", out)
    out = re.sub(r"\s+([.,;:])", r"\1", out)
    return out.strip()


def quick_checks(text: str) -> list[str]:
    """Return a list of issues. Empty list = passes."""
    problems: list[str] = []
    if "—" in text or "–" in text:
        problems.append("em/en dash present")
    cliches = [
        "leverage", "delve", "cutting-edge", "synergy", "harness",
        "transformative", "embark", "ever-evolving", "in today's fast-paced",
        "passionate about", "perfect fit", "perfect intersection",
        "elevate", "empower", "navigate the complexities",
        "robust", "honed", "innovative approach", "modern solutions",
        "scalable, efficient", "scalable and efficient", "user-friendly",
        "paramount", "intricacies", "valuable insights", "cohesive",
    ]
    found = [w for w in cliches if re.search(rf"\b{re.escape(w)}\b", text, re.IGNORECASE)]
    if found:
        problems.append(f"AI-cliche words: {found}")
    if re.search(r"\bthe company('s)?\b|\bthis company\b|\bthe firm('s)?\b", text, re.IGNORECASE):
        problems.append("third-person 'the company' — use 'your team' or '<Company>'s'")
    return problems


# ---------- Step 1: picker --------------------------------------------------

def _picker_prompt(profile: dict) -> str:
    name = profile.get("name", "the user")
    return f"""You pick 2-3 projects from {name}'s project library that BEST FIT a
specific job description.

Selection rules, in priority order:
1. JD's specific technical asks must match a project that DIRECTLY uses
   that tech (read tags + bullets).
2. JD's domain (fintech / healthcare / robotics / vision / infra / frontend)
   must match the project's domain.
3. Surface brand-name projects (tagged [BRAND]) preferentially when the JD
   has any data-pipeline / production-AI / large-scale signal.
4. Honor each project's gates:
   - `only_for: [...]` — surface ONLY if the JD matches one of those flavors
   - `never_for: [...]` — hide from JDs matching those flavors
5. Among matches, prefer projects with concrete numbers / outcomes the user
   can defend in an interview.
6. Show breadth: prefer 2-3 projects from different contexts.

PROJECT COUNT — TWO MODES:
- SDE / backend / full-stack / generic engineering roles: pick 3-4.
- Specialized roles (NLP-only / robotics-only / pure CV): pick 2-3 (depth).

NEVER pick a project whose `never_for` list matches the JD's flavor.

Output a SINGLE JSON object:
  {{"picks": ["project_id_1", "project_id_2", ...], "rationale": "1 sentence"}}

`picks` must contain 2-4 project_ids exactly as named in the library.
"""


def pick_projects(jd: str, title: str, company: str) -> tuple[list[str], str]:
    """Returns (project_ids, rationale). Falls back to alphabetic order of
    available projects if the LLM call fails."""
    profile = load_profile()
    pool = load_full_pool()
    if not pool:
        return [], "no projects/*.md files found — fill projects/ first"
    pool_dump = format_pool_for_prompt(pool)
    user = f"""COMPANY: {company}
TITLE: {title}

JD (first 4000 chars):
{jd[:4000]}

Project library:
{pool_dump}

Pick 2-4 project_ids. JSON only."""
    try:
        raw = llm_call(_picker_prompt(profile), user, model="gpt-4o-mini", max_tokens=500)
        obj = parse_json_from_response(raw)
        picks = obj.get("picks") or []
        rationale = obj.get("rationale", "")
        valid_ids = set(pool.keys())
        picks = [p for p in picks if p in valid_ids][:4]
        if len(picks) >= 1:
            return picks, rationale
    except Exception as e:
        print(f"  [pick_projects] LLM picker failed: {e}", file=sys.stderr)
    # Safe fallback: pick first N projects
    return list(pool.keys())[:3], "fallback (LLM unavailable)"


# ---------- Step 2: writer --------------------------------------------------

def _writer_prompt(profile: dict) -> str:
    bio = bio_block(profile)
    voice = voice_hint(profile)
    forbidden = forbidden_topics_block(profile)
    return f"""You write a FIT-PITCH blurb answering "What interests you about
working for this company?" on Wellfound.

About the user:
{bio}

Voice: {voice}

OUTPUT SHAPE — body length depends on how many project KBs are provided:
  - 2 KBs  → 4 paragraphs, 260–400 words
  - 3 KBs  → 4 paragraphs, 300–470 words
  - 4 KBs  → 5 paragraphs, 350–550 words (one extra supporting paragraph)

1. **Opening (1-2 SENTENCES MAX, ≤200 characters).** One concrete hook on
   the company's product / domain / market + the role title verbatim. NO
   filler like "is exciting", "is particularly compelling", "has the
   potential to revolutionize", "is incredibly appealing", "aligns
   perfectly with", "I'm passionate / thrilled / drawn to". One sentence
   is BETTER than two.
2. **Lead-project paragraph (3-4 sentences):** introduce project #1. Cite
   3 SPECIFIC technical details from its KB (component names, real
   numbers, real architecture decisions). Bridge to why that matters for
   this company.
3. **Supporting-project paragraph #1 (3-4 sentences):** project #2. Two
   specific facts from its KB. Surface a brand-name project here if one
   was picked.
4. **Supporting-project paragraph #2 (3-4 sentences, ONLY IF 4 KBs were
   provided):** project #3.
5. **Closing (1-2 sentences).** Pick ONE of: a specific bet on what to
   ship in week 1; a question to learn from the team; logistics-only
   (visa + location + availability); a bridge from the JD's biggest
   unknown to one of the user's projects. NEVER write "From week one, I
   would bring...", "eager to contribute", "meaningful impact", "from
   day one".

HARD WRITING RULES — VIOLATING ANY = REWRITE:
- NEVER use em dashes (—) or en dashes (–).
- NEVER use these words/phrases (case-insensitive): leverage, leveraging,
  delve, delving, robust, cutting-edge, synergy, harness, transformative,
  embark, ever-evolving, in today's, passionate about, perfect fit,
  elevate, empower, navigate the complexities, the realm of, eager to
  contribute, meaningful impact, from day one, honed, innovative approach,
  modern solutions, scalable and efficient, user-friendly, paramount,
  intricacies, valuable insights, cohesive, particularly compelling.
- NEVER use AI rhetorical structures: "is not only X but Y", "which is
  why I'm applying for the X role", "is the type/kind of X I want to".
- NEVER use third-person "the company" — use "your team" or "<Company>'s".
- NEVER mention: {forbidden}.
- NEVER reference git/PR/branch artifacts or coursework codes.
- GROUNDING: every concrete claim must trace to a line in the provided KB.

Output: ONLY the blurb prose. No JSON, no preamble, no markdown headers,
no bullet lists. Plain paragraphs separated by blank lines.
"""


def write_blurb(*, jd: str, title: str, company: str,
                project_ids: list[str], critic_feedback: str = "") -> str:
    """Single LLM call that produces the full FIT-PITCH blurb."""
    profile = load_profile()
    kb_blocks: list[str] = []
    for pid in project_ids[:4]:
        kb = load_project_kb(pid, max_chars=2800)
        if kb:
            kb_blocks.append(f"═══ KB for `{pid}` ═══\n{kb}\n═══ end ═══")
    kb_dump = "\n\n".join(kb_blocks) if kb_blocks else "(no KBs available — write from JD only)"

    critic_block = ""
    if critic_feedback:
        critic_block = (
            f"\n\n--- PREVIOUS DRAFT FAILED CRITIC. Fix these and "
            f"re-output the entire blurb:\n{critic_feedback}\n"
        )

    user = f"""COMPANY: {company}
ROLE TITLE: {title}

JD (first 4000 chars):
{jd[:4000]}

PROJECT KBs to ground from (use ONLY facts from these):
{kb_dump}
{critic_block}
Write the FIT-PITCH blurb now. Plain prose, no markdown headers."""

    client = openai_client()
    resp = client.chat.completions.create(
        model="gpt-4o",
        max_tokens=1400,
        temperature=0.75,
        messages=[
            {"role": "system", "content": _writer_prompt(profile)},
            {"role": "user", "content": user},
        ],
    )
    out = (resp.choices[0].message.content or "").strip()
    if out.startswith("```"):
        out = re.sub(r"^```[a-z]*\s*", "", out).rstrip("`").strip()
    return out


# ---------- Top-level composer ---------------------------------------------

def compose_blurb(jd: str, title: str, company: str,
                  min_projects: int = 2, max_projects: int = 4) -> str:
    """Two-step LLM pipeline + one critic retry. Returns clean prose."""
    picks, _rationale = pick_projects(jd, title, company)
    if len(picks) < min_projects:
        pool_ids = list(load_full_pool().keys())
        for fid in pool_ids:
            if fid not in picks:
                picks.append(fid)
            if len(picks) >= max_projects:
                break
    picks = picks[:max_projects]

    blurb = write_blurb(jd=jd, title=title, company=company, project_ids=picks)
    issues = quick_checks(blurb)
    if issues:
        feedback = "\n".join(f"  • {p}" for p in issues)
        blurb = write_blurb(jd=jd, title=title, company=company,
                            project_ids=picks, critic_feedback=feedback)

    return scrub_ai_tells(blurb)


# ---------- Per-question answer (used by the dashboard endpoint) -----------

def answer_question_prompt(profile: dict) -> str:
    bio = bio_block(profile)
    voice = voice_hint(profile)
    forbidden = forbidden_topics_block(profile)
    return f"""You answer Wellfound application questions for a candidate.

About the candidate:
{bio}

Voice: {voice}

For EACH question:
- Match the answer's length to the question's nature:
  - Yes/no: 1-2 sentences
  - "Why this company/role" / open-ended: 3-5 sentences with one project bridge
  - "Describe a project" / "Tell me about a time...": 4-6 sentences, ONE
    project, cite 3 specific facts from the provided KB pool
  - Salary / availability / dates: short, factual
- Pick a different project per question when possible
- For non-engineering questions (e.g. 5-year vision), be brief and connect
  to wanting to keep building production work

HARD RULES (violating = rewrite):
- NEVER use em dashes — / en dashes –
- NEVER: leverage(s/d/ing), delve, robust, cutting-edge, synergy, harness,
  transformative, embark, ever-evolving, in today's, passionate about,
  perfect fit, elevate, empower, navigate the complexities, eager to
  contribute, meaningful impact, from day one, honed, innovative approach,
  modern solutions, scalable and efficient, user-friendly, paramount
- NEVER use: "is the type of X I want to", "From week one, I would bring..."
- NEVER mention: {forbidden}
- GROUNDING: every concrete claim must trace to a bullet in the project pool

OUTPUT JSON only: {{"answers": [{{"name": "<exact textarea name from input>", "answer": "<text>"}}, ...]}}
"""
