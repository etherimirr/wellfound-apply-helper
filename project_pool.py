"""
Loads the user's project pool from ./projects/*.md.

Each Markdown file is one project. The file is parsed for:
  - "## Header"  → one-line title
  - "## Tags"    → comma-separated keywords
  - "## Bullets" → list of `- ` bulleted facts
  - "## Gates"   → optional yaml-style key/values:
        brand: true|false
        only_for: [list of JD-flavor terms]
        never_for: [list of JD-flavor terms]

The whole .md file is also loaded verbatim for the writer step (so the LLM
can ground in everything below the structured sections too).
"""
from __future__ import annotations

import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECTS_DIR = HERE / "projects"


def _split_sections(text: str) -> dict[str, str]:
    """Split a Markdown file by ## headings. Returns {heading_lower: body}."""
    out: dict[str, str] = {}
    current = "_preamble"
    buf: list[str] = []
    for line in text.splitlines():
        h = re.match(r"^##\s+(.+?)\s*$", line)
        if h:
            if buf:
                out[current.lower()] = "\n".join(buf).strip()
            current = h.group(1).strip()
            buf = []
        else:
            buf.append(line)
    if buf:
        out[current.lower()] = "\n".join(buf).strip()
    return out


def _parse_gates(gates_body: str) -> dict:
    """Parse the yaml-ish Gates section. We accept either fenced ```yaml ```
    blocks or plain key: value lines."""
    if not gates_body:
        return {}
    # Strip fenced block markers
    body = re.sub(r"```(?:yaml)?\s*", "", gates_body)
    body = body.replace("```", "")
    out: dict = {}
    for line in body.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = re.match(r"^([a-zA-Z_][a-zA-Z0-9_]*)\s*:\s*(.*)$", line)
        if not m:
            continue
        key, val = m.group(1), m.group(2).strip()
        if val.lower() in ("true", "yes"):
            out[key] = True
        elif val.lower() in ("false", "no"):
            out[key] = False
        elif val.startswith("[") and val.endswith("]"):
            inner = val[1:-1].strip()
            if not inner:
                out[key] = []
            else:
                out[key] = [s.strip().strip("\"'") for s in inner.split(",")]
        else:
            out[key] = val
    return out


def _parse_bullets(bullets_body: str) -> list[str]:
    """Pull every `- ...` line."""
    out: list[str] = []
    for line in bullets_body.splitlines():
        m = re.match(r"^\s*[-*]\s+(.+)$", line)
        if m:
            out.append(m.group(1).strip())
    return out


def load_project_kb(project_id: str, max_chars: int = 6000) -> str:
    """Read the full Markdown file for one project. Used by the writer step
    so the LLM can ground in every line of the user's authored project doc."""
    p = PROJECTS_DIR / f"{project_id}.md"
    if not p.exists():
        return ""
    text = p.read_text(encoding="utf-8", errors="ignore")
    if len(text) > max_chars:
        text = text[:max_chars] + "\n\n[...truncated...]"
    return text


def load_full_pool() -> dict[str, dict]:
    """Read every .md in projects/ and return a dict keyed by project_id.

    Each value: {id, header, tags, bullets, gates, full_text}.
    """
    out: dict[str, dict] = {}
    if not PROJECTS_DIR.exists():
        return out
    for p in sorted(PROJECTS_DIR.glob("*.md")):
        pid = p.stem
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        sections = _split_sections(text)
        out[pid] = {
            "id": pid,
            "header": sections.get("header", "").strip().split("\n")[0],
            "tags": [t.strip() for t in re.split(r"[,\n]", sections.get("tags", "")) if t.strip()],
            "bullets": _parse_bullets(sections.get("bullets", "")),
            "gates": _parse_gates(sections.get("gates", "")),
            "full_text": text[:6000],
        }
    return out


def format_pool_for_prompt(pool: dict | None = None) -> str:
    """Render the pool as a compact prompt block for the picker step."""
    if pool is None:
        pool = load_full_pool()
    lines: list[str] = []
    for pid, entry in pool.items():
        gates = entry.get("gates") or {}
        brand_tag = " [BRAND]" if gates.get("brand") else ""
        only_for = gates.get("only_for") or []
        never_for = gates.get("never_for") or []
        gate_str = ""
        if only_for:
            gate_str += f"  only-for: {only_for}\n"
        if never_for:
            gate_str += f"  never-for: {never_for}\n"
        tags_str = ", ".join(entry.get("tags", []))
        lines.append(f"\n### {pid}{brand_tag}: {entry.get('header', '')}")
        if tags_str:
            lines.append(f"  tags: {tags_str}")
        if gate_str:
            lines.append(gate_str.rstrip())
        for b in entry.get("bullets", [])[:6]:
            lines.append(f"  - {b}")
    return "\n".join(lines)


if __name__ == "__main__":
    pool = load_full_pool()
    print(f"Loaded {len(pool)} project(s):")
    for pid, entry in pool.items():
        print(f"  {pid:30s}  brand={entry.get('gates', {}).get('brand', False)}  "
              f"tags={entry.get('tags', [])[:4]}")
