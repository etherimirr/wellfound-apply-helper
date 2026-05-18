# Project: backend_intern_example

Replace this file with your real project. The filename (minus `.md`) is
the project ID the LLM picker will refer to.

## Header

Example Co — Backend Engineer Intern (Summer 2025)

## Tags

python, fastapi, postgres, distributed-systems, production

## Bullets

- Built a Python + PostgreSQL data pipeline that ingested and normalized
  heterogeneous event data from 12 upstream services, processing ~2M
  events/day with end-to-end latency under 1.5s.
- Implemented rule-based + ML-driven validation workflows that reduced
  manual data-review effort by ~50% and surfaced 3 previously-undetected
  upstream bugs.
- Designed a backfill / replay command for the pipeline so the data team
  could re-process a historical window without human review, used during 4
  separate incident reviews.

## Gates

```yaml
brand: true
# Set brand=true if this is a name-recognition internship (e.g. FAANG,
# well-known startups). The picker will preferentially surface brand=true
# projects when the JD has data-pipeline / ML / production-AI signal.

only_for: []
# Leave empty to surface this project for any role.
# Or list specific JD-flavors to gate it, e.g.:
#   only_for: ["game studio", "pure frontend", "UX-only"]
# means "ONLY surface this project for JDs matching those flavors".

never_for: []
# Inverse: hide this project from JDs matching these flavors.
# Useful for niche projects that don't fit most roles.
```

## How the LLM uses this

When asked "What interests you about working for $COMPANY?":

1. **Picker call** (gpt-4o-mini): given the JD + your full project pool,
   pick 2-3 project IDs that best fit. The picker reads the `Header` and
   `Tags` of each project plus the `Gates`.
2. **Writer call** (gpt-4o): given the picked IDs and their full Markdown
   (this whole file, including Bullets), generate 3-5 sentences per
   project grounded in the specific bullets you wrote here.

The more **specific** your bullets are (real tech names, real numbers, real
component names), the better the blurb. Vague bullets like "improved
efficiency" produce vague blurbs.
