# Project: personal_side_project_example

Replace this file with your real project.

## Header

LLM Agent for Personal Productivity — Multi-Agent System on macOS

## Tags

python, llm, agent, claude-code, multi-agent, async

## Bullets

- Built a multi-agent personal assistant orchestrating two specialized
  agents (planner + executor) over a typed event bus, persisting state
  in SQLite + FTS5 with sub-100ms lookups at 50K+ entries.
- Designed a 3-layer rate limiter with circuit-breaker that keeps total
  cost under $20/month while running 24/7 against the Anthropic API.
- Implemented an overnight consolidation job (launchd-scheduled) that
  compresses the day's interactions into long-term memory traces; system
  has run for 90+ days without manual restart.

## Gates

```yaml
brand: false
only_for: []
never_for:
  - "games"
  - "pure UX"

# This is a personal project (not a brand-name internship), so brand=false.
# It surfaces well for any LLM / agent / Claude Code / multi-agent role.
# Hidden for games and pure UX roles where it doesn't fit.
```
