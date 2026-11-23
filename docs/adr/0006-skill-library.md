# 6. A library of 15 agent skills, grouped by category

Date: 2026-11-23 · Status: accepted

## Context
The first Agent shop had 6 skills. A real quest ("today's Guardian headlines") showed
agents could not reach the web, and agents could not read the PDF they found. phil
asked for a library of 10-15 genuinely useful skills.

## Decision
15 skills in 5 categories, defined in `core/skill_catalog.py` and carried out by
`services/skills/` (one module per area):

| Category | Skills |
|---|---|
| Files | read file, list folder, search files, write file, move or rename |
| Web | fetch web page, read news feed (RSS/Atom), search the web |
| Documents | read PDF, take note |
| Desktop | open with app, copy to clipboard, run command |
| Utility | calculator, date and time |

- **Approval:** every skill that touches the computer or the internet asks each time
  (phil's rule). Only take note, calculator and date/time run without asking, because
  they read and change nothing outside the app.
- **Standard library and system tools, no new Python dependencies:** html.parser and
  xml.etree for pages and feeds; `pdftotext` (poppler), `xdg-open` and `wl-copy`
  (wl-clipboard) for desktop skills. Missing tools give an install hint.
- **Web search** reads DuckDuckGo's no-JavaScript results page because it needs no API
  key. That page is not a stable API: if it changes, the skill returns "No results"
  rather than crashing, and the parser in `core/text_extract.py` is the one place to fix.
- **Safety choices:** move never overwrites; there is deliberately no delete skill;
  web skills accept only http/https; downloads are capped at 3 MB and all results are
  truncated so they fit a small model's context.
- **Calculator** parses with `ast` and allows only numbers, operators and named maths
  functions (no `eval`), because small models get arithmetic wrong.
- The 5-slot limit stays: 15 skills to choose from, 5 per agent.

## Consequences
- A new starter agent, Herald, reads the news. Mira can now read PDFs.
- Adding a skill = one catalog entry + one handler; the executor refuses to start if
  a catalog entry has no handler, so a missing one is caught by tests, not mid-quest.
