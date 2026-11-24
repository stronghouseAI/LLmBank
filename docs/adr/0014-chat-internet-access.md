# 14. Internet access in Chat

Date: 2026-11-24 · Status: accepted

## Context
phil asked for an "access internet" option in Chat, used when the model needs it. Agents
already have three web skills (web_search, fetch_web_page, read_news_feed) that work live on
phil's laptop (ADR 0006), and the app's rule is that every action reaching outside asks first.

## Decision
- A **Use internet** switch next to **Think first** on the Chat page. It is off by default, and
  while it is off the request carries no tools, so chat stays fully offline.
- When it is on, `POST /api/chat` with `web: true` adds the same three web skills as tools
  (`core/chat_web.py`, taken from the skill catalog's web category) plus a short system note:
  use them only when the answer needs current or outside information, and name the source.
- The model decides when it needs the internet. Its tool calls stream to the page, which shows
  a card ("Search the web for ...") with **Allow** and **Deny** for every single call. Only after
  Allow does the page call `POST /api/chat/web`, which runs the agents' `SkillExecutor` and
  refuses any skill outside the web category. A deny or a failed fetch goes back to the model as
  text, so it can answer anyway or try another route.
- At most 6 internet steps per reply; after that the tools are not offered and the model must
  answer. Stop works while a card is waiting.
- Internet steps appear as `Internet:` blocks in Copy chat and Export .txt.

## Consequences
- Needs a model whose chat template supports tools (llama-server already runs with `--jinja`).
  Models without tool support simply answer without searching.
- The page holds the loop, so the server stays stateless; a reload mid-reply loses that reply.
