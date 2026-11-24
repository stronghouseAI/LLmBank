# 10. "Studio" UI look and one error shape for the API

Date: 2026-11-24 · Status: accepted

## Context
phil asked for a more professional GUI, "not as cartoony", plus hardening. The first UI
used warm stone-and-brass colours, emoji portraits and game words (Guild, Quest, XP,
Dismiss). Several failure paths were silent or confusing: validation errors showed as
"[object Object]", a failed request stopped quest polling (so an approval could go unseen),
and buttons could be double-clicked into duplicate requests.

## Decision
- **Look**: a "Studio" theme. Graphite surfaces, one steel-blue accent, left sidebar with
  line icons, a page header on every page, tabular numbers. Follows the system light/dark
  setting. All colours, sizes and radii are tokens in `:root` in `style.css`; a test checks
  that no raw colour appears outside them. No web fonts or CDNs: the app works offline.
- **Words**: the Agent shop is shown as "Agents", quests as "tasks", Creativity and
  Diligence as the real Temperature and Max steps they map to (the API sends the mapping,
  so the numbers live only in `core/agents.py`). Agents get initials avatars instead of
  emoji. Only the visible text changed: API routes, database and `.agent.json` cards keep
  their names, so existing agents and cards keep working.
- **Errors**: every API error is `{"detail": "<one sentence>"}`. Validation errors are
  flattened to `field: message`; unexpected crashes are logged with a traceback and reported
  by exception type only, so file paths or tokens inside a message are never echoed. A chat
  stream that breaks mid-reply ends with an `error` event.
- **Dialogs**: the browser's `prompt()` and `confirm()` are replaced by one in-app dialog.

## Consequences
- Every agent action still waits for Allow once / Deny; nothing about approvals changed.
- The UI is still plain HTML/CSS/JS with no build step (ADR 0004).
- The words "quest" and "guild" remain in code and API paths; a rename there would break
  `shai` and saved cards for no user benefit.
