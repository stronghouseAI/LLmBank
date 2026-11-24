# 13. Copy and export chat outputs

Date: 2026-11-24 · Status: accepted

## Context
phil wants to keep what the model writes in Chat. The only way was to select text by hand,
and while a reply streamed in the page rewrote the whole reply text on every token, which
wiped any selection (the same bug the Run page engine log had, ADR 0012).

## Decision
- **Copy** under each finished reply copies just that answer. It sits in a row outside the
  answer text with `user-select: none`, so a hand selection never picks up the button label.
- **Copy chat** and **Export .txt** on the Chat page header. Both use one plain-text
  transcript built by `core/chat_transcript.py` through `POST /api/chat/transcript`, so the
  format is tested with pytest: a header (model file, date and time, message count), then
  `You:` / `Model:` blocks, with thinking marked `[Thinking] ... [End of thinking]`.
- The browser saves the file itself (a Blob download named like
  `chat-Qwen3-30B-A3B-Q4_K_M-2026-11-24-1350.txt`), so nothing leaves the computer and the
  server writes no files.
- Streaming appends new text nodes instead of rewriting `textContent`, so text can be
  selected and copied while a reply is still arriving. Chat bubbles are `user-select: text`.

## Consequences
- The transcript lives only in the page: Clear or a reload loses it unless it was exported.
  Saving chats on disk inside the app would be a separate decision.
- Failed replies (engine errors) are not in the transcript, matching what is sent to the model.
