# 16. Tool calls written as text

Date: 2026-11-25 · Status: accepted

## Context
After PR 21 phil ran `shai chat --web --files` with an Nvidia model and asked for his Download
folder. The tools were sent (llama-server runs with `--jinja`), but the model answered "I cannot
access your files" and never called one. Many local models do this: their chat template drops
the tool list, llama.cpp does not recognise their call format so the call arrives as plain text,
or their training makes them refuse before they think of the tools.

## Decision
- When the terminal offers tools it also sends `text_tool_calls: true`. The server then adds a
  system note: you really have these tools, never say you cannot access files or the internet,
  the tools with their arguments, and a plain-text format to call one:
  `<tool_call>{"name": ..., "arguments": {...}}</tool_call>` (the tag Qwen and Hermes already use).
- `core/text_tool_calls.py` reads that format back, plus Nemotron's `<TOOLCALL>[...]</TOOLCALL>`
  and a reply that is only one JSON call. Only tools that are switched on count.
- A text call goes through the same `Allow once? [y/N]` question. Its result goes back as a
  user message, so templates that know nothing about tool messages still work. The call text
  itself is not printed.
- If tools were on, nothing was called and the reply says it cannot access files or the
  internet, the terminal prints a note that the model ignored its tools, with a tip.

## Consequences
- Native tool calls still win; the text format is only read when there are none.
- The browser Chat page does not send `text_tool_calls` yet.
