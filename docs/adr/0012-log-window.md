# 12. Log window with copy, and a log file for the app

Date: 2026-11-24 · Status: accepted

## Context
When something breaks, phil pastes logs to Claude to fix it. The only log in the GUI was the
Engine log on the Run page, and its text could not be copied: it was redrawn every 3 seconds,
which wiped any selection. The app server itself only logged to the terminal, so job and agent
errors were lost once the terminal closed, and the `extra={...}` details (job id, model, command)
were never printed at all.

## Decision
- **app.log**: the server logs to `<data dir>/app.log` as well as the terminal, rotated at 5 MB
  with two old files kept (`config.py`). Lines are `date time LEVEL logger: message key=value`,
  so the structured `extra` fields are now visible. httpx request lines are turned down to
  WARNING, or the engine health check would fill the file.
- **Logs page**: App, Engine, Jobs, Agents and Startup (`server.log`, when `shai` started the
  app). Jobs and Agents are App filtered by logger name, with tracebacks kept with their record.
  Options: last 200/500/2000 lines, only warnings and errors, follow. **Copy** puts one log on
  the clipboard as a Markdown code block; **Copy all logs** adds GPU, RAM, CPU and engine facts.
- **Run page**: the Engine log is not redrawn while text in it is selected, keeps its scroll
  position, and has its own Copy button.
- **Secrets**: every line is passed through `core/log_view.redact` before it leaves the server
  (Hugging Face, OpenAI-style and GitHub tokens, Bearer headers, `user:pass@` in URLs and
  `token=`/`api_key=`/`password=`/`secret=` values) and the home folder is shown as `~`.
- **`shai logs [LOG] [--lines N] [--problems]`** reads the files directly, so it works when
  the app will not start.

## Consequences
- The app only reads log files; it never deletes or edits them.
- Redaction is pattern based. A secret in an unusual shape could still get through, so the
  page says what is hidden rather than promising everything is.
