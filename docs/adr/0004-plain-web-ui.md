# 4. Python + FastAPI backend with a plain HTML/JS UI

Date: 2026-11-23 · Status: accepted

## Context
The project's coding rules favour few, boring, long-lived dependencies and Python
tooling (ruff, pytest). Options considered: Tauri (Rust + web UI), Electron, Qt
(PySide6), Python + web UI. phil chose Python + FastAPI + web UI.

## Decision
FastAPI serves a JSON API and a static UI written in plain HTML, CSS and JavaScript
with no build step and no framework. Runtime dependencies: fastapi, uvicorn, httpx.
The app opens in the default browser.

## Consequences
- Nothing to compile; the UI can be read and edited directly.
- If the UI grows a lot, a small framework may become worth it; revisit then.
- A dedicated desktop window (e.g. pywebview, or `chromium --app=`) can be added later
  without changing the backend.
