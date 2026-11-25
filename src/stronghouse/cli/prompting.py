"""Reading answers typed at the terminal. Tests replace prompt_user."""

from __future__ import annotations


def prompt_user(prompt: str) -> str:
    """Read one line from the person at the terminal."""
    return input(prompt)
