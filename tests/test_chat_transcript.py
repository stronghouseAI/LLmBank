"""Saving a chat: the plain-text transcript, its file name, and the API behind Copy chat/Export."""

from datetime import datetime
from pathlib import Path

from stronghouse.core.chat_transcript import ChatTurn, transcript_filename, transcript_text
from tests.test_api import client  # noqa: F401  (pytest fixture)

WHEN = datetime(2026, 11, 24, 13, 5)
TURNS = [
    ChatTurn("user", "What is 2+2?"),
    ChatTurn("assistant", "4.\n\nIt is basic addition.", thinking="  Simple sum.  "),
]


def test_transcript_has_header_and_every_message_in_order():
    text = transcript_text(TURNS, "Qwen3-30B.gguf", WHEN)
    assert text.startswith("StronghouseAI chat\nModel: Qwen3-30B.gguf\nSaved: 2026-11-24 13:05\n")
    assert "Messages: 2" in text
    assert text.index("You:\nWhat is 2+2?") < text.index("Model:\n")
    assert "4.\n\nIt is basic addition." in text  # paragraphs kept as written
    assert text.endswith("It is basic addition.\n")


def test_thinking_is_marked_and_can_be_left_out():
    with_thinking = transcript_text(TURNS, None, WHEN)
    assert "[Thinking]\nSimple sum.\n[End of thinking]\n4." in with_thinking
    assert "Model: unknown" in with_thinking
    assert "Thinking" not in transcript_text(TURNS, None, WHEN, include_thinking=False)


def test_empty_reply_and_unknown_role_are_named():
    text = transcript_text([ChatTurn("assistant", "  "), ChatTurn("function", "x")], None, WHEN)
    assert "Model:\n(empty)" in text
    assert "Function:\nx" in text


def test_filename_is_safe_and_dated():
    assert transcript_filename("Qwen3-30B-A3B-Q4_K_M.gguf", WHEN) == (
        "chat-Qwen3-30B-A3B-Q4_K_M-2026-11-24-1305.txt"
    )
    assert transcript_filename("DeepSeek-V3-00001-of-00009.gguf", WHEN) == (
        "chat-DeepSeek-V3-2026-11-24-1305.txt"
    )
    assert transcript_filename("../../etc/pass wd", WHEN) == "chat-etc-pass-wd-2026-11-24-1305.txt"
    assert transcript_filename(None, WHEN) == "chat-2026-11-24-1305.txt"


def test_transcript_api_names_the_loaded_model(client):  # noqa: F811
    test_client, services = client
    services.engine.started = (Path("/models/m.gguf"), None)
    body = test_client.post(
        "/api/chat/transcript",
        json={
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
            ]
        },
    ).json()
    assert "Model: m.gguf" in body["text"]
    assert "You:\nhi" in body["text"] and "Model:\nhello" in body["text"]
    assert body["filename"].startswith("chat-m-") and body["filename"].endswith(".txt")


def test_transcript_api_rejects_an_empty_chat(client):  # noqa: F811
    test_client, _ = client
    assert test_client.post("/api/chat/transcript", json={"messages": []}).status_code == 422
