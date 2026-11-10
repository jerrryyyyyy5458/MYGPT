"""
Test suite for the Gemini chatbot.

The Gemini network call is mocked, so the whole suite runs offline with no
API key. Run with:  pytest -q
"""

import base64
from contextlib import closing
import importlib
import io
import os
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture()
def app_module(tmp_path, monkeypatch):
    """Import app.py fresh with a temp DB and no API key."""
    monkeypatch.setenv("CHATBOT_DB", str(tmp_path / "test.db"))
    monkeypatch.setenv("APP_ENV", "testing")
    monkeypatch.setenv("SECRET_KEY", "test-only-stable-secret")
    monkeypatch.setenv("MAX_UPLOAD_SIZE_MB", "10")
    monkeypatch.setenv("PYTHON_DOTENV_DISABLED", "1")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    import app as app_mod

    importlib.reload(app_mod)
    return app_mod


@pytest.fixture()
def client(app_module):
    app_module.app.config.update(TESTING=True)
    client = app_module.app.test_client()
    _register_and_login(app_module, client, "test-user")
    return client


def _register_and_login(app_module, client, username):
    response = client.post("/api/register", json={
        "username": username,
        "email": f"{username}@example.test",
        "password": "correct-horse",
    })
    assert response.status_code == 201
    response = client.post("/api/login", json={
        "username": username,
        "password": "correct-horse",
    })
    assert response.status_code == 200


def _client_user_id(client):
    with client.session_transaction() as sess:
        return sess["user_id"]


def test_production_startup_requires_secret_key(tmp_path):
    env = os.environ.copy()
    env.pop("SECRET_KEY", None)
    env["APP_ENV"] = "production"
    env["PYTHON_DOTENV_DISABLED"] = "1"
    env["CHATBOT_DB"] = str(tmp_path / "missing-secret.db")
    result = subprocess.run(
        [sys.executable, "-c", "import app"],
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "SECRET_KEY must be configured in production" in result.stderr


def test_forged_session_signed_with_attacker_key_is_rejected(client, app_module):
    from flask import Flask

    user_id = _client_user_id(client)
    attacker_app = Flask("attacker")
    attacker_app.secret_key = "attacker-controlled-key"
    serializer = app_module.app.session_interface.get_signing_serializer(attacker_app)
    forged_cookie = serializer.dumps({"user_id": user_id})

    attacker = app_module.app.test_client()
    attacker.set_cookie("session", forged_cookie)
    assert attacker.get("/api/me").status_code == 401


def test_session_cookie_survives_app_reload_with_same_secret(client, app_module):
    cookie = client.get_cookie("session")
    assert cookie is not None

    importlib.reload(app_module)
    app_module.app.config.update(TESTING=True)
    restarted_client = app_module.app.test_client()
    restarted_client.set_cookie("session", cookie.value)
    response = restarted_client.get("/api/me")
    assert response.status_code == 200
    assert response.get_json()["user"]["username"] == "test-user"


# --------------------------------------------------------------------------
# Pure utility functions
# --------------------------------------------------------------------------
def test_split_text_short_returns_single_chunk(app_module):
    assert app_module.split_text("hello world") == ["hello world"]


def test_split_text_empty_returns_nothing(app_module):
    assert app_module.split_text("   ") == []


def test_split_text_long_chunks_with_overlap(app_module):
    text = ("word " * 6000).strip()  # ~30k chars
    chunks = app_module.split_text(text, chunk_size=10000, overlap=1000)
    assert len(chunks) > 1
    assert all(len(c) <= 10000 for c in chunks)


def test_format_response_wraps_code(app_module):
    out = app_module.format_response("show ```python", "print(1)")
    assert out.startswith("```") and out.endswith("```")


def test_format_response_plain_passthrough(app_module):
    assert app_module.format_response("hi", "hello") == "hello"


# --------------------------------------------------------------------------
# Database
# --------------------------------------------------------------------------
def test_db_roundtrip(app_module):
    app_module.add_chat_history("q", "a", user_id=1)
    rows = app_module.get_chat_history(1)
    assert rows[-1][1] == "q" and rows[-1][2] == "a"


def test_add_chat_history_requires_owner(app_module):
    with pytest.raises(TypeError):
        app_module.add_chat_history("q", "a")


def test_conversation_migration_is_repeatable_and_enforces_foreign_keys(app_module):
    before_conn = app_module._connect_database()
    try:
        assert before_conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        before_users = before_conn.execute("SELECT * FROM users ORDER BY id").fetchall()
        before_chats = before_conn.execute(
            "SELECT id, prompt, response, image, user_id FROM chat_history ORDER BY id"
        ).fetchall()
    finally:
        before_conn.close()

    app_module.create_database()
    app_module.create_database()

    conn = app_module._connect_database()
    try:
        assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
        assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert conn.execute("SELECT * FROM users ORDER BY id").fetchall() == before_users
        assert conn.execute(
            "SELECT id, prompt, response, image, user_id FROM chat_history ORDER BY id"
        ).fetchall() == before_chats
        assert conn.execute("SELECT COUNT(*) FROM conversations").fetchone()[0] == 0
        assert "conversation_id" in {
            row[1] for row in conn.execute("PRAGMA table_info(chat_history)")
        }
        with pytest.raises(app_module.sqlite3.IntegrityError):
            conn.execute("INSERT INTO conversations(user_id, title) VALUES(-1, 'invalid')")
        with pytest.raises(app_module.sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO chat_history(prompt, response, user_id, conversation_id) "
                "VALUES('invalid', 'invalid', NULL, -1)"
            )
    finally:
        conn.close()


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------
def test_home_serves_static_page(client):
    r = client.get("/")
    assert r.status_code == 200
    assert b"Gemini" in r.data
    # No unrendered Jinja tags should ever reach the browser.
    assert b"{{" not in r.data and b"{%" not in r.data


def test_history_endpoint_shape(client):
    r = client.get("/api/history")
    assert r.status_code == 200
    body = r.get_json()
    assert body["model"] == "gemini-3.5-flash"
    assert body["api_ready"] is False
    assert body["history"] == []


def test_duplicate_registration_login_and_logout(client, app_module):
    duplicate_username = client.post("/api/register", json={
        "username": "test-user",
        "email": "new-address@example.test",
        "password": "another-password",
    })
    duplicate_email = client.post("/api/register", json={
        "username": "new-user",
        "email": "test-user@example.test",
        "password": "another-password",
    })
    assert duplicate_username.status_code == 409
    assert duplicate_email.status_code == 409

    wrong_password = app_module.app.test_client()
    failed_login = wrong_password.post("/api/login", json={
        "username": "test-user", "password": "incorrect-password"
    })
    assert failed_login.status_code == 401
    assert wrong_password.get("/api/me").status_code == 401

    assert client.post("/api/logout").status_code == 200
    assert client.get("/api/me").status_code == 401
    assert client.get("/api/history").status_code == 401


def test_unauthenticated_users_cannot_read_history_or_chat(app_module, monkeypatch):
    app_module.app.config.update(TESTING=True)
    client = app_module.app.test_client()
    monkeypatch.setattr(app_module, "generate_reply", lambda *a, **k: "reply")

    assert client.get("/api/history").status_code == 401
    assert client.post("/api/chat", data={"prompt": "hello"}).status_code == 401


def test_chat_history_is_isolated_and_request_user_id_is_ignored(
    client, app_module, monkeypatch
):
    monkeypatch.setattr(
        app_module, "generate_reply", lambda prompt, **kwargs: f"reply to {prompt}"
    )
    alice_id = _client_user_id(client)

    bob = app_module.app.test_client()
    _register_and_login(app_module, bob, "bob")
    bob_id = _client_user_id(bob)
    assert alice_id != bob_id

    alice_chat = client.post("/api/chat", data={
        "prompt": "alice secret", "user_id": str(bob_id)
    })
    bob_chat = bob.post("/api/chat", data={"prompt": "bob secret"})
    assert alice_chat.status_code == bob_chat.status_code == 200

    alice_history = client.get(f"/api/history?user_id={bob_id}").get_json()["history"]
    bob_history = bob.get(f"/api/history?user_id={alice_id}").get_json()["history"]
    assert [row["prompt"] for row in alice_history] == ["alice secret"]
    assert [row["prompt"] for row in bob_history] == ["bob secret"]
    alice_saved_reply = app_module.get_chat_history(alice_id)[0][2]
    bob_saved_reply = app_module.get_chat_history(bob_id)[0][2]
    assert "alice secret" in alice_saved_reply and "bob secret" not in alice_saved_reply
    assert "bob secret" in bob_saved_reply and "alice secret" not in bob_saved_reply


def test_conversation_crud_search_titles_and_message_persistence(client, app_module, monkeypatch):
    monkeypatch.setattr(app_module, "generate_reply", lambda prompt, **kwargs: f"Answer: {prompt}")
    created = client.post("/api/conversations", json={"user_id": 987654})
    other_created = client.post("/api/conversations", json={})
    assert created.status_code == other_created.status_code == 201
    first_id = created.get_json()["conversation"]["id"]
    second_id = other_created.get_json()["conversation"]["id"]
    assert first_id != second_id

    first = client.post("/api/chat", data={
        "conversation_id": str(first_id), "prompt": "Plan a quiet Kyoto trip"
    })
    second = client.post("/api/chat", data={
        "conversation_id": str(second_id), "prompt": "Find a solar charger"
    })
    followup = client.post("/api/chat", data={
        "conversation_id": str(first_id), "prompt": "Include a garden visit"
    })
    assert first.status_code == followup.status_code == second.status_code == 200

    listing = client.get("/api/conversations").get_json()["conversations"]
    by_id = {item["id"]: item for item in listing}
    assert listing[0]["id"] == first_id
    assert by_id[first_id]["title"] == "Plan a quiet Kyoto trip"
    assert by_id[second_id]["title"] == "Find a solar charger"
    assert by_id[first_id]["legacy"] is False
    assert by_id[first_id]["last_prompt"] == "Include a garden visit"

    opened = client.get(f"/api/conversations/{first_id}")
    assert opened.status_code == 200
    messages = opened.get_json()["messages"]
    assert [message["prompt"] for message in messages] == [
        "Plan a quiet Kyoto trip", "Include a garden visit"
    ]

    renamed = client.patch(f"/api/conversations/{first_id}", json={"title": "Kyoto Weekend"})
    assert renamed.status_code == 200
    assert renamed.get_json()["conversation"]["title"] == "Kyoto Weekend"
    assert client.get("/api/conversations?q=kyoto").get_json()["conversations"][0]["id"] == first_id
    assert client.get("/api/conversations?q=solar").get_json()["conversations"][0]["id"] == second_id
    assert client.get("/api/conversations?q=Include%20a%20garden").get_json()["conversations"][0]["id"] == first_id

    deleted = client.delete(f"/api/conversations/{first_id}")
    assert deleted.status_code == 200
    assert client.get(f"/api/conversations/{first_id}").status_code == 404
    assert client.get(f"/api/conversations/{second_id}").status_code == 200
    remaining_prompts = [row[1] for row in app_module.get_chat_history(_client_user_id(client))]
    assert remaining_prompts == ["Find a solar charger"]


def test_conversation_ownership_and_search_are_isolated(client, app_module, monkeypatch):
    generated = []
    monkeypatch.setattr(
        app_module, "generate_reply", lambda *_a, **_k: generated.append(True) or "private answer"
    )
    alice_id = client.post("/api/conversations").get_json()["conversation"]["id"]
    assert client.post("/api/chat", data={
        "conversation_id": str(alice_id), "prompt": "Alice confidential topic"
    }).status_code == 200

    bob = app_module.app.test_client()
    _register_and_login(app_module, bob, "conversation-bob")
    bob_id = bob.post("/api/conversations").get_json()["conversation"]["id"]
    assert bob_id != alice_id

    assert bob.get(f"/api/conversations/{alice_id}").status_code == 404
    assert bob.patch(f"/api/conversations/{alice_id}", json={"title": "Stolen"}).status_code == 404
    assert bob.delete(f"/api/conversations/{alice_id}").status_code == 404
    assert bob.post("/api/chat", data={
        "conversation_id": str(alice_id), "prompt": "Try to append to Alice's chat"
    }).status_code == 404
    assert len(generated) == 1
    assert bob.get("/api/conversations?q=confidential").get_json()["conversations"] == []
    assert client.get(f"/api/conversations/{alice_id}").status_code == 200
    assert client.get(f"/api/conversations/{bob_id}").status_code == 404


def test_legacy_history_is_read_only_owner_scoped_and_unassigned_hidden(client, app_module):
    owner_id = _client_user_id(client)
    with closing(app_module._connect_database()) as conn:
        conn.execute(
            "INSERT INTO chat_history(prompt, response, user_id) VALUES(?, ?, ?)",
            ("Assigned legacy prompt", "Assigned legacy answer", owner_id),
        )
        conn.execute(
            "INSERT INTO chat_history(prompt, response, user_id) VALUES(?, ?, NULL)",
            ("Unassigned private prompt", "Unassigned private response"),
        )
        conn.commit()

    legacy_entries = client.get("/api/conversations").get_json()["conversations"]
    legacy = next(item for item in legacy_entries if item["legacy"])
    assert legacy["title"] == "Assigned legacy prompt"
    opened = client.get(f"/api/conversations/{legacy['id']}")
    assert opened.status_code == 200
    assert opened.get_json()["messages"][0]["response"] == "Assigned legacy answer"
    assert client.patch(f"/api/conversations/{legacy['id']}", json={"title": "Changed"}).status_code == 404
    assert client.delete(f"/api/conversations/{legacy['id']}").status_code == 404

    bob = app_module.app.test_client()
    _register_and_login(app_module, bob, "legacy-bob")
    assert bob.get(f"/api/conversations/{legacy['id']}").status_code == 404
    assert bob.get("/api/conversations?q=Assigned%20legacy").get_json()["conversations"] == []
    assert client.get("/api/conversations?q=Unassigned").get_json()["conversations"] == []


def test_conversation_routes_require_auth_and_validate_ids(client, app_module):
    unauthenticated = app_module.app.test_client()
    assert unauthenticated.get("/api/conversations").status_code == 401
    assert unauthenticated.post("/api/conversations").status_code == 401
    assert unauthenticated.get("/api/conversations/1").status_code == 401
    assert unauthenticated.patch("/api/conversations/1", json={"title": "x"}).status_code == 401
    assert unauthenticated.delete("/api/conversations/1").status_code == 401
    assert client.get("/api/conversations/not-an-id").status_code == 404
    missing_id = client.get("/api/conversations/")
    assert missing_id.status_code == 404 and missing_id.is_json
    assert client.get("/api/conversations/999999").status_code == 404
    assert client.post("/api/chat", data={
        "conversation_id": "not-an-id", "prompt": "Invalid ID"
    }).status_code == 400
    assert client.get("/api/conversations?q=" + "x" * 201).status_code == 400
    assert client.patch("/api/conversations/1", json={"title": []}).status_code == 400
    assert client.patch("/api/conversations/1", json={"title": " "}).status_code == 400


def test_legacy_chat_post_without_conversation_id_creates_one(client, app_module, monkeypatch):
    monkeypatch.setattr(app_module, "generate_reply", lambda *_a, **_k: "reply")
    response = client.post("/api/chat", data={"prompt": "Compatibility-created conversation"})
    assert response.status_code == 200
    body = response.get_json()
    assert body["conversation_id"] > 0
    assert body["title"] == "Compatibility-created conversation"
    assert client.get(f"/api/conversations/{body['conversation_id']}").get_json()["messages"][0]["prompt"] == "Compatibility-created conversation"


def test_conversation_frontend_controls_use_existing_mygpt_shell(client):
    page = client.get("/").get_data(as_text=True)
    for marker in (
        'id="conversation-search"',
        'id="new-chat"',
        'fetch("/api/conversations"',
        'method: "PATCH"',
        'method: "DELETE"',
        'fd.append("conversation_id"',
        "function openConversation(key)",
        "function renderHistoryList(items)",
    ):
        assert marker in page


def test_profiles_and_settings_are_private_and_persist(client, app_module):
    alice_id = _client_user_id(client)
    bob = app_module.app.test_client()
    _register_and_login(app_module, bob, "profile-bob")
    bob_id = _client_user_id(bob)

    defaults = bob.get("/api/settings").get_json()["settings"]
    assert defaults == {
        "assistant_name": "MYGPT",
        "response_style": "friendly",
        "custom_instructions": "",
    }

    profile = client.patch("/api/me", json={
        "full_name": "Alice Example", "user_id": bob_id
    })
    assert profile.status_code == 200
    saved = client.put("/api/settings", json={
        "assistant_name": "Nova",
        "response_style": "detailed",
        "custom_instructions": "Use practical examples.",
        "user_id": bob_id,
    })
    assert saved.status_code == 200

    assert bob.get("/api/me").get_json()["user"]["full_name"] is None
    assert bob.get("/api/settings").get_json()["settings"] == defaults
    assert app_module.get_user_settings(alice_id)["assistant_name"] == "Nova"

    client.post("/api/logout")
    logged_in = client.post("/api/login", json={
        "username": "test-user", "password": "correct-horse"
    })
    assert logged_in.status_code == 200
    assert client.get("/api/me").get_json()["user"]["full_name"] == "Alice Example"
    assert client.get("/api/settings").get_json()["settings"]["assistant_name"] == "Nova"


def test_saved_personalisation_is_sent_with_chat_prompt(client, app_module, monkeypatch):
    client.put("/api/settings", json={
        "assistant_name": "Nova",
        "response_style": "concise",
        "custom_instructions": "Use short bullet points.",
    })
    captured = {}

    def fake_generate(prompt, **kwargs):
        captured["prompt"] = prompt
        return "personalised reply"

    monkeypatch.setattr(app_module, "generate_reply", fake_generate)
    response = client.post("/api/chat", data={"prompt": "Explain photosynthesis"})
    assert response.status_code == 200
    assert "You are Nova" in captured["prompt"]
    assert "concise response style" in captured["prompt"]
    assert "Use short bullet points" in captured["prompt"]
    assert "Explain photosynthesis" in captured["prompt"]


def test_settings_panel_is_present_in_existing_frontend(client):
    page = client.get("/").get_data(as_text=True)
    for marker in (
        'id="settings-dialog"',
        'id="settings-full-name"',
        'id="settings-assistant-name"',
        'id="settings-response-style"',
        'id="settings-custom-instructions"',
        'fetch("/api/settings"',
    ):
        assert marker in page


def test_upload_limit_defaults_to_ten_mib(app_module):
    assert app_module.app.config["MAX_CONTENT_LENGTH"] == 10 * 1024 * 1024


def test_upload_limit_can_be_configured_with_environment(tmp_path):
    env = os.environ.copy()
    env.update({
        "APP_ENV": "testing",
        "SECRET_KEY": "subprocess-test-key",
        "MAX_UPLOAD_SIZE_MB": "2",
        "CHATBOT_DB": str(tmp_path / "configured-limit.db"),
        "PYTHON_DOTENV_DISABLED": "1",
    })
    result = subprocess.run(
        [sys.executable, "-c", "import app; print(app.app.config['MAX_CONTENT_LENGTH'])"],
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == str(2 * 1024 * 1024)


def test_upload_over_limit_returns_json_413(client, app_module, monkeypatch):
    monkeypatch.setitem(app_module.app.config, "MAX_CONTENT_LENGTH", 256)
    response = client.post("/api/chat", data={
        "prompt": "large",
        "image": (io.BytesIO(b"x" * 1024), "large.png"),
    }, content_type="multipart/form-data")
    assert response.status_code == 413
    assert response.is_json
    assert "too large" in response.get_json()["error"].lower()


def test_image_upload_requires_allowed_extension_and_signature(client, app_module, monkeypatch):
    called = []
    monkeypatch.setattr(app_module, "generate_reply", lambda *a, **k: called.append(True) or "ok")
    png_signature = b"\x89PNG\r\n\x1a\n" + b"test-data"

    bad_extension = client.post("/api/chat", data={
        "prompt": "describe", "image": (io.BytesIO(png_signature), "payload.exe")
    }, content_type="multipart/form-data")
    bad_content = client.post("/api/chat", data={
        "prompt": "describe", "image": (io.BytesIO(b"not an image"), "photo.png")
    }, content_type="multipart/form-data")

    assert bad_extension.status_code == bad_content.status_code == 400
    assert called == []


def test_valid_image_upload_is_forwarded_with_detected_mime(client, app_module, monkeypatch):
    captured = {}
    def fake_generate(prompt, image_bytes=None, image_mime=None, pdf_file=None):
        captured.update(image_bytes=image_bytes, image_mime=image_mime)
        return "image received"
    monkeypatch.setattr(app_module, "generate_reply", fake_generate)
    png_data = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
    )
    response = client.post("/api/chat", data={
        "prompt": "describe", "image": (io.BytesIO(png_data), "photo.png")
    }, content_type="multipart/form-data")
    assert response.status_code == 200
    assert captured == {"image_bytes": png_data, "image_mime": "image/png"}


@pytest.mark.parametrize(("filename", "contents", "expected_mime"), [
    ("photo.jpg", b"\xff\xd8\xffimage-data", "image/jpeg"),
    ("photo.jpeg", b"\xff\xd8\xffimage-data", "image/jpeg"),
    ("photo.webp", b"RIFF\x00\x00\x00\x00WEBPdata", "image/webp"),
    ("photo.heic", b"\x00\x00\x00\x18ftypheic0000", "image/heic"),
    ("photo.heif", b"\x00\x00\x00\x18ftypmif10000", "image/heif"),
])
def test_supported_image_signatures_are_detected(app_module, filename, contents, expected_mime):
    assert app_module.validate_image_upload(filename, contents) == expected_mime


def test_valid_pdf_upload_is_forwarded(client, app_module, monkeypatch):
    from pypdf import PdfWriter

    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    pdf_stream = io.BytesIO()
    writer.write(pdf_stream)
    pdf_stream.seek(0)
    captured = {}
    def fake_generate(prompt, image_bytes=None, image_mime=None, pdf_file=None):
        captured["pdf_bytes"] = pdf_file.read()
        return "PDF received"
    monkeypatch.setattr(app_module, "generate_reply", fake_generate)

    response = client.post("/api/chat", data={
        "prompt": "summarize", "pdf_file": (pdf_stream, "report.pdf")
    }, content_type="multipart/form-data")
    assert response.status_code == 200
    assert captured["pdf_bytes"].startswith(b"%PDF-")


def test_malformed_pdf_is_rejected_before_gemini(client, app_module, monkeypatch):
    called = []
    monkeypatch.setattr(app_module, "generate_reply", lambda *a, **k: called.append(True) or "ok")
    response = client.post("/api/chat", data={
        "prompt": "summarize", "pdf_file": (io.BytesIO(b"%PDF-1.7\nnot a valid pdf"), "report.pdf")
    }, content_type="multipart/form-data")
    assert response.status_code == 400
    assert "PDF" in response.get_json()["error"]
    assert called == []


@pytest.mark.parametrize("payload", [[], "text", 42, None])
def test_auth_routes_reject_non_object_json(client, payload):
    for route in ("/api/register", "/api/login"):
        response = client.post(route, json=payload)
        assert response.status_code == 400
        assert response.is_json


@pytest.mark.parametrize("payload", [
    {"username": 7, "email": "a@example.test", "password": "correct-horse"},
    {"username": "name", "email": [], "password": "correct-horse"},
    {"username": "name", "email": "a@example.test", "password": {"x": 1}},
    {"username": None, "email": "a@example.test", "password": "correct-horse"},
])
def test_registration_rejects_incorrect_field_types(client, payload):
    assert client.post("/api/register", json=payload).status_code == 400


def test_registration_rejects_overlong_fields(client):
    assert client.post("/api/register", json={
        "username": "u" * 65, "email": "a@example.test", "password": "correct-horse"
    }).status_code == 400
    assert client.post("/api/register", json={
        "username": "new-user", "email": "a" * 250 + "@test", "password": "correct-horse"
    }).status_code == 400


@pytest.mark.parametrize("payload", [
    {"username": 7, "password": "correct-horse"},
    {"username": "test-user", "password": []},
    {"username": "test-user", "password": None},
])
def test_login_rejects_incorrect_field_types(client, payload):
    assert client.post("/api/login", json=payload).status_code == 400


def test_unknown_path_falls_back_to_app(client):
    r = client.get("/whatever")
    assert r.status_code == 200
    assert b"Gemini" in r.data


def test_api_empty_prompt_rejected(client):
    r = client.post("/api/chat", data={"prompt": "   "})
    assert r.status_code == 400
    assert "error" in r.get_json()


def test_api_without_key_returns_setup_message(client):
    r = client.post("/api/chat", data={"prompt": "hello"})
    assert r.status_code == 500
    assert r.get_json()["error"] == "MYGPT could not process that request. Please try again."


# --------------------------------------------------------------------------
# End-to-end request flow with a mocked model
# --------------------------------------------------------------------------
def test_api_text_prompt_mocked(client, app_module, monkeypatch):
    monkeypatch.setattr(app_module, "generate_reply", lambda *a, **k: "mocked reply")
    r = client.post("/api/chat", data={"prompt": "hello"})
    assert r.status_code == 200
    assert r.get_json()["response"] == "mocked reply"
    user_id = _client_user_id(client)
    assert app_module.get_chat_history(user_id)[-1][2] == "mocked reply"
    # and it shows up in history endpoint
    hist = client.get("/api/history").get_json()["history"]
    assert hist[-1]["prompt"] == "hello"


def test_api_image_prompt_persists_base64(client, app_module, monkeypatch):
    captured = {}

    def fake_generate(prompt, image_bytes=None, image_mime=None, pdf_file=None):
        captured["image_bytes"] = image_bytes
        captured["mime"] = image_mime
        return "saw the image"

    monkeypatch.setattr(app_module, "generate_reply", fake_generate)

    fake_png = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
    )
    data = {"prompt": "describe this", "image": (io.BytesIO(fake_png), "pic.png")}
    r = client.post("/api/chat", data=data, content_type="multipart/form-data")
    assert r.status_code == 200
    assert captured["image_bytes"] == fake_png
    stored_b64 = app_module.get_chat_history(_client_user_id(client))[-1][3]
    assert base64.b64decode(stored_b64) == fake_png


def test_api_reports_model_error_cleanly(client, app_module, monkeypatch, caplog):
    def boom(*a, **k):
        raise RuntimeError("internal provider detail and fake API key secret-value")

    monkeypatch.setattr(app_module, "generate_reply", boom)
    r = client.post("/api/chat", data={"prompt": "hi"})
    assert r.status_code == 500
    assert r.get_json()["error"] == "MYGPT could not process that request. Please try again."
    assert "secret-value" not in r.get_data(as_text=True)
    assert "provider detail" not in r.get_data(as_text=True)
    assert "exception_type=RuntimeError" in caplog.text
    assert "secret-value" not in caplog.text


def test_unhandled_route_error_returns_safe_json(client, app_module, monkeypatch, caplog):
    def fail(_user_id):
        raise RuntimeError("database detail containing private-value")
    monkeypatch.setattr(app_module, "get_user_settings", fail)
    response = client.post("/api/chat", data={"prompt": "hello"})
    assert response.status_code == 500
    assert response.is_json
    assert "private-value" not in response.get_data(as_text=True)
    assert "exception_type=RuntimeError" in caplog.text
    assert "private-value" not in caplog.text
