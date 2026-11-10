"""
Gemini-powered multimodal chatbot — Flask backend (API + static host).

The frontend (static/index.html) is a self-contained page with no server-side
templating, so it renders correctly whether it's opened directly, previewed, or
served here. This backend does three things:

  * serves the frontend at  GET  /
  * returns chat history at  GET  /api/history
  * answers prompts at       POST /api/chat   (text, image, or PDF)

Chat turns persist to SQLite. The app degrades gracefully: with no API key it
still serves every page and returns a clear setup message instead of crashing.

Built on the unified Google Gen AI SDK (`google-genai`). The legacy
`google-generativeai` package and its `gemini-pro` / `gemini-pro-vision` model
names are both retired.
"""

from werkzeug.security import generate_password_hash, check_password_hash
import base64
import io
import logging
import os
import re
import sqlite3
from contextlib import closing

from flask import Flask, g, jsonify, request, send_from_directory, session
from werkzeug.exceptions import HTTPException, RequestEntityTooLarge
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # dotenv is optional at runtime
    pass

try:
    from google import genai
    from google.genai import types

    _GENAI_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - only hit when dep is absent
    genai = None
    types = None
    _GENAI_IMPORT_ERROR = exc

from pypdf import PdfReader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("chatbot")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
API_KEY = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
MODEL_NAME = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash")
DATABASE = os.environ.get("CHATBOT_DB", "chatbot_data.db")
RESPONSE_STYLES = {"friendly", "professional", "concise", "detailed"}

try:
    MAX_UPLOAD_SIZE_MB = int(os.environ.get("MAX_UPLOAD_SIZE_MB", "10"))
except ValueError as exc:
    raise RuntimeError("MAX_UPLOAD_SIZE_MB must be a positive integer.") from exc
if MAX_UPLOAD_SIZE_MB < 1:
    raise RuntimeError("MAX_UPLOAD_SIZE_MB must be a positive integer.")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")

app = Flask(__name__, static_folder=STATIC_DIR, static_url_path="/static")
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_SIZE_MB * 1024 * 1024
APP_ENV = (os.environ.get("APP_ENV") or os.environ.get("FLASK_ENV") or "development").strip().lower()
SECRET_KEY = os.environ.get("SECRET_KEY")
if not SECRET_KEY:
    if APP_ENV in {"production", "prod"}:
        raise RuntimeError("SECRET_KEY must be configured in production.")
    raise RuntimeError(
        "SECRET_KEY is required. Add a persistent SECRET_KEY to your local .env file."
    )
app.secret_key = SECRET_KEY


def get_client():
    """Return a cached Gen AI client, or None if it cannot be created."""
    if genai is None or not API_KEY:
        return None
    if "genai_client" not in g:
        g.genai_client = genai.Client(api_key=API_KEY)
    return g.genai_client


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

def _connect_database():
    """Open a SQLite connection with foreign-key enforcement enabled."""
    conn = sqlite3.connect(DATABASE)
    try:
        conn.execute("PRAGMA foreign_keys = ON")
        enabled = conn.execute("PRAGMA foreign_keys").fetchone()
        if enabled is None or enabled[0] != 1:
            raise RuntimeError("SQLite foreign-key enforcement is unavailable.")
        return conn
    except Exception:
        conn.close()
        raise


def create_database():
    with closing(_connect_database()) as conn:
        conn.execute("BEGIN IMMEDIATE")
        try:
            # Keep one canonical account schema. IF NOT EXISTS leaves existing
            # account rows untouched.
            conn.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    email TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    full_name TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Add personalization columns without changing existing values.
            user_columns = {
                row[1] for row in conn.execute("PRAGMA table_info(users)").fetchall()
            }
            preference_columns = {
                "assistant_name": "TEXT NOT NULL DEFAULT 'MYGPT'",
                "response_style": "TEXT NOT NULL DEFAULT 'friendly'",
                "custom_instructions": "TEXT NOT NULL DEFAULT ''",
            }
            for column, definition in preference_columns.items():
                if column not in user_columns:
                    conn.execute(f"ALTER TABLE users ADD COLUMN {column} {definition}")

            conn.execute("""
                CREATE TABLE IF NOT EXISTS chat_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    prompt TEXT,
                    response TEXT,
                    image TEXT,
                    user_id INTEGER
                )
            """)

            # Add legacy ownership columns in place, preserving existing rows.
            columns = {
                row[1] for row in conn.execute("PRAGMA table_info(chat_history)").fetchall()
            }
            if "user_id" not in columns:
                conn.execute("ALTER TABLE chat_history ADD COLUMN user_id INTEGER")

            # Conversation parents are new and intentionally remain empty at
            # migration time. Historical turns are not assigned or regrouped.
            conn.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL REFERENCES users(id),
                    title TEXT,
                    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
            """)

            columns = {
                row[1] for row in conn.execute("PRAGMA table_info(chat_history)").fetchall()
            }
            if "conversation_id" not in columns:
                conn.execute("""
                    ALTER TABLE chat_history
                    ADD COLUMN conversation_id INTEGER
                    REFERENCES conversations(id) ON DELETE SET NULL
                """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_chat_user
                ON chat_history(user_id)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_conversations_owner_updated
                ON conversations(user_id, updated_at DESC, created_at DESC, id DESC)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_chat_conversation_order
                ON chat_history(conversation_id, id)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_chat_legacy_owner_order
                ON chat_history(user_id, id)
            """)

            conn.commit()
        except Exception:
            conn.rollback()
            raise

def _make_conversation_title(prompt, has_image=False, has_pdf=False):
    title = " ".join((prompt or "").split())
    if not title:
        if has_pdf:
            return "PDF conversation"
        if has_image:
            return "Image conversation"
        return "New chat"
    if len(title) <= 64:
        return title
    shortened = title[:61].rsplit(" ", 1)[0].rstrip()
    return (shortened or title[:61].rstrip()) + "…"


def create_conversation(user_id):
    """Create an empty conversation for the authenticated account."""
    if user_id is None:
        raise ValueError("user_id is required to create a conversation")
    with closing(_connect_database()) as conn:
        cursor = conn.execute(
            "INSERT INTO conversations (user_id, title) VALUES (?, NULL)",
            (user_id,),
        )
        conversation_id = cursor.lastrowid
        conn.commit()
    return conversation_id


def add_chat_history(
    prompt, response, image_b64=None, *, user_id, conversation_id=None, has_pdf=False
):
    """Store a chat turn, optionally attached to an owned conversation."""
    if user_id is None:
        raise ValueError("user_id is required to save chat history")

    with closing(_connect_database()) as conn:
        conn.execute("BEGIN IMMEDIATE")
        try:
            first_turn = False
            if conversation_id is not None:
                conversation = conn.execute(
                    "SELECT title FROM conversations WHERE id = ? AND user_id = ?",
                    (conversation_id, user_id),
                ).fetchone()
                if conversation is None:
                    raise ValueError("conversation is not owned by this user")
                first_turn = conn.execute(
                    "SELECT 1 FROM chat_history WHERE conversation_id = ? LIMIT 1",
                    (conversation_id,),
                ).fetchone() is None

            conn.execute(
                """
                INSERT INTO chat_history
                    (prompt, response, image, user_id, conversation_id)
                VALUES (?, ?, ?, ?, ?)
                """,
                (prompt, response, image_b64, user_id, conversation_id),
            )
            if conversation_id is not None:
                if first_turn and not (conversation[0] or "").strip():
                    title = _make_conversation_title(
                        prompt, has_image=bool(image_b64), has_pdf=has_pdf
                    )
                    conn.execute(
                        """UPDATE conversations
                           SET title = ?, updated_at = strftime('%Y-%m-%d %H:%M:%f', 'now')
                           WHERE id = ? AND user_id = ?""",
                        (title, conversation_id, user_id),
                    )
                else:
                    conn.execute(
                        """UPDATE conversations
                           SET updated_at = strftime('%Y-%m-%d %H:%M:%f', 'now')
                           WHERE id = ? AND user_id = ?""",
                        (conversation_id, user_id),
                    )
            conn.commit()
        except Exception:
            conn.rollback()
            raise


def get_chat_history(user_id):
    with closing(_connect_database()) as conn:
        rows = conn.execute(
            """
            SELECT id, prompt, response, image
            FROM chat_history
            WHERE user_id = ?
            ORDER BY id
            """,
            (user_id,)
        ).fetchall()

    return rows


def list_conversations(user_id, search=""):
    """List owned conversations and each assigned legacy turn separately."""
    with closing(_connect_database()) as conn:
        conversations = conn.execute(
            """
            SELECT c.id, c.title, c.created_at, c.updated_at,
                   (SELECT m.prompt FROM chat_history AS m
                    WHERE m.conversation_id = c.id AND m.user_id = c.user_id
                    ORDER BY m.id DESC LIMIT 1) AS last_prompt
            FROM conversations AS c
            WHERE c.user_id = ?
              AND (? = '' OR instr(lower(COALESCE(c.title, '')), lower(?)) > 0
                   OR EXISTS (
                       SELECT 1 FROM chat_history AS m
                       WHERE m.conversation_id = c.id AND m.user_id = c.user_id
                         AND (instr(lower(COALESCE(m.prompt, '')), lower(?)) > 0
                              OR instr(lower(COALESCE(m.response, '')), lower(?)) > 0)
                   ))
            ORDER BY c.updated_at DESC, c.id DESC
            """,
            (user_id, search, search, search, search),
        ).fetchall()
        legacy = conn.execute(
            """
            SELECT id, prompt, response, image
            FROM chat_history
            WHERE user_id = ? AND conversation_id IS NULL
              AND (? = '' OR instr(lower(COALESCE(prompt, '')), lower(?)) > 0
                   OR instr(lower(COALESCE(response, '')), lower(?)) > 0)
            ORDER BY id DESC
            """,
            (user_id, search, search, search),
        ).fetchall()

    results = [{
        "id": row[0], "title": row[1] or "New chat", "created_at": row[2],
        "updated_at": row[3], "last_prompt": row[4], "legacy": False,
    } for row in conversations]
    results.extend({
        "id": f"legacy-{row[0]}",
        "title": _make_conversation_title(row[1], has_image=bool(row[3])),
        "created_at": None, "updated_at": None, "last_prompt": row[1], "legacy": True,
    } for row in legacy)
    def recency_key(item):
        item_id = item["id"]
        if item["legacy"]:
            item_id = int(str(item_id).removeprefix("legacy-"))
            return ("", 0, item_id)
        return (item["updated_at"] or "", 1, int(item_id))

    results.sort(key=recency_key, reverse=True)
    return results


def get_conversation(user_id, conversation_key):
    """Return an owned conversation, or one owned legacy turn, if present."""
    if conversation_key.startswith("legacy-"):
        suffix = conversation_key[7:]
        if not suffix.isdecimal() or int(suffix) < 1:
            return None
        legacy_id = int(suffix)
        with closing(_connect_database()) as conn:
            row = conn.execute(
                """SELECT id, prompt, response, image FROM chat_history
                   WHERE id = ? AND user_id = ? AND conversation_id IS NULL""",
                (legacy_id, user_id),
            ).fetchone()
        if row is None:
            return None
        return {
            "conversation": {
                "id": f"legacy-{row[0]}",
                "title": _make_conversation_title(row[1], has_image=bool(row[3])),
                "legacy": True,
            },
            "messages": [{"prompt": row[1], "response": row[2], "image": row[3]}],
        }

    if not conversation_key.isdecimal() or int(conversation_key) < 1:
        return None
    conversation_id = int(conversation_key)
    with closing(_connect_database()) as conn:
        conversation = conn.execute(
            """SELECT id, title, created_at, updated_at FROM conversations
               WHERE id = ? AND user_id = ?""",
            (conversation_id, user_id),
        ).fetchone()
        if conversation is None:
            return None
        messages = conn.execute(
            """SELECT prompt, response, image FROM chat_history
               WHERE conversation_id = ? AND user_id = ? ORDER BY id""",
            (conversation_id, user_id),
        ).fetchall()
    return {
        "conversation": {
            "id": conversation[0], "title": conversation[1] or "New chat",
            "created_at": conversation[2], "updated_at": conversation[3],
            "legacy": False,
        },
        "messages": [
            {"prompt": row[0], "response": row[1], "image": row[2]}
            for row in messages
        ],
    }


def get_user_settings(user_id):
    with closing(_connect_database()) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            """SELECT assistant_name, response_style, custom_instructions
               FROM users WHERE id = ?""",
            (user_id,),
        ).fetchone()
    return dict(row) if row else None


def build_personalized_prompt(prompt, settings):
    """Apply an account's saved preferences as user-level response guidance."""
    guidance = [
        f"You are {settings['assistant_name']}, the user's AI assistant.",
        f"Use a {settings['response_style']} response style.",
    ]
    if settings["custom_instructions"]:
        guidance.append(f"User preferences: {settings['custom_instructions']}")
    guidance.extend([
        "Use these preferences when they fit the request.",
        "The user's message begins below:",
        prompt,
    ])
    return "\n\n".join(guidance)

# ---------------- AUTHENTICATION ----------------

@app.route("/api/register", methods=["POST"])
def register():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "Send registration details as a JSON object."}), 400

    username = data.get("username")
    email = data.get("email")
    password = data.get("password")
    if not all(isinstance(value, str) for value in (username, email, password)):
        return jsonify({"error": "Username, email, and password must be text."}), 400

    username = username.strip()
    email = email.strip().lower()

    if not username or not email or not password:
        return jsonify({"error": "All fields are required"}), 400
    if len(username) > 64:
        return jsonify({"error": "Username must be 64 characters or fewer."}), 400
    if len(email) > 254:
        return jsonify({"error": "Email must be 254 characters or fewer."}), 400
    if len(password) > 128:
        return jsonify({"error": "Password must be 128 characters or fewer."}), 400

    if len(password) < 8:
        return jsonify({"error": "Password must be at least 8 characters"}), 400

    hashed_password = generate_password_hash(password)

    try:
        with closing(_connect_database()) as conn:
            conn.execute(
                """INSERT INTO users (username, email, password_hash)
                   VALUES (?, ?, ?)""",
                (username, email, hashed_password)
            )
            conn.commit()

        return jsonify({"message": "Account created successfully!"}), 201

    except sqlite3.IntegrityError:
        return jsonify({"error": "Username or email already exists"}), 409


@app.route("/api/login", methods=["POST"])
def login():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "Send login details as a JSON object."}), 400

    username = data.get("username")
    password = data.get("password")
    if not isinstance(username, str) or not isinstance(password, str):
        return jsonify({"error": "Username and password must be text."}), 400
    username = username.strip()

    if not username or not password:
        return jsonify({"error": "Username and password are required"}), 400
    if len(username) > 254 or len(password) > 1024:
        return jsonify({"error": "Username or password is too long."}), 400

    with closing(_connect_database()) as conn:
        conn.row_factory = sqlite3.Row

        user = conn.execute(
            "SELECT * FROM users WHERE username = ?",
            (username,)
        ).fetchone()

    if not user or not check_password_hash(user["password_hash"], password):
        return jsonify({"error": "Invalid username or password"}), 401

    session["user_id"] = user["id"]

    return jsonify({
        "message": "Login successful!",
        "user": {
            "id": user["id"],
            "username": user["username"],
            "full_name": user["full_name"],
            "assistant_name": user["assistant_name"],
            "response_style": user["response_style"],
            "custom_instructions": user["custom_instructions"],
        }
    })


@app.route("/api/logout", methods=["POST"])
def logout():
    session.clear()
    return jsonify({"message": "Logged out successfully!"})


@app.route("/api/me", methods=["GET", "PATCH"])
def current_user():
    if "user_id" not in session:
        return jsonify({"logged_in": False}), 401

    data = request.get_json(silent=True) if request.method == "PATCH" else {}
    full_name = None
    if request.method == "PATCH":
        if not isinstance(data, dict):
            return jsonify({"error": "Profile updates must be a JSON object"}), 400
        if "full_name" not in data:
            return jsonify({"error": "Full name is required"}), 400
        if not isinstance(data["full_name"], str):
            return jsonify({"error": "Full name must be text"}), 400
        full_name = data["full_name"].strip()
        if len(full_name) > 100:
            return jsonify({"error": "Full name must be 100 characters or fewer"}), 400

    with closing(_connect_database()) as conn:
        conn.row_factory = sqlite3.Row
        if request.method == "PATCH":
            conn.execute(
                "UPDATE users SET full_name = ? WHERE id = ?",
                (full_name or None, session["user_id"]),
            )
            conn.commit()

        user = conn.execute(
            """SELECT id, username, email, full_name, assistant_name,
                      response_style, custom_instructions
               FROM users WHERE id = ?""",
            (session["user_id"],)
        ).fetchone()

    if not user:
        session.clear()
        return jsonify({"logged_in": False}), 401

    return jsonify({
        "logged_in": True,
        "user": dict(user)
    })


@app.route("/api/settings", methods=["GET", "PUT"])
def user_settings():
    if "user_id" not in session:
        return jsonify({"error": "Please log in first"}), 401

    allowed_fields = {"assistant_name", "response_style", "custom_instructions"}
    updates = {}
    if request.method == "PUT":
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return jsonify({"error": "Settings must be sent as a JSON object"}), 400
        if not data or not (set(data) & allowed_fields):
            return jsonify({"error": "Provide at least one supported setting"}), 400

        if "assistant_name" in data:
            value = data["assistant_name"]
            if not isinstance(value, str) or not value.strip() or len(value.strip()) > 40:
                return jsonify({"error": "Assistant name must be 1 to 40 characters"}), 400
            updates["assistant_name"] = value.strip()

        if "response_style" in data:
            value = data["response_style"]
            if not isinstance(value, str) or value not in RESPONSE_STYLES:
                return jsonify({"error": "Choose a supported response style"}), 400
            updates["response_style"] = value

        if "custom_instructions" in data:
            value = data["custom_instructions"]
            if not isinstance(value, str) or len(value) > 4000:
                return jsonify({"error": "Custom instructions must be 4,000 characters or fewer"}), 400
            updates["custom_instructions"] = value.strip()

    with closing(_connect_database()) as conn:
        conn.row_factory = sqlite3.Row
        if updates:
            assignments = ", ".join(f"{field} = ?" for field in updates)
            conn.execute(
                f"UPDATE users SET {assignments} WHERE id = ?",
                (*updates.values(), session["user_id"]),
            )
            conn.commit()
        settings = conn.execute(
            """SELECT assistant_name, response_style, custom_instructions
               FROM users WHERE id = ?""",
            (session["user_id"],),
        ).fetchone()

    if not settings:
        session.clear()
        return jsonify({"error": "Please log in again"}), 401
    return jsonify({"settings": dict(settings)})

# ---------------------------------------------------------------------------
# Text / PDF utilities
# ---------------------------------------------------------------------------
def get_pdf_text(pdf_file):
    reader = PdfReader(pdf_file)
    return "\n".join((page.extract_text() or "") for page in reader.pages)


class InvalidUpload(ValueError):
    """A user-supplied upload has an unsupported type or invalid contents."""

    def __init__(self, client_message):
        super().__init__(client_message)
        self.client_message = client_message


def validate_image_upload(filename, image_bytes):
    extension = os.path.splitext(filename or "")[1].lower()
    if extension not in {".png", ".jpg", ".jpeg", ".webp", ".heic", ".heif"}:
        raise InvalidUpload("Choose a PNG, JPEG, WebP, HEIC, or HEIF image.")

    if extension == ".png" and image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if extension in {".jpg", ".jpeg"} and image_bytes.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if extension == ".webp" and image_bytes.startswith(b"RIFF") and image_bytes[8:12] == b"WEBP":
        return "image/webp"

    if len(image_bytes) >= 16 and image_bytes[4:8] == b"ftyp":
        brands = {image_bytes[offset:offset + 4] for offset in range(8, min(len(image_bytes), 32), 4)}
        if extension == ".heic" and brands & {b"heic", b"heix", b"hevc", b"hevx", b"heim", b"heis"}:
            return "image/heic"
        if extension == ".heif" and brands & {b"mif1", b"msf1"}:
            return "image/heif"

    raise InvalidUpload("The image extension does not match a supported image file.")


def validate_pdf_upload(filename, pdf_bytes):
    if os.path.splitext(filename or "")[1].lower() != ".pdf":
        raise InvalidUpload("Choose a PDF file with a .pdf extension.")
    if not pdf_bytes.startswith(b"%PDF-"):
        raise InvalidUpload("This file is not a valid PDF.")

    try:
        reader = PdfReader(io.BytesIO(pdf_bytes), strict=True)
        if len(reader.pages) == 0:
            raise InvalidUpload("This PDF does not contain any pages.")
    except InvalidUpload:
        raise
    except Exception as exc:
        logger.warning("Rejected malformed PDF (parser=%s)", type(exc).__name__)
        raise InvalidUpload("This PDF is malformed or unsupported. Choose a valid PDF.") from None


def split_text(text, chunk_size=10000, overlap=1000):
    """Lightweight recursive-style splitter (drops the heavy langchain dep)."""
    if len(text) <= chunk_size:
        return [text] if text.strip() else []

    chunks, start = [], 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            for sep in ("\n\n", "\n", ". ", " "):
                pivot = text.rfind(sep, start, end)
                if pivot > start:
                    end = pivot + len(sep)
                    break
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def format_response(prompt, text):
    if re.search(r"^\s*\d+\.", prompt, re.MULTILINE):
        return text
    if "```" in prompt and "```" not in text:
        return f"```\n{text}\n```"
    return text


# ---------------------------------------------------------------------------
# Gemini call
# ---------------------------------------------------------------------------
def generate_reply(prompt, image_bytes=None, image_mime=None, pdf_file=None):
    client = get_client()
    if client is None:
        raise RuntimeError(
            "Gemini is not configured. Set GEMINI_API_KEY in your environment "
            "(see .env.example) to enable live responses."
        )

    if pdf_file is not None:
        context = "\n".join(split_text(get_pdf_text(pdf_file)))
        contents = [f"{prompt}\n\nContext from the uploaded PDF:\n{context}"]
    elif image_bytes is not None:
        contents = [
            prompt,
            types.Part.from_bytes(data=image_bytes, mime_type=image_mime or "image/png"),
        ]
    else:
        contents = [prompt]

    response = client.models.generate_content(model=MODEL_NAME, contents=contents)
    return (response.text or "").strip()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/")
def home():
    return send_from_directory(STATIC_DIR, "index.html")


@app.errorhandler(RequestEntityTooLarge)
def request_too_large(_error):
    return jsonify({
        "error": f"The request is too large. Uploads must fit within {MAX_UPLOAD_SIZE_MB} MB."
    }), 413

@app.route("/api/conversations", methods=["GET", "POST"])
def api_conversations():
    if "user_id" not in session:
        return jsonify({"error": "Please log in first"}), 401
    user_id = session["user_id"]

    if request.method == "GET":
        search = (request.args.get("q") or "").strip()
        if len(search) > 200:
            return jsonify({"error": "Search text must be 200 characters or fewer."}), 400
        return jsonify({
            "model": MODEL_NAME,
            "api_ready": bool(API_KEY),
            "conversations": list_conversations(user_id, search),
        })

    payload = request.get_json(silent=True)
    if request.data and not isinstance(payload, dict):
        return jsonify({"error": "Send conversation details as a JSON object."}), 400
    if payload is not None and not isinstance(payload, dict):
        return jsonify({"error": "Send conversation details as a JSON object."}), 400
    try:
        conversation_id = create_conversation(user_id)
    except Exception as exc:
        logger.error("Conversation creation failed (exception_type=%s)", type(exc).__name__)
        return jsonify({"error": "Could not create a conversation. Please try again."}), 500
    return jsonify({"conversation": {
        "id": conversation_id, "title": "New chat", "legacy": False,
    }}), 201


@app.route(
    "/api/conversations/<conversation_key>",
    methods=["GET", "PATCH", "DELETE"],
)
def api_conversation(conversation_key):
    if "user_id" not in session:
        return jsonify({"error": "Please log in first"}), 401
    user_id = session["user_id"]

    if request.method == "GET":
        conversation = get_conversation(user_id, conversation_key)
        if conversation is None:
            return jsonify({"error": "Conversation not found."}), 404
        return jsonify(conversation)

    # Legacy records are read-only compatibility entries, never formal
    # conversations that can be renamed or deleted through this endpoint.
    if not conversation_key.isdecimal() or int(conversation_key) < 1:
        return jsonify({"error": "Conversation not found."}), 404
    conversation_id = int(conversation_key)

    if request.method == "PATCH":
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify({"error": "Send a JSON object containing a title."}), 400
        title = payload.get("title")
        if not isinstance(title, str):
            return jsonify({"error": "Conversation title must be text."}), 400
        title = " ".join(title.split())
        if not title or len(title) > 120:
            return jsonify({"error": "Conversation title must be 1 to 120 characters."}), 400
        with closing(_connect_database()) as conn:
            cursor = conn.execute(
                """UPDATE conversations
                   SET title = ?, updated_at = strftime('%Y-%m-%d %H:%M:%f', 'now')
                   WHERE id = ? AND user_id = ?""",
                (title, conversation_id, user_id),
            )
            if cursor.rowcount != 1:
                conn.rollback()
                return jsonify({"error": "Conversation not found."}), 404
            conn.commit()
        return jsonify({"conversation": {
            "id": conversation_id, "title": title, "legacy": False,
        }})

    with closing(_connect_database()) as conn:
        conn.execute("BEGIN IMMEDIATE")
        try:
            exists = conn.execute(
                "SELECT 1 FROM conversations WHERE id = ? AND user_id = ?",
                (conversation_id, user_id),
            ).fetchone()
            if exists is None:
                conn.rollback()
                return jsonify({"error": "Conversation not found."}), 404
            conn.execute(
                "DELETE FROM chat_history WHERE conversation_id = ? AND user_id = ?",
                (conversation_id, user_id),
            )
            conn.execute(
                "DELETE FROM conversations WHERE id = ? AND user_id = ?",
                (conversation_id, user_id),
            )
            conn.commit()
        except Exception:
            conn.rollback()
            raise
    return jsonify({"message": "Conversation deleted."})


@app.route("/api/history")
def api_history():

    if "user_id" not in session:
        return jsonify({"error": "Please log in first"}), 401

    history = [
        {"prompt": r[1], "response": r[2], "image": r[3]}
        for r in get_chat_history(session["user_id"])
    ]

    return jsonify({
        "model": MODEL_NAME,
        "api_ready": bool(API_KEY),
        "history": history
    })



@app.route("/api/chat", methods=["POST"])
def api_chat():
    if "user_id" not in session:
        return jsonify({"error": "Please log in first"}), 401
    user_id = session["user_id"]
    settings = get_user_settings(user_id)
    if settings is None:
        session.clear()
        return jsonify({"error": "Please log in again"}), 401

    prompt = (request.form.get("prompt") or "").strip()
    conversation_value = request.form.get("conversation_id")
    conversation_id = None
    if conversation_value:
        if not conversation_value.isdecimal() or int(conversation_value) < 1:
            return jsonify({"error": "Conversation ID is invalid."}), 400
        conversation_id = int(conversation_value)
        with closing(_connect_database()) as conn:
            owned = conn.execute(
                "SELECT 1 FROM conversations WHERE id = ? AND user_id = ?",
                (conversation_id, user_id),
            ).fetchone()
        if owned is None:
            return jsonify({"error": "Conversation not found."}), 404

    image = request.files.get("image")
    pdf_file = request.files.get("pdf_file")

    if not prompt and not image and not pdf_file:
        return jsonify({"error": "Please enter a message."}), 400
    personalized_prompt = build_personalized_prompt(prompt, settings)

    image_bytes = image_mime = image_b64 = None
    validated_pdf = None

    try:
        if image and image.filename:
            image_bytes = image.read()
            image_mime = validate_image_upload(image.filename, image_bytes)
            image_b64 = base64.b64encode(image_bytes).decode("ascii")

        if pdf_file and pdf_file.filename:
            pdf_bytes = pdf_file.read()
            validate_pdf_upload(pdf_file.filename, pdf_bytes)
            validated_pdf = io.BytesIO(pdf_bytes)
    except InvalidUpload as exc:
        return jsonify({"error": exc.client_message}), 400

    try:
        reply = generate_reply(
            personalized_prompt,
            image_bytes=image_bytes,
            image_mime=image_mime,
            pdf_file=validated_pdf,
        )

        reply = format_response(prompt, reply) or "Gemini returned an empty response."

        if conversation_id is None:
            conversation_id = create_conversation(user_id)
        add_chat_history(
            prompt,
            reply,
            image_b64,
            user_id=user_id,
            conversation_id=conversation_id,
            has_pdf=bool(pdf_file and pdf_file.filename),
        )
        conversation = get_conversation(user_id, str(conversation_id))
        return jsonify({
            "response": reply,
            "conversation_id": conversation_id,
            "title": conversation["conversation"]["title"],
        })

    except Exception as exc:
        logger.error("Chat generation failed (exception_type=%s)", type(exc).__name__)
        return jsonify({"error": "MYGPT could not process that request. Please try again."}), 500



@app.errorhandler(404)
def not_found(_e):
    if request.path.startswith("/api/"):
        return jsonify({"error": "API endpoint or resource not found."}), 404
    return send_from_directory(STATIC_DIR, "index.html"), 200


@app.errorhandler(Exception)
def unexpected_error(error):
    if isinstance(error, HTTPException):
        return error
    logger.error("Unhandled request failure (exception_type=%s)", type(error).__name__)
    return jsonify({"error": "An unexpected server error occurred. Please try again."}), 500


# Initialise the DB at import time so `flask run`, gunicorn and tests all work.
create_database()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
