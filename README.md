# MYGPT

MYGPT is a personal, multi-user AI chat application built with Flask, vanilla JavaScript, SQLite, and the Google Gemini API. It keeps each account's profile, preferences, conversations, and messages private to that account.

## Features

- Account registration, login, logout, and signed Flask sessions.
- User profiles and per-account assistant name, response style, and custom instructions.
- Separate conversations with automatic titles, rename, delete, and search.
- Conversation history stored in SQLite and scoped to the authenticated user.
- Gemini text responses, image uploads, and PDF question answering.
- Browser voice input, responsive MYGPT interface, typing animation, and Markdown rendering.
- Backend validation for uploads and request inputs, with safe JSON errors.

## Technologies

- Python, Flask, and Gunicorn
- Google Gen AI SDK (`google-genai`)
- SQLite
- HTML, CSS, and vanilla JavaScript
- `pypdf` for PDF processing
- pytest and Node.js tests

## Project structure

```text
chatbot/
├── app.py                 # Flask routes, Gemini integration, and SQLite access
├── static/
│   └── index.html         # MYGPT interface, styles, and browser JavaScript
├── tests/
│   ├── test_app.py        # API, database, auth, isolation, and upload tests
│   └── test_markdown_security.js
├── requirements.txt       # Runtime dependencies
├── requirements-dev.txt   # Test dependencies
├── pytest.ini
├── .env.example           # Safe configuration template
├── Procfile
└── render.yaml
```

## Local installation

Use Python 3.9 or newer and Node.js to run the JavaScript security tests.
Clone the repository and change into its directory before following the setup steps:

```bash
git clone https://github.com/21pravi/chatbot.git
cd chatbot
```

### Windows PowerShell

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt -r requirements-dev.txt
Copy-Item .env.example .env
```

### macOS or Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt -r requirements-dev.txt
cp .env.example .env
```

## Environment configuration

Edit `.env` and set a persistent Flask signing key and your Gemini API key. Generate a strong signing key locally with:

```bash
python -c "import secrets; print(secrets.token_urlsafe(48))"
```

Paste the generated value into `SECRET_KEY` in `.env`. Keep `.env` private and do not commit it. Keep the same signing key between restarts so existing sessions remain valid. Production startup fails when `SECRET_KEY` is missing.

| Variable | Default | Purpose |
| --- | --- | --- |
| `SECRET_KEY` | Required | Signs Flask session cookies; never sent to the browser. |
| `GEMINI_API_KEY` | None | Enables live Gemini responses. `GOOGLE_API_KEY` is also accepted. |
| `GEMINI_MODEL` | `gemini-3.5-flash` | Gemini model name. |
| `CHATBOT_DB` | `chatbot_data.db` | SQLite database path. |
| `MAX_UPLOAD_SIZE_MB` | `10` | Maximum total request size in MiB. |
| `APP_ENV` | `development` | Set to `production` for production startup checks. |
| `PORT` | `5000` | Local or hosting server port. |

`MAX_UPLOAD_SIZE_MB` must be a positive integer. Image uploads are restricted to supported formats and checked on the server. PDF files are signature and parser validated.

## Database setup and data

On startup, MYGPT initializes the SQLite schema and applies safe, repeatable additive migrations. The `users` table stores profiles and personalization. `conversations` stores owner, title, and timestamps. `chat_history` stores each prompt/response pair and links it to its user and, when applicable, a conversation.

Foreign-key enforcement is enabled and verified for every application database connection. Existing history without a conversation ID remains available as individual legacy entries only to its recorded owner. Rows without an owner are not exposed to users. Back up an existing database before moving or replacing it; never commit database files or backups.

## Conversation API

All conversation endpoints require an authenticated session. Ownership always comes from the session, not a request parameter.

| Method | Route | Purpose |
| --- | --- | --- |
| `GET` | `/api/conversations?q=...` | List or search the current user's conversations and owned legacy entries. |
| `POST` | `/api/conversations` | Create an empty conversation. |
| `GET` | `/api/conversations/<id>` | Load an owned conversation and its messages. |
| `PATCH` | `/api/conversations/<id>` | Rename an owned conversation. |
| `DELETE` | `/api/conversations/<id>` | Delete an owned conversation and its messages. |
| `POST` | `/api/chat` | Send a prompt/upload; include `conversation_id` to continue a conversation. |

New conversation titles are generated from the first user prompt. Legacy rows with an assigned owner remain individually viewable but read-only; unassigned rows are not listed or searchable.

## Run MYGPT

```bash
python app.py
```

Open [http://localhost:5000](http://localhost:5000). Set `GEMINI_API_KEY` to enable live AI replies. The interface can still be previewed without a running backend by opening `static/index.html` directly, but authentication and persistence require Flask.

## Run tests

```bash
python -m pytest -q
node --test tests/test_markdown_security.js
```

The Python suite uses temporary databases, synthetic accounts, and mocked Gemini calls. It covers authentication, multi-user ownership, profiles and preferences, conversations, uploads, validation, and safe errors. The Node.js suite checks Markdown link safety and inline frontend JavaScript syntax.

## Deployment

The repository includes a `Procfile` and Render configuration. Configure `SECRET_KEY` and `GEMINI_API_KEY` in the hosting provider's environment settings. Use persistent storage for SQLite if the host's filesystem is ephemeral, and back up the database before upgrades.


<img width="1920" height="903" alt="Screenshot 2026-11-10 235820" src="https://github.com/user-attachments/assets/158cbb05-acb2-465d-94ae-52afd0503631" />
<img width="1920" height="903" alt="Screenshot 2026-11-10 235804" src="https://github.com/user-attachments/assets/4139a957-6af9-4ec5-95c5-6c6cdbcc7c39" />
<img width="1920" height="907" alt="Screenshot 2026-11-10 235502" src="https://github.com/user-attachments/assets/9aa691be-6b80-4f86-a3e9-786c0a033ee4" />
<img width="1920" height="911" alt="Screenshot 2026-11-10 235353" src="https://github.com/user-attachments/assets/c4c9dda4-974f-4e1a-a165-22ed2470b164" />

