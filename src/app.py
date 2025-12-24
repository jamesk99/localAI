import logging
import os
import sys
import time
import uuid
import json

from logging.handlers import RotatingFileHandler
from flask import Flask, request, jsonify, send_from_directory, Response

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

from query import initialize_rag_system, create_query_engine, format_response

app = Flask(__name__)

LOG_LEVEL = os.getenv("RAG_LOG_LEVEL", "INFO").upper()

# Default log file location (always-on file logging for production)
LOG_DIR = os.path.join(os.path.dirname(CURRENT_DIR), "logs")
DEFAULT_LOG_FILE = os.path.join(LOG_DIR, "rag_app.log")
LOG_FILE = os.getenv("RAG_LOG_FILE", DEFAULT_LOG_FILE)

LOG_QUESTIONS = os.getenv("RAG_LOG_QUESTIONS", "true").lower() == "true"

# NOTE: Future logging extensibility points:
# - Add a separate logger for user activity (login/logout events) → logger_user_activity
# - Add a logger for chat/conversation history → logger_chat_history
# - Add a logger for RAG performance metrics (latency, token counts) → logger_metrics
# - Consider structured JSON logging for easier parsing and aggregation
# - For scaling: ship logs to centralized system (ELK, Loki, CloudWatch, etc.)

logger = logging.getLogger("rag_app")
if not logger.handlers:
    level = getattr(logging, LOG_LEVEL, logging.INFO)
    logger.setLevel(level)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    # Console handler (always on)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # File handler (always on, with rotation)
    if LOG_FILE:
        # Ensure log directory exists
        log_dir = os.path.dirname(LOG_FILE)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        
        file_handler = RotatingFileHandler(
            LOG_FILE,
            maxBytes=5 * 1024 * 1024,  # 5 MB per file
            backupCount=10,             # Keep 10 backups (~50 MB total)
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)


AUTH_USERS = None
AUTH_USERS_FILE = os.getenv("RAG_USER_FILE")
if not AUTH_USERS_FILE:
    default_auth_file = os.path.join(CURRENT_DIR, "auth_users.json")
    if os.path.exists(default_auth_file):
        AUTH_USERS_FILE = default_auth_file


def load_auth_users():
    global AUTH_USERS
    if not AUTH_USERS_FILE:
        AUTH_USERS = None
        return
    try:
        with open(AUTH_USERS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            AUTH_USERS = {str(k): str(v) for k, v in data.items()}
        else:
            AUTH_USERS = None
            logger.error(
                "auth user file %s must contain a JSON object of username: password pairs",
                AUTH_USERS_FILE,
            )
    except Exception as exc:
        AUTH_USERS = None
        logger.error("error loading auth user file %s: %s", AUTH_USERS_FILE, exc)


load_auth_users()

# Log auth mode on startup
if AUTH_USERS is not None:
    logger.info(
        "auth_mode=json_file user_count=%d file=%s",
        len(AUTH_USERS),
        AUTH_USERS_FILE,
    )
else:
    logger.info("auth_mode=env_single_user user=%s", os.getenv("RAG_USER", "raguser"))

USERNAME = os.getenv("RAG_USER", "raguser")
PASSWORD = os.getenv("RAG_PASS", "changeme")


def check_auth(username, password):
    if AUTH_USERS is not None:
        expected = AUTH_USERS.get(username)
        return expected is not None and expected == password
    return username == USERNAME and password == PASSWORD


def authenticate():
    return Response(
        "Authentication required",
        401,
        {"WWW-Authenticate": 'Basic realm="Local RAG AI System"'},
    )


def requires_auth(f):
    from functools import wraps

    @wraps(f)
    def wrapper(*args, **kwargs):
        auth = request.authorization
        if not auth or not check_auth(auth.username, auth.password):
            return authenticate()
        return f(*args, **kwargs)

    return wrapper


index = initialize_rag_system()
query_engine = create_query_engine(index)

# Log RAG system startup info
from config import LLM_MODEL, EMBED_MODEL, OLLAMA_BASE_URL
logger.info(
    "rag_system_initialized llm_model=%s embed_model=%s ollama_url=%s",
    LLM_MODEL,
    EMBED_MODEL,
    OLLAMA_BASE_URL,
)


@app.route("/", methods=["GET"])
@requires_auth
def serve_index():
    auth = request.authorization
    username = auth.username if auth else "unknown"
    client_ip = request.remote_addr or "unknown"
    logger.info("event=serve_index user=%s ip=%s", username, client_ip)
    return send_from_directory(CURRENT_DIR, "index.html")


@app.route("/api/query", methods=["POST"])
@requires_auth
def api_query():
    start_time = time.time()
    request_id = uuid.uuid4().hex
    auth = request.authorization
    username = auth.username if auth else "unknown"
    client_ip = request.remote_addr or "unknown"

    data = request.get_json(silent=True) or {}
    question = data.get("question", "").strip()
    if not question:
        return jsonify({"error": "Missing 'question'"}), 400

    try:
        preview = question[:200]
        if not LOG_QUESTIONS:
            preview = ""
        logger.info(
            "event=query_received request_id=%s user=%s ip=%s q_len=%s q_preview=%r",
            request_id,
            username,
            client_ip,
            len(question),
            preview,
        )

        response = query_engine.query(question)
        result = format_response(response)
        duration_ms = int((time.time() - start_time) * 1000)
        sources = result.get("sources") or []
        top_score = None
        if sources and isinstance(sources[0], dict):
            top_score = sources[0].get("score")
        logger.info(
            "event=query_success request_id=%s duration_ms=%d source_count=%d top_score=%s",
            request_id,
            duration_ms,
            len(sources),
            top_score,
        )
        return jsonify(result)
    except Exception as exc:
        logger.exception("event=query_error request_id=%s", request_id)
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    host = os.environ.get("RAG_APP_HOST", "0.0.0.0")
    port = int(os.environ.get("RAG_APP_PORT", "8000"))
    debug_env = os.environ.get("FLASK_DEBUG", "").lower()
    debug = debug_env in ("1", "true", "yes", "on")
    
    logger.info(
        "flask_app_starting host=%s port=%d debug=%s log_file=%s",
        host,
        port,
        debug,
        LOG_FILE,
    )
    
    app.run(host=host, port=port, debug=debug, use_reloader=debug)
