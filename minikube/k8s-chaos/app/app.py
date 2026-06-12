from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import sqlite3, os, time, random, threading, logging, json
from datetime import datetime
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
POD_NAME   = os.environ.get("HOSTNAME", "pod-local")
LOG_DIR    = Path("/app/logs")
DB_PATH    = LOG_DIR / "events.db"
APP_LOG    = LOG_DIR / f"{POD_NAME}.log"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ── Logger ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(APP_LOG),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("demo-app")

# ── SQLite ─────────────────────────────────────────────────────────────────────
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with get_db() as db:
        db.execute("""
            CREATE TABLE IF NOT EXISTS events (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                pod       TEXT,
                type      TEXT,
                message   TEXT,
                status    TEXT DEFAULT 'ok',
                latency   REAL,
                ts        TEXT DEFAULT (datetime('now','localtime'))
            )
        """)
        db.commit()
    logger.info(f"DB inicializado em {DB_PATH}")

def insert_event(type: str, message: str, status: str = "ok", latency: float = 0.0):
    try:
        with get_db() as db:
            db.execute(
                "INSERT INTO events (pod, type, message, status, latency) VALUES (?,?,?,?,?)",
                (POD_NAME, type, message, status, round(latency, 4))
            )
            db.commit()
        logger.info(f"[DB INSERT] type={type} status={status} msg={message}")
    except Exception as e:
        logger.error(f"[DB ERROR] {e}")

# ── Background: gera eventos periódicos ───────────────────────────────────────
EVENT_TYPES = [
    ("request",  "GET /api/orders",       "ok"),
    ("request",  "POST /api/checkout",    "ok"),
    ("request",  "GET /api/products",     "ok"),
    ("cache",    "Cache hit: user_42",    "ok"),
    ("cache",    "Cache miss: product_7", "warn"),
    ("db",       "SELECT users WHERE id=88", "ok"),
    ("db",       "UPDATE orders SET status='shipped'", "ok"),
    ("auth",     "Token validated: svc-worker", "ok"),
    ("error",    "Timeout connecting to payment-svc", "error"),
    ("error",    "Retry 1/3: inventory-svc", "warn"),
]

def background_event_generator():
    time.sleep(3)
    logger.info("Gerador de eventos iniciado")
    while True:
        try:
            etype, msg, status = random.choice(EVENT_TYPES)
            latency = round(random.uniform(0.002, 0.45), 4)
            insert_event(etype, msg, status, latency)
            time.sleep(random.uniform(2, 6))
        except Exception as e:
            logger.error(f"[GENERATOR] {e}")
            time.sleep(5)

# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(title="Demo App", version="1.0.0")

@app.on_event("startup")
def startup():
    init_db()
    insert_event("system", f"Pod {POD_NAME} iniciado", "ok", 0)
    t = threading.Thread(target=background_event_generator, daemon=True)
    t.start()

# ── Middleware: loga cada request ──────────────────────────────────────────────
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    latency = time.time() - start
    if request.url.path not in ("/health", "/favicon.ico"):
        status = "ok" if response.status_code < 400 else "error"
        insert_event("http", f"{request.method} {request.url.path}", status, latency)
        logger.info(f"{request.method} {request.url.path} → {response.status_code} ({latency:.3f}s)")
    return response

# ── API endpoints ──────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "pod": POD_NAME}

@app.get("/api/status")
def status():
    with get_db() as db:
        total   = db.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        errors  = db.execute("SELECT COUNT(*) FROM events WHERE status='error'").fetchone()[0]
        last_ts = db.execute("SELECT ts FROM events ORDER BY id DESC LIMIT 1").fetchone()
    return {
        "pod": POD_NAME,
        "uptime_since": datetime.now().isoformat(),
        "total_events": total,
        "total_errors": errors,
        "last_event": last_ts[0] if last_ts else None
    }

@app.get("/api/events")
def events(limit: int = 50):
    with get_db() as db:
        rows = db.execute(
            "SELECT * FROM events ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]

@app.get("/api/events/stats")
def stats():
    with get_db() as db:
        by_type   = db.execute("SELECT type, COUNT(*) as n FROM events GROUP BY type").fetchall()
        by_status = db.execute("SELECT status, COUNT(*) as n FROM events GROUP BY status").fetchall()
        avg_lat   = db.execute("SELECT AVG(latency) FROM events WHERE latency > 0").fetchone()[0]
    return {
        "by_type":   [dict(r) for r in by_type],
        "by_status": [dict(r) for r in by_status],
        "avg_latency_ms": round((avg_lat or 0) * 1000, 2)
    }

@app.post("/api/stress")
def stress():
    start = time.time()
    _ = [x**2 for x in range(500_000)]
    latency = time.time() - start
    insert_event("stress", "CPU stress test executado", "ok", latency)
    return {"pod": POD_NAME, "duration_ms": round(latency * 1000, 1)}

# ── Dashboard HTML ─────────────────────────────────────────────────────────────
DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Demo App — Monitor</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  :root {
    --bg:       #0d1117;
    --surface:  #161b22;
    --surface2: #21262d;
    --border:   #30363d;
    --text:     #e6edf3;
    --muted:    #7d8590;
    --ok:       #3fb950;
    --warn:     #d29922;
    --error:    #f85149;
    --accent:   #58a6ff;
    --purple:   #bc8cff;
    --mono:     'JetBrains Mono', monospace;
    --sans:     'Inter', sans-serif;
  }

  body {
    background: var(--bg);
    color: var(--text);
    font-family: var(--sans);
    font-size: 14px;
    line-height: 1.5;
    min-height: 100vh;
  }

  header {
    border-bottom: 1px solid var(--border);
    padding: 14px 28px;
    display: flex;
    align-items: center;
    gap: 12px;
    position: sticky;
    top: 0;
    background: var(--bg);
    z-index: 10;
  }
  header .logo {
    width: 28px; height: 28px;
    background: var(--accent);
    border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    font-size: 14px; font-weight: 700; color: #0d1117;
  }
  header h1 { font-size: 15px; font-weight: 600; letter-spacing: -0.2px; }
  header .pod-badge {
    margin-left: auto;
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 3px 10px;
    font-family: var(--mono);
    font-size: 11px;
    color: var(--accent);
  }
  .live-dot {
    width: 7px; height: 7px;
    background: var(--ok);
    border-radius: 50%;
    animation: pulse 2s infinite;
    margin-left: 10px;
  }
  @keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.3; }
  }

  main { padding: 24px 28px; max-width: 1200px; margin: 0 auto; }

  .stats-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
    gap: 12px;
    margin-bottom: 24px;
  }
  .stat-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 16px;
  }
  .stat-card .label {
    font-size: 11px;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 8px;
  }
  .stat-card .value {
    font-size: 26px;
    font-weight: 600;
    font-family: var(--mono);
    line-height: 1;
  }
  .stat-card .sub {
    font-size: 11px;
    color: var(--muted);
    margin-top: 6px;
  }
  .val-ok     { color: var(--ok); }
  .val-warn   { color: var(--warn); }
  .val-error  { color: var(--error); }
  .val-accent { color: var(--accent); }
  .val-purple { color: var(--purple); }

  .section-title {
    font-size: 12px;
    font-weight: 500;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 10px;
  }

  .log-panel {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    overflow: hidden;
    margin-bottom: 24px;
  }
  .log-panel-header {
    padding: 10px 16px;
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: center;
    gap: 8px;
  }
  .log-panel-header span { font-size: 12px; font-weight: 500; }
  .log-panel-header .count-badge {
    margin-left: auto;
    background: var(--surface2);
    border-radius: 10px;
    padding: 2px 8px;
    font-size: 11px;
    color: var(--muted);
    font-family: var(--mono);
  }
  .log-table-wrap { overflow-x: auto; max-height: 360px; overflow-y: auto; }
  table { width: 100%; border-collapse: collapse; font-size: 12px; font-family: var(--mono); }
  th {
    position: sticky; top: 0;
    background: var(--surface2);
    padding: 8px 14px;
    text-align: left;
    font-weight: 500;
    color: var(--muted);
    font-size: 11px;
    letter-spacing: 0.04em;
    white-space: nowrap;
    border-bottom: 1px solid var(--border);
  }
  td {
    padding: 7px 14px;
    border-bottom: 1px solid rgba(48,54,61,0.5);
    white-space: nowrap;
    max-width: 300px;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  tr:last-child td { border-bottom: none; }
  tr:hover td { background: rgba(255,255,255,0.02); }

  .badge {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    padding: 2px 7px;
    border-radius: 20px;
    font-size: 10px;
    font-weight: 500;
    font-family: var(--sans);
  }
  .badge-ok     { background: rgba(63,185,80,.15);   color: var(--ok);     border: 1px solid rgba(63,185,80,.3); }
  .badge-warn   { background: rgba(210,153,34,.15);  color: var(--warn);   border: 1px solid rgba(210,153,34,.3); }
  .badge-error  { background: rgba(248,81,73,.15);   color: var(--error);  border: 1px solid rgba(248,81,73,.3); }
  .badge-http   { background: rgba(88,166,255,.1);   color: var(--accent); border: 1px solid rgba(88,166,255,.25); }
  .badge-db     { background: rgba(188,140,255,.1);  color: var(--purple); border: 1px solid rgba(188,140,255,.25); }
  .badge-system { background: rgba(125,133,144,.1);  color: var(--muted);  border: 1px solid rgba(125,133,144,.2); }
  .badge-cache  { background: rgba(210,153,34,.1);   color: var(--warn);   border: 1px solid rgba(210,153,34,.25); }
  .badge-auth   { background: rgba(63,185,80,.1);    color: var(--ok);     border: 1px solid rgba(63,185,80,.25); }
  .badge-stress { background: rgba(248,81,73,.1);    color: var(--error);  border: 1px solid rgba(248,81,73,.25); }

  .latency { color: var(--muted); }
  .latency.slow { color: var(--warn); }

  .actions { display: flex; gap: 8px; margin-bottom: 20px; flex-wrap: wrap; }
  button {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 7px;
    color: var(--text);
    font-family: var(--sans);
    font-size: 12px;
    font-weight: 500;
    padding: 7px 14px;
    cursor: pointer;
    transition: background 0.15s, border-color 0.15s;
  }
  button:hover { background: var(--surface2); border-color: var(--accent); color: var(--accent); }
  button.primary {
    background: var(--accent);
    border-color: var(--accent);
    color: #0d1117;
    font-weight: 600;
  }
  button.primary:hover { background: #79b8ff; border-color: #79b8ff; color: #0d1117; }

  #refresh-timer { font-size: 11px; color: var(--muted); margin-left: auto; align-self: center; font-family: var(--mono); }

  .empty { padding: 40px; text-align: center; color: var(--muted); font-size: 13px; }

  footer {
    border-top: 1px solid var(--border);
    padding: 12px 28px;
    font-size: 11px;
    color: var(--muted);
    display: flex;
    gap: 16px;
    flex-wrap: wrap;
  }
  footer a { color: var(--accent); text-decoration: none; }
  footer a:hover { text-decoration: underline; }
</style>
</head>
<body>

<header>
  <div class="logo">D</div>
  <h1>Demo App</h1>
  <div class="live-dot" title="Live"></div>
  <span class="pod-badge" id="pod-name">carregando...</span>
</header>

<main>
  <div class="stats-grid" id="stats-grid">
    <div class="stat-card"><div class="label">Total eventos</div><div class="value val-accent" id="s-total">—</div><div class="sub">desde o início</div></div>
    <div class="stat-card"><div class="label">Erros</div><div class="value val-error" id="s-errors">—</div><div class="sub">status error</div></div>
    <div class="stat-card"><div class="label">Latência média</div><div class="value val-purple" id="s-latency">—</div><div class="sub">ms por evento</div></div>
    <div class="stat-card"><div class="label">Último evento</div><div class="value val-ok" style="font-size:13px;padding-top:4px" id="s-last">—</div><div class="sub">horário local</div></div>
  </div>

  <div class="actions">
    <button class="primary" onclick="loadEvents()">↻ Atualizar</button>
    <button onclick="triggerStress()">⚡ CPU Stress</button>
    <button onclick="window.open('/docs','_blank')">📄 OpenAPI Docs</button>
    <span id="refresh-timer">auto-refresh: 5s</span>
  </div>

  <div class="section-title">Eventos recentes — banco de dados</div>
  <div class="log-panel">
    <div class="log-panel-header">
      <span>events</span>
      <span style="font-size:11px;color:var(--muted);font-family:var(--mono)">/app/logs/events.db</span>
      <span class="count-badge" id="event-count">0 rows</span>
    </div>
    <div class="log-table-wrap">
      <table>
        <thead>
          <tr>
            <th>#</th>
            <th>timestamp</th>
            <th>type</th>
            <th>status</th>
            <th>latency</th>
            <th>message</th>
            <th>pod</th>
          </tr>
        </thead>
        <tbody id="events-body">
          <tr><td colspan="7" class="empty">Carregando eventos...</td></tr>
        </tbody>
      </table>
    </div>
  </div>
</main>

<footer>
  <span>Demo App v1.0</span>
  <a href="/api/status">GET /api/status</a>
  <a href="/api/events">GET /api/events</a>
  <a href="/api/events/stats">GET /api/events/stats</a>
  <a href="/docs">Swagger UI</a>
</footer>

<script>
const typeBadge = t => {
  const map = {http:'badge-http', db:'badge-db', cache:'badge-cache',
                auth:'badge-auth', error:'badge-error', system:'badge-system',
                stress:'badge-stress', request:'badge-http'};
  return `<span class="badge ${map[t]||'badge-system'}">${t}</span>`;
};
const statusBadge = s => {
  const map = {ok:'badge-ok', warn:'badge-warn', error:'badge-error'};
  return `<span class="badge ${map[s]||'badge-system'}">${s}</span>`;
};
const latencyCell = ms => {
  const cls = ms > 200 ? 'slow' : '';
  return ms > 0 ? `<span class="latency ${cls}">${ms.toFixed(1)}ms</span>` : '<span class="latency">—</span>';
};

async function loadStats() {
  try {
    const [st, stats] = await Promise.all([
      fetch('/api/status').then(r=>r.json()),
      fetch('/api/events/stats').then(r=>r.json())
    ]);
    document.getElementById('pod-name').textContent = st.pod;
    document.getElementById('s-total').textContent = st.total_events.toLocaleString();
    document.getElementById('s-errors').textContent = st.total_errors.toLocaleString();
    document.getElementById('s-latency').textContent = stats.avg_latency_ms + ' ms';
    document.getElementById('s-last').textContent = st.last_event ? st.last_event.slice(11,19) : '—';
  } catch(e) { console.error(e); }
}

async function loadEvents() {
  try {
    const rows = await fetch('/api/events?limit=80').then(r=>r.json());
    document.getElementById('event-count').textContent = rows.length + ' rows';
    if (!rows.length) {
      document.getElementById('events-body').innerHTML = '<tr><td colspan="7" class="empty">Nenhum evento ainda.</td></tr>';
      return;
    }
    document.getElementById('events-body').innerHTML = rows.map(r => `
      <tr>
        <td style="color:var(--muted)">${r.id}</td>
        <td style="color:var(--muted)">${r.ts.slice(11,19)}</td>
        <td>${typeBadge(r.type)}</td>
        <td>${statusBadge(r.status)}</td>
        <td>${latencyCell(r.latency * 1000)}</td>
        <td style="color:var(--text);max-width:280px;overflow:hidden;text-overflow:ellipsis">${r.message}</td>
        <td style="color:var(--muted);font-size:11px">${r.pod}</td>
      </tr>
    `).join('');
  } catch(e) { console.error(e); }
}

async function triggerStress() {
  const btn = event.target;
  btn.textContent = '⏳ Rodando...';
  btn.disabled = true;
  try {
    const r = await fetch('/api/stress', {method:'POST'}).then(r=>r.json());
    btn.textContent = `✅ ${r.duration_ms}ms`;
    setTimeout(() => { btn.textContent = '⚡ CPU Stress'; btn.disabled = false; }, 2000);
    loadEvents(); loadStats();
  } catch(e) {
    btn.textContent = '⚡ CPU Stress';
    btn.disabled = false;
  }
}

// Auto-refresh a cada 5s
let countdown = 5;
setInterval(() => {
  countdown--;
  document.getElementById('refresh-timer').textContent = `auto-refresh: ${countdown}s`;
  if (countdown <= 0) {
    countdown = 5;
    loadEvents();
    loadStats();
  }
}, 1000);

loadStats();
loadEvents();
</script>
</body>
</html>"""

@app.get("/", response_class=HTMLResponse)
def dashboard():
    return DASHBOARD_HTML
