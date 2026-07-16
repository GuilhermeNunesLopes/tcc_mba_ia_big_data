import logging
from logging.handlers import RotatingFileHandler
import time, random, sys, json, signal, os, uuid, threading
from pathlib import Path
from datetime import datetime

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import uvicorn

# ── Diretório de logs ──────────────────────────────────────────────────────────
LOG_DIR  = Path("/app/logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "app_operation.log"
POD_NAME = os.environ.get("HOSTNAME", "pod-local")

# ── Formatter JSON ─────────────────────────────────────────────────────────────
class JSONFormatter(logging.Formatter):
    def format(self, record):
        return json.dumps({
            "timestamp": self.formatTime(record, self.datefmt),
            "level":     record.levelname,
            "module":    record.module,
            "pod":       POD_NAME,
            "trace_id":  getattr(record, "trace_id", "N/A"),
            "message":   record.getMessage(),
        })

# ── Logger ─────────────────────────────────────────────────────────────────────
logger = logging.getLogger("ChaosApp")
logger.setLevel(logging.DEBUG)

file_handler   = RotatingFileHandler(LOG_FILE, maxBytes=10*1024*1024, backupCount=5)
file_handler.setFormatter(JSONFormatter())
stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setFormatter(JSONFormatter())
logger.addHandler(file_handler)
logger.addHandler(stdout_handler)

# ── Estado em memória (para exibir na UI) ─────────────────────────────────────
recent_logs: list[dict] = []
MAX_RECENT  = 200
counters    = {"info": 0, "debug": 0, "warning": 0, "error": 0, "total": 0}

def record_log(level: str, message: str, trace_id: str):
    entry = {
        "ts":       datetime.now().strftime("%H:%M:%S"),
        "level":    level,
        "pod":      POD_NAME,
        "trace_id": trace_id[:8],
        "message":  message,
    }
    recent_logs.insert(0, entry)
    if len(recent_logs) > MAX_RECENT:
        recent_logs.pop()
    counters[level.lower()] = counters.get(level.lower(), 0) + 1
    counters["total"] += 1

# ── Gerador contínuo de logs (thread em background) ───────────────────────────
ACTIONS = ["process_payment", "fetch_user_data", "update_cache", "send_email", "connect_db"]

def simulate_work():
    logger.info("Aplicação iniciada. Gerando carga de logs...",
                extra={"trace_id": str(uuid.uuid4())})
    while True:
        action    = random.choice(ACTIONS)
        roll      = random.random()
        trace_id  = str(uuid.uuid4())

        if roll < 0.70:
            msg = f"Sucesso ao executar a acao: {action}"
            logger.info(msg, extra={"trace_id": trace_id})
            record_log("INFO", msg, trace_id)
        elif roll < 0.85:
            msg = f"Detalhes internos de {action} - payload size {random.randint(10, 1000)}kb"
            logger.debug(msg, extra={"trace_id": trace_id})
            record_log("DEBUG", msg, trace_id)
        elif roll < 0.95:
            msg = f"Lentidao detectada em {action} (duracao {random.uniform(1.0, 5.0):.2f}s)"
            logger.warning(msg, extra={"trace_id": trace_id})
            record_log("WARNING", msg, trace_id)
        else:
            msg = f"Falha critica ao executar {action} - Connection Timeout ou Deadlock"
            logger.error(msg, extra={"trace_id": trace_id})
            record_log("ERROR", msg, trace_id)

        time.sleep(random.uniform(0.1, 0.5))

# ── FastAPI ────────────────────────────────────────────────────────────────────
app = FastAPI(title="Chaos Log App", version="2.0.0")

@app.on_event("startup")
def startup():
    t = threading.Thread(target=simulate_work, daemon=True)
    t.start()

def handle_sigterm(*args):
    logger.info("SIGTERM recebido. Graceful shutdown...",
                extra={"trace_id": str(uuid.uuid4())})
    sys.exit(0)

signal.signal(signal.SIGTERM, handle_sigterm)

# ── API endpoints ──────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "pod": POD_NAME}

@app.get("/api/logs")
def get_logs(limit: int = 80):
    return recent_logs[:limit]

@app.get("/api/stats")
def get_stats():
    return {"pod": POD_NAME, **counters}

# ── Dashboard HTML ─────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
def dashboard():
    return """<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Chaos Log App</title>
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg:#0d1117;--sf:#161b22;--sf2:#21262d;--bd:#30363d;
  --tx:#e6edf3;--mu:#7d8590;
  --ok:#3fb950;--warn:#d29922;--err:#f85149;--ac:#58a6ff;--pu:#bc8cff;
  --mono:'JetBrains Mono','Fira Mono',monospace;
}
body{background:var(--bg);color:var(--tx);font-family:system-ui,sans-serif;font-size:13px;line-height:1.5;min-height:100vh}
header{border-bottom:1px solid var(--bd);padding:11px 22px;display:flex;align-items:center;gap:10px;position:sticky;top:0;background:var(--bg);z-index:9}
.logo{width:26px;height:26px;background:var(--ac);border-radius:7px;display:flex;align-items:center;justify-content:center;font-size:13px;font-weight:700;color:#0d1117}
header h1{font-size:14px;font-weight:600}
.dot{width:6px;height:6px;background:var(--ok);border-radius:50%;animation:pu 2s infinite}
@keyframes pu{0%,100%{opacity:1}50%{opacity:.2}}
.pod{margin-left:auto;background:var(--sf2);border:1px solid var(--bd);border-radius:20px;padding:2px 10px;font-family:var(--mono);font-size:10px;color:var(--ac)}
main{padding:18px 22px;max-width:1100px}
.grid{display:grid;grid-template-columns:repeat(5,1fr);gap:10px;margin-bottom:18px}
.card{background:var(--sf);border:1px solid var(--bd);border-radius:9px;padding:13px}
.card .lbl{font-size:10px;color:var(--mu);text-transform:uppercase;letter-spacing:.06em;margin-bottom:6px}
.card .val{font-size:22px;font-weight:600;font-family:var(--mono);line-height:1}
.card .sub{font-size:10px;color:var(--mu);margin-top:5px}
.c-ac{color:var(--ac)}.c-ok{color:var(--ok)}.c-mu{color:var(--mu)}.c-warn{color:var(--warn)}.c-err{color:var(--err)}.c-pu{color:var(--pu)}
.row{display:flex;gap:8px;margin-bottom:14px;align-items:center;flex-wrap:wrap}
button{background:var(--sf);border:1px solid var(--bd);border-radius:6px;color:var(--tx);font-size:11px;font-weight:500;padding:6px 13px;cursor:pointer;transition:.15s}
button:hover{border-color:var(--ac);color:var(--ac)}
.btn-p{background:var(--ac)!important;border-color:var(--ac)!important;color:#0d1117!important;font-weight:600!important}
.btn-p:hover{background:#79b8ff!important}
.timer{font-size:10px;color:var(--mu);margin-left:auto;font-family:var(--mono)}
.sec-title{font-size:10px;font-weight:500;color:var(--mu);text-transform:uppercase;letter-spacing:.08em;margin-bottom:8px}
.panel{background:var(--sf);border:1px solid var(--bd);border-radius:9px;overflow:hidden}
.phead{padding:8px 14px;border-bottom:1px solid var(--bd);display:flex;align-items:center;gap:8px}
.phead span{font-size:11px;font-weight:500}
.ppath{font-size:10px;color:var(--mu);font-family:var(--mono)}
.cnt{margin-left:auto;background:var(--sf2);border-radius:10px;padding:1px 7px;font-size:10px;color:var(--mu);font-family:var(--mono)}
.twrap{overflow-x:auto;max-height:420px;overflow-y:auto}
table{width:100%;border-collapse:collapse;font-size:11px;font-family:var(--mono)}
th{position:sticky;top:0;background:var(--sf2);padding:7px 12px;text-align:left;font-weight:500;color:var(--mu);font-size:10px;letter-spacing:.04em;border-bottom:1px solid var(--bd);white-space:nowrap}
td{padding:6px 12px;border-bottom:1px solid rgba(48,54,61,.45);white-space:nowrap;max-width:340px;overflow:hidden;text-overflow:ellipsis}
tr:last-child td{border-bottom:none}
tr:hover td{background:rgba(255,255,255,.018)}
.badge{display:inline-flex;padding:1px 6px;border-radius:20px;font-size:9px;font-weight:600;font-family:system-ui}
.b-info{background:rgba(88,166,255,.12);color:var(--ac);border:1px solid rgba(88,166,255,.28)}
.b-debug{background:rgba(188,140,255,.1);color:var(--pu);border:1px solid rgba(188,140,255,.28)}
.b-warning{background:rgba(210,153,34,.12);color:var(--warn);border:1px solid rgba(210,153,34,.28)}
.b-error{background:rgba(248,81,73,.12);color:var(--err);border:1px solid rgba(248,81,73,.28)}
footer{border-top:1px solid var(--bd);padding:10px 22px;font-size:10px;color:var(--mu);display:flex;gap:14px;margin-top:0}
footer a{color:var(--ac);text-decoration:none}
footer a:hover{text-decoration:underline}
</style>
</head>
<body>
<header>
  <div class="logo">C</div>
  <h1>Chaos Log App</h1>
  <div class="dot"></div>
  <span class="pod" id="pod-name">carregando...</span>
</header>
<main>
  <div class="grid">
    <div class="card"><div class="lbl">Total logs</div><div class="val c-ac" id="s-total">—</div><div class="sub">desde o início</div></div>
    <div class="card"><div class="lbl">INFO</div><div class="val c-ok" id="s-info">—</div><div class="sub">~70% dos eventos</div></div>
    <div class="card"><div class="lbl">DEBUG</div><div class="val c-pu" id="s-debug">—</div><div class="sub">detalhes internos</div></div>
    <div class="card"><div class="lbl">WARNING</div><div class="val c-warn" id="s-warning">—</div><div class="sub">lentidões</div></div>
    <div class="card"><div class="lbl">ERROR</div><div class="val c-err" id="s-error">—</div><div class="sub">falhas críticas</div></div>
  </div>

  <div class="row">
    <button class="btn-p" onclick="load()">↻ Atualizar</button>
    <button onclick="window.open('/docs','_blank')">📄 OpenAPI Docs</button>
    <button onclick="window.open('/api/logs','_blank')">🔗 JSON /api/logs</button>
    <span class="timer" id="timer">auto-refresh: 3s</span>
  </div>

  <div class="sec-title">Stream de logs — gerado continuamente em /app/logs/app_operation.log</div>
  <div class="panel">
    <div class="phead">
      <span>app_operation.log</span>
      <span class="ppath">/app/logs/app_operation.log</span>
      <span class="cnt" id="cnt">0 entradas</span>
    </div>
    <div class="twrap">
      <table>
        <thead><tr><th>hora</th><th>level</th><th>trace_id</th><th>mensagem</th><th>pod</th></tr></thead>
        <tbody id="tbody"><tr><td colspan="5" style="padding:30px;text-align:center;color:var(--mu)">Carregando logs...</td></tr></tbody>
      </table>
    </div>
  </div>
</main>
<footer>
  <span>Chaos Log App v2.0</span>
  <a href="/api/logs">GET /api/logs</a>
  <a href="/api/stats">GET /api/stats</a>
  <a href="/health">GET /health</a>
  <a href="/docs">Swagger UI</a>
</footer>
<script>
const BC = {INFO:'b-info',DEBUG:'b-debug',WARNING:'b-warning',ERROR:'b-error'};

async function load(){
  try{
    const [logs, stats] = await Promise.all([
      fetch('/api/logs?limit=100').then(r=>r.json()),
      fetch('/api/stats').then(r=>r.json())
    ]);
    document.getElementById('pod-name').textContent = stats.pod;
    document.getElementById('s-total').textContent   = (stats.total||0).toLocaleString();
    document.getElementById('s-info').textContent    = (stats.info||0).toLocaleString();
    document.getElementById('s-debug').textContent   = (stats.debug||0).toLocaleString();
    document.getElementById('s-warning').textContent = (stats.warning||0).toLocaleString();
    document.getElementById('s-error').textContent   = (stats.error||0).toLocaleString();
    document.getElementById('cnt').textContent       = logs.length + ' entradas';
    document.getElementById('tbody').innerHTML = logs.map(r=>`
      <tr>
        <td style="color:var(--mu)">${r.ts}</td>
        <td><span class="badge ${BC[r.level]||'b-info'}">${r.level}</span></td>
        <td style="color:var(--mu)">${r.trace_id}</td>
        <td style="color:var(--tx)">${r.message}</td>
        <td style="color:var(--mu);font-size:10px">${r.pod}</td>
      </tr>`).join('');
  }catch(e){console.error(e)}
}

let cd=3;
setInterval(()=>{
  cd--;
  document.getElementById('timer').textContent='auto-refresh: '+cd+'s';
  if(cd<=0){cd=3;load()}
},1000);
load();
</script>
</body>
</html>"""

# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8087, log_level="warning")
