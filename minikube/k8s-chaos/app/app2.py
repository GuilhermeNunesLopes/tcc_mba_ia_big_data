import atexit
import json
import logging
import os
import random
import sys
import threading
import time
import uuid
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import uvicorn

# ── Diretório de logs ──────────────────────────────────────────────────────────
POD_NAME = os.environ.get("HOSTNAME", "pod-local")

# LOG_DIR é configurável via env var para poder apontar para um volume externo
# (ex.: hostPath do Kubernetes mapeado para ./k8s-chaos/logs no host).
LOG_DIR = Path(os.environ.get("LOG_DIR", "/app/logs"))
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Nomes de arquivo por pod: evita colisão quando várias réplicas compartilham
# o mesmo diretório externo montado.
LOG_FILE   = LOG_DIR / f"{POD_NAME}_app_operation.log"
STATE_FILE = LOG_DIR / f"{POD_NAME}_state.json"


# ── Formatter JSON ─────────────────────────────────────────────────────────────
class JSONFormatter(logging.Formatter):
    def format(self, record):
        payload = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level":     record.levelname,
            "module":    record.module,
            "pod":       POD_NAME,
            "trace_id":  getattr(record, "trace_id", "N/A"),
            "event":     getattr(record, "event", "log"),
            "message":   record.getMessage(),
        }
        if record.exc_info:
            exc_type, exc_value, _ = record.exc_info
            payload["exception"] = {
                "type":        exc_type.__name__ if exc_type else None,
                "message":     str(exc_value),
                "stack_trace": self.formatException(record.exc_info),
            }
        return json.dumps(payload, ensure_ascii=False)


# ── Logger ─────────────────────────────────────────────────────────────────────
logger = logging.getLogger("ChaosApp")
logger.setLevel(logging.DEBUG)

file_handler   = RotatingFileHandler(LOG_FILE, maxBytes=10 * 1024 * 1024, backupCount=5)
file_handler.setFormatter(JSONFormatter())
stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setFormatter(JSONFormatter())
logger.addHandler(file_handler)
logger.addHandler(stdout_handler)


# ── Estado em memória (para exibir na UI) ─────────────────────────────────────
recent_logs: list[dict] = []
MAX_RECENT  = 200
counters    = {"info": 0, "debug": 0, "warning": 0, "error": 0, "critical": 0, "total": 0}


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


_LEVELS = {"DEBUG": logging.DEBUG, "INFO": logging.INFO, "WARNING": logging.WARNING,
           "ERROR": logging.ERROR, "CRITICAL": logging.CRITICAL}


def log_event(level: str, message: str, trace_id: str = None, event: str = "log", exc_info=None):
    """Loga (arquivo JSON + stdout) e também registra na lista em memória usada pelo dashboard."""
    trace_id = trace_id or str(uuid.uuid4())
    logger.log(_LEVELS[level], message, extra={"trace_id": trace_id, "event": event}, exc_info=exc_info)
    record_log(level, message, trace_id)
    return trace_id


# ── Estado persistido (detecta encerramento anormal do processo anterior) ─────
# Sinais como SIGKILL/OOMKill não podem ser capturados em código — a única
# forma confiável de perceber que o pod "caiu" é checar, na próxima
# inicialização, se o estado anterior nunca chegou a ser marcado como
# "stopped_clean". Como STATE_FILE fica no LOG_DIR (volume externo), essa
# informação sobrevive ao restart do container.
RECOVERY_INFO = None


def _write_state(status: str, extra: dict | None = None):
    try:
        data = {
            "pod": POD_NAME,
            "pid": os.getpid(),
            "status": status,
            "updated_at": datetime.now().isoformat(),
        }
        if extra:
            data.update(extra)
        STATE_FILE.write_text(json.dumps(data, ensure_ascii=False))
    except Exception:
        pass  # persistir estado nunca deve derrubar a aplicação


def _check_previous_crash():
    global RECOVERY_INFO
    if not STATE_FILE.exists():
        return
    try:
        prev = json.loads(STATE_FILE.read_text())
    except Exception:
        return
    if prev.get("status") == "running":
        gap = "desconhecido"
        try:
            last = datetime.fromisoformat(prev["updated_at"])
            gap = f"{(datetime.now() - last).total_seconds():.1f}s"
        except Exception:
            pass
        RECOVERY_INFO = {
            "previous_pid": prev.get("pid"),
            "last_heartbeat": prev.get("updated_at"),
            "gap_seconds": gap,
        }
        log_event(
            "CRITICAL",
            f"Reinicio detectado sem encerramento gracioso do processo anterior "
            f"(pid={prev.get('pid', '?')}), ultimo heartbeat ha {gap}. "
            f"Provavel OOMKill, SIGKILL ou crash do container.",
            event="unclean_shutdown_detected",
        )


# ── Captura de falhas não tratadas (thread principal e threads em background) ─
def _log_fatal(exc_type, exc_value, exc_tb, origin: str):
    log_event(
        "CRITICAL",
        f"Falha fatal nao tratada em {origin}: {exc_type.__name__}: {exc_value}",
        event="fatal_crash",
        exc_info=(exc_type, exc_value, exc_tb),
    )
    _write_state("crashed", {"error": f"{exc_type.__name__}: {exc_value}", "origin": origin})
    for h in logger.handlers:
        h.flush()


def _excepthook(exc_type, exc_value, exc_tb):
    _log_fatal(exc_type, exc_value, exc_tb, origin="thread principal")
    sys.__excepthook__(exc_type, exc_value, exc_tb)


def _thread_excepthook(args):
    _log_fatal(args.exc_type, args.exc_value, args.exc_traceback, origin=f"thread '{args.thread.name}'")


sys.excepthook = _excepthook
threading.excepthook = _thread_excepthook
atexit.register(logging.shutdown)


# ── Gerador contínuo de logs (thread em background) ───────────────────────────
ACTIONS = ["process_payment", "fetch_user_data", "update_cache", "send_email", "connect_db"]


def simulate_work():
    log_event("INFO", "Aplicacao iniciada. Gerando carga de logs...", event="lifecycle")
    last_heartbeat = 0.0
    while True:
        try:
            action = random.choice(ACTIONS)
            roll   = random.random()

            if roll < 0.70:
                log_event("INFO", f"Sucesso ao executar a acao: {action}")
            elif roll < 0.85:
                log_event("DEBUG", f"Detalhes internos de {action} - payload size {random.randint(10, 1000)}kb")
            elif roll < 0.95:
                log_event("WARNING", f"Lentidao detectada em {action} (duracao {random.uniform(1.0, 5.0):.2f}s)")
            else:
                log_event("ERROR", f"Falha critica ao executar {action} - Connection Timeout ou Deadlock")

            now = time.time()
            if now - last_heartbeat > 5:
                _write_state("running")
                last_heartbeat = now

            time.sleep(random.uniform(0.1, 0.5))
        except Exception:
            # Erro inesperado no próprio loop (diferente das falhas simuladas acima)
            log_event(
                "CRITICAL",
                "Erro inesperado no loop de simulacao",
                event="generator_error",
                exc_info=sys.exc_info(),
            )
            time.sleep(2)


# ── FastAPI ────────────────────────────────────────────────────────────────────
app = FastAPI(title="Chaos Log App", version="2.0.0")


@app.on_event("startup")
def startup():
    _check_previous_crash()
    _write_state("running")
    t = threading.Thread(target=simulate_work, daemon=True, name="log-simulator")
    t.start()


@app.on_event("shutdown")
def shutdown():
    # Disparado pelo lifespan do ASGI quando o uvicorn recebe SIGTERM/SIGINT e
    # drena as conexões — é o jeito correto de reagir ao sinal aqui, já que um
    # signal.signal() manual seria sobrescrito pelo próprio uvicorn.
    log_event("WARNING", "Encerramento gracioso solicitado (shutdown do lifespan).",
              event="graceful_shutdown")
    _write_state("stopped_clean", {"reason": "graceful_shutdown"})
    for h in logger.handlers:
        h.flush()


# ── API endpoints ──────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "pod": POD_NAME}


@app.get("/api/logs")
def get_logs(limit: int = 80):
    return recent_logs[:limit]


@app.get("/api/stats")
def get_stats():
    return {"pod": POD_NAME, "recovery": RECOVERY_INFO, **counters}


@app.get("/api/state")
def api_state():
    """Estado persistido no volume externo — útil para confirmar que o mount está ativo."""
    try:
        return json.loads(STATE_FILE.read_text())
    except Exception:
        return {"status": "unknown"}


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
.recovery-banner{display:none;background:rgba(248,81,73,.12);border:1px solid var(--err);color:var(--err);border-radius:9px;padding:10px 14px;font-size:12px;font-weight:500;margin-bottom:16px}
.recovery-banner.show{display:block}
.grid{display:grid;grid-template-columns:repeat(6,1fr);gap:10px;margin-bottom:18px}
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
.b-critical{background:rgba(248,81,73,.3);color:#fff;border:1px solid var(--err);font-weight:700;animation:blink 1.2s infinite}
@keyframes blink{0%,100%{opacity:1}50%{opacity:.55}}
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
  <div class="recovery-banner" id="recovery-banner"></div>
  <div class="grid">
    <div class="card"><div class="lbl">Total logs</div><div class="val c-ac" id="s-total">—</div><div class="sub">desde o início</div></div>
    <div class="card"><div class="lbl">INFO</div><div class="val c-ok" id="s-info">—</div><div class="sub">~70% dos eventos</div></div>
    <div class="card"><div class="lbl">DEBUG</div><div class="val c-pu" id="s-debug">—</div><div class="sub">detalhes internos</div></div>
    <div class="card"><div class="lbl">WARNING</div><div class="val c-warn" id="s-warning">—</div><div class="sub">lentidões</div></div>
    <div class="card"><div class="lbl">ERROR</div><div class="val c-err" id="s-error">—</div><div class="sub">falhas simuladas</div></div>
    <div class="card"><div class="lbl">CRITICAL</div><div class="val c-err" id="s-critical">—</div><div class="sub">crashes reais</div></div>
  </div>

  <div class="row">
    <button class="btn-p" onclick="load()">↻ Atualizar</button>
    <button onclick="window.open('/docs','_blank')">📄 OpenAPI Docs</button>
    <button onclick="window.open('/api/logs','_blank')">🔗 JSON /api/logs</button>
    <span class="timer" id="timer">auto-refresh: 3s</span>
  </div>

  <div class="sec-title">Stream de logs — gerado continuamente em <span id="log-path">/app/logs/</span></div>
  <div class="panel">
    <div class="phead">
      <span id="log-name">app_operation.log</span>
      <span class="ppath" id="log-fullpath">/app/logs/app_operation.log</span>
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
  <a href="/api/state">GET /api/state</a>
  <a href="/health">GET /health</a>
  <a href="/docs">Swagger UI</a>
</footer>
<script>
const BC = {INFO:'b-info',DEBUG:'b-debug',WARNING:'b-warning',ERROR:'b-error',CRITICAL:'b-critical'};

async function load(){
  try{
    const [logs, stats] = await Promise.all([
      fetch('/api/logs?limit=100').then(r=>r.json()),
      fetch('/api/stats').then(r=>r.json())
    ]);
    document.getElementById('pod-name').textContent = stats.pod;
    document.getElementById('log-name').textContent = stats.pod + '_app_operation.log';
    document.getElementById('log-fullpath').textContent = '/app/logs/' + stats.pod + '_app_operation.log';
    document.getElementById('s-total').textContent   = (stats.total||0).toLocaleString();
    document.getElementById('s-info').textContent    = (stats.info||0).toLocaleString();
    document.getElementById('s-debug').textContent   = (stats.debug||0).toLocaleString();
    document.getElementById('s-warning').textContent = (stats.warning||0).toLocaleString();
    document.getElementById('s-error').textContent   = (stats.error||0).toLocaleString();
    document.getElementById('s-critical').textContent = (stats.critical||0).toLocaleString();
    document.getElementById('cnt').textContent       = logs.length + ' entradas';

    const banner = document.getElementById('recovery-banner');
    if (stats.recovery) {
      banner.textContent = `⚠ Este pod (${stats.pod}) reiniciou apos um encerramento anormal do processo anterior — ultimo heartbeat ha ${stats.recovery.gap_seconds}. Provavel OOMKill, SIGKILL ou crash.`;
      banner.classList.add('show');
    } else {
      banner.classList.remove('show');
    }

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
