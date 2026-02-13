"""Dashboard — localhost web interface for agent introspection.

Real-time view of the agent's cognitive state via SSE + REST API.
Serves a single-page dark-themed dashboard at http://0.0.0.0:8080.

Uses aiohttp — runs as another coroutine in asyncio.gather alongside
the cognitive loop, consolidation, peripherals. Dashboard crash does
not kill the agent.

Memory browser uses direct asyncpg pool.fetch() for read-only access —
no side effects on the cognitive system (no access counts, no mutation).
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from aiohttp import web

logger = logging.getLogger("agent.dashboard")

DASHBOARD_PORT = 8080


# ── Shared state ──────────────────────────────────────────────────────


@dataclass
class AgentState:
    """Shared state readable by both cognitive loop and dashboard."""

    # Set by main.py at startup
    config: Any = None
    layers: Any = None
    memory: Any = None

    # Set by cognitive_loop after initialization
    attention: Any = None
    gut: Any = None
    safety: Any = None
    outcome_tracker: Any = None
    bootstrap: Any = None

    # Shared with cognitive_loop (same list object)
    conversation: list = field(default_factory=list)
    exchange_count: int = 0
    escalation_stats: dict = field(default_factory=lambda: {"retries": 0, "retry_successes": 0, "escalations": 0})

    # SSE broadcast
    _sse_subscribers: list = field(default_factory=list)

    def publish_event(self, event: dict):
        """Broadcast event to all SSE subscribers. Fire-and-forget."""
        dead = []
        for q in self._sse_subscribers:
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                dead.append(q)
        for q in dead:
            self._sse_subscribers.remove(q)

    def subscribe(self) -> asyncio.Queue:
        """Register a new SSE subscriber. Returns its event queue."""
        q: asyncio.Queue = asyncio.Queue(maxsize=200)
        self._sse_subscribers.append(q)
        return q

    def unsubscribe(self, q: asyncio.Queue):
        """Remove an SSE subscriber on disconnect."""
        if q in self._sse_subscribers:
            self._sse_subscribers.remove(q)


# ── Route handlers ────────────────────────────────────────────────────


async def index_handler(request):
    """Serve the dashboard HTML page."""
    return web.Response(text=DASHBOARD_HTML, content_type="text/html")


async def sse_handler(request):
    """Server-Sent Events stream for real-time dashboard updates."""
    response = web.StreamResponse(
        status=200,
        reason="OK",
        headers={
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
    await response.prepare(request)

    state: AgentState = request.app["agent_state"]
    q = state.subscribe()

    try:
        while True:
            try:
                event = await asyncio.wait_for(q.get(), timeout=15.0)
                data = json.dumps(event, default=_json_default)
                await response.write(
                    f"event: {event.get('type', 'update')}\ndata: {data}\n\n".encode()
                )
            except asyncio.TimeoutError:
                await response.write(b": keepalive\n\n")
            except ConnectionResetError:
                break
    finally:
        state.unsubscribe(q)

    return response


async def api_status(request):
    """JSON snapshot of current agent state."""
    state: AgentState = request.app["agent_state"]

    data = {
        "agent_id": None,
        "phase": None,
        "model": None,
        "memory_count": 0,
        "conversation_length": len(state.conversation),
        "exchange_count": state.exchange_count,
        "attention_queue_size": 0,
        "bootstrap": None,
        "gut": None,
        "energy": None,
    }

    if state.layers:
        data["agent_id"] = state.layers.manifest.get("agent_id")
        data["phase"] = state.layers.manifest.get("phase")
        data["uptime_hours"] = state.layers.manifest.get("uptime_total_hours", 0)
        data["times_restarted"] = state.layers.manifest.get("times_restarted", 0)

    if state.config:
        data["model"] = state.config.models.system1.model
        data["model_s2"] = state.config.models.system2.model if state.config.models.system2 else None

    if state.memory:
        try:
            data["memory_count"] = await state.memory.memory_count()
        except Exception:
            pass

    if state.attention:
        data["attention_queue_size"] = state.attention.queue_size

    if state.escalation_stats:
        data["escalation"] = state.escalation_stats

    if state.bootstrap:
        achieved, total = state.bootstrap.progress
        data["bootstrap"] = {"achieved": achieved, "total": total}

    if state.gut:
        data["gut"] = {
            "summary": state.gut.gut_summary(),
            "emotional_charge": round(state.gut.emotional_charge, 3),
            "emotional_alignment": round(state.gut.emotional_alignment, 3),
        }

    try:
        from .llm import energy_tracker
        data["energy"] = {
            "session_cost": round(energy_tracker.session_cost, 6),
            "cost_24h": round(energy_tracker.cost_24h, 6),
            "total_calls": len(energy_tracker.entries),
            "breakdown": {
                model: {
                    "calls": s["calls"],
                    "tokens_in": s["tokens_in"],
                    "tokens_out": s["tokens_out"],
                    "cost": round(s["cost"], 6),
                }
                for model, s in energy_tracker.breakdown().items()
            },
        }
    except Exception:
        pass

    return web.json_response(data)


async def api_memories(request):
    """Paginated memory list (read-only, no side effects)."""
    state: AgentState = request.app["agent_state"]
    if not state.memory or not state.memory.pool:
        return web.json_response({"memories": [], "total": 0})

    limit = min(int(request.query.get("limit", "20")), 100)
    offset = int(request.query.get("offset", "0"))

    rows = await state.memory.pool.fetch(
        """
        SELECT id, LEFT(content, 200) as content_preview, type, confidence,
               importance, depth_weight_alpha, depth_weight_beta,
               access_count, created_at, source_tag, immutable
        FROM memories
        ORDER BY created_at DESC
        LIMIT $1 OFFSET $2
        """,
        limit,
        offset,
    )
    total = await state.memory.memory_count()

    memories = []
    for r in rows:
        alpha = r["depth_weight_alpha"] or 1.0
        beta = r["depth_weight_beta"] or 4.0
        memories.append({
            "id": r["id"],
            "content_preview": r["content_preview"],
            "type": r["type"],
            "confidence": round(r["confidence"] or 0, 3),
            "importance": round(r["importance"] or 0, 3),
            "depth_center": round(alpha / (alpha + beta), 3),
            "access_count": r["access_count"] or 0,
            "created_at": r["created_at"].isoformat() if r["created_at"] else None,
            "source_tag": r["source_tag"],
            "immutable": r["immutable"],
        })

    return web.json_response({
        "memories": memories,
        "total": total,
        "limit": limit,
        "offset": offset,
    })


async def api_memory_detail(request):
    """Single memory full detail."""
    state: AgentState = request.app["agent_state"]
    if not state.memory or not state.memory.pool:
        return web.json_response({"error": "no memory store"}, status=503)

    mem_id = request.match_info["id"]
    row = await state.memory.pool.fetchrow(
        """
        SELECT id, content, type, confidence, importance,
               depth_weight_alpha, depth_weight_beta,
               access_count, last_accessed, created_at, updated_at,
               source, source_tag, tags, immutable, compressed,
               evidence_count, metadata
        FROM memories WHERE id = $1
        """,
        mem_id,
    )
    if not row:
        return web.json_response({"error": "not found"}, status=404)

    alpha = row["depth_weight_alpha"] or 1.0
    beta = row["depth_weight_beta"] or 4.0
    data = {
        "id": row["id"],
        "content": row["content"],
        "type": row["type"],
        "confidence": round(row["confidence"] or 0, 3),
        "importance": round(row["importance"] or 0, 3),
        "depth_center": round(alpha / (alpha + beta), 3),
        "depth_alpha": round(alpha, 2),
        "depth_beta": round(beta, 2),
        "access_count": row["access_count"] or 0,
        "last_accessed": row["last_accessed"].isoformat() if row["last_accessed"] else None,
        "created_at": row["created_at"].isoformat() if row["created_at"] else None,
        "updated_at": row["updated_at"].isoformat() if row["updated_at"] else None,
        "source": row["source"],
        "source_tag": row["source_tag"],
        "tags": row["tags"],
        "immutable": row["immutable"],
        "compressed": row["compressed"],
        "evidence_count": row["evidence_count"] or 0,
        "metadata": row["metadata"],
    }
    return web.json_response(data)


async def api_attention(request):
    """Attention queue state."""
    state: AgentState = request.app["agent_state"]
    if not state.attention:
        return web.json_response({"queue": [], "queue_size": 0})

    queue_items = []
    for cand in state.attention._queue:
        queue_items.append({
            "content": cand.content[:100],
            "source_tag": cand.source_tag,
            "urgency": round(cand.urgency, 3),
            "salience": round(cand.salience, 3),
            "created_at": cand.created_at.isoformat() if cand.created_at else None,
        })

    return web.json_response({
        "queue": queue_items,
        "queue_size": state.attention.queue_size,
        "has_centroid": state.attention.attention_centroid is not None,
    })


async def api_gut(request):
    """Gut feeling state + delta log."""
    state: AgentState = request.app["agent_state"]
    if not state.gut:
        return web.json_response({"summary": "not initialized", "deltas": []})

    deltas = []
    for d in state.gut.get_delta_log(last_n=20):
        ts = None
        if hasattr(d, "timestamp") and d.timestamp:
            ts = datetime.fromtimestamp(d.timestamp, tz=timezone.utc).isoformat()
        deltas.append({
            "magnitude": round(d.magnitude, 4),
            "context": d.context[:80] if d.context else None,
            "timestamp": ts,
            "outcome_id": d.outcome_id,
        })

    return web.json_response({
        "summary": state.gut.gut_summary(),
        "emotional_charge": round(state.gut.emotional_charge, 3),
        "emotional_alignment": round(state.gut.emotional_alignment, 3),
        "deltas": deltas,
    })


async def api_conversation(request):
    """Current conversation window."""
    state: AgentState = request.app["agent_state"]
    msgs = []
    for msg in state.conversation[-50:]:
        msgs.append({
            "role": msg.get("role"),
            "content": msg.get("content", "")[:300],
            "source_tag": msg.get("source_tag"),
            "timestamp": msg.get("timestamp"),
        })
    return web.json_response({"messages": msgs, "total": len(state.conversation)})


async def api_energy(request):
    """Energy tracker breakdown."""
    try:
        from .llm import energy_tracker
        return web.json_response({
            "session_cost": round(energy_tracker.session_cost, 6),
            "cost_24h": round(energy_tracker.cost_24h, 6),
            "total_calls": len(energy_tracker.entries),
            "breakdown": {
                model: {
                    "calls": s["calls"],
                    "tokens_in": s["tokens_in"],
                    "tokens_out": s["tokens_out"],
                    "cost": round(s["cost"], 6),
                }
                for model, s in energy_tracker.breakdown().items()
            },
        })
    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)


# ── Server lifecycle ──────────────────────────────────────────────────


async def run_dashboard(agent_state: AgentState, shutdown_event: asyncio.Event):
    """Run the dashboard web server as an asyncio task."""
    app = web.Application()
    app["agent_state"] = agent_state

    app.router.add_get("/", index_handler)
    app.router.add_get("/events", sse_handler)
    app.router.add_get("/api/status", api_status)
    app.router.add_get("/api/memories", api_memories)
    app.router.add_get("/api/memory/{id}", api_memory_detail)
    app.router.add_get("/api/attention", api_attention)
    app.router.add_get("/api/gut", api_gut)
    app.router.add_get("/api/conversation", api_conversation)
    app.router.add_get("/api/energy", api_energy)

    runner = web.AppRunner(app, access_log=None)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", DASHBOARD_PORT)

    try:
        await site.start()
        logger.info(f"Dashboard started on http://0.0.0.0:{DASHBOARD_PORT}")
        await shutdown_event.wait()
    except Exception as e:
        logger.error(f"Dashboard error: {e}", exc_info=True)
    finally:
        await runner.cleanup()
        logger.info("Dashboard stopped.")


# ── Helpers ───────────────────────────────────────────────────────────


def _json_default(obj):
    """JSON serializer for non-standard types."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    if hasattr(obj, "__float__"):
        return float(obj)
    return str(obj)


# ── Inline HTML/CSS/JS ────────────────────────────────────────────────
# Note: All user-facing data is escaped via textContent in the JS
# to prevent XSS. The dashboard is localhost-only (Tailscale).

DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Agent Dashboard</title>
<style>
:root {
  --bg: #0d1117;
  --bg2: #161b22;
  --bg3: #21262d;
  --border: #30363d;
  --text: #c9d1d9;
  --text-dim: #8b949e;
  --accent: #58a6ff;
  --green: #3fb950;
  --orange: #d29922;
  --red: #f85149;
  --purple: #bc8cff;
}
* { margin: 0; padding: 0; box-sizing: border-box; }
body {
  font-family: 'SF Mono', 'Fira Code', 'Cascadia Code', monospace;
  background: var(--bg);
  color: var(--text);
  font-size: 13px;
  line-height: 1.5;
}
.header {
  background: var(--bg2);
  border-bottom: 1px solid var(--border);
  padding: 8px 16px;
  display: flex;
  justify-content: space-between;
  align-items: center;
}
.header h1 { font-size: 14px; color: var(--accent); font-weight: 600; }
.header .status { font-size: 12px; color: var(--text-dim); }
.status .dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 4px; }
.dot.on { background: var(--green); }
.dot.off { background: var(--red); }
.grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  grid-template-rows: 1fr 1fr;
  gap: 1px;
  height: calc(100vh - 37px);
  background: var(--border);
}
.panel {
  background: var(--bg);
  overflow: hidden;
  display: flex;
  flex-direction: column;
}
.panel-title {
  background: var(--bg2);
  padding: 6px 12px;
  font-size: 11px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  color: var(--text-dim);
  border-bottom: 1px solid var(--border);
  flex-shrink: 0;
}
.panel-body {
  padding: 8px 12px;
  overflow-y: auto;
  flex: 1;
}
.event { padding: 4px 0; border-bottom: 1px solid var(--bg3); font-size: 12px; }
.stat-row {
  display: flex;
  justify-content: space-between;
  padding: 3px 0;
  border-bottom: 1px solid var(--bg3);
}
.stat-row .label { color: var(--text-dim); }
.stat-row .value { color: var(--text); font-weight: 500; }
.mem-table { width: 100%; border-collapse: collapse; font-size: 12px; }
.mem-table th {
  text-align: left; padding: 4px 6px; border-bottom: 1px solid var(--border);
  color: var(--text-dim); font-weight: 500; position: sticky; top: 0; background: var(--bg);
}
.mem-table td { padding: 4px 6px; border-bottom: 1px solid var(--bg3); }
.mem-table tr:hover { background: var(--bg2); cursor: pointer; }
.depth-bar {
  display: inline-block; width: 40px; height: 6px;
  background: var(--bg3); border-radius: 3px; overflow: hidden; vertical-align: middle;
}
.depth-fill { height: 100%; border-radius: 3px; }
.pagination {
  padding: 6px 12px; background: var(--bg2); border-top: 1px solid var(--border);
  font-size: 11px; display: flex; justify-content: space-between; flex-shrink: 0;
}
.pagination button {
  background: var(--bg3); border: 1px solid var(--border); color: var(--text);
  padding: 2px 10px; cursor: pointer; font-size: 11px; border-radius: 3px;
}
.pagination button:hover { background: var(--border); }
.pagination button:disabled { opacity: 0.3; cursor: default; }
.modal-overlay {
  display: none; position: fixed; top: 0; left: 0; right: 0; bottom: 0;
  background: rgba(0,0,0,0.7); z-index: 100; justify-content: center; align-items: center;
}
.modal-overlay.active { display: flex; }
.modal {
  background: var(--bg2); border: 1px solid var(--border); border-radius: 6px;
  max-width: 600px; width: 90%; max-height: 80vh; overflow-y: auto; padding: 16px;
}
.modal h3 { color: var(--accent); margin-bottom: 12px; font-size: 13px; }
.modal pre {
  background: var(--bg); padding: 8px; border-radius: 4px;
  white-space: pre-wrap; word-break: break-word; font-size: 12px;
}
.modal .close { float: right; cursor: pointer; color: var(--text-dim); font-size: 16px; }
.modal .close:hover { color: var(--text); }
.conv-msg { padding: 4px 0; border-bottom: 1px solid var(--bg3); }
.conv-msg .role { font-weight: 600; font-size: 11px; }
.conv-msg .role-user { color: var(--accent); }
.conv-msg .role-assistant { color: var(--green); }
.conv-msg .msg-content { font-size: 12px; margin-top: 2px; }
</style>
</head>
<body>

<div class="header">
  <h1>Agent Consciousness</h1>
  <div class="status">
    <span class="dot" id="sse-dot"></span>
    <span id="sse-status">connecting...</span>
    &nbsp;|&nbsp;
    <span id="header-cost">$0.0000</span>
  </div>
</div>

<div class="grid">
  <div class="panel">
    <div class="panel-title">Live Feed</div>
    <div class="panel-body" id="feed"></div>
  </div>

  <div class="panel">
    <div class="panel-title">Agent Status</div>
    <div class="panel-body" id="status-panel"></div>
  </div>

  <div class="panel">
    <div class="panel-title">Context Window</div>
    <div class="panel-body" id="conversation-panel"></div>
  </div>

  <div class="panel">
    <div class="panel-title">Memory Store</div>
    <div class="panel-body" id="memory-panel"></div>
    <div class="pagination">
      <span id="mem-info">-</span>
      <span>
        <button id="mem-prev" disabled>&lt; prev</button>
        <button id="mem-next">next &gt;</button>
      </span>
    </div>
  </div>
</div>

<div class="modal-overlay" id="modal">
  <div class="modal">
    <span class="close" id="modal-close">&times;</span>
    <div id="modal-content"></div>
  </div>
</div>

<script>
// All dynamic content uses textContent (safe) or DOM construction.
// No raw HTML injection from API data.

const feed = document.getElementById('feed');
const dot = document.getElementById('sse-dot');
const sseStatus = document.getElementById('sse-status');

// ── SSE ──
function connectSSE() {
  const es = new EventSource('/events');
  es.onopen = function() {
    dot.className = 'dot on';
    sseStatus.textContent = 'connected';
  };
  es.onerror = function() {
    dot.className = 'dot off';
    sseStatus.textContent = 'reconnecting...';
  };

  ['cycle_start', 'llm_response', 'escalation', 'gate_flush'].forEach(function(type) {
    es.addEventListener(type, function(e) {
      addEvent(type, JSON.parse(e.data));
    });
  });
}

function addEvent(type, data) {
  var div = document.createElement('div');
  div.className = 'event';

  var ts = document.createElement('span');
  ts.style.color = 'var(--text-dim)';
  ts.style.fontSize = '11px';
  ts.textContent = data.ts ? new Date(data.ts).toLocaleTimeString() + ' ' : '';
  div.appendChild(ts);

  var tag = document.createElement('span');
  tag.style.fontWeight = '600';

  if (type === 'cycle_start') {
    tag.style.color = 'var(--green)';
    tag.textContent = '[ATTEND] ';
    div.appendChild(tag);
    var src = document.createElement('span');
    src.style.color = data.source === 'external_user' ? 'var(--accent)' : 'var(--purple)';
    src.textContent = data.source + ' ';
    div.appendChild(src);
    var info = document.createElement('span');
    info.style.color = 'var(--text-dim)';
    info.textContent = '(sal=' + (data.salience||0).toFixed(3) + ', q=' + (data.queue_size||0) + ') ';
    div.appendChild(info);
    var ct = document.createElement('span');
    ct.textContent = data.content || '';
    div.appendChild(ct);
  } else if (type === 'llm_response') {
    tag.style.color = data.escalated ? 'var(--orange)' : 'var(--accent)';
    tag.textContent = data.escalated ? '[S2] ' : '[S1] ';
    div.appendChild(tag);
    var rp = document.createElement('span');
    rp.textContent = data.reply || '';
    div.appendChild(rp);
  } else if (type === 'escalation') {
    tag.style.color = 'var(--red)';
    tag.textContent = '[ESCALATE] ';
    div.appendChild(tag);
    var tr = document.createElement('span');
    tr.textContent = 'triggers: ' + (data.triggers||[]).join(', ');
    div.appendChild(tr);
  } else if (type === 'gate_flush') {
    tag.style.color = 'var(--orange)';
    tag.textContent = '[GATE] ';
    div.appendChild(tag);
    var gf = document.createElement('span');
    gf.textContent = data.persisted + ' persisted, ' + data.dropped + ' dropped';
    div.appendChild(gf);
  }

  feed.insertBefore(div, feed.firstChild);
  while (feed.children.length > 200) feed.lastChild.remove();
}

// ── Status polling ──
function mkStat(label, value) {
  var row = document.createElement('div');
  row.className = 'stat-row';
  var l = document.createElement('span');
  l.className = 'label';
  l.textContent = label;
  var v = document.createElement('span');
  v.className = 'value';
  v.textContent = String(value);
  row.appendChild(l);
  row.appendChild(v);
  return row;
}

async function refreshStatus() {
  try {
    var r = await fetch('/api/status');
    var d = await r.json();
    var p = document.getElementById('status-panel');
    p.replaceChildren();
    p.appendChild(mkStat('Agent', d.agent_id || '-'));
    p.appendChild(mkStat('Phase', d.phase || '-'));
    p.appendChild(mkStat('System 1', d.model || '-'));
    p.appendChild(mkStat('System 2', d.model_s2 || '-'));
    p.appendChild(mkStat('Memories', d.memory_count));
    p.appendChild(mkStat('Conversation', d.conversation_length + ' msgs'));
    p.appendChild(mkStat('Exchanges/Flush', d.exchange_count + '/5'));
    p.appendChild(mkStat('Attention Queue', d.attention_queue_size));
    if (d.escalation) p.appendChild(mkStat('Escalations', 'S2: ' + d.escalation.escalations + ', retries: ' + d.escalation.retries + ' (' + d.escalation.retry_successes + ' ok)'));
    if (d.bootstrap) p.appendChild(mkStat('Bootstrap', d.bootstrap.achieved + '/' + d.bootstrap.total));
    if (d.gut) {
      p.appendChild(mkStat('Gut', d.gut.summary));
      p.appendChild(mkStat('Emotional Charge', d.gut.emotional_charge));
      p.appendChild(mkStat('Alignment', d.gut.emotional_alignment));
    }
    if (d.energy) {
      p.appendChild(mkStat('Session Cost', '$' + d.energy.session_cost.toFixed(4)));
      p.appendChild(mkStat('24h Cost', '$' + d.energy.cost_24h.toFixed(4)));
      p.appendChild(mkStat('API Calls', d.energy.total_calls));
      document.getElementById('header-cost').textContent = '$' + d.energy.session_cost.toFixed(4);
      if (d.energy.breakdown) {
        for (var model in d.energy.breakdown) {
          var s = d.energy.breakdown[model];
          var short = model.split('-').slice(-2).join('-');
          p.appendChild(mkStat('  ' + short, s.calls + ' calls, $' + s.cost.toFixed(4)));
        }
      }
    }
    p.appendChild(mkStat('Restarts', d.times_restarted || 0));
  } catch(e) {}
}

// ── Conversation ──
async function refreshConversation() {
  try {
    var r = await fetch('/api/conversation');
    var d = await r.json();
    var p = document.getElementById('conversation-panel');
    p.replaceChildren();
    if (!d.messages || d.messages.length === 0) {
      var empty = document.createElement('div');
      empty.style.color = 'var(--text-dim)';
      empty.style.padding = '8px';
      empty.textContent = 'No conversation yet';
      p.appendChild(empty);
      return;
    }
    var info = document.createElement('div');
    info.style.cssText = 'color:var(--text-dim);font-size:11px;margin-bottom:6px;';
    info.textContent = d.total + ' messages in window';
    p.appendChild(info);
    d.messages.forEach(function(m) {
      var div = document.createElement('div');
      div.className = 'conv-msg';
      var role = document.createElement('span');
      role.className = 'role role-' + m.role;
      role.textContent = m.role;
      div.appendChild(role);
      if (m.source_tag) {
        var st = document.createElement('span');
        st.style.cssText = 'color:var(--text-dim);font-size:10px;margin-left:4px;';
        st.textContent = '[' + m.source_tag + ']';
        div.appendChild(st);
      }
      var ct = document.createElement('div');
      ct.className = 'msg-content';
      ct.textContent = m.content;
      div.appendChild(ct);
      p.appendChild(div);
    });
    p.scrollTop = p.scrollHeight;
  } catch(e) {}
}

// ── Memory browser ──
var memOffset = 0;
var memLimit = 15;

async function loadMemories() {
  try {
    var r = await fetch('/api/memories?limit=' + memLimit + '&offset=' + memOffset);
    var d = await r.json();
    var p = document.getElementById('memory-panel');
    p.replaceChildren();

    var table = document.createElement('table');
    table.className = 'mem-table';
    var thead = document.createElement('thead');
    var hr = document.createElement('tr');
    ['Type', 'Content', 'Depth', 'Imp', 'Access'].forEach(function(h) {
      var th = document.createElement('th');
      th.textContent = h;
      hr.appendChild(th);
    });
    thead.appendChild(hr);
    table.appendChild(thead);

    var tbody = document.createElement('tbody');
    d.memories.forEach(function(m) {
      var tr = document.createElement('tr');
      tr.addEventListener('click', function() { showMemory(m.id); });

      var tdType = document.createElement('td');
      tdType.style.color = 'var(--purple)';
      tdType.textContent = m.type || '-';
      tr.appendChild(tdType);

      var tdContent = document.createElement('td');
      tdContent.textContent = (m.content_preview || '').substring(0, 80);
      tr.appendChild(tdContent);

      var pct = Math.round(m.depth_center * 100);
      var color = pct > 70 ? 'var(--green)' : pct > 40 ? 'var(--orange)' : 'var(--text-dim)';
      var tdDepth = document.createElement('td');
      var bar = document.createElement('span');
      bar.className = 'depth-bar';
      var fill = document.createElement('span');
      fill.className = 'depth-fill';
      fill.style.width = pct + '%';
      fill.style.background = color;
      bar.appendChild(fill);
      tdDepth.appendChild(bar);
      var pctText = document.createTextNode(' ' + pct + '%');
      tdDepth.appendChild(pctText);
      tr.appendChild(tdDepth);

      var tdImp = document.createElement('td');
      tdImp.textContent = m.importance;
      tr.appendChild(tdImp);

      var tdAcc = document.createElement('td');
      tdAcc.textContent = m.access_count;
      tr.appendChild(tdAcc);

      tbody.appendChild(tr);
    });
    table.appendChild(tbody);
    p.appendChild(table);

    document.getElementById('mem-info').textContent =
      (memOffset+1) + '-' + Math.min(memOffset+memLimit, d.total) + ' of ' + d.total;
    document.getElementById('mem-prev').disabled = memOffset === 0;
    document.getElementById('mem-next').disabled = memOffset + memLimit >= d.total;
  } catch(e) {}
}

document.getElementById('mem-prev').addEventListener('click', function() {
  memOffset = Math.max(0, memOffset - memLimit);
  loadMemories();
});
document.getElementById('mem-next').addEventListener('click', function() {
  memOffset += memLimit;
  loadMemories();
});

async function showMemory(id) {
  try {
    var r = await fetch('/api/memory/' + encodeURIComponent(id));
    var d = await r.json();
    var mc = document.getElementById('modal-content');
    mc.replaceChildren();

    var h = document.createElement('h3');
    h.textContent = d.id;
    mc.appendChild(h);

    var fields = [
      ['Type', d.type], ['Depth', d.depth_center + ' (a=' + d.depth_alpha + ', b=' + d.depth_beta + ')'],
      ['Importance', d.importance], ['Confidence', d.confidence],
      ['Access Count', d.access_count], ['Source', d.source_tag || d.source || '-'],
      ['Immutable', d.immutable], ['Created', d.created_at || '-'],
    ];
    if (d.compressed) fields.push(['Compressed', d.compressed]);

    fields.forEach(function(f) { mc.appendChild(mkStat(f[0], f[1])); });

    var label = document.createElement('div');
    label.style.marginTop = '12px';
    label.style.fontWeight = 'bold';
    label.textContent = 'Content:';
    mc.appendChild(label);

    var pre = document.createElement('pre');
    pre.textContent = d.content;
    mc.appendChild(pre);

    document.getElementById('modal').classList.add('active');
  } catch(e) {}
}

document.getElementById('modal').addEventListener('click', function(e) {
  if (e.target === this) this.classList.remove('active');
});
document.getElementById('modal-close').addEventListener('click', function() {
  document.getElementById('modal').classList.remove('active');
});
document.addEventListener('keydown', function(e) {
  if (e.key === 'Escape') document.getElementById('modal').classList.remove('active');
});

// ── Init ──
connectSSE();
refreshStatus();
refreshConversation();
loadMemories();
setInterval(refreshStatus, 5000);
setInterval(refreshConversation, 3000);
setInterval(loadMemories, 10000);
</script>
</body>
</html>
"""
