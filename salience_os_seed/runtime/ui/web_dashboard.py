"""HTML dashboard for SalienceRuntime using the standard library."""

from __future__ import annotations

import argparse
import json
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse

from ...conversation.session import ConversationConfig, ConversationSession, ConversationSnapshot
from ...proto_lm.trainer import TrainingConfig
from ...runtime.orchestrator import RuntimeMetrics
from ...training.cot_curriculum.loader import iter_examples
from ...telemetry import (
    BUS,
    ParameterEvent,
    TelemetryEvent,
    render_ingestion_event,
    render_parameter_event,
    render_training_event,
)

BASE_DIR = Path(__file__).resolve().parents[2]


def _run_cot_curriculum(session: ConversationSession, *, limit: Optional[int] = None) -> Dict[str, int]:
    curriculum_root = BASE_DIR / "training" / "cot_curriculum"
    count = 0
    for example in iter_examples(str(curriculum_root)):
        trace_lines = [
            f"Task: {example.task}",
            "Trace:",
            *example.reasoning_trace,
            f"Answer: {example.answer}",
            f"Meta: {example.meta_lesson}",
        ]
        payload = "\n".join(trace_lines)
        segments_processed, _ = session.ingest_text(
            payload,
            source="cot_curriculum",
            max_chars=768,
        )
        count += 1
        if limit is not None and count >= limit:
            break
    return {"examples": count}


def _chunk_text(text: str, *, max_chars: int = 768) -> List[str]:
    lines = text.splitlines()
    chunks: List[str] = []
    buffer: List[str] = []
    length = 0
    for line in lines:
        line = line.strip()
        if not line:
            continue
        line_len = len(line) + 1
        if length + line_len > max_chars and buffer:
            chunks.append("\n".join(buffer))
            buffer = []
            length = 0
        buffer.append(line)
        length += line_len
    if buffer:
        chunks.append("\n".join(buffer))
    return chunks


def _ingest_food_corpus(session: ConversationSession) -> Dict[str, int]:
    food_root = BASE_DIR / "food"
    files = 0
    chunks = 0
    for path in sorted(food_root.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in {".txt", ".md", ".log", ".json"}:
            continue
        try:
            content = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        content = content.strip()
        if not content:
            continue
        for chunk in _chunk_text(content):
            processed, _ = session.ingest_text(
                chunk,
                source=f"food::{path.name}",
                max_chars=768,
            )
            chunks += processed
        files += 1
    return {"files": files, "chunks": chunks}


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <title>Salience Runtime Dashboard</title>
  <style>
    :root { color-scheme: dark; }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      padding: 0;
      background: radial-gradient(circle at 20% 20%, #0c1a3a, #01020d 55%);
      color: #f0f6ff;
      font-family: 'Lucida Console', 'Courier New', monospace;
      letter-spacing: 0.02em;
      position: relative;
      min-height: 100vh;
    }
    body::before {
      content: '';
      position: fixed;
      inset: 0;
      background-image: linear-gradient(
        rgba(255, 255, 255, 0.03) 1px,
        transparent 1px
      );
      background-size: 100% 4px;
      pointer-events: none;
      mix-blend-mode: screen;
      opacity: 0.6;
    }
    header {
      padding: 16px 24px;
      background: linear-gradient(90deg, #051a49 0%, #0e063b 50%, #051a49 100%);
      border-bottom: 3px solid #4effd2;
      box-shadow: 0 8px 32px rgba(0, 0, 0, 0.45);
      position: sticky;
      top: 0;
      z-index: 10;
    }
    h1 {
      margin: 0;
      font-size: 1.8rem;
      text-transform: uppercase;
      color: #9fffd7;
      text-shadow: 0 0 6px rgba(79, 255, 215, 0.65);
    }
    .meta {
      margin-top: 10px;
      color: #8ab4ff;
      font-size: 0.88rem;
      white-space: pre-wrap;
    }
    main {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
      gap: 18px;
      padding: 22px;
    }
    section.panel {
      position: relative;
      background: rgba(5, 12, 32, 0.88);
      border: 2px solid #3cdfff;
      border-radius: 12px;
      padding: 16px 18px;
      box-shadow: 0 0 0 2px rgba(14, 255, 197, 0.12), 0 10px 22px rgba(0, 0, 0, 0.55);
      display: flex;
      flex-direction: column;
      gap: 12px;
      overflow: hidden;
    }
    section.panel::after {
      content: '';
      position: absolute;
      inset: 0;
      border: 1px solid rgba(0, 255, 200, 0.15);
      border-radius: 10px;
      pointer-events: none;
    }
    section.panel h2 {
      margin: 0;
      font-size: 1rem;
      letter-spacing: 0.16em;
      text-transform: uppercase;
      color: #7df9ff;
    }
    .mono { font-family: 'Lucida Console', 'Courier New', monospace; }
    .scroll { overflow-y: auto; max-height: 360px; padding-right: 6px; }
    .kv-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
      gap: 8px 12px;
      font-size: 0.88rem;
    }
    .kv-row {
      display: flex;
      flex-direction: column;
      gap: 2px;
      padding: 6px 10px;
      background: rgba(18, 36, 78, 0.6);
      border: 1px solid rgba(94, 255, 222, 0.25);
      border-radius: 6px;
    }
    .kv-label {
      text-transform: uppercase;
      font-size: 0.7rem;
      color: #70d6ff;
      letter-spacing: 0.1em;
    }
    .kv-value {
      font-size: 0.95rem;
      color: #f6fffc;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 0.85rem;
      border: 1px solid rgba(81, 255, 218, 0.35);
    }
    thead { background: rgba(43, 124, 255, 0.25); }
    th, td { padding: 6px 6px; border-bottom: 1px solid rgba(63, 205, 255, 0.25); }
    tr:last-child td { border-bottom: none; }
    th { text-transform: uppercase; font-size: 0.7rem; letter-spacing: 0.12em; color: #8dd7ff; }
    td { color: #f0f9ff; }
    .error { color: #ff8a8a; font-size: 0.85rem; }
    .conversation-panel { min-height: 320px; }
    .conversation-wrapper {
      flex: 1;
      overflow-y: auto;
      padding-right: 6px;
      display: flex;
      flex-direction: column;
      gap: 10px;
    }
    #conversation {
      display: flex;
      flex-direction: column;
      gap: 10px;
    }
    .bubble {
      border-radius: 10px;
      padding: 12px 14px;
      background: rgba(9, 41, 117, 0.55);
      border: 1px solid rgba(124, 255, 246, 0.3);
      margin-bottom: 8px;
      box-shadow: inset 0 0 12px rgba(0, 255, 200, 0.08);
    }
    .bubble.user { background: rgba(26, 64, 180, 0.45); align-self: flex-end; }
    .bubble.assistant { background: rgba(9, 92, 73, 0.45); align-self: flex-start; }
    .bubble .speaker {
      font-weight: 600;
      display: block;
      margin-bottom: 6px;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      color: #92fdfc;
    }
    .bubble pre {
      margin: 0;
      background: transparent;
      padding: 0;
      color: #f0faff;
      white-space: pre-wrap;
      word-break: break-word;
    }
    form { display: flex; flex-direction: column; gap: 10px; }
    textarea {
      resize: vertical;
      min-height: 88px;
      border-radius: 6px;
      border: 1px solid rgba(115, 255, 217, 0.4);
      padding: 10px;
      background: rgba(5, 18, 52, 0.75);
      color: #f5feff;
    }
    input[type="file"] {
      color: #92fdfc;
    }
    button {
      align-self: flex-start;
      padding: 8px 18px;
      border: 2px solid #5fffe2;
      border-radius: 6px;
      background: linear-gradient(90deg, rgba(12, 35, 96, 0.8), rgba(16, 92, 81, 0.8));
      color: #affff7;
      font-weight: 600;
      cursor: pointer;
      text-transform: uppercase;
      letter-spacing: 0.12em;
      transition: transform 0.1s ease, box-shadow 0.1s ease;
      box-shadow: 0 0 10px rgba(95, 255, 226, 0.35);
    }
    button:hover { transform: translateY(-1px); }
    button:disabled { opacity: 0.55; cursor: wait; box-shadow: none; }
    .status { font-size: 0.8rem; color: #7edaff; min-height: 1.2rem; }
    .meter-block { display: flex; flex-direction: column; gap: 6px; }
    .meter-label { font-size: 0.76rem; text-transform: uppercase; color: #5fceff; letter-spacing: 0.14em; }
    .meter {
      position: relative;
      height: 12px;
      width: 100%;
      border: 1px solid rgba(98, 255, 225, 0.7);
      border-radius: 10px;
      overflow: hidden;
      background: rgba(8, 23, 64, 0.8);
    }
    .meter-fill {
      position: absolute;
      inset: 0;
      width: 0%;
      background: linear-gradient(90deg, #22ffe0 0%, #72ffe3 50%, #22ffe0 100%);
      box-shadow: 0 0 12px rgba(61, 255, 221, 0.6);
      transition: width 0.4s ease;
    }
    .meter.active .meter-fill {
      animation: progress-stripes 1.1s linear infinite;
      background-size: 28px 100%;
      opacity: 0.8;
      width: 100%;
    }
    .meter-readout { font-size: 0.78rem; color: #9dfcff; }
    @keyframes progress-stripes {
      0% { background-position: 0 0; }
      100% { background-position: 28px 0; }
    }
    .telemetry-entry {
      margin-bottom: 8px;
      padding: 8px;
      background: rgba(9, 22, 58, 0.85);
      border: 1px solid rgba(87, 255, 235, 0.25);
      border-radius: 6px;
      font-size: 0.78rem;
      line-height: 1.35;
    }
    .telemetry-wrapper {
      flex: 1;
      max-height: 360px;
      overflow-y: auto;
      padding-right: 6px;
      display: flex;
      flex-direction: column;
      gap: 8px;
    }
    .badge {
      display: inline-block;
      padding: 2px 8px;
      border-radius: 999px;
      margin-right: 6px;
      font-size: 0.65rem;
      text-transform: uppercase;
      letter-spacing: 0.18em;
      color: #04102a;
      font-weight: 700;
    }
    .badge.param { background: #7fffd0; }
    .badge.train { background: #ffdb6e; }
    .badge.ingest { background: #7ecbff; }
  </style>
</head>
<body>
  <header>
    <h1>Salience Runtime Dashboard</h1>
    <div id=\"meta\" class=\"meta\"></div>
  </header>
  <main>
    <section class=\"panel\">
      <h2>Decision</h2>
      <div id=\"decision\"></div>
    </section>
    <section class=\"panel\">
      <h2>Status</h2>
      <div class=\"meter-block\">
        <span class=\"meter-label\">Budget Utilisation</span>
        <div class=\"meter\" id=\"budgetMeter\">
          <div class=\"meter-fill\" id=\"budgetFill\"></div>
        </div>
        <div class=\"meter-readout mono\" id=\"budgetPercent\">0%</div>
      </div>
      <div id=\"status\"></div>
    </section>
    <section class=\"panel scroll\">
      <h2>Scratchpad</h2>
      <div id=\"scratchpad\"></div>
    </section>
    <section class=\"panel scroll\">
      <h2>Todos</h2>
      <div id=\"todos\"></div>
    </section>
    <section class=\"panel scroll\">
      <h2>Maintenance</h2>
      <div id=\"maintenance\"></div>
    </section>
    <section class=\"panel scroll\">
      <h2>Experiments</h2>
      <div id=\"experiments\"></div>
    </section>
    <section class="panel conversation-panel">
      <h2>Conversation</h2>
      <div class="conversation-wrapper" id="conversationWrapper">
        <div id="conversation"></div>
      </div>
    </section>
    <section class=\"panel\">
      <h2>Send Message</h2>
      <form id=\"utterForm\">
        <textarea id=\"utterInput\" placeholder=\"Type a message for the agent...\"></textarea>
        <div class=\"status\" id=\"utterStatus\"></div>
        <button type=\"submit\" id=\"utterSubmit\">Send</button>
      </form>
    </section>
    <section class=\"panel\">
      <h2>Think</h2>
      <button type=\"button\" id=\"thinkButton\">Poke Reflect</button>
      <div class=\"status\" id=\"thinkStatus\"></div>
    </section>
    <section class=\"panel\">
      <h2>Gym &amp; Food</h2>
      <div class=\"control-block\">
        <button type=\"button\" id=\"runGymButton\">Run Gym</button>
        <div class=\"meter\" id=\"runGymMeter\"><div class=\"meter-fill\"></div></div>
        <div class=\"status\" id=\"runGymStatus\"></div>
      </div>
      <div class=\"control-block\">
        <button type=\"button\" id=\"eatFoodButton\">Eat Food</button>
        <div class=\"meter\" id=\"eatFoodMeter\"><div class=\"meter-fill\"></div></div>
      </div>
    </section>
    <section class="panel">
      <h2>Upload Training Data</h2>
      <form id="uploadForm">
        <input type="file" id="uploadInput" accept=".txt,.md,.log,.json,.csv" multiple />
        <label class="checkbox">
          <input type="checkbox" id="allowDuplicates" /> Allow duplicate ingest (testing)
        </label>
        <div class="status" id="uploadStatus"></div>
        <button type="submit" id="uploadSubmit">Upload &amp; Ingest</button>
      </form>
    </section>
    <section class="panel">
      <h2>Telemetry</h2>
      <div class="telemetry-wrapper" id="telemetryWrapper">
        <div id="telemetry"></div>
      </div>
    </section>
  </main>
  <script>
    async function fetchJson(path) {
      const resp = await fetch(path, { cache: 'no-cache' });
      if (!resp.ok) { throw new Error(`HTTP ${resp.status}`); }
      return resp.json();
    }

    function renderKeyValue(container, record) {
      if (!record || Object.keys(record).length === 0) {
        container.innerHTML = '<span class="error">No data</span>';
        return;
      }
      let html = '<div class="kv-grid">';
      for (const [key, value] of Object.entries(record)) {
        html += `<div class="kv-row"><span class="kv-label">${key.replace(/_/g, ' ')}</span><span class="kv-value">${value ?? ''}</span></div>`;
      }
      html += '</div>';
      container.innerHTML = html;
    }

    function renderTable(container, rows) {
      if (!rows || rows.length === 0) {
        container.innerHTML = '<span class="error">No data</span>';
        return;
      }
      const header = Object.keys(rows[0]);
      let html = '<table><thead><tr>' + header.map(h => `<th>${h.replace(/_/g, ' ')}</th>`).join('') + '</tr></thead><tbody>';
      for (const row of rows) {
        html += '<tr>' + header.map(h => `<td>${row[h] ?? ''}</td>`).join('') + '</tr>';
      }
      html += '</tbody></table>';
      container.innerHTML = html;
    }

    function renderList(container, entries) {
      if (!entries || entries.length === 0) {
        container.innerHTML = '<span class="error">No entries</span>';
        return;
      }
      container.innerHTML = `<pre class="mono">${entries.join('\\n')}</pre>`;
    }

    function renderTodos(container, todos) {
      if (!todos || todos.length === 0) {
        container.innerHTML = '<span class="error">No todos</span>';
        return;
      }
      let html = '<table><thead><tr><th>#</th><th>todo</th><th>score</th></tr></thead><tbody>';
      for (const todo of todos) {
        html += `<tr><td>${todo.id}</td><td>${todo.text}</td><td>${todo.score}</td></tr>`;
      }
      html += '</tbody></table>';
      container.innerHTML = html;
    }

    function escapeHtml(str) {
      const map = {
        '&': '&amp;',
        '<': '&lt;',
        '>': '&gt;',
        '"': '&quot;',
        "'": '&#39;'
      };
      return str.replace(/[&<>'"]/g, (char) => map[char] || char);
    }

    function renderConversation(container, history) {
      if (!history || history.length === 0) {
        container.innerHTML = '<span class="error">No conversation yet</span>';
        return;
      }
      const fragments = history.map(entry => {
        const speaker = escapeHtml(entry.speaker || '?');
        const text = escapeHtml(entry.text || '');
        const roleClass = entry.speaker === 'assistant' ? 'assistant' : 'user';
        return `<div class="bubble ${roleClass}"><span class="speaker">${speaker}</span><pre>${text}</pre></div>`;
      });
      container.innerHTML = fragments.join('');
      const wrapper = container.parentElement;
      if (wrapper && wrapper.classList.contains('conversation-wrapper')) {
        wrapper.scrollTop = wrapper.scrollHeight;
      }
    }

    function updateMeterFromRatio(ratioValue) {
      const fill = document.getElementById('budgetFill');
      const label = document.getElementById('budgetPercent');
      let ratio = Number.parseFloat(ratioValue);
      if (!Number.isFinite(ratio)) {
        ratio = 0;
      }
      const clamped = Math.max(0, Math.min(1, ratio));
      fill.style.width = `${(clamped * 100).toFixed(0)}%`;
      label.textContent = `${(clamped * 100).toFixed(0)}%`;
    }

    async function refresh() {
      try {
        const data = await fetchJson('/metrics');
        document.getElementById('meta').textContent = data.meta_report || '';
        renderKeyValue(document.getElementById('decision'), data.decision);
        renderKeyValue(document.getElementById('status'), data.status);
        renderList(document.getElementById('scratchpad'), data.scratchpad);
        renderTodos(document.getElementById('todos'), data.todos);
        renderTable(document.getElementById('maintenance'), data.maintenance);
        renderTable(document.getElementById('experiments'), data.experiments);
        renderConversation(document.getElementById('conversation'), data.history);
        updateMeterFromRatio(data.status?.budget_ratio ?? 0);
      } catch (err) {
        console.error(err);
      }

      try {
        const telem = await fetchJson('/telemetry');
        const wrapper = document.getElementById('telemetryWrapper');
        const root = document.getElementById('telemetry');
        root.innerHTML = '';
        for (const entry of telem.entries) {
          const div = document.createElement('div');
          div.className = 'telemetry-entry';
          const badgeClass = entry.type === 'parameters/update' ? 'param' : entry.type === 'training/step' ? 'train' : 'ingest';
          const badgeLabel = entry.type === 'parameters/update' ? 'params' : entry.type === 'training/step' ? 'train' : 'ingest';
          div.innerHTML = `<span class="badge ${badgeClass}">${badgeLabel}</span><pre>${entry.rendered}</pre>`;
          root.appendChild(div);
        }
        if (wrapper) {
          wrapper.scrollTop = wrapper.scrollHeight;
        }
      } catch (err) {
        console.error(err);
      }
    }

    refresh();
    setInterval(refresh, 1000);

    document.getElementById('utterForm').addEventListener('submit', async (event) => {
      event.preventDefault();
      const input = document.getElementById('utterInput');
      const status = document.getElementById('utterStatus');
      const button = document.getElementById('utterSubmit');
      const text = input.value.trim();
      if (!text) {
        status.textContent = 'Please enter a message.';
        return;
      }
      button.disabled = true;
      status.textContent = 'Sending...';
      try {
        const resp = await fetch('/utter', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ text }),
        });
        if (!resp.ok) {
          const errText = await resp.text();
          throw new Error(errText || `HTTP ${resp.status}`);
        }
        const payload = await resp.json();
        status.textContent = `agent> ${payload.response || '(no reply)'}`;
        input.value = '';
        await refresh();
      } catch (err) {
        console.error(err);
        status.textContent = `Error: ${err.message}`;
      } finally {
        button.disabled = false;
      }
    });

    document.getElementById('uploadForm').addEventListener('submit', async (event) => {
      event.preventDefault();
      const files = document.getElementById('uploadInput').files;
      if (!files.length) {
        return;
      }
      const allowDuplicates = document.getElementById('allowDuplicates').checked;
      const formData = new FormData();
      for (const file of files) {
        formData.append('files', file, file.name);
      }
      formData.append('allow_duplicates', allowDuplicates ? '1' : '0');
      try {
        const resp = await fetch('/ingest', {
          method: 'POST',
          body: formData,
        });
        if (!resp.ok) {
          const errText = await resp.text();
          throw new Error(errText || `HTTP ${resp.status}`);
        }
        const payload = await resp.json();
        status.textContent = `Ingested ${payload.segments} segments from ${payload.files.length} file(s).`;
        input.value = '';
        await refresh();
      } catch (err) {
        console.error(err);
        status.textContent = `Error: ${err.message}`;
      } finally {
        button.disabled = false;
      }
    });

    document.getElementById('thinkButton').addEventListener('click', async () => {
      const status = document.getElementById('thinkStatus');
      const button = document.getElementById('thinkButton');
      button.disabled = true;
      status.textContent = 'Nudging...';
      try {
        const resp = await fetch('/think', { method: 'POST' });
        if (!resp.ok) {
          const errText = await resp.text();
          throw new Error(errText || `HTTP ${resp.status}`);
        }
        status.textContent = 'Reflection poke queued.';
        await refresh();
      } catch (err) {
        console.error(err);
        status.textContent = `Error: ${err.message}`;
      } finally {
        button.disabled = false;
      }
    });

    function activateMeter(meter, active) {
      if (!meter) { return; }
      const fill = meter.querySelector('.meter-fill');
      if (active) {
        meter.classList.add('active');
        if (fill) { fill.style.width = '0%'; }
      } else {
        meter.classList.remove('active');
        if (fill) { fill.style.width = '0%'; }
      }
    }

    async function triggerIngestion(endpoint, statusElement, buttonElement, meterElement, successMessage) {
      buttonElement.disabled = true;
      statusElement.textContent = 'Running...';
      activateMeter(meterElement, true);
      try {
        const resp = await fetch(endpoint, { method: 'POST' });
        if (!resp.ok) {
          const errText = await resp.text();
          throw new Error(errText || `HTTP ${resp.status}`);
        }
        const payload = await resp.json();
        statusElement.textContent = successMessage(payload);
        if (meterElement) {
          const fill = meterElement.querySelector('.meter-fill');
          if (fill) {
            fill.style.width = '100%';
            setTimeout(() => {
              activateMeter(meterElement, false);
            }, 1000);
          }
        }
        await refresh();
      } catch (err) {
        console.error(err);
        statusElement.textContent = `Error: ${err.message}`;
        activateMeter(meterElement, false);
      } finally {
        buttonElement.disabled = false;
      }
    }

    document.getElementById('runGymButton').addEventListener('click', async () => {
      await triggerIngestion(
        '/run_gym',
        document.getElementById('runGymStatus'),
        document.getElementById('runGymButton'),
        document.getElementById('runGymMeter'),
        (payload) => `Processed ${payload.examples || 0} examples.`
      );
    });

    document.getElementById('eatFoodButton').addEventListener('click', async () => {
      await triggerIngestion(
        '/eat_food',
        document.getElementById('eatFoodStatus'),
        document.getElementById('eatFoodButton'),
        document.getElementById('eatFoodMeter'),
        (payload) => `Digested ${payload.files || 0} files / ${payload.chunks || 0} chunks.`
      );
    });
  </script>
</body>
</html>
"""


class DashboardState:
    """Thread-safe container for conversation-aware snapshots and telemetry."""

    def __init__(self, session: ConversationSession) -> None:
        self._session = session
        self._lock = threading.Lock()
        self._latest_metrics: Optional[RuntimeMetrics] = None
        self._latest_snapshot: Optional[ConversationSnapshot] = None
        self._telemetry: List[Dict[str, object]] = []
        self._max_telemetry = 200
        self._cached_payload: Dict[str, object] = self._build_default_payload()

    def _build_default_payload(self) -> Dict[str, object]:
        return {
            "meta_report": "(awaiting conversation)",
            "decision": {
                "action": "-",
                "score": "0.00",
                "cooldown": 0,
                "hysteresis_delta": "0.00",
            },
            "status": {
                "step": 0,
                "budget_left": "0.0",
                "idea_acceptances": 0,
                "verification_passed": None,
                "budget_ratio": "0.00",
                "events": "<none>",
            },
            "scratchpad": ["<empty scratchpad>"],
            "todos": [{"id": "-", "text": "<empty>", "score": "0.00"}],
            "maintenance": [{"category": "<none>", "count": "0"}],
            "experiments": [{"name": "<none>", "conclusion": ""}],
            "history": [],
        }

    def refresh_from_session(
        self,
        *,
        metrics: Optional[RuntimeMetrics] = None,
        snapshot: Optional[ConversationSnapshot] = None,
    ) -> None:
        with self._lock:
            if metrics is not None:
                self._latest_metrics = metrics
            if snapshot is not None:
                self._latest_snapshot = snapshot

            metrics_obj = self._latest_snapshot.metrics if self._latest_snapshot else self._latest_metrics
            payload = self._build_default_payload()

            if metrics_obj:
                decision = metrics_obj.decision
                payload["meta_report"] = metrics_obj.meta_report
                payload["decision"] = {
                    "action": str(decision.action),
                    "score": f"{decision.score:.2f}",
                    "cooldown": decision.cooldown_steps,
                    "hysteresis_delta": f"{decision.hysteresis_delta:.2f}",
                }
                scheduler = metrics_obj.scheduler_snapshot
                payload["status"] = {
                    "step": metrics_obj.step,
                    "budget_left": f"{metrics_obj.budget_left:.1f}",
                    "idea_acceptances": metrics_obj.idea_acceptances,
                    "verification_passed": metrics_obj.verification_passed,
                    "budget_ratio": f"{scheduler.get('budget_ratio', 0.0):.2f}",
                    "events": ", ".join(scheduler.get("events", ())) or "<none>",
                }
                maintenance = metrics_obj.maintenance_report or {}
                payload["maintenance"] = [
                    {"category": key, "count": value}
                    for key, value in maintenance.items()
                ] or [{"category": "<none>", "count": "0"}]
                experiments = metrics_obj.experiment_reports or ()
                payload["experiments"] = [
                    {
                        "name": report.get("name", "<unnamed>"),
                        "conclusion": report.get("conclusion", ""),
                    }
                    for report in experiments
                ] or [{"name": "<none>", "conclusion": ""}]

            proto_lm = getattr(self._session, "proto_lm", None)
            if proto_lm is not None:
                status = payload.setdefault("status", {})
                status.setdefault("step", proto_lm.step)
                status["model_step"] = proto_lm.step
                status["generator"] = getattr(
                    self._latest_snapshot,
                    "generator_description",
                    f"vocab_size={proto_lm.vocab.size()} step={proto_lm.step}",
                )

            if self._latest_snapshot:
                todos_seq = list(self._latest_snapshot.todos)
                payload["todos"] = [
                    {
                        "id": idx + 1,
                        "text": getattr(item, "text", str(item)),
                        "score": f"{getattr(item, 'score', 0.0):.2f}",
                    }
                    for idx, item in enumerate(todos_seq)
                ] or [{"id": "-", "text": "<empty>", "score": "0.00"}]

            scratchpad = getattr(self._session.runtime, "scratchpad", None)
            if scratchpad is not None:
                scratch_lines = list(scratchpad.current_trace[-16:])
                if not scratch_lines:
                    scratch_lines = ["<empty>"]
                scratch_lines.append("\nSummary: " + scratchpad.summarize(max_traces=5))
                payload["scratchpad"] = scratch_lines

            history_items = list(self._session.history)
            payload["history"] = [
                {"speaker": speaker, "text": text}
                for speaker, text in history_items[-50:]
            ]

            self._cached_payload = payload

    def snapshot(self) -> Dict[str, object]:
        with self._lock:
            return dict(self._cached_payload)

    def record_telemetry(self, event: TelemetryEvent) -> None:
        if isinstance(event, ParameterEvent):
            rendered = render_parameter_event(event)
        elif event.type == "training/step":
            rendered = "\n".join(render_training_event(event))
        elif event.type == "ingestion/chunk":
            rendered = render_ingestion_event(event)
        else:
            rendered = json.dumps(event.payload)
        entry = {
            "type": event.type,
            "rendered": rendered,
            "timestamp": time.time(),
        }
        with self._lock:
            self._telemetry.append(entry)
            if len(self._telemetry) > self._max_telemetry:
                self._telemetry = self._telemetry[-self._max_telemetry :]

    def telemetry(self) -> Dict[str, object]:
        with self._lock:
            return {"entries": list(self._telemetry)}


STATE: Optional[DashboardState] = None


def telemetry_sink(event: TelemetryEvent) -> None:
    if STATE is not None:
        STATE.record_telemetry(event)


class DashboardHandler(BaseHTTPRequestHandler):
    state: Optional[DashboardState] = None
    session: Optional[ConversationSession] = None

    def do_GET(self) -> None:  # pragma: no cover - IO heavy
        parsed = urlparse(self.path)
        path = parsed.path
        if path in {"/", "/index.html"}:
            self._send_html(HTML_TEMPLATE)
            return
        if path == "/metrics":
            if self.__class__.state is not None:
                self.__class__.state.refresh_from_session()
                payload = self.__class__.state.snapshot()
            else:
                payload = {"error": "dashboard not initialized"}
            self._send_json(payload if payload else {"error": "snapshot not ready"})
            return
        if path == "/telemetry":
            payload = self.__class__.state.telemetry() if self.__class__.state else {"entries": []}
            self._send_json(payload)
            return
        self.send_error(HTTPStatus.NOT_FOUND, "Unknown endpoint")

    def do_POST(self) -> None:  # pragma: no cover - IO heavy
        parsed = urlparse(self.path)
        path = parsed.path
        if path == "/utter":
            self._handle_utter()
            return
        if path == "/ingest":
            self._handle_ingest()
            return
        if path == "/think":
            self._handle_think()
            return
        if path == "/run_gym":
            self._handle_run_gym()
            return
        if path == "/eat_food":
            self._handle_eat_food()
            return
        self.send_error(HTTPStatus.NOT_FOUND, "Unknown endpoint")

    def _handle_utter(self) -> None:
        if self.__class__.state is None or self.__class__.session is None:
            self.send_error(HTTPStatus.INTERNAL_SERVER_ERROR, "Dashboard not initialized")
            return
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length)
        try:
            payload = json.loads(raw.decode("utf-8")) if raw else {}
        except json.JSONDecodeError:
            self.send_error(HTTPStatus.BAD_REQUEST, "Invalid JSON")
            return
        text = str(payload.get("text", "")).strip()
        if not text:
            self.send_error(HTTPStatus.BAD_REQUEST, "Text is required")
            return
        try:
            user_metrics = self.__class__.session.process_user_input(text)
            self.__class__.state.refresh_from_session(metrics=user_metrics)
            snapshot = self.__class__.session.generate_response()
            self.__class__.state.refresh_from_session(snapshot=snapshot)
        except Exception as exc:  # pragma: no cover - runtime safety
            self.send_error(HTTPStatus.INTERNAL_SERVER_ERROR, f"Conversation error: {exc}")
            return
        response = {
            "response": snapshot.response,
            "meta_report": snapshot.metrics.meta_report,
        }
        self._send_json(response)

    def _handle_ingest(self) -> None:
        if self.__class__.state is None or self.__class__.session is None:
            self.send_error(HTTPStatus.INTERNAL_SERVER_ERROR, "Dashboard not initialized")
            return
        content_type = self.headers.get("Content-Type", "")
        if "multipart/form-data" not in content_type:
            self.send_error(HTTPStatus.BAD_REQUEST, "Expected multipart/form-data upload")
            return

        boundary_token = "boundary="
        boundary_idx = content_type.find(boundary_token)
        if boundary_idx == -1:
            self.send_error(HTTPStatus.BAD_REQUEST, "Missing multipart boundary")
            return
        boundary = content_type[boundary_idx + len(boundary_token):]
        if boundary.startswith('"') and boundary.endswith('"'):
            boundary = boundary[1:-1]
        boundary_bytes = ("--" + boundary).encode("utf-8")
        length = int(self.headers.get("Content-Length", "0"))
        payload = self.rfile.read(length)

        segments_total = 0
        files_processed: List[str] = []
        allow_duplicates = False
        buffered_files: List[Tuple[str, str]] = []

        parts = payload.split(boundary_bytes)
        for part in parts:
            if not part or part == b"--\r\n" or part == b"--":
                continue
            header_section, _, body = part.partition(b"\r\n\r\n")
            if not body:
                continue
            headers = header_section.decode("utf-8", errors="ignore")
            if "Content-Disposition" not in headers:
                continue
            name_marker = "name=\""
            name_start = headers.find(name_marker)
            name_end = headers.find("\"", name_start + len(name_marker)) if name_start != -1 else -1
            field_name = headers[name_start + len(name_marker): name_end] if name_start != -1 and name_end != -1 else ""
            if "filename=" not in headers:
                if field_name == "allow_duplicates":
                    value = body.rstrip(b"\r\n--").decode("utf-8", errors="ignore").strip()
                    allow_duplicates = value in {"1", "true", "True"}
                continue

            filename_marker = "filename="
            start = headers.find(filename_marker)
            end = headers.find("\r\n", start)
            filename = headers[start + len(filename_marker): end].strip().strip('"')
            text = body.rstrip(b"\r\n--").decode("utf-8", errors="ignore")
            buffered_files.append((filename or "upload", text))

        for filename, text in buffered_files:
            processed, metrics = self.__class__.session.ingest_text(
                text,
                source=filename,
                allow_duplicates=allow_duplicates,
            )
            segments_total += processed
            if metrics is not None:
                self.__class__.state.refresh_from_session(snapshot=ConversationSnapshot(metrics=metrics, response="", meta_report=metrics.meta_report, todos=(), generator_description="ingest"))
            files_processed.append(filename)

        self._send_json({
            "segments": segments_total,
            "files": files_processed,
        })

    def _handle_think(self) -> None:
        if self.__class__.state is None or self.__class__.session is None:
            self.send_error(HTTPStatus.INTERNAL_SERVER_ERROR, "Dashboard not initialized")
            return
        runtime = self.__class__.session.runtime
        runtime.introspection.request_reflection(boost=1.0)
        self._send_json({"status": "queued"})

    def _handle_run_gym(self) -> None:
        if self.__class__.state is None or self.__class__.session is None:
            self.send_error(HTTPStatus.INTERNAL_SERVER_ERROR, "Dashboard not initialized")
            return
        session = self.__class__.session
        result = _run_cot_curriculum(session)
        if self.__class__.state is not None:
            self.__class__.state.refresh_from_session()
        self._send_json(result | {"status": "ok"})

    def _handle_eat_food(self) -> None:
        if self.__class__.state is None or self.__class__.session is None:
            self.send_error(HTTPStatus.INTERNAL_SERVER_ERROR, "Dashboard not initialized")
            return
        session = self.__class__.session
        result = _ingest_food_corpus(session)
        if self.__class__.state is not None:
            self.__class__.state.refresh_from_session()
        self._send_json(result | {"status": "ok"})

    def log_message(self, format: str, *args: object) -> None:
        return  # suppress console spam

    def _send_html(self, html: str) -> None:
        encoded = html.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def _send_json(self, payload: Dict[str, object]) -> None:
        encoded = json.dumps(payload).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


def run_dashboard(
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    auto_save_path: Optional[str] = None,
    checkpoint_path: Optional[str] = "storage/proto_lm/tinystories.pt",
    device: str = "auto",
    learning_enabled: bool = False,
    archive_checkpoint_on_start: bool = False,
) -> None:
    training_config = TrainingConfig()
    training_config.device = device
    if checkpoint_path is not None:
        resolved = Path(checkpoint_path)
        if not resolved.is_absolute():
            resolved = BASE_DIR / resolved
        training_config.checkpoint_path = str(resolved)
    session_config = ConversationConfig(
        auto_save_path=auto_save_path,
        lm=training_config,
        learning_enabled=learning_enabled,
        archive_checkpoint_on_start=archive_checkpoint_on_start,
    )
    session = ConversationSession(config=session_config)
    run_dashboard_from_session(
        session=session,
        host=host,
        port=port,
    )


def run_dashboard_from_session(
    *,
    session: ConversationSession,
    host: str = "127.0.0.1",
    port: int = 8765,
) -> None:
    state = DashboardState(session)
    session_snapshot = session.generate_response("") if session.history else None
    if session_snapshot is not None:
        state.refresh_from_session(snapshot=session_snapshot)
    else:
        state.refresh_from_session()

    global STATE
    STATE = state
    DashboardHandler.state = state
    DashboardHandler.session = session

    unsubscribe = BUS.subscribe(telemetry_sink)
    try:
        with ThreadingHTTPServer((host, port), DashboardHandler) as server:
            server.serve_forever()
    except KeyboardInterrupt:  # pragma: no cover - interactive
        pass
    finally:
        unsubscribe()

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--auto-save-path")
    parser.add_argument("--checkpoint", default="storage/proto_lm/tinystories.pt")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--learn", action="store_true")
    parser.add_argument("--archive", action="store_true")
    args = parser.parse_args()
    run_dashboard(
        host=args.host,
        port=args.port,
        auto_save_path=args.auto_save_path,
        checkpoint_path=args.checkpoint,
        device=args.device,
        learning_enabled=args.learn,
        archive_checkpoint_on_start=args.archive,
    )


if __name__ == "__main__":
    main()
