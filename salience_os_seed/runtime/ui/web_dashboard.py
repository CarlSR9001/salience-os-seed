"""HTML dashboard for SalienceRuntime using the standard library."""

from __future__ import annotations

import argparse
import json
import threading
import time
from collections import deque
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple
from urllib.parse import urlparse

from ...conversation.session import ConversationConfig, ConversationSession, ConversationSnapshot
from ...proto_lm.trainer import TrainingConfig
from ...runtime.orchestrator import RuntimeMetrics
from ...training.cot_curriculum.loader import iter_examples
from ...telemetry import (
    BUS,
    ParameterEvent,
    SpatialEvent,
    TelemetryEvent,
    render_ingestion_event,
    render_parameter_event,
    render_spatial_event,
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
        count += segments_processed
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


_SPARKLINE_BARS = "▁▂▃▄▅▆▇█"


def _render_sparkline(values: Deque[float], *, width: int = 32) -> str:
    if not values:
        return "—"
    tail = list(values)[-width:]
    minimum = min(tail)
    maximum = max(tail)
    if maximum - minimum < 1e-6:
        return _SPARKLINE_BARS[0] * len(tail)
    span = maximum - minimum
    scale = len(_SPARKLINE_BARS) - 1
    bars = []
    for value in tail:
        index = int((value - minimum) / span * scale)
        index = max(0, min(scale, index))
        bars.append(_SPARKLINE_BARS[index])
    return "".join(bars)


def _format_salience(salience_raw: Dict[str, float]) -> Dict[str, object]:
    if not salience_raw:
        return {"top": [], "raw": {}}
    positive_total = sum(max(value, 0.0) for value in salience_raw.values()) or 1.0
    top = sorted(salience_raw.items(), key=lambda item: item[1], reverse=True)[:8]
    return {
        "top": [
            {
                "dimension": key,
                "value": float(value),
                "percent": max(value, 0.0) / positive_total,
            }
            for key, value in top
        ],
        "raw": {key: float(value) for key, value in salience_raw.items()},
    }

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <title>Salience Runtime Observatory</title>
  <style>
    :root {
      color-scheme: dark;
      font-family: 'DM Mono', 'Fira Code', 'SFMono-Regular', Consolas, monospace;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      background: radial-gradient(circle at 16% 20%, #10244d, #040513 70%);
      color: #eaf8ff;
      letter-spacing: 0.01em;
    }
    body::before {
      content: '';
      position: fixed;
      inset: 0;
      pointer-events: none;
      background-image:
        radial-gradient(circle at 20% 20%, rgba(72, 255, 210, 0.07) 0, transparent 55%),
        linear-gradient(rgba(255, 255, 255, 0.05) 1px, transparent 1px);
      background-size: 640px 640px, 100% 4px;
      mix-blend-mode: screen;
      opacity: 0.6;
    }
    .app-header {
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      gap: 24px;
      padding: 22px 28px 18px;
      background: linear-gradient(135deg, rgba(8, 31, 72, 0.92), rgba(15, 96, 112, 0.55));
      border-bottom: 2px solid rgba(95, 255, 230, 0.5);
      position: sticky;
      top: 0;
      z-index: 20;
      backdrop-filter: blur(10px);
    }
    .brand h1 {
      margin: 0;
      font-size: 1.6rem;
      text-transform: uppercase;
      color: #92ffe0;
      letter-spacing: 0.26em;
      text-shadow: 0 0 12px rgba(80, 255, 225, 0.7);
    }
    .meta {
      margin-top: 8px;
      color: #9ec9ff;
      font-size: 0.86rem;
      white-space: pre-wrap;
    }
    .status-pills {
      display: flex;
      gap: 12px;
      flex-wrap: wrap;
    }
    .pill {
      padding: 10px 16px;
      border-radius: 999px;
      border: 1px solid rgba(120, 255, 230, 0.4);
      background: rgba(12, 38, 72, 0.75);
      color: #c7f9ff;
      min-width: 140px;
      display: flex;
      flex-direction: column;
      gap: 4px;
      box-shadow: 0 0 12px rgba(30, 140, 180, 0.35);
      text-transform: uppercase;
      font-size: 0.64rem;
      letter-spacing: 0.18em;
    }
    .pill .value {
      font-size: 0.95rem;
      letter-spacing: 0.08em;
      color: #f5fffb;
    }
    main.grid {
      display: grid;
      grid-template-columns: repeat(12, minmax(0, 1fr));
      gap: 20px;
      padding: 26px;
    }
    .panel {
      position: relative;
      grid-column: span 4;
      background: rgba(6, 20, 44, 0.88);
      border: 1px solid rgba(96, 246, 255, 0.28);
      border-radius: 14px;
      padding: 18px 20px;
      display: flex;
      flex-direction: column;
      gap: 14px;
      box-shadow: 0 18px 45px rgba(0, 0, 0, 0.45), inset 0 0 22px rgba(40, 120, 160, 0.15);
      overflow: hidden;
    }
    .panel::after {
      content: '';
      position: absolute;
      inset: 0;
      border-radius: 12px;
      border: 1px solid rgba(95, 255, 210, 0.12);
      pointer-events: none;
    }
    .panel-title {
      text-transform: uppercase;
      letter-spacing: 0.22em;
      font-size: 0.82rem;
      color: #7fffe0;
    }
    .span-2 { grid-column: span 8; }
    .span-3 { grid-column: span 12; }
    .sparkline-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
      gap: 12px;
    }
    .spark-card {
      padding: 10px 12px;
      border-radius: 10px;
      background: rgba(12, 42, 84, 0.55);
      border: 1px solid rgba(120, 255, 230, 0.2);
    }
    .spark-label {
      font-size: 0.68rem;
      text-transform: uppercase;
      letter-spacing: 0.16em;
      color: #74d5ff;
    }
    .sparkline {
      margin-top: 4px;
      font-size: 1.1rem;
      letter-spacing: 0.08em;
      color: #e0fffe;
      white-space: nowrap;
    }
    .kv-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
      gap: 8px 14px;
    }
    .kv {
      padding: 8px 10px;
      border-radius: 10px;
      background: rgba(15, 40, 74, 0.55);
      border: 1px solid rgba(120, 255, 240, 0.25);
      display: flex;
      flex-direction: column;
      gap: 6px;
    }
    .kv .label {
      font-size: 0.7rem;
      letter-spacing: 0.14em;
      text-transform: uppercase;
      color: #6fd9ff;
    }
    .kv .value {
      font-size: 1rem;
      color: #f4fffd;
    }
    .list-table {
      width: 100%;
      border-collapse: collapse;
      font-size: 0.86rem;
    }
    .list-table tr + tr { border-top: 1px solid rgba(120, 220, 255, 0.22); }
    .list-table td {
      padding: 6px 0;
      vertical-align: top;
    }
    .list-table td:first-child {
      width: 72px;
      color: #7dd9ff;
      font-size: 0.72rem;
      letter-spacing: 0.1em;
      text-transform: uppercase;
    }
    .chip-row {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
    }
    .chip {
      padding: 4px 10px;
      border-radius: 999px;
      border: 1px solid rgba(120, 255, 240, 0.25);
      background: rgba(22, 62, 110, 0.6);
      font-size: 0.72rem;
      letter-spacing: 0.1em;
      text-transform: uppercase;
      color: #dffcff;
    }
    .conversation {
      display: flex;
      flex-direction: column;
      gap: 12px;
      max-height: 520px;
      overflow-y: auto;
      padding-right: 6px;
    }
    .bubble {
      border-radius: 12px;
      padding: 12px 14px;
      background: rgba(18, 48, 98, 0.6);
      border: 1px solid rgba(120, 255, 240, 0.25);
      box-shadow: inset 0 0 18px rgba(70, 170, 210, 0.25);
    }
    .bubble.user { background: rgba(26, 94, 160, 0.6); align-self: flex-end; }
    .bubble.assistant { background: rgba(14, 92, 82, 0.55); }
    .bubble .speaker {
      font-size: 0.7rem;
      text-transform: uppercase;
      letter-spacing: 0.16em;
      color: #a7f6ff;
      margin-bottom: 4px;
    }
    .bubble pre {
      margin: 0;
      white-space: pre-wrap;
      word-break: break-word;
      color: #f1faff;
    }
    textarea {
      resize: vertical;
      min-height: 88px;
      padding: 12px;
      border-radius: 8px;
      border: 1px solid rgba(110, 255, 220, 0.3);
      background: rgba(10, 28, 58, 0.75);
      color: #e9faff;
    }
    button {
      align-self: flex-start;
      padding: 9px 18px;
      border-radius: 8px;
      border: 1px solid rgba(120, 255, 230, 0.55);
      background: linear-gradient(120deg, rgba(16, 60, 120, 0.9), rgba(18, 112, 108, 0.9));
      color: #a6fff1;
      text-transform: uppercase;
      letter-spacing: 0.2em;
      font-size: 0.7rem;
      cursor: pointer;
      box-shadow: 0 10px 24px rgba(0, 0, 0, 0.35);
      transition: transform 0.1s ease, box-shadow 0.2s ease;
    }
    button:hover { transform: translateY(-1px); }
    button:disabled { opacity: 0.55; cursor: wait; box-shadow: none; }
    .status-line {
      font-size: 0.78rem;
      min-height: 1.1rem;
      color: #86d6ff;
    }
    .meter {
      position: relative;
      height: 12px;
      width: 100%;
      border: 1px solid rgba(120, 255, 230, 0.5);
      border-radius: 999px;
      overflow: hidden;
      background: rgba(8, 24, 58, 0.8);
    }
    .meter-fill {
      position: absolute;
      inset: 0;
      width: 0%;
      background: linear-gradient(90deg, #21ffe6 0%, #78ffe9 50%, #21ffe6 100%);
      transition: width 0.4s ease;
    }
    .meter.active .meter-fill {
      animation: slide 1s linear infinite;
      background-size: 32px 100%;
    }
    @keyframes slide {
      from { background-position: 0 0; }
      to { background-position: 32px 0; }
    }
    .telemetry-feed {
      display: flex;
      flex-direction: column;
      gap: 10px;
      max-height: 420px;
      overflow-y: auto;
      padding-right: 6px;
    }
    .telemetry-entry {
      padding: 10px;
      border-radius: 10px;
      background: rgba(12, 32, 68, 0.7);
      border: 1px solid rgba(120, 255, 240, 0.22);
      font-size: 0.78rem;
      line-height: 1.35;
    }
    .badge {
      display: inline-block;
      padding: 2px 8px;
      border-radius: 999px;
      font-size: 0.62rem;
      letter-spacing: 0.18em;
      margin-bottom: 6px;
      font-weight: 700;
      text-transform: uppercase;
      color: #02122a;
    }
    .badge.param { background: #6dffd4; }
    .badge.train { background: #ffe480; }
    .badge.ingest { background: #7fd1ff; }
    .badge.spatial { background: #d9a5ff; color: #1c012c; }
    canvas {
      border-radius: 10px;
      border: 1px solid rgba(120, 255, 230, 0.25);
      background: radial-gradient(circle at 45% 20%, rgba(56, 150, 200, 0.35), rgba(6, 18, 44, 0.95));
      width: 100%;
      height: 220px;
    }
    pre.mono {
      margin: 0;
      max-height: 180px;
      overflow-y: auto;
      padding: 10px;
      background: rgba(12, 32, 68, 0.6);
      border-radius: 8px;
      border: 1px solid rgba(120, 255, 240, 0.18);
    }
    .control-group {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
      gap: 12px;
      align-items: start;
    }
    .control {
      display: flex;
      flex-direction: column;
      gap: 6px;
    }
    label.checkbox {
      font-size: 0.78rem;
      display: flex;
      gap: 6px;
      align-items: center;
      color: #9fe2ff;
    }
    input[type=\"file\"] {
      color: #a6faff;
    }
    @media (max-width: 1280px) {
      main.grid { grid-template-columns: repeat(6, minmax(0, 1fr)); }
      .panel { grid-column: span 6; }
      .span-2, .span-3 { grid-column: span 6; }
    }
    @media (max-width: 760px) {
      main.grid { grid-template-columns: repeat(2, minmax(0, 1fr)); padding: 18px; }
      .panel { grid-column: span 2; }
    }
  </style>
</head>
<body>
  <header class=\"app-header\">
    <div class=\"brand\">
      <h1>Salience Runtime Observatory</h1>
      <div id=\"meta\" class=\"meta\"></div>
    </div>
    <div class=\"status-pills\">
      <div class=\"pill\" id=\"learningPill\"><span class=\"label\">Learning</span><span class=\"value\" id=\"learningState\">Idle</span></div>
      <div class=\"pill\" id=\"filterPill\"><span class=\"label\">Ingest Filter</span><span class=\"value\" id=\"filterState\">Unknown</span></div>
      <div class=\"pill\" id=\"trainingPill\"><span class=\"label\">Training Active</span><span class=\"value\" id=\"trainingState\">Unknown</span></div>
    </div>
  </header>
  <main class=\"grid\">
    <section class=\"panel span-2\">
      <div class=\"panel-title\">Overview</div>
      <div class=\"kv-grid\" id=\"overviewStats\"></div>
      <div class=\"sparkline-grid\">
        <div class=\"spark-card\"><div class=\"spark-label\">Decision score</div><div class=\"sparkline\" id=\"scoreSpark\">—</div></div>
        <div class=\"spark-card\"><div class=\"spark-label\">Budget ratio</div><div class=\"sparkline\" id=\"budgetSpark\">—</div></div>
        <div class=\"spark-card\"><div class=\"spark-label\">Idea acceptance</div><div class=\"sparkline\" id=\"ideaSpark\">—</div></div>
        <div class=\"spark-card\"><div class=\"spark-label\">Verification rate</div><div class=\"sparkline\" id=\"verificationSpark\">—</div></div>
      </div>
    </section>
    <section class=\"panel\">
      <div class=\"panel-title\">Decision</div>
      <div id=\"decision\"></div>
    </section>
    <section class=\"panel\">
      <div class=\"panel-title\">Status</div>
      <div class=\"kv-grid\" id=\"status\"></div>
      <div>
        <div class=\"label\" style=\"text-transform:uppercase;font-size:0.7rem;letter-spacing:0.14em;color:#6fd9ff;\">Budget utilisation</div>
        <div class=\"meter\" id=\"budgetMeter\"><div class=\"meter-fill\" id=\"budgetFill\"></div></div>
        <div class=\"status-line\" id=\"budgetPercent\">0%</div>
      </div>
    </section>
    <section class=\"panel\">
      <div class=\"panel-title\">Salience</div>
      <div id=\"salience\"></div>
      <div class=\"chip-row\" id=\"yearning\"></div>
    </section>
    <section class=\"panel\">
      <div class=\"panel-title\">Todos</div>
      <div id=\"todos\"></div>
    </section>
    <section class=\"panel\">
      <div class=\"panel-title\">Maintenance</div>
      <div id=\"maintenance\"></div>
    </section>
    <section class=\"panel\">
      <div class=\"panel-title\">Experiments</div>
      <div id=\"experiments\"></div>
    </section>
    <section class=\"panel span-2\">
      <div class=\"panel-title\">Scratchpad</div>
      <div id=\"scratchpad\"></div>
    </section>
    <section class=\"panel span-2\">
      <div class=\"panel-title\">Conversation</div>
      <div class=\"conversation\" id=\"conversationWrapper\"><div id=\"conversation\"></div></div>
    </section>
    <section class=\"panel\">
      <div class=\"panel-title\">4D Trajectory</div>
      <canvas id=\"spatialCanvas\" width=\"360\" height=\"220\"></canvas>
      <pre class=\"mono\" id=\"spatialAscii\"></pre>
      <div class=\"status-line\" id=\"spatialSummary\"></div>
    </section>
    <section class=\"panel\">
      <div class=\"panel-title\">Send Message</div>
      <form id=\"utterForm\">
        <textarea id=\"utterInput\" placeholder=\"Type a message for the runtime...\"></textarea>
        <div class=\"status-line\" id=\"utterStatus\"></div>
        <button type=\"submit\" id=\"utterSubmit\">Transmit</button>
      </form>
    </section>
    <section class=\"panel\">
      <div class=\"panel-title\">Runtime Controls</div>
      <div class=\"control-group\">
        <div class=\"control\"><button type=\"button\" id=\"thinkButton\">Reflection pulse</button><div class=\"status-line\" id=\"thinkStatus\"></div></div>
        <div class=\"control\"><button type=\"button\" id=\"runGymButton\">Curriculum sweep</button><div class=\"status-line\" id=\"runGymStatus\"></div><div class=\"meter\" id=\"runGymMeter\"><div class=\"meter-fill\"></div></div></div>
        <div class=\"control\"><button type=\"button\" id=\"eatFoodButton\">Digest food corpus</button><div class=\"status-line\" id=\"eatFoodStatus\"></div><div class=\"meter\" id=\"eatFoodMeter\"><div class=\"meter-fill\"></div></div></div>
      </div>
    </section>
    <section class=\"panel\">
      <div class=\"panel-title\">Learning + Filters</div>
      <div class=\"control-group\">
        <div class=\"control\"><button type=\"button\" id=\"toggleLearningButton\">Toggle lightweight learning</button><div class=\"status-line\" id=\"learningStatus\"></div></div>
        <div class=\"control\"><button type=\"button\" id=\"flushLearningButton\">Flush buffer to train</button><div class=\"status-line\" id=\"flushStatus\"></div></div>
        <div class=\"control\"><button type=\"button\" id=\"trainingActiveButton\">Toggle training active</button><div class=\"status-line\" id=\"trainingStatus\"></div></div>
        <div class=\"control\"><button type=\"button\" id=\"toggleFilterButton\">Toggle ingest filter</button><div class=\"status-line\" id=\"filterStatus\"></div></div>
      </div>
      <div class=\"kv-grid\" id=\"trainingStats\"></div>
    </section>
    <section class=\"panel\">
      <div class=\"panel-title\">Upload &amp; Ingest</div>
      <form id=\"uploadForm\">
        <input type=\"file\" id=\"uploadInput\" accept=\".txt,.md,.log,.json,.csv\" multiple />
        <label class=\"checkbox\"><input type=\"checkbox\" id=\"allowDuplicates\" /> Allow duplicate ingest</label>
        <div class=\"status-line\" id=\"uploadStatus\"></div>
        <button type=\"submit\" id=\"uploadSubmit\">Upload &amp; ingest</button>
      </form>
    </section>
    <section class=\"panel span-2\">
      <div class=\"panel-title\">Telemetry</div>
      <div class=\"telemetry-feed\" id=\"telemetryWrapper\"><div id=\"telemetry\"></div></div>
    </section>
  </main>
  <script>
    async function fetchJson(path, options = {}) {
      const resp = await fetch(path, { cache: 'no-cache', ...options });
      if (!resp.ok) {
        const text = await resp.text();
        throw new Error(text || `HTTP ${resp.status}`);
      }
      return resp.json();
    }

    function renderKeyValues(container, entries) {
      if (!entries || entries.length === 0) {
        container.innerHTML = '<div class="status-line">No data</div>';
        return;
      }
      container.innerHTML = entries.map(({ label, value }) => (
        `<div class="kv"><div class="label">${label}</div><div class="value">${value ?? ''}</div></div>`
      )).join('');
    }

    function renderTable(container, rows) {
      if (!rows || rows.length === 0) {
        container.innerHTML = '<div class="status-line">No entries</div>';
        return;
      }
      const header = Object.keys(rows[0]);
      let html = '<table class="list-table">';
      for (const row of rows) {
        html += '<tr>';
        for (const key of header) {
          html += `<td>${row[key] ?? ''}</td>`;
        }
        html += '</tr>';
      }
      html += '</table>';
      container.innerHTML = html;
    }

    function renderList(container, lines) {
      if (!lines || lines.length === 0) {
        container.innerHTML = '<div class="status-line">No signal yet.</div>';
        return;
      }
      container.innerHTML = `<pre class="mono">${lines.join('\\n')}</pre>`;
    }

    function renderConversation(container, history) {
      if (!history || history.length === 0) {
        container.innerHTML = '<div class="status-line">No dialogue yet.</div>';
        return;
      }
      container.innerHTML = history.map((entry) => {
        const speaker = entry.speaker || 'agent';
        const klass = speaker.toLowerCase().includes('user') ? 'bubble user' : 'bubble assistant';
        return `<div class="${klass}"><span class="speaker">${speaker}</span><pre>${entry.text || ''}</pre></div>`;
      }).join('');
    }

    function renderSalience(container, payload) {
      if (!payload || !payload.top || payload.top.length === 0) {
        container.innerHTML = '<div class="status-line">No salience vector.</div>';
        return;
      }
      const rows = payload.top.map((item) => (
        `<tr><td>${item.dimension}</td><td>${item.value.toFixed(3)}</td><td>${(item.percent * 100).toFixed(1)}%</td></tr>`
      ));
      container.innerHTML = `<table class="list-table"><tr><td>Dim</td><td>Value</td><td>Share</td></tr>${rows.join('')}</table>`;
    }

    function renderYearning(container, payload) {
      if (!payload || Object.keys(payload).length === 0) {
        container.innerHTML = '<div class="status-line">No yearning snapshot.</div>';
        return;
      }
      container.innerHTML = Object.entries(payload).map(([name, bands]) => {
        const score = (bands.score ?? bands.confidence ?? 0).toFixed(2);
        return `<span class="chip">${name}: ${score}</span>`;
      }).join('');
    }

    function renderTelemetry(container, entries) {
      container.innerHTML = '';
      if (!entries || entries.length === 0) {
        container.innerHTML = '<div class="status-line">No telemetry yet.</div>';
        return;
      }
      for (const entry of entries) {
        const badgeClass = entry.type.startsWith('parameters') ? 'param'
          : entry.type.startsWith('training') ? 'train'
          : entry.type.startsWith('spatial') ? 'spatial'
          : 'ingest';
        const label = entry.type.split('/')[0] || 'event';
        const div = document.createElement('div');
        div.className = 'telemetry-entry';
        div.innerHTML = `<span class="badge ${badgeClass}">${label}</span><pre>${entry.rendered}</pre>`;
        container.appendChild(div);
      }
    }

    function renderSparklines(data = {}) {
      const { score = '—', budget = '—', acceptance = '—', verification = '—' } = data;
      document.getElementById('scoreSpark').textContent = score;
      document.getElementById('budgetSpark').textContent = budget;
      document.getElementById('ideaSpark').textContent = acceptance;
      document.getElementById('verificationSpark').textContent = verification;
    }

    function renderSpatial(payload) {
      const canvas = document.getElementById('spatialCanvas');
      const ctx = canvas.getContext('2d');
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      if (payload && Array.isArray(payload.points) && payload.points.length) {
        const xs = payload.points.map(p => p.x ?? 0);
        const ys = payload.points.map(p => p.y ?? 0);
        const minX = Math.min(...xs);
        const maxX = Math.max(...xs);
        const minY = Math.min(...ys);
        const maxY = Math.max(...ys);
        const pad = 12;
        const toCanvasX = (x) => {
          if (maxX === minX) return canvas.width / 2;
          return pad + ((x - minX) / (maxX - minX)) * (canvas.width - pad * 2);
        };
        const toCanvasY = (y) => {
          if (maxY === minY) return canvas.height / 2;
          return canvas.height - (pad + ((y - minY) / (maxY - minY)) * (canvas.height - pad * 2));
        };
        ctx.beginPath();
        payload.points.forEach((pt, idx) => {
          const x = toCanvasX(pt.x ?? 0);
          const y = toCanvasY(pt.y ?? 0);
          if (idx === 0) {
            ctx.moveTo(x, y);
          } else {
            ctx.lineTo(x, y);
          }
        });
        ctx.lineWidth = 2;
        ctx.strokeStyle = 'rgba(80, 255, 220, 0.7)';
        ctx.stroke();
        const last = payload.points[payload.points.length - 1];
        ctx.beginPath();
        ctx.fillStyle = 'rgba(255, 255, 255, 0.9)';
        ctx.arc(toCanvasX(last.x ?? 0), toCanvasY(last.y ?? 0), 5, 0, Math.PI * 2);
        ctx.fill();
      }
      document.getElementById('spatialAscii').textContent = payload?.ascii || '';
      document.getElementById('spatialSummary').textContent = payload?.summary || '';
    }

    function updateMeter(ratio) {
      const clamp = Math.max(0, Math.min(1, Number(ratio) || 0));
      document.getElementById('budgetFill').style.width = `${(clamp * 100).toFixed(1)}%`;
      document.getElementById('budgetPercent').textContent = `${(clamp * 100).toFixed(1)}% budget consumed`;
    }

    async function refresh() {
      try {
        const data = await fetchJson('/metrics');
        document.getElementById('meta').textContent = data.meta_report || '';
        renderKeyValues(document.getElementById('overviewStats'), data.overview || []);
        renderKeyValues(document.getElementById('decision'), data.decision_rows || []);
        renderKeyValues(document.getElementById('status'), data.status_rows || []);
        renderSparklines(data.sparklines || {});
        renderSalience(document.getElementById('salience'), data.salience);
        renderYearning(document.getElementById('yearning'), data.yearning);
        renderList(document.getElementById('scratchpad'), data.scratchpad || []);
        renderTable(document.getElementById('todos'), data.todos || []);
        renderTable(document.getElementById('maintenance'), data.maintenance || []);
        renderTable(document.getElementById('experiments'), data.experiments || []);
        renderConversation(document.getElementById('conversation'), data.history || []);
        renderSpatial(data.spatial);
        renderKeyValues(document.getElementById('trainingStats'), data.training_rows || []);
        updateMeter(data.status?.budget_ratio_numeric ?? 0);
        document.getElementById('learningState').textContent = data.training?.learning_enabled ? 'Enabled' : 'Disabled';
        document.getElementById('filterState').textContent = data.filter?.enabled ? 'Active' : 'Bypassed';
        document.getElementById('trainingState').textContent = data.training?.runtime_active ? 'Active' : 'Idle';
      } catch (err) {
        console.error(err);
      }

      try {
        const telem = await fetchJson('/telemetry');
        const wrapper = document.getElementById('telemetryWrapper');
        renderTelemetry(document.getElementById('telemetry'), telem.entries || []);
        wrapper.scrollTop = wrapper.scrollHeight;
      } catch (err) {
        console.error(err);
      }
    }

    refresh();
    setInterval(refresh, 1200);

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
      status.textContent = 'Transmitting...';
      try {
        const payload = await fetchJson('/utter', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ text })
        });
        status.textContent = `agent> ${payload.response || '(no reply)'}`;
        input.value = '';
        await refresh();
      } catch (err) {
        status.textContent = `Error: ${err.message}`;
      } finally {
        button.disabled = false;
      }
    });

    async function simplePost(path, statusEl) {
      statusEl.textContent = 'Running...';
      try {
        const payload = await fetchJson(path, { method: 'POST' });
        statusEl.textContent = payload.status || 'Done.';
        await refresh();
      } catch (err) {
        statusEl.textContent = `Error: ${err.message}`;
      }
    }

    function activateMeter(meter, active) {
      if (!meter) return;
      meter.classList.toggle('active', active);
      if (!active) {
        const fill = meter.querySelector('.meter-fill');
        if (fill) {
          fill.style.width = '0%';
        }
      }
    }

    async function triggerIngestion(endpoint, statusId, meterId, successMessage) {
      const status = document.getElementById(statusId);
      const meter = document.getElementById(meterId);
      activateMeter(meter, true);
      status.textContent = 'Running...';
      try {
        const payload = await fetchJson(endpoint, { method: 'POST' });
        status.textContent = successMessage(payload);
        const fill = meter.querySelector('.meter-fill');
        if (fill) {
          fill.style.width = '100%';
          setTimeout(() => activateMeter(meter, false), 900);
        } else {
          activateMeter(meter, false);
        }
        await refresh();
      } catch (err) {
        status.textContent = `Error: ${err.message}`;
        activateMeter(meter, false);
      }
    }

    document.getElementById('thinkButton').addEventListener('click', () => {
      simplePost('/think', document.getElementById('thinkStatus'));
    });

    document.getElementById('runGymButton').addEventListener('click', () => {
      triggerIngestion(
        '/run_gym',
        'runGymStatus',
        'runGymMeter',
        (payload) => `Processed ${payload.examples || 0} examples.`
      );
    });

    document.getElementById('eatFoodButton').addEventListener('click', () => {
      triggerIngestion(
        '/eat_food',
        'eatFoodStatus',
        'eatFoodMeter',
        (payload) => `Digested ${payload.files || 0} files / ${payload.chunks || 0} chunks.`
      );
    });

    document.getElementById('toggleLearningButton').addEventListener('click', () => {
      simplePost('/toggle_learning', document.getElementById('learningStatus'));
    });

    document.getElementById('flushLearningButton').addEventListener('click', () => {
      simplePost('/flush_learning', document.getElementById('flushStatus'));
    });

    document.getElementById('trainingActiveButton').addEventListener('click', () => {
      simplePost('/training_active', document.getElementById('trainingStatus'));
    });

    document.getElementById('toggleFilterButton').addEventListener('click', () => {
      simplePost('/toggle_filter', document.getElementById('filterStatus'));
    });

    document.getElementById('uploadForm').addEventListener('submit', async (event) => {
      event.preventDefault();
      const input = document.getElementById('uploadInput');
      const files = input.files;
      const status = document.getElementById('uploadStatus');
      const button = document.getElementById('uploadSubmit');
      if (!files.length) {
        status.textContent = 'Select file(s) first.';
        return;
      }
      const allowDuplicates = document.getElementById('allowDuplicates').checked;
      const formData = new FormData();
      for (const file of files) {
        formData.append('files', file, file.name);
      }
      formData.append('allow_duplicates', allowDuplicates ? '1' : '0');
      button.disabled = true;
      status.textContent = 'Uploading...';
      try {
        const payload = await fetchJson('/ingest', { method: 'POST', body: formData });
        status.textContent = `Ingested ${payload.segments || 0} segments from ${(payload.files || []).length} file(s).`;
        input.value = '';
        await refresh();
      } catch (err) {
        status.textContent = `Error: ${err.message}`;
      } finally {
        button.disabled = false;
      }
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
        self._max_telemetry = 250
        self._cached_payload: Dict[str, object] = self._build_default_payload()
        self._score_history: Deque[float] = deque(maxlen=96)
        self._budget_history: Deque[float] = deque(maxlen=96)
        self._accept_history: Deque[float] = deque(maxlen=96)
        self._verification_history: Deque[float] = deque(maxlen=96)

    def _build_default_payload(self) -> Dict[str, object]:
        return {
            "meta_report": "(awaiting conversation)",
            "overview": [],
            "decision_rows": [],
            "status": {"budget_ratio_numeric": 0.0},
            "status_rows": [],
            "scratchpad": ["<empty>"],
            "salience": {"top": [], "raw": {}},
            "yearning": {},
            "todos": [{"Rank": "—", "Todo": "<empty>", "Score": "0.00"}],
            "maintenance": [{"Category": "—", "Count": "0"}],
            "experiments": [{"Experiment": "—", "Conclusion": ""}],
            "history": [],
            "sparklines": {
                "score": "—",
                "budget": "—",
                "acceptance": "—",
                "verification": "—",
            },
            "training_rows": [],
            "training": {"learning_enabled": False, "runtime_active": False},
            "filter": {"enabled": False, "thresholds": {}},
            "spatial": {"summary": "no path", "ascii": "", "points": []},
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

            if metrics_obj is not None:
                decision = metrics_obj.decision
                scheduler = dict(metrics_obj.scheduler_snapshot or {})
                budget_ratio = float(scheduler.get("budget_ratio", 0.0))
                verification = metrics_obj.verification_passed
                verification_value = (
                    1.0
                    if verification is True
                    else 0.0
                    if verification is False
                    else 0.5
                )
                self._score_history.append(float(decision.score))
                self._budget_history.append(budget_ratio)
                self._accept_history.append(float(metrics_obj.idea_acceptances))
                self._verification_history.append(verification_value)

                payload["meta_report"] = metrics_obj.meta_report
                overview_rows = [
                    {"label": "Step", "value": str(metrics_obj.step)},
                    {"label": "Budget left", "value": f"{metrics_obj.budget_left:.1f}"},
                ]
                generator_desc = getattr(
                    self._latest_snapshot,
                    "generator_description",
                    "",
                )
                if generator_desc:
                    overview_rows.append({"label": "Generator", "value": generator_desc})
                if metrics_obj.episode_recorded:
                    overview_rows.append({"label": "Episode", "value": metrics_obj.episode_recorded})
                payload["overview"] = overview_rows

                payload["decision_rows"] = [
                    {"label": "Action", "value": str(decision.action)},
                    {"label": "Score", "value": f"{decision.score:.2f}"},
                    {"label": "Hysteresis Δ", "value": f"{decision.hysteresis_delta:.2f}"},
                    {"label": "Cooldown", "value": str(decision.cooldown_steps)},
                ]

                events = ", ".join(str(evt) for evt in scheduler.get("events", ())) or "—"
                verification_label = (
                    "Passed" if verification is True else "Failed" if verification is False else "Pending"
                )
                payload["status"] = {
                    "step": metrics_obj.step,
                    "budget_left": metrics_obj.budget_left,
                    "idea_acceptances": metrics_obj.idea_acceptances,
                    "verification": verification_label,
                    "events": events,
                    "budget_ratio_numeric": budget_ratio,
                }
                payload["status_rows"] = [
                    {"label": "Budget left", "value": f"{metrics_obj.budget_left:.1f}"},
                    {"label": "Idea accepts", "value": str(metrics_obj.idea_acceptances)},
                    {"label": "Verification", "value": verification_label},
                    {"label": "Scheduler events", "value": events},
                ]

                maintenance = metrics_obj.maintenance_report or {}
                payload["maintenance"] = [
                    {"Category": key, "Count": str(value)}
                    for key, value in sorted(maintenance.items())
                ] or [{"Category": "—", "Count": "0"}]

                experiments = metrics_obj.experiment_reports or ()
                payload["experiments"] = [
                    {
                        "Experiment": report.get("name", "—"),
                        "Conclusion": report.get("conclusion", ""),
                    }
                    for report in experiments
                ] or [{"Experiment": "—", "Conclusion": ""}]

                payload["salience"] = _format_salience(dict(metrics_obj.salience_raw or {}))
                payload["yearning"] = dict(metrics_obj.yearning_snapshot or {})

            scratchpad = getattr(self._session.runtime, "scratchpad", None)
            if scratchpad is not None:
                scratch_lines = list(scratchpad.current_trace[-20:])
                if not scratch_lines:
                    scratch_lines = ["<empty>"]
                summary = scratchpad.summarize(max_traces=5)
                if summary:
                    scratch_lines.append("")
                    scratch_lines.append("Summary: " + summary)
                payload["scratchpad"] = scratch_lines
                latest_path = (
                    scratchpad.latest_four_d_path() if hasattr(scratchpad, "latest_four_d_path") else None
                )
                if latest_path:
                    path_dict = latest_path.to_dict()
                    payload["spatial"] = {
                        "summary": latest_path.summary(),
                        "ascii": latest_path.ascii_projection(),
                        "points": path_dict.get("points", []),
                    }
                else:
                    payload["spatial"] = {"summary": "no path", "ascii": "", "points": []}
            else:
                payload["scratchpad"] = ["<no scratchpad>"]
                payload["spatial"] = {"summary": "unavailable", "ascii": "", "points": []}

            if self._latest_snapshot is not None:
                todos_seq = list(self._latest_snapshot.todos)
                payload["todos"] = [
                    {
                        "Rank": f"#{idx + 1}",
                        "Todo": getattr(item, "text", str(item)),
                        "Score": f"{getattr(item, 'score', 0.0):.2f}",
                    }
                    for idx, item in enumerate(todos_seq)
                ] or [{"Rank": "—", "Todo": "<empty>", "Score": "0.00"}]
                payload["history"] = [
                    {"speaker": speaker, "text": text}
                    for speaker, text in list(self._session.history)[-64:]
                ]
                if payload["overview"]:
                    payload["overview"].append(
                        {
                            "label": "Response length",
                            "value": str(len(self._latest_snapshot.response or "")),
                        }
                    )
            else:
                payload["history"] = []

            payload["sparklines"] = {
                "score": _render_sparkline(self._score_history),
                "budget": _render_sparkline(self._budget_history),
                "acceptance": _render_sparkline(self._accept_history),
                "verification": _render_sparkline(self._verification_history),
            }

            training_rows, training_state = self._build_training_payload()
            payload["training_rows"] = training_rows
            payload["training"] = training_state
            payload["filter"] = self._build_filter_payload()

            self._cached_payload = payload

    def _build_training_payload(self) -> Tuple[List[Dict[str, str]], Dict[str, object]]:
        config = getattr(self._session, "config", None)
        learning_enabled = bool(getattr(config, "learning_enabled", False))
        buffer_items = len(getattr(self._session, "_learning_buffer", []))
        buffer_chars = int(getattr(self._session, "_learning_accumulated_chars", 0))
        last_flush = int(getattr(self._session, "_learning_last_flush_step", 0))
        proto = getattr(self._session, "proto_lm", None)
        model_step = int(getattr(proto, "step", 0))
        runtime_active = bool(getattr(self._session.runtime.action_context, "training_active", False))
        rows = [
            {"label": "Learning enabled", "value": "yes" if learning_enabled else "no"},
            {"label": "Runtime active", "value": "yes" if runtime_active else "no"},
            {"label": "Buffer items", "value": str(buffer_items)},
            {"label": "Buffered chars", "value": str(buffer_chars)},
            {"label": "Last flush step", "value": str(last_flush)},
            {"label": "Model step", "value": str(model_step)},
        ]
        state = {
            "learning_enabled": learning_enabled,
            "runtime_active": runtime_active,
            "buffer_items": buffer_items,
            "buffer_chars": buffer_chars,
            "last_flush_step": last_flush,
            "model_step": model_step,
        }
        return rows, state

    def _build_filter_payload(self) -> Dict[str, object]:
        enabled = False
        thresholds: Dict[str, Optional[float]] = {}
        if hasattr(self._session, "is_filter_enabled"):
            enabled = bool(self._session.is_filter_enabled())
        filter_obj = getattr(self._session, "_filter", None)
        if filter_obj is not None:
            thresholds_obj = getattr(filter_obj, "thresholds", None)
            if thresholds_obj is not None:
                thresholds = {
                    "min_uncertainty": getattr(thresholds_obj, "min_uncertainty", None),
                    "min_novelty": getattr(thresholds_obj, "min_novelty", None),
                    "max_drag": getattr(thresholds_obj, "max_drag", None),
                }
        return {"enabled": enabled, "thresholds": thresholds}

    def snapshot(self) -> Dict[str, object]:
        with self._lock:
            return dict(self._cached_payload)

    def record_telemetry(self, event: TelemetryEvent) -> None:
        if isinstance(event, ParameterEvent):
            rendered = render_parameter_event(event)
        elif isinstance(event, SpatialEvent):
            rendered = render_spatial_event(event)
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
            "payload": event.payload,
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
        if path == "/toggle_learning":
            self._handle_toggle_learning()
            return
        if path == "/flush_learning":
            self._handle_flush_learning()
            return
        if path == "/training_active":
            self._handle_training_active()
            return
        if path == "/toggle_filter":
            self._handle_toggle_filter()
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
        session = self.__class__.session
        try:
            user_metrics = session.process_user_input(text)
            self.__class__.state.refresh_from_session(metrics=user_metrics)
            snapshot = session.generate_response()
            self.__class__.state.refresh_from_session(snapshot=snapshot)
        except Exception as exc:  # pragma: no cover - runtime safety
            self.send_error(HTTPStatus.INTERNAL_SERVER_ERROR, f"Conversation error: {exc}")
            return
        response = {
            "response": snapshot.response,
            "meta_report": snapshot.metrics.meta_report,
            "status": "ok",
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
        boundary = content_type[boundary_idx + len(boundary_token) :]
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
            if not part or part in {b"--\r\n", b"--"}:
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
            field_name = headers[name_start + len(name_marker) : name_end] if name_start != -1 and name_end != -1 else ""
            if "filename=" not in headers:
                if field_name == "allow_duplicates":
                    value = body.rstrip(b"\r\n--").decode("utf-8", errors="ignore").strip()
                    allow_duplicates = value in {"1", "true", "True"}
                continue

            filename_marker = "filename="
            start = headers.find(filename_marker)
            end = headers.find("\r\n", start)
            filename = headers[start + len(filename_marker) : end].strip().strip('"')
            text = body.rstrip(b"\r\n--").decode("utf-8", errors="ignore")
            buffered_files.append((filename or "upload", text))

        session = self.__class__.session
        for filename, text in buffered_files:
            processed, metrics = session.ingest_text(
                text,
                source=filename,
                allow_duplicates=allow_duplicates,
            )
            segments_total += processed
            if metrics is not None:
                snapshot = ConversationSnapshot(
                    metrics=metrics,
                    response="",
                    meta_report=metrics.meta_report,
                    todos=tuple(),
                    generator_description="ingest",
                )
                self.__class__.state.refresh_from_session(snapshot=snapshot)
            files_processed.append(filename)

        self.__class__.state.refresh_from_session()
        self._send_json({
            "segments": segments_total,
            "files": files_processed,
            "status": f"Ingested {segments_total} segment(s)",
        })

    def _handle_think(self) -> None:
        if self.__class__.state is None or self.__class__.session is None:
            self.send_error(HTTPStatus.INTERNAL_SERVER_ERROR, "Dashboard not initialized")
            return
        runtime = self.__class__.session.runtime
        runtime.introspection.request_reflection(boost=1.0)
        self._send_json({"status": "Reflection queued"})

    def _handle_run_gym(self) -> None:
        if self.__class__.state is None or self.__class__.session is None:
            self.send_error(HTTPStatus.INTERNAL_SERVER_ERROR, "Dashboard not initialized")
            return
        session = self.__class__.session
        result = _run_cot_curriculum(session)
        self.__class__.state.refresh_from_session()
        self._send_json(result | {"status": f"Processed {result.get('examples', 0)} examples"})

    def _handle_eat_food(self) -> None:
        if self.__class__.state is None or self.__class__.session is None:
            self.send_error(HTTPStatus.INTERNAL_SERVER_ERROR, "Dashboard not initialized")
            return
        session = self.__class__.session
        result = _ingest_food_corpus(session)
        self.__class__.state.refresh_from_session()
        self._send_json(result | {"status": "Food corpus digested"})

    def _handle_toggle_learning(self) -> None:
        session = self.__class__.session
        if self.__class__.state is None or session is None:
            self.send_error(HTTPStatus.INTERNAL_SERVER_ERROR, "Dashboard not initialized")
            return
        current = bool(getattr(session.config, "learning_enabled", False))
        session.config.learning_enabled = not current
        status = "Learning enabled" if session.config.learning_enabled else "Learning disabled"
        self.__class__.state.refresh_from_session()
        self._send_json({"status": status})

    def _handle_flush_learning(self) -> None:
        session = self.__class__.session
        if self.__class__.state is None or session is None:
            self.send_error(HTTPStatus.INTERNAL_SERVER_ERROR, "Dashboard not initialized")
            return
        buffered = len(getattr(session, "_learning_buffer", []))
        try:
            session._maybe_train_on_buffer()  # type: ignore[attr-defined]
        except Exception as exc:  # pragma: no cover - runtime safety
            self.send_error(HTTPStatus.INTERNAL_SERVER_ERROR, f"Training error: {exc}")
            return
        self.__class__.state.refresh_from_session()
        self._send_json({"status": f"Flushed {buffered} buffered chunk(s)"})

    def _handle_training_active(self) -> None:
        session = self.__class__.session
        if self.__class__.state is None or session is None:
            self.send_error(HTTPStatus.INTERNAL_SERVER_ERROR, "Dashboard not initialized")
            return
        runtime = session.runtime
        current = bool(getattr(runtime.action_context, "training_active", False))
        runtime.set_training_active(not current)
        self.__class__.state.refresh_from_session()
        status = "Training active" if not current else "Training idle"
        self._send_json({"status": status})

    def _handle_toggle_filter(self) -> None:
        session = self.__class__.session
        if self.__class__.state is None or session is None:
            self.send_error(HTTPStatus.INTERNAL_SERVER_ERROR, "Dashboard not initialized")
            return
        current = bool(session.is_filter_enabled()) if hasattr(session, "is_filter_enabled") else False
        toggle_target = not current
        if hasattr(session, "_set_filter_enabled"):
            session._set_filter_enabled(toggle_target)  # type: ignore[attr-defined]
        self.__class__.state.refresh_from_session()
        status = "Ingest filter enabled" if toggle_target else "Ingest filter bypassed"
        self._send_json({"status": status})

    def log_message(self, format: str, *args: object) -> None:  # pragma: no cover - silence
        return

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
