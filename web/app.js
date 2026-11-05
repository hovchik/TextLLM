// Text LLM Studio front end (vanilla JS, no build step).
const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const main = $("#main");
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const num = (v, d = 0) => (v == null || isNaN(v) ? "—" : (+v).toLocaleString(undefined, { maximumFractionDigits: d, minimumFractionDigits: d }));
const dur = (s) => {
  if (s == null) return "—";
  s = Math.round(s);
  if (s < 60) return `${s}s`;
  if (s < 3600) return `${Math.floor(s / 60)}m ${s % 60}s`;
  return `${Math.floor(s / 3600)}h ${Math.floor((s % 3600) / 60)}m`;
};
const ago = (t) => {
  if (!t) return "";
  const s = Date.now() / 1000 - t;
  if (s < 60) return "just now";
  if (s < 3600) return `${Math.floor(s / 60)}m ago`;
  if (s < 86400) return `${Math.floor(s / 3600)}h ago`;
  return new Date(t * 1000).toLocaleDateString();
};
const I = {
  star: '<svg viewBox="0 0 24 24"><path d="m12 3 2.7 5.6 6.1.9-4.4 4.3 1 6.1L12 17l-5.4 2.9 1-6.1-4.4-4.3 6.1-.9z"/></svg>',
  copy: '<svg viewBox="0 0 24 24"><rect x="8" y="8" width="12" height="12" rx="2"/><path d="M16 8V5a1 1 0 0 0-1-1H5a1 1 0 0 0-1 1v10a1 1 0 0 0 1 1h3"/></svg>',
  redo: '<svg viewBox="0 0 24 24"><path d="M20 11a8 8 0 1 0-2.3 5.7M20 4v7h-7"/></svg>',
  edit: '<svg viewBox="0 0 24 24"><path d="M4 20h4L19 9l-4-4L4 16zM14 6l4 4"/></svg>',
  send: '<svg viewBox="0 0 24 24"><path d="M5 12h14M13 6l6 6-6 6"/></svg>',
  plus: '<svg viewBox="0 0 24 24"><path d="M12 5v14M5 12h14"/></svg>',
  play: '<svg viewBox="0 0 24 24"><path d="M7 4v16l13-8z"/></svg>',
  stop: '<svg viewBox="0 0 24 24"><rect x="6" y="6" width="12" height="12" rx="2"/></svg>',
  upload: '<svg viewBox="0 0 24 24"><path d="M12 16V4m-5 5 5-5 5 5M4 16v3a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-3"/></svg>',
  model: '<svg viewBox="0 0 24 24"><path d="M12 3 4 7.5v9L12 21l8-4.5v-9zM4 7.5l8 4.5 8-4.5M12 12v9"/></svg>',
  chat: '<svg viewBox="0 0 24 24"><path d="M4 5h16v11H8l-4 4z"/></svg>',
  spark: '<svg viewBox="0 0 24 24"><path d="M12 3v4m0 10v4M3 12h4m10 0h4M6 6l2.5 2.5m7 7L18 18M6 18l2.5-2.5m7-7L18 6"/></svg>',
  trend: '<svg viewBox="0 0 24 24"><path d="m3 17 6-6 4 4 8-8M15 7h6v6"/></svg>',
  clock: '<svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/></svg>',
  logo: '<svg viewBox="0 0 32 32"><path d="M9 11h14M9 16h10M9 21h12" stroke="currentColor" stroke-width="2.6" stroke-linecap="round"/></svg>',
  trash: '<svg viewBox="0 0 24 24"><path d="M4 7h16M9 7V4h6v3m-8 0 1 13h8l1-13"/></svg>',
  sliders: '<svg viewBox="0 0 24 24"><path d="M4 6h10m4 0h2M4 12h4m4 0h8M4 18h12m4 0h0"/><circle cx="16" cy="6" r="2"/><circle cx="10" cy="12" r="2"/><circle cx="18" cy="18" r="2"/></svg>',
};

async function api(path, opts = {}) {
  const o = { ...opts };
  if (o.json) { o.method = o.method || "POST"; o.headers = { "Content-Type": "application/json" }; o.body = JSON.stringify(o.json); }
  const r = await fetch(path, o);
  if (!r.ok) {
    let msg = r.statusText;
    try { msg = (await r.json()).detail || msg; } catch {}
    throw new Error(typeof msg === "string" ? msg : JSON.stringify(msg));
  }
  return r.json();
}
function toast(msg, kind = "") {
  const t = document.createElement("div");
  t.className = `toast ${kind}`;
  t.innerHTML = msg;
  $("#toasts").appendChild(t);
  setTimeout(() => t.remove(), 4500);
}
const fail = (e) => toast(esc(e.message || e), "err");
function modal(html, onMount) {
  const root = $("#modal-root");
  root.innerHTML = `<div class="modal-bg"><div class="modal">${html}</div></div>`;
  const close = () => (root.innerHTML = "");
  $(".modal-bg", root).addEventListener("mousedown", (e) => e.target.classList.contains("modal-bg") && close());
  onMount && onMount($(".modal", root), close);
  return close;
}
const levelBar = (lv) => {
  if (!lv) return '<span class="faint">—</span>';
  if (lv.n == null) return `<span class="pill accent">${esc(lv.label)}</span>`;
  return `<div class="row" style="gap:8px"><div class="level">${[1, 2, 3, 4, 5, 6].map((i) => `<i class="${i <= lv.n ? "on" : ""}"></i>`).join("")}</div><span class="muted" style="font-size:12.5px">${lv.n}/6 · ${esc(lv.label)}</span></div>`;
};
const starsText = (v) => (v == null ? "—" : `<span style="color:var(--star)">★</span> ${(+v).toFixed(1)}`);
const typePill = (t) => (t === "lora" ? '<span class="pill accent">Fine-tuned</span>' : '<span class="pill">From scratch</span>');

// ---------------------------------------------------------------- theme
(function theme() {
  const saved = (() => { try { return localStorage.getItem("theme"); } catch { return null; } })();
  if (saved) document.documentElement.dataset.theme = saved;
  else if (matchMedia("(prefers-color-scheme: light)").matches) document.documentElement.dataset.theme = "light";
  $("#theme-toggle").onclick = () => {
    const t = document.documentElement.dataset.theme === "light" ? "dark" : "light";
    document.documentElement.dataset.theme = t;
    try { localStorage.setItem("theme", t); } catch {}
    route();
  };
})();

// ---------------------------------------------------------------- status polling
const S = { status: null, page: null, onStatus: null };
async function poll() {
  try {
    S.status = await api("/api/status");
    const j = S.status.job;
    const pill = $("#jobpill");
    pill.classList.toggle("on", j.running);
    $(".jobtext", pill).textContent = j.running ? `${labelKind(j.kind)}: ${j.name}` : "Idle · no job running";
    const b = $("#nav-pending");
    b.hidden = !S.status.pending_feedback;
    b.textContent = S.status.pending_feedback;
    S.onStatus && S.onStatus(S.status);
  } catch {}
}
const labelKind = (k) => ({ training: "Training", data: "Preparing data", texts: "Learning texts", feedback: "Learning from ratings" }[k] || k);
setInterval(poll, 3000);
$("#jobpill").onclick = () => (location.hash = "#/training/live");

// ---------------------------------------------------------------- router
const pages = {};
function route() {
  const [page = "dashboard", sub] = location.hash.replace(/^#\/?/, "").split("/");
  S.onStatus = null;
  $("#modal-root").innerHTML = "";
  $$("#nav a").forEach((a) => a.classList.toggle("active", a.dataset.page === page));
  (pages[page] || pages.dashboard)(sub);
}
window.addEventListener("hashchange", route);

// ---------------------------------------------------------------- live job card (shared)
function liveCard(run) {
  const j = S.status?.job;
  if (!j?.running) return "";
  const r = run && run.name === j.run ? run : null;
  return `<div class="card live" id="live">
    <div class="row" style="gap:18px;align-items:center">
      ${Charts.ring(r ? r.progress : 0)}
      <div style="flex:1;min-width:0">
        <div class="row" style="gap:8px"><span class="pill good">● Running</span><b>${labelKind(j.kind)}</b><span class="muted">${esc(j.name)}</span></div>
        <div class="muted" style="margin-top:4px">${r ? `Step ${num(r.step)} of ${num(r.max_steps)} · ETA ${dur(r.eta_sec)} · loss ${num(r.train_loss, 3)}` : `Started ${ago(j.started)}`}</div>
      </div>
      <button class="btn danger sm" id="live-stop">${I.stop} Stop</button>
    </div></div>`;
}
function bindLive() {
  const b = $("#live-stop");
  if (b) b.onclick = async () => { b.disabled = true; await api("/api/stop", { method: "POST" }).catch(fail); toast("Stopped. The last checkpoint is kept."); poll(); route(); };
}

// ================================================================ DASHBOARD
pages.dashboard = async () => {
  main.innerHTML = `<div class="page"><div class="empty">Loading…</div></div>`;
  let a, models;
  try { [a, models] = await Promise.all([api("/api/analytics"), api("/api/models")]); } catch (e) { return fail(e); }
  await poll();
  const running = a.runs.find((r) => r.running);
  const best = models.filter((m) => m.level?.n).sort((x, y) => y.level.n - x.level.n)[0];
  const d = S.status.data;
  const setup = [
    { done: d.raw_files > 0, t: "Get text", s: d.raw_files ? `${d.raw_files} files · ${num(d.raw_mb)} MB` : "Download books or add your own", go: "#/training/data" },
    { done: d.tokenizer, t: "Build the tokenizer", s: d.tokenizer ? `${num(d.vocab)} tokens in the vocabulary` : "Teaches the model how to split words", go: "#/training/data" },
    { done: d.train_tokens > 0, t: "Encode the corpus", s: d.train_tokens ? `${num(d.train_tokens / 1e6, 1)}M training tokens` : "Turns the text into numbers", go: "#/training/data" },
    { done: models.length > 0, t: "Train your first model", s: models.length ? `${models.length} model(s) ready` : "From scratch or fine-tune an open model", go: "#/training/new" },
  ];
  main.innerHTML = `<div class="page">
    <div class="page-head"><div><h1>Welcome back</h1><p>Train writing models on books and your own texts, chat with them, and rate their answers so they keep improving.</p></div>
      <div class="row"><a class="btn" href="#/chat">${I.chat} Open chat</a><a class="btn primary" href="#/training/new">${I.plus} New model</a></div></div>
    ${liveCard(running)}
    <div class="grid g5">
      ${kpi("Models ready", models.length, I.model, `${a.runs.length} runs in total`)}
      ${kpi("Best level", best ? `${best.level.n}/6` : "—", I.trend, best ? esc(best.level.label) : "train a model to see")}
      ${kpi("Chats", num(a.counts.chats), I.chat, `${num(a.counts.answers)} answers written`)}
      ${kpi("Average rating", a.counts.avg_rating ? `${(+a.counts.avg_rating).toFixed(1)}★` : "—", I.star, `${num(a.counts.rated)} ratings`)}
      ${kpi("Waiting to learn", num(a.pending_feedback), I.spark, a.pending_feedback ? "ratings not trained on yet" : "all feedback learned")}
    </div>
    <div class="grid g3">
      <div class="card span2"><div class="card-head"><h2>Validation loss by model</h2><span class="hint">lower is better</span></div><div id="dash-curves"></div></div>
      <div class="card"><div class="card-head"><h2>Getting started</h2></div><div class="steps">
        ${setup.map((x, i) => `<a class="step ${x.done ? "done" : ""}" href="${x.go}"><div class="num">${x.done ? "✓" : i + 1}</div><div><b>${x.t}</b><div class="hint">${x.s}</div></div><span></span></a>`).join("")}
      </div></div>
    </div>
    <div><div class="card-head"><h2>Your models</h2><a class="btn ghost sm" href="#/training/runs">All runs →</a></div>
      <div class="grid g3">${models.length ? models.map(modelCard).join("") : `<div class="card empty span2"><div class="big">🧠</div>No trained models yet.<br><a class="btn primary" style="margin-top:14px" href="#/training/new">Train your first model</a></div>`}</div></div>
  </div>`;
  bindLive();
  Charts.line($("#dash-curves"), Object.entries(a.val_curves).map(([name, pts]) => ({ name, points: pts.map((p) => ({ x: p.step, y: p.loss })) })), { xLabel: "step", empty: "Train a model to see its learning curve" });
  $$("[data-chat]").forEach((b) => (b.onclick = () => { localStorage.setItem("chatModel", b.dataset.chat); location.hash = "#/chat"; }));
  $$("[data-run]").forEach((b) => (b.onclick = () => openRun(b.dataset.run)));
  S.onStatus = (s) => { if (!!s.job.running !== !!$("#live")) route(); };
};
const kpi = (l, v, ic, sub) => `<div class="card kpi"><div class="ic">${ic}</div><div class="l">${l}</div><div class="v">${v}</div><div class="sub">${sub || ""}</div></div>`;
const modelCard = (m) => `<div class="card model-card">
  <div class="row" style="justify-content:space-between"><div class="name">${esc(m.name)}</div>${typePill(m.type)}</div>
  <div>${levelBar(m.level)}</div>
  <div class="stats">
    <div class="stat"><div class="l">Val loss</div><div class="v">${num(m.val_loss, 2)}</div></div>
    <div class="stat"><div class="l">Rating</div><div class="v">${starsText(m.avg_rating)}</div></div>
  </div>
  <div class="row"><button class="btn sm primary" data-chat="${esc(m.name)}">${I.chat} Chat</button><button class="btn sm" data-run="${esc(m.name)}">Details</button>
    ${m.pending_feedback ? `<span class="pill warn" title="ratings not trained on yet">${m.pending_feedback} new ratings</span>` : ""}</div>
</div>`;

// ================================================================ TRAINING
const PRESETS = [
  { id: "tiny", n: "Tiny", p: "4M", h: "Testing only · minutes on CPU" },
  { id: "cpu", n: "CPU", p: "14M", h: "Laptop/desktop · ~1 day", tag: "CPU" },
  { id: "small", n: "Small", p: "110M", h: "1× 24 GB GPU · 2–4 days" },
  { id: "medium", n: "Medium", p: "340M", h: "1–2× A100 · ~1 week" },
  { id: "1b", n: "1B", p: "1.3B", h: "8× A100 · several days" },
];
pages.training = async (sub = "new") => {
  if (sub === "live") sub = "runs";
  main.innerHTML = `<div class="page">
    <div class="page-head"><div><h1>Training</h1><p>Prepare data, train new models, and teach existing models from your own files and links.</p></div>
      <div class="tabs" id="ttabs">${[["new", "New model"], ["data", "Data"], ["texts", "Your texts"], ["runs", "Runs"]].map(([k, l]) => `<button data-k="${k}" class="${k === sub ? "on" : ""}">${l}</button>`).join("")}</div></div>
    <div id="tlive"></div><div id="tbody"></div></div>`;
  $$("#ttabs button").forEach((b) => (b.onclick = () => (location.hash = `#/training/${b.dataset.k}`)));
  await poll();
  const runsList = await api("/api/runs").catch(() => []);
  const renderLive = (rs) => { $("#tlive").innerHTML = liveCard(rs.find((r) => r.running)); bindLive(); };
  renderLive(runsList);
  S.onStatus = async () => { const rs = await api("/api/runs").catch(() => []); renderLive(rs); if (sub === "runs") drawRuns(rs); };
  ({ new: trainNew, data: trainData, texts: trainTexts, runs: () => drawRuns(runsList) }[sub] || trainNew)();
};

function trainNew() {
  const st = { mode: "scratch", preset: "cpu" };
  const body = $("#tbody");
  body.innerHTML = `<div class="grid g3">
    <div class="card span2" style="display:flex;flex-direction:column;gap:20px">
      <div><h3>1 · Approach</h3><div class="choice g2" style="margin-top:10px" id="modes">
        <div class="opt on" data-v="scratch"><b>🧱 Train from scratch</b><span>Your own GPT, built from zero on books. You own every weight; quality depends on compute.</span></div>
        <div class="opt" data-v="lora"><span class="tag">Best quality</span><b>✨ Fine-tune an open model</b><span>Teach Qwen (or another open model) your style with LoRA. Follows requests like "write a post about…".</span></div>
      </div></div>
      <div id="sizebox"><h3>2 · Model size</h3><div class="choice g5" style="margin-top:10px" id="presets">
        ${PRESETS.map((p) => `<div class="opt ${p.id === "cpu" ? "on" : ""}" data-v="${p.id}">${p.tag ? `<span class="tag">${p.tag}</span>` : ""}<b>${p.n}</b><span>${p.p} params</span><span class="hint">${p.h}</span></div>`).join("")}
      </div></div>
      <div id="basebox" hidden><h3>2 · Base model</h3><div class="grid g2" style="margin-top:10px">
        <label class="field">Hugging Face model<input class="input" id="f-base" value="Qwen/Qwen2.5-0.5B-Instruct"><span class="hint">0.5B runs on CPU; try Qwen/Qwen2.5-1.5B-Instruct on a GPU.</span></label>
        <label class="field">Book examples<input class="input" id="f-examples" type="number" value="4000"><span class="hint">Passages taken from your corpus</span></label>
      </div></div>
      <div><h3>3 · Settings</h3><div class="grid g3" style="margin-top:10px">
        <label class="field">Model name<input class="input" id="f-name" value="my-writer"></label>
        <label class="field">Training steps<input class="input" id="f-steps" type="number" value="20000"></label>
        <label class="field">Learning rate<input class="input" id="f-lr" type="number" step="any" value="0.001"></label>
      </div>
      <details style="margin-top:14px"><summary>Advanced</summary><div class="grid g4" style="margin-top:12px">
        <label class="field">Batch size<input class="input" id="f-batch" type="number" value="16"></label>
        <label class="field">Grad accumulation<input class="input" id="f-accum" type="number" value="4"></label>
        <label class="field">Vocab size<input class="input" id="f-vocab" type="number" value="${S.status?.data?.vocab || 8192}"><span class="hint">must match tokenizer</span></label>
        <label class="field">Tokens / example<input class="input" id="f-seq" type="number" value="512"></label>
      </div><label class="row" style="margin-top:12px"><span class="switch"><input type="checkbox" id="f-resume"><span></span></span> Resume from the last checkpoint (from scratch only)</label></details></div>
      <div class="row end"><button class="btn primary" id="start">${I.play} Start training</button></div>
    </div>
    <div class="card" style="display:flex;flex-direction:column;gap:14px"><h2>What to expect</h2><div id="expect" class="muted"></div></div>
  </div>`;
  const expect = () => {
    $("#expect").innerHTML = st.mode === "lora"
      ? `<p>The model already writes well; fine-tuning shifts its <b>style</b> toward your books and texts.</p><p>On CPU each step takes roughly a minute, so 300 steps is a few hours. On a free Colab/Kaggle GPU it's minutes.</p><p>After training, chat with it and <b>rate answers</b>: every rating teaches it what you like.</p>`
      : `<p>A from-scratch model starts from <b>random letters</b> and climbs the 6 quality levels as it reads.</p>${levelBar({ n: 4, label: "Grammatical sentences" })}<p style="margin-top:12px">With the <b>CPU</b> size, expect book-like sentences that drift after a few lines, after about a day. Bigger sizes need a GPU.</p><p>It continues the text you type rather than following instructions.</p>`;
  };
  const setMode = (m) => {
    st.mode = m;
    $$("#modes .opt").forEach((o) => o.classList.toggle("on", o.dataset.v === m));
    $("#sizebox").hidden = m !== "scratch"; $("#basebox").hidden = m === "scratch";
    Object.entries(m === "scratch" ? { steps: 20000, lr: 0.001, batch: 16, accum: 4, name: "my-writer" } : { steps: 300, lr: 0.0002, batch: 2, accum: 8, name: "my-qwen-writer" })
      .forEach(([k, v]) => ($("#f-" + k).value = v));
    expect();
  };
  $$("#modes .opt").forEach((o) => (o.onclick = () => setMode(o.dataset.v)));
  $$("#presets .opt").forEach((o) => (o.onclick = () => { st.preset = o.dataset.v; $$("#presets .opt").forEach((x) => x.classList.toggle("on", x === o)); }));
  expect();
  $("#start").onclick = async () => {
    const v = (k) => $("#f-" + k).value;
    try {
      await api("/api/train", { json: { mode: st.mode, name: v("name"), resume: $("#f-resume").checked, preset: st.preset, vocab: +v("vocab"), steps: +v("steps"), batch: +v("batch"), accum: +v("accum"), lr: +v("lr"), base: v("base"), examples: +v("examples"), seq: +v("seq") } });
      toast(`Training <b>${esc(v("name"))}</b> started`, "ok");
      location.hash = "#/training/runs";
      setTimeout(() => openRun(v("name")), 400);
    } catch (e) { fail(e); }
  };
}

function trainData() {
  const d = S.status.data;
  $("#tbody").innerHTML = `<div class="grid g3"><div class="card span2"><div class="card-head"><h2>Prepare the training corpus</h2><span class="hint">Run the steps in order</span></div>
    <div class="steps">
      <div class="step ${d.raw_files ? "done" : ""}"><div class="num">${d.raw_files ? "✓" : 1}</div><div><b>Download books and web text</b><div class="hint">${d.raw_files ? `${d.raw_files} files, ${num(d.raw_mb)} MB (${d.custom_files} of your own)` : "Public-domain books from Project Gutenberg plus openly licensed FineWeb-Edu articles"}</div>
        <div class="row" style="margin-top:8px"><label class="field" style="width:140px">Books<input class="input" id="d-books" type="number" value="3000"></label><label class="field" style="width:160px">Web text (GB)<input class="input" id="d-web" type="number" step="0.1" value="0.3"></label></div></div>
        <button class="btn" data-step="download">${I.play} Run</button></div>
      <div class="step ${d.tokenizer ? "done" : ""}"><div class="num">${d.tokenizer ? "✓" : 2}</div><div><b>Build the tokenizer</b><div class="hint">${d.tokenizer ? `${num(d.vocab)}-token vocabulary` : "Learns the word pieces the model reads and writes"}</div>
        <div class="row" style="margin-top:8px"><label class="field" style="width:200px">Vocabulary<select class="input" id="d-vocab"><option value="8192">8,192 (CPU)</option><option value="32000">32,000 (GPU)</option></select></label></div></div>
        <button class="btn" data-step="tokenizer">${I.play} Run</button></div>
      <div class="step ${d.train_tokens ? "done" : ""}"><div class="num">${d.train_tokens ? "✓" : 3}</div><div><b>Encode the corpus</b><div class="hint">${d.train_tokens ? `${num(d.train_tokens / 1e6, 1)}M tokens ready to train` : "Converts every text into token files for fast training"}</div></div>
        <button class="btn" data-step="encode">${I.play} Run</button></div>
    </div></div>
    <div class="card"><h2>Tips</h2><ul class="muted" style="padding-left:18px">
      <li>Only <b>from-scratch</b> training needs steps 2 and 3. Fine-tuning reads the downloaded text directly.</li>
      <li>Want your own books in the corpus? Add them under <a href="#/training/texts"><u>Your texts</u></a> → "Add to training data", then re-run steps 2–3.</li>
      <li>Progress and logs show at the top of this page while a step runs.</li></ul></div></div>
    <div class="card" style="margin-top:16px"><div class="card-head"><h2>Log</h2></div><pre class="console" id="dlog">${esc(S.status.job.kind === "data" ? S.status.job.log : "No data job yet.")}</pre></div>`;
  $$("[data-step]").forEach((b) => (b.onclick = async () => {
    try {
      await api("/api/data", { json: { step: b.dataset.step, books: +$("#d-books").value, web_gb: +$("#d-web").value, vocab: +$("#d-vocab").value } });
      toast("Started", "ok"); poll();
    } catch (e) { fail(e); }
  }));
  const prev = S.onStatus;
  S.onStatus = (s) => { prev && prev(s); const l = $("#dlog"); if (l && s.job.kind === "data") { l.textContent = s.job.log; l.scrollTop = 1e9; } };
}

async function trainTexts() {
  const models = await api("/api/models").catch(() => []);
  const files = [];
  $("#tbody").innerHTML = `<div class="grid g3"><div class="card span2" style="display:flex;flex-direction:column;gap:18px">
    <div><h2>Learn from your own texts</h2><p class="muted" style="margin:4px 0 0">PDF, Word (.doc/.docx), TXT, Markdown or HTML files, links to books and articles, or a whole folder.</p></div>
    <div class="drop" id="drop">${I.upload}<div><b>Drop files here</b> or click to browse</div><div class="hint">.pdf · .docx · .doc · .txt · .md · .html</div><input type="file" id="fileinput" multiple accept=".pdf,.doc,.docx,.txt,.md,.markdown,.html,.htm" hidden></div>
    <div class="files" id="filelist"></div>
    <div class="grid g2">
      <label class="field">Links, one per line<textarea class="input" id="x-urls" placeholder="https://www.gutenberg.org/cache/epub/1342/pg1342.txt&#10;https://example.com/article"></textarea></label>
      <label class="field">Files or folders on this computer<textarea class="input" id="x-paths" placeholder="C:\\Books&#10;/home/me/articles"></textarea>
        <span class="row hint"><span class="switch"><input type="checkbox" id="x-del"><span></span></span> Delete these local files after reading</span></label>
    </div>
    <div><h3>What should happen?</h3><div class="choice g2" style="margin-top:10px" id="xaction">
      <div class="opt on" data-v="learn"><b>Teach a model now</b><span>Download → train the model → delete the texts. The model keeps what it learned.</span></div>
      <div class="opt" data-v="add"><b>Add to training data</b><span>Keep them in the corpus for the next full training.</span></div></div></div>
    <div class="grid g2" id="xlearn">
      <label class="field">Model to teach<select class="input" id="x-model">${models.map((m) => `<option value="${esc(m.name)}">${esc(m.label)}</option>`).join("") || "<option value=''>No trained models yet</option>"}</select></label>
      <label class="field"><span>Read each text <b id="x-pv">3</b> times</span><input type="range" id="x-passes" min="1" max="10" value="3"></label>
    </div>
    <div class="row end"><button class="btn primary" id="x-start">${I.play} Start</button></div>
  </div>
  <div class="card"><h2>Good to know</h2><ul class="muted" style="padding-left:18px">
    <li>Scanned PDFs have no text layer; run OCR first (e.g. <span class="mono">ocrmypdf</span>).</li>
    <li>Old <b>.doc</b> files need LibreOffice installed.</li>
    <li>From-scratch models mix in old data while learning so they don't forget the rest.</li>
    <li>Use texts you have the right to use.</li></ul></div></div>`;
  let action = "learn";
  const drawFiles = () => {
    $("#filelist").innerHTML = files.map((f, i) => `<span class="file">📄 ${esc(f.name)} <span class="faint">${num(f.size / 1024)} KB</span><button data-i="${i}">✕</button></span>`).join("");
    $$("#filelist button").forEach((b) => (b.onclick = () => { files.splice(+b.dataset.i, 1); drawFiles(); }));
  };
  const drop = $("#drop"), input = $("#fileinput");
  drop.onclick = () => input.click();
  input.onchange = () => { files.push(...input.files); input.value = ""; drawFiles(); };
  drop.ondragover = (e) => { e.preventDefault(); drop.classList.add("over"); };
  drop.ondragleave = () => drop.classList.remove("over");
  drop.ondrop = (e) => { e.preventDefault(); drop.classList.remove("over"); files.push(...e.dataTransfer.files); drawFiles(); };
  $$("#xaction .opt").forEach((o) => (o.onclick = () => { action = o.dataset.v; $$("#xaction .opt").forEach((x) => x.classList.toggle("on", x === o)); $("#xlearn").hidden = action !== "learn"; }));
  $("#x-passes").oninput = (e) => ($("#x-pv").textContent = e.target.value);
  $("#x-start").onclick = async () => {
    const fd = new FormData();
    files.forEach((f) => fd.append("files", f));
    fd.append("urls", $("#x-urls").value); fd.append("paths", $("#x-paths").value);
    fd.append("action", action); fd.append("model", $("#x-model").value);
    fd.append("passes", $("#x-passes").value); fd.append("delete_local", $("#x-del").checked);
    try {
      await api("/api/texts", { method: "POST", body: fd });
      toast(action === "learn" ? "Reading your texts and training…" : "Adding texts to the corpus…", "ok");
      files.length = 0; drawFiles(); poll();
      if (action === "learn") setTimeout(() => openRun($("#x-model").value), 500);
    } catch (e) { fail(e); }
  };
}

function drawRuns(rs) {
  const body = $("#tbody");
  if (!body) return;
  if (!rs.length) { body.innerHTML = `<div class="card empty"><div class="big">🏁</div>No runs yet.<br><a class="btn primary" style="margin-top:14px" href="#/training/new">Start a training</a></div>`; return; }
  body.innerHTML = `<div class="card"><table><thead><tr><th>Model</th><th>Type</th><th>Progress</th><th>Level</th><th>Val loss</th><th>Updated</th></tr></thead><tbody>
    ${rs.map((r) => `<tr class="click" data-run="${esc(r.name)}"><td><b>${esc(r.name)}</b> ${r.running ? '<span class="pill good">● running</span>' : r.ready ? "" : '<span class="pill">no weights yet</span>'}</td>
      <td>${typePill(r.type)}</td><td style="min-width:160px"><div class="bar"><i style="width:${(r.progress * 100).toFixed(1)}%"></i></div><div class="hint">${num(r.step)} / ${num(r.max_steps)}${r.eta_sec ? " · ETA " + dur(r.eta_sec) : ""}</div></td>
      <td>${levelBar(r.level)}</td><td>${num(r.val_loss, 3)}</td><td class="muted">${ago(r.updated)}</td></tr>`).join("")}</tbody></table></div>`;
  $$("tr[data-run]").forEach((t) => (t.onclick = () => openRun(t.dataset.run)));
}

async function openRun(name) {
  let timer;
  const close = modal(`<div class="row" style="justify-content:space-between"><div><h1 style="font-size:22px" id="r-title"></h1><div class="muted" id="r-sub"></div></div><div class="row" id="r-btns"></div></div>
    <div class="stats" id="r-stats"></div><div class="card"><div class="card-head"><h2>Loss</h2><span class="hint">training · validation · learning from ratings</span></div><div id="r-chart"></div></div>
    <div><div class="card-head"><h2>Log</h2></div><pre class="console" id="r-log"></pre></div>`, (m) => {
    const load = async () => {
      if (!document.body.contains(m)) return clearInterval(timer);
      let d;
      try { d = await api(`/api/runs/${encodeURIComponent(name)}`); } catch (e) { clearInterval(timer); return fail(e); }
      const s = d.summary;
      $("#r-title", m).textContent = s.name;
      $("#r-sub", m).innerHTML = `${typePill(s.type)} ${s.base ? esc(s.base) : s.preset ? esc(s.preset) + " size" : ""} ${s.running ? '<span class="pill good">● training</span>' : ""}`;
      $("#r-btns", m).innerHTML = `${s.ready ? `<button class="btn sm primary" id="r-chat">${I.chat} Chat</button>` : ""}${s.type === "scratch" && !s.running ? `<button class="btn sm" id="r-resume">${I.play} Resume</button>` : ""}${s.running ? `<button class="btn sm danger" id="r-stop">${I.stop} Stop</button>` : `<button class="btn sm ghost" id="r-del">${I.trash}</button>`}`;
      $("#r-stats", m).innerHTML = [
        ["Progress", `${(s.progress * 100).toFixed(0)}%`], ["Step", `${num(s.step)} / ${num(s.max_steps)}`], ["ETA", s.running ? dur(s.eta_sec) : "—"],
        ["Train loss", num(s.train_loss, 3)], ["Val loss", num(s.val_loss, 3)], ["Perplexity", num(s.perplexity, 1)],
        ["Speed", s.tokens_per_sec ? num(s.tokens_per_sec) + " tok/s" : "—"], ["Trained for", dur(s.train_seconds)],
      ].map(([l, v]) => `<div class="stat"><div class="l">${l}</div><div class="v">${v}</div></div>`).join("") + `<div class="stat" style="grid-column:span 2"><div class="l">Level</div><div style="margin-top:6px">${levelBar(s.level)}</div></div>`;
      const kinds = [["train", "Training", Charts.PALETTE[0]], ["val", "Validation", Charts.PALETTE[1]], ["feedback", "From ratings", Charts.PALETTE[2]]];
      Charts.line($("#r-chart", m), kinds.map(([k, n, c]) => ({ name: n, color: c, area: k === "train", points: d.metrics.filter((x) => x.kind === k).map((x) => ({ x: x.step, y: x.loss })) })).filter((x) => x.points.length), { xLabel: "step", empty: "Waiting for the first steps…" });
      const log = $("#r-log", m); const atEnd = log.scrollTop + log.clientHeight >= log.scrollHeight - 30;
      log.textContent = d.log || "No log yet."; if (atEnd) log.scrollTop = 1e9;
      const on = (id, f) => { const b = $(id, m); if (b) b.onclick = f; };
      on("#r-chat", () => { close(); localStorage.setItem("chatModel", s.name); location.hash = "#/chat"; });
      on("#r-stop", async () => { await api("/api/stop", { method: "POST" }).catch(fail); toast("Stopped. The last checkpoint is kept."); load(); });
      on("#r-resume", async () => {
        try { await api("/api/train", { json: { mode: "scratch", name: s.name, resume: true, preset: s.preset || "cpu", steps: s.max_steps, vocab: S.status?.data?.vocab || 8192 } }); toast("Resumed", "ok"); load(); } catch (e) { fail(e); }
      });
      on("#r-del", async () => { if (!confirm(`Delete ${s.name} and its weights?`)) return; await api(`/api/runs/${encodeURIComponent(s.name)}`, { method: "DELETE" }).catch(fail); close(); route(); });
    };
    load();
    timer = setInterval(load, 4000);
  });
}

// ================================================================ CHAT
const SUGGEST = [
  ["Write a short story", "about a lighthouse keeper who finds a message in a bottle"],
  ["Write a LinkedIn post", "announcing that I trained my own language model"],
  ["Describe a place", "a rainy harbour town at dawn, in the style of a 19th-century novel"],
  ["Continue this opening", "\"It was the last train of the night, and only two passengers remained.\""],
];
const C = { conv: null, models: [], busy: false, temp: 0.8, max: 300 };

pages.chat = async () => {
  main.innerHTML = `<div class="chat">
    <div class="convs"><div class="convs-head"><button class="btn primary" style="width:100%" id="newchat">${I.plus} New chat</button></div><div class="convs-list" id="convs"></div></div>
    <div class="thread">
      <div class="thread-head"><select class="input" id="model"></select><span id="model-info" class="row" style="gap:6px"></span><span class="spacer"></span>
        <div style="position:relative"><button class="iconbtn" id="settings-btn" title="Settings">${I.sliders}</button><div class="popover" id="settings" hidden>
          <label class="field"><span>Creativity <b id="tv">${C.temp}</b></span><input type="range" id="temp" min="0" max="1.5" step="0.05" value="${C.temp}"><span class="hint">Low = focused and predictable · high = surprising</span></label>
          <label class="field"><span>Max length <b id="mv">${C.max}</b> tokens</span><input type="range" id="max" min="32" max="1024" step="16" value="${C.max}"></label></div></div></div>
      <div class="msgs" id="msgs"><div class="msgs-inner" id="inner"></div></div>
      <div class="composer"><div class="composer-box"><textarea id="input" rows="1" placeholder="Ask for a story, a post, an article…"></textarea><button class="send" id="send" title="Send">${I.send}</button></div>
        <div class="note">Rate answers with ★ — each rating teaches the model what you like.</div></div>
    </div></div>`;
  C.models = await api("/api/models").catch(() => []);
  const pref = (() => { try { return localStorage.getItem("chatModel"); } catch { return null; } })();
  $("#model").innerHTML = C.models.length ? C.models.map((m) => `<option value="${esc(m.name)}" ${m.name === pref ? "selected" : ""}>${esc(m.name)} · ${m.type === "lora" ? "fine-tuned" : "from scratch"}</option>`).join("") : `<option value="">No trained models yet</option>`;
  const modelInfo = () => {
    const m = C.models.find((x) => x.name === $("#model").value);
    $("#model-info").innerHTML = m ? `${m.type === "lora" ? "" : levelBar(m.level)} <span class="pill">${starsText(m.avg_rating)} · ${m.ratings} ratings</span>` : "";
  };
  $("#model").onchange = () => { try { localStorage.setItem("chatModel", $("#model").value); } catch {} modelInfo(); };
  modelInfo();
  $("#settings-btn").onclick = (e) => { e.stopPropagation(); $("#settings").hidden = !$("#settings").hidden; };
  document.addEventListener("click", (e) => { const p = $("#settings"); if (p && !p.contains(e.target)) p.hidden = true; }, { once: false });
  $("#temp").oninput = (e) => ($("#tv").textContent = C.temp = +e.target.value);
  $("#max").oninput = (e) => ($("#mv").textContent = C.max = +e.target.value);
  const input = $("#input");
  input.oninput = () => { input.style.height = "auto"; input.style.height = Math.min(input.scrollHeight, 200) + "px"; };
  input.onkeydown = (e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); } };
  $("#send").onclick = send;
  $("#newchat").onclick = () => { C.conv = null; drawConvs(); drawThread([]); input.focus(); };
  await drawConvs();
  if (C.conv) openConv(C.conv); else drawThread([]);
};

async function drawConvs() {
  const list = await api("/api/conversations").catch(() => []);
  const el = $("#convs");
  if (!el) return;
  el.innerHTML = list.length ? list.map((c) => `<div class="conv ${c.id === C.conv ? "on" : ""}" data-id="${c.id}"><span class="t">${esc(c.title)}</span><button class="x" data-del="${c.id}" title="Delete">✕</button></div>`).join("") : `<div class="hint" style="padding:10px">Your chats will appear here.</div>`;
  $$(".conv", el).forEach((c) => (c.onclick = (e) => { if (e.target.dataset.del) return; openConv(+c.dataset.id); }));
  $$("[data-del]", el).forEach((b) => (b.onclick = async () => { await api(`/api/conversations/${b.dataset.del}`, { method: "DELETE" }); if (C.conv === +b.dataset.del) { C.conv = null; drawThread([]); } drawConvs(); }));
}
async function openConv(id) {
  C.conv = id;
  const c = await api(`/api/conversations/${id}`).catch(fail);
  if (!c) return;
  $$(".conv").forEach((x) => x.classList.toggle("on", +x.dataset.id === id));
  drawThread(c.messages);
}
function drawThread(messages) {
  const inner = $("#inner");
  if (!messages.length) {
    inner.innerHTML = `<div class="welcome"><div class="logo" style="width:54px;height:54px;margin:0 auto 18px;border-radius:16px">${I.logo}</div><h1>What should we write today?</h1>
      <p class="muted">${C.models.length ? "Pick a model above and ask for a story, a post or any text." : 'No trained model yet. <a href="#/training/new"><u>Train one first</u></a>.'}</p>
      <div class="chips">${SUGGEST.map(([t, s]) => `<div class="chip" data-p="${esc(t + " " + s)}"><b>${t}</b><span>${esc(s)}</span></div>`).join("")}</div></div>`;
    $$(".chip").forEach((c) => (c.onclick = () => { $("#input").value = c.dataset.p; send(); }));
    return;
  }
  const answers = {};
  messages.filter((m) => m.role === "assistant").forEach((m) => (answers[m.prompt_id] = answers[m.prompt_id] || []).push(m));
  inner.innerHTML = messages.filter((m) => m.role === "user").map((u) => userHtml(u) + (answers[u.id] || []).map((a, i, arr) => answerHtml(a, arr.length > 1 ? `Answer ${i + 1} of ${arr.length}` : "")).join("")).join("");
  bindAnswers();
  $("#msgs").scrollTop = 1e9;
}
const userHtml = (u) => `<div class="msg user" data-uid="${u.id}"><div class="bubble">${esc(u.content)}</div></div>`;
const fmtBody = (t) => esc(t).split(/\n{2,}/).map((p) => `<p>${p}</p>`).join("");
function answerHtml(a, alt = "") {
  return `<div class="msg" data-mid="${a.id}" data-uid="${a.prompt_id}"><div class="avatar">${I.logo}</div><div class="answer">
    ${alt ? `<div class="alt">${alt}</div>` : ""}<div class="body">${fmtBody(a.content)}</div>
    ${a.correction ? `<div class="correction"><b>Your better answer · the model learns from this</b>${esc(a.correction)}</div>` : ""}
    <div class="actions"><span class="stars" title="Rate this answer">${[1, 2, 3, 4, 5].map((n) => `<button data-r="${n}" class="${a.rating >= n ? "on" : ""}" title="${["Bad", "Weak", "OK", "Good", "Great"][n - 1]}">${I.star}</button>`).join("")}</span>
      <button class="act" data-act="improve">${I.edit} Improve</button><button class="act" data-act="regen">${I.redo} Try again</button><button class="act" data-act="copy">${I.copy} Copy</button>
      <span class="hint" data-status>${a.trained_in ? "✓ learned" : a.rating ? "Rated · will be learned" : ""}</span></div>
  </div></div>`;
}
function bindAnswers() {
  $$(".msg[data-mid]").forEach((el) => {
    if (el.dataset.bound) return;
    el.dataset.bound = 1;
    const mid = +el.dataset.mid;
    $$(".stars button", el).forEach((b) => (b.onclick = async () => {
      const r = +b.dataset.r;
      $$(".stars button", el).forEach((x) => x.classList.toggle("on", +x.dataset.r <= r));
      try {
        const res = await api(`/api/messages/${mid}/rate`, { json: { rating: r } });
        $("[data-status]", el).textContent = "Rated · will be learned";
        const th = S.status?.settings?.auto_threshold;
        if (res.auto_training) toast("🎓 Enough new ratings: the model is now <b>learning from your feedback</b>.", "ok");
        else toast(r >= 4 ? "Thanks! The model will learn to write more like this." : r <= 2 ? "Got it. Tip: click <b>Improve</b> to show a better answer." : "Thanks for rating.", "ok");
        if (r <= 2 && !$(".improve", el)) openImprove(el, mid);
        poll();
      } catch (e) { fail(e); }
    }));
    $$(".act", el).forEach((b) => (b.onclick = () => {
      const a = b.dataset.act;
      if (a === "copy") { navigator.clipboard?.writeText($(".body", el).innerText); toast("Copied"); }
      if (a === "improve") openImprove(el, mid);
      if (a === "regen") send(+el.dataset.uid);
    }));
  });
}
function openImprove(el, mid) {
  if ($(".improve", el)) return;
  const box = document.createElement("div");
  box.className = "improve";
  box.innerHTML = `<b style="font-size:13px">Show the model a better answer</b><textarea class="input" placeholder="Write or paste the answer you wanted…">${esc($(".body", el).innerText)}</textarea>
    <input class="input" placeholder="Optional: what was wrong? (too long, off-topic, wrong tone…)"><div class="row end"><button class="btn sm ghost">Cancel</button><button class="btn sm primary">Save & teach</button></div>`;
  $(".actions", el).after(box);
  const [ta, cm] = [$("textarea", box), $("input", box)];
  ta.focus();
  const [cancel, save] = $$("button", box);
  cancel.onclick = () => box.remove();
  save.onclick = async () => {
    try {
      const res = await api(`/api/messages/${mid}/rate`, { json: { correction: ta.value.trim(), comment: cm.value.trim() || null } });
      box.outerHTML = `<div class="correction"><b>Your better answer · the model learns from this</b>${esc(ta.value.trim())}</div>`;
      toast(res.auto_training ? "🎓 The model is now learning from your feedback." : "Saved. The model will learn from your answer.", "ok");
      poll();
    } catch (e) { fail(e); }
  };
}
async function send(regenUid) {
  if (C.busy) return;
  const model = $("#model").value;
  if (!model) return toast("Train a model first.", "err");
  const input = $("#input");
  const text = input.value.trim();
  if (!regenUid && !text) return;
  C.busy = true; $("#send").disabled = true;
  const inner = $("#inner");
  if (!regenUid) {
    if (!C.conv) inner.innerHTML = "";
    inner.insertAdjacentHTML("beforeend", `<div class="msg user"><div class="bubble">${esc(text)}</div></div>`);
    input.value = ""; input.style.height = "auto";
  }
  const holder = document.createElement("div");
  holder.className = "msg";
  holder.innerHTML = `<div class="avatar">${I.logo}</div><div class="answer"><div class="body cursor"></div></div>`;
  if (regenUid) {
    const sibs = $$(`.msg[data-uid="${regenUid}"]`);
    sibs[sibs.length - 1].after(holder);
  } else inner.appendChild(holder);
  const body = $(".body", holder);
  $("#msgs").scrollTop = 1e9;
  let full = "", done = null;
  try {
    const r = await fetch("/api/chat", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ model, message: text, conversation_id: C.conv, regenerate: regenUid || null, temperature: C.temp, max_tokens: C.max }) });
    if (!r.ok) throw new Error((await r.json()).detail || r.statusText);
    const rd = r.body.getReader(), dec = new TextDecoder();
    let buf = "";
    for (;;) {
      const { value, done: end } = await rd.read();
      if (end) break;
      buf += dec.decode(value, { stream: true });
      let i;
      while ((i = buf.indexOf("\n")) >= 0) {
        const ev = JSON.parse(buf.slice(0, i)); buf = buf.slice(i + 1);
        if (ev.type === "start") { if (!C.conv) { C.conv = ev.conversation_id; drawConvs(); } }
        if (ev.type === "delta") { full += ev.text; body.innerHTML = fmtBody(full); const m = $("#msgs"); if (m.scrollHeight - m.scrollTop - m.clientHeight < 120) m.scrollTop = 1e9; }
        if (ev.type === "error") toast(esc(ev.error), "err");
        if (ev.type === "done") done = ev;
      }
    }
  } catch (e) { fail(e); body.classList.remove("cursor"); body.textContent = "⚠️ " + e.message; }
  C.busy = false; $("#send").disabled = false;
  if (done) await openConv(C.conv);
}

// ================================================================ ANALYTICS
pages.analytics = async () => {
  main.innerHTML = `<div class="page"><div class="empty">Loading…</div></div>`;
  let a;
  try { a = await api("/api/analytics"); } catch (e) { return fail(e); }
  const dist = [1, 2, 3, 4, 5].map((n) => ({ label: "★".repeat(n), value: a.rating_distribution[n] || 0, color: ["#f87171", "#fb923c", "#fbbf24", "#a3e635", "#34d399"][n - 1] }));
  const liked = (a.rating_distribution[4] || 0) + (a.rating_distribution[5] || 0);
  main.innerHTML = `<div class="page">
    <div class="page-head"><div><h1>Analytics</h1><p>How your models learn and how people rate what they write.</p></div></div>
    <div class="grid g5">
      ${kpi("Training time", `${num(a.train_hours, 1)}h`, I.clock, `${a.runs.length} runs`)}
      ${kpi("Best val loss", num(Math.min(...a.runs.map((r) => r.best_val ?? Infinity).filter(isFinite)), 3).replace("∞", "—"), I.trend, "lower is better")}
      ${kpi("Answers written", num(a.counts.answers), I.chat, `${num(a.counts.chats)} chats`)}
      ${kpi("Liked answers", a.counts.rated ? `${Math.round((liked / a.counts.rated) * 100)}%` : "—", I.star, "rated 4–5 stars")}
      ${kpi("Average rating", a.counts.avg_rating ? `${(+a.counts.avg_rating).toFixed(2)}★` : "—", I.spark, `${num(a.counts.rated)} ratings`)}
    </div>
    <div class="grid g2">
      <div class="card"><div class="card-head"><h2>Validation loss</h2><span class="hint">all models · lower is better</span></div><div id="a-curves"></div></div>
      <div class="card"><div class="card-head"><h2>Ratings per day</h2><span class="hint">bar = count · hover for average</span></div><div id="a-daily"></div></div>
      <div class="card"><div class="card-head"><h2>Rating distribution</h2></div><div id="a-dist"></div></div>
      <div class="card"><div class="card-head"><h2>Did feedback training help?</h2><span class="hint">average rating per training round</span></div><div id="a-trend"></div></div>
    </div>
    <div class="card"><div class="card-head"><h2>Model leaderboard</h2></div><table><thead><tr><th>Model</th><th>Type</th><th>Level</th><th>Val loss</th><th>Answers</th><th>Avg rating</th><th>👍 / 👎</th><th>Corrections</th></tr></thead><tbody>
      ${a.runs.map((r) => { const p = a.per_model.find((x) => x.model === r.name) || {}; return `<tr class="click" data-run="${esc(r.name)}"><td><b>${esc(r.name)}</b></td><td>${typePill(r.type)}</td><td>${levelBar(r.level)}</td><td>${num(r.best_val, 3)}</td><td>${num(p.answers || 0)}</td><td>${starsText(p.avg)}</td><td><span style="color:var(--good)">${p.liked || 0}</span> / <span style="color:var(--bad)">${p.disliked || 0}</span></td><td>${p.corrected || 0}</td></tr>`; }).join("") || `<tr><td colspan="8" class="empty">No models yet</td></tr>`}
    </tbody></table></div>
    <div class="card"><div class="card-head"><h2>Recent jobs</h2></div><table><thead><tr><th>Job</th><th>Target</th><th>Started</th><th>Duration</th><th>Result</th></tr></thead><tbody>
      ${a.jobs.slice().reverse().map((j) => `<tr><td>${labelKind(j.kind)}</td><td>${esc(j.name)}</td><td class="muted">${ago(j.started)}</td><td>${dur(j.ended - j.started)}</td><td>${j.exit === 0 ? '<span class="pill good">done</span>' : j.exit < 0 || j.exit === 130 ? '<span class="pill warn">stopped</span>' : '<span class="pill bad">failed</span>'}</td></tr>`).join("") || `<tr><td colspan="5" class="empty">No jobs yet</td></tr>`}
    </tbody></table></div></div>`;
  Charts.line($("#a-curves"), Object.entries(a.val_curves).map(([name, pts]) => ({ name, points: pts.map((p) => ({ x: p.step, y: p.loss })) })), { xLabel: "step" });
  Charts.bars($("#a-daily"), a.daily.map((d) => ({ label: d.day.slice(5), value: d.n, note: `avg ${(+d.avg).toFixed(1)}★` })), { empty: "Rate answers in Chat to see this" });
  Charts.bars($("#a-dist"), dist, { empty: "No ratings yet" });
  const rounds = a.trend.map((t) => ({ label: `${t.model}${t.trained_in ? " · " + t.trained_in.split("@")[1].slice(5) : " · new"}`, value: +(+t.avg).toFixed(2), color: t.trained_in ? Charts.PALETTE[0] : Charts.PALETTE[1], note: `${t.n} ratings` }));
  Charts.bars($("#a-trend"), rounds, { max: 5, fmt: (v) => v.toFixed(1) + "★", empty: "Shows up after the first feedback training" });
  $$("tr[data-run]").forEach((t) => (t.onclick = () => openRun(t.dataset.run)));
};

// ================================================================ FEEDBACK
pages.feedback = async () => {
  main.innerHTML = `<div class="page"><div class="empty">Loading…</div></div>`;
  await poll();
  let models, list;
  try { [models, list] = await Promise.all([api("/api/models"), api("/api/feedback")]); } catch (e) { return fail(e); }
  const st = S.status.settings;
  main.innerHTML = `<div class="page">
    <div class="page-head"><div><h1>Feedback</h1><p>Every rating and correction becomes training data. Liked answers and your corrections teach the model what to write; disliked answers teach it what to avoid.</p></div></div>
    <div class="grid g3">
      <div class="card span2"><div class="card-head"><h2>Learn from ratings</h2></div>
        <table><thead><tr><th>Model</th><th>New ratings</th><th>Avg rating</th><th></th></tr></thead><tbody>
        ${models.map((m) => `<tr><td><b>${esc(m.name)}</b> ${typePill(m.type)}</td><td>${m.pending_feedback ? `<span class="pill warn">${m.pending_feedback} waiting</span>` : '<span class="pill good">up to date</span>'}</td><td>${starsText(m.avg_rating)}</td>
          <td style="text-align:right"><button class="btn sm primary" data-train="${esc(m.name)}" ${m.pending_feedback ? "" : "disabled"}>${I.spark} Train now</button></td></tr>`).join("") || `<tr><td colspan="4" class="empty">No trained models yet</td></tr>`}
        </tbody></table></div>
      <div class="card" style="display:flex;flex-direction:column;gap:14px"><h2>Automatic learning</h2>
        <label class="row"><span class="switch"><input type="checkbox" id="auto" ${st.auto_train ? "checked" : ""}><span></span></span> Train automatically</label>
        <label class="field">After this many new ratings<input class="input" type="number" id="thr" min="1" value="${st.auto_threshold}"></label>
        <p class="hint" style="margin:0">Starts when the app is idle. Fine-tuned models use supervised + preference (DPO) training; from-scratch models re-read liked answers.</p></div>
    </div>
    <div class="card"><div class="card-head"><h2>All rated answers</h2><div class="tabs" id="ff"><button class="on" data-f="all">All</button><button data-f="liked">Liked</button><button data-f="disliked">Disliked</button><button data-f="corrected">Corrected</button><button data-f="pending">Not learned yet</button></div></div>
      <div id="fb-table"></div></div></div>`;
  const draw = (f) => {
    const rows = list.filter((r) => f === "all" || (f === "liked" && r.rating >= 4) || (f === "disliked" && r.rating <= 2) || (f === "corrected" && r.correction) || (f === "pending" && !r.trained_in));
    $("#fb-table").innerHTML = rows.length ? `<table><thead><tr><th style="width:24%">Prompt</th><th>Answer</th><th>Rating</th><th>Model</th><th>Status</th></tr></thead><tbody>
      ${rows.map((r) => `<tr class="click" data-conv="${r.conversation_id}"><td><div class="clamp">${esc(r.prompt)}</div></td><td><div class="clamp">${esc(r.content)}</div>${r.correction ? `<div class="hint" style="color:var(--good)">✎ corrected</div>` : ""}${r.comment ? `<div class="hint">“${esc(r.comment)}”</div>` : ""}</td>
        <td style="white-space:nowrap;color:var(--star)">${r.rating ? "★".repeat(r.rating) + '<span class="faint">' + "★".repeat(5 - r.rating) + "</span>" : "—"}</td><td>${esc(r.model)}</td>
        <td>${r.trained_in ? `<span class="pill good" title="${esc(r.trained_in)}">learned</span>` : '<span class="pill warn">waiting</span>'}</td></tr>`).join("")}</tbody></table>`
      : `<div class="empty"><div class="big">⭐</div>Nothing here yet. Rate answers in <a href="#/chat"><u>Chat</u></a>.</div>`;
    $$("tr[data-conv]").forEach((t) => (t.onclick = () => { C.conv = +t.dataset.conv; location.hash = "#/chat"; }));
  };
  draw("all");
  $$("#ff button").forEach((b) => (b.onclick = () => { $$("#ff button").forEach((x) => x.classList.toggle("on", x === b)); draw(b.dataset.f); }));
  const save = async () => { try { S.status.settings = await api("/api/settings", { json: { auto_train: $("#auto").checked, auto_threshold: Math.max(1, +$("#thr").value) } }); toast("Saved", "ok"); } catch (e) { fail(e); } };
  $("#auto").onchange = save; $("#thr").onchange = save;
  $$("[data-train]").forEach((b) => (b.onclick = async () => {
    try { await api("/api/feedback/train", { json: { model: b.dataset.train } }); toast(`🎓 <b>${esc(b.dataset.train)}</b> is learning from your ratings`, "ok"); openRun(b.dataset.train); poll(); } catch (e) { fail(e); }
  }));
};

poll().then(route);
