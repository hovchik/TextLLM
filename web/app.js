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
  download: '<svg viewBox="0 0 24 24"><path d="M12 4v12m-5-5 5 5 5-5M4 16v3a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-3"/></svg>',
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
  if (r.status === 401 && !path.startsWith("/api/auth/")) { showAuth(); throw new Error("Please sign in."); }
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
const S = { status: null, page: null, onStatus: null, user: null, timer: null };
const isAdmin = () => S.user?.role === "admin";
async function poll() {
  if (!S.user) return;
  try {
    S.status = await api("/api/status");
    if (S.status.user) setUser(S.status.user);
    if (!isAdmin()) return;
    const j = S.status.job;
    const pill = $("#jobpill");
    pill.classList.toggle("on", j.running);
    $(".jobtext", pill).textContent = j.running ? `${labelKind(j.kind)}: ${j.name}` : j.external ? `Scheduled crawler: ${j.external.name || "training"}` : "Idle · no job running";
    const b = $("#nav-pending");
    b.hidden = !S.status.pending_feedback;
    b.textContent = S.status.pending_feedback;
    S.onStatus && S.onStatus(S.status);
  } catch {}
}
const labelKind = (k) => ({ crawl: "Reading news & web", training: "Training", data: "Preparing data", texts: "Learning texts", feedback: "Learning from ratings", eval: "Evaluating", export: "Making fast CPU copy", cluster: "Cluster training", worker: "Helping a cluster" }[k] || k);
$("#jobpill").onclick = () => (location.hash = "#/training/live");

// ---------------------------------------------------------------- router
const pages = {};
const USER_PAGES = ["chat", "settings"];  // all a regular user can open
function route() {
  if (!S.user) return;
  let [page = "", sub] = location.hash.replace(/^#\/?/, "").split("/");
  if (!isAdmin() && !USER_PAGES.includes(page)) {
    if (location.hash !== "#/chat") return void (location.hash = "#/chat");
    page = "chat";
  }
  if (!pages[page]) page = isAdmin() ? "dashboard" : "chat";
  S.onStatus = null;
  $("#modal-root").innerHTML = "";
  $$("#nav a").forEach((a) => a.classList.toggle("active", a.dataset.page === page));
  pages[page](sub);
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
        <div class="muted" style="margin-top:4px">${r ? `${j.kind === "cluster" ? "Round" : "Step"} ${num(r.step)} of ${num(r.max_steps)} · ETA ${dur(r.eta_sec)} · loss ${num(r.train_loss, 3)}` : `Started ${ago(j.started)}`}</div>
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
    ${deviceNote(S.status.device)}
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
const deviceNote = (dv) => !dv ? "" : `<div class="hint">${dv.gpu ? "⚡ Training and chat run on the GPU" : "Training and chat run on the CPU"}: <b>${esc(dv.name)}</b>${dv.hint ? ` · ${esc(dv.hint)}` : ""} · <a href="#/settings"><u>change</u></a></div>`;
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
// Base models for fine-tuning (all Apache 2.0). Qwen3.5 and Gemma 4 load through their multimodal class.
const BASES = [
  { id: "Qwen/Qwen3-0.6B", n: "Qwen3 0.6B · CPU default", h: "Apache 2.0 · text-only · fast on CPU, GGUF works", r: 8 },
  { id: "Qwen/Qwen3-1.7B", n: "Qwen3 1.7B · better writing", h: "Apache 2.0 · LoRA on CPU overnight", r: 16 },
  { id: "Qwen/Qwen3.5-0.8B", n: "Qwen3.5 0.8B · newest (experimental)", h: "Apache 2.0 · multimodal checkpoint; needs the latest transformers and a llama.cpp that supports it", r: 8 },
  { id: "Qwen/Qwen3.5-2B", n: "Qwen3.5 2B · newest (experimental)", h: "Apache 2.0 · as above; slower on CPU", r: 16 },
  { id: "HuggingFaceTB/SmolLM3-3B", n: "SmolLM3 3B · fully open", h: "Apache 2.0 · train on a free GPU, chat via GGUF on CPU", r: 16 },
  { id: "Qwen/Qwen2.5-0.5B-Instruct", n: "Qwen2.5 0.5B · previous default", h: "Apache 2.0 · kept for comparison", r: 8 },
];
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
      <div class="tabs" id="ttabs">${[["new", "New model"], ["data", "Data"], ["texts", "Your texts"], ["news", "News & web"], ["style", "Your style"], ["corpus", "Corpus"], ["cluster", "Cluster"], ["runs", "Runs"]].map(([k, l]) => `<button data-k="${k}" class="${k === sub ? "on" : ""}">${l}</button>`).join("")}</div></div>
    <div id="tlive"></div><div id="tbody"></div></div>`;
  $$("#ttabs button").forEach((b) => (b.onclick = () => (location.hash = `#/training/${b.dataset.k}`)));
  await poll();
  const runsList = await api("/api/runs").catch(() => []);
  const renderLive = (rs) => { $("#tlive").innerHTML = liveCard(rs.find((r) => r.running)); bindLive(); };
  renderLive(runsList);
  S.onStatus = async () => { const rs = await api("/api/runs").catch(() => []); renderLive(rs); if (sub === "runs") drawRuns(rs); if (sub === "cluster") drawClusterLive(); };
  ({ new: trainNew, data: trainData, texts: trainTexts, news: trainNews, style: trainStyle, corpus: trainCorpus, cluster: trainCluster, runs: () => drawRuns(runsList) }[sub] || trainNew)();
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
        <label class="field">Base model<select class="input" id="f-basepick">${BASES.map((b) => `<option value="${b.id}">${esc(b.n)}</option>`).join("")}<option value="">Other (type a Hugging Face id)…</option></select><span class="hint" id="f-basehint"></span></label>
        <label class="field">Hugging Face model id<input class="input" id="f-base" value="${BASES[0].id}"></label>
        <label class="row" style="flex-wrap:nowrap"><span class="switch"><input type="checkbox" id="f-inst" checked><span></span></span> Use the writing instructions (posts, stories, essays)</label>
        <label class="field"><span>Book passages in the mix <b id="f-bsv">20</b>%</span><input type="range" id="f-bookshare" min="0" max="100" step="5" value="20"><span class="hint">Keeps a literary style; the plan suggests at most 20%</span></label>
        <label class="row" style="flex-wrap:nowrap"><span class="switch"><input type="checkbox" id="f-gguf" checked><span></span></span> Make a fast CPU copy after training (GGUF, 4-bit)</label>
        <label class="field">Book examples (no instructions)<input class="input" id="f-examples" type="number" value="4000"><span class="hint">Used when there is no instruction set</span></label>
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
        <label class="field">LoRA rank<input class="input" id="f-lorar" type="number" value="8"><span class="hint">8 below 1B · 16 for 2B+</span></label>
        <label class="field">Passes over the data<input class="input" id="f-epochs" type="number" step="0.5" value="0"><span class="hint">0 = use training steps</span></label>
      </div>
      <div class="grid g4" style="margin-top:12px" id="adv-scratch">
        <label class="field">Warmup steps<input class="input" id="f-warmup" type="number" min="0" value="0"><span class="hint">0 = automatic (steps ÷ 40)</span></label>
        <label class="field">Final learning rate<input class="input" id="f-minlr" type="number" step="any" min="0" value="0"><span class="hint">0 = 0.00006</span></label>
        <label class="field">Weight decay<input class="input" id="f-wd" type="number" step="0.01" min="0" value="0.1"></label>
        <label class="row" style="align-self:center"><span class="switch"><input type="checkbox" id="f-compile"><span></span></span> torch.compile <span class="hint">faster on a GPU, slow start</span></label>
      </div>
      <div class="grid g3" style="margin-top:12px" id="adv-lora" hidden>
        <label class="field">Continue from a fine-tuned model<select class="input" id="f-initfrom"><option value="">No, start a fresh adapter</option></select><span class="hint">Keeps what it learned and trains more on top. Same base model.</span></label>
        <label class="field">Examples per book<input class="input" id="f-ppd" type="number" min="1" value="8"><span class="hint">max passages taken from one book</span></label>
        <label class="field">Shortest book (characters)<input class="input" id="f-mindoc" type="number" min="0" step="500" value="5000"></label>
      </div><label class="row" style="margin-top:12px"><span class="switch"><input type="checkbox" id="f-resume"><span></span></span> Resume from the last checkpoint</label></details></div>
      <div class="row end"><button class="btn primary" id="start">${I.play} Start training</button></div>
    </div>
    <div class="card" style="display:flex;flex-direction:column;gap:14px"><h2>What to expect</h2><div id="expect" class="muted"></div></div>
  </div>`;
  const expect = () => {
    $("#expect").innerHTML = st.mode === "lora"
      ? `<p>The base model already writes well; fine-tuning teaches it <b>formats</b> (post lengths, hooks, essay structure) from the writing instructions, plus a little <b>style</b> from your books.</p><p>Build the instructions first under <a href="#/training/data"><u>Data</u></a> → step 4. ${S.status?.data?.instructions ? `<b>${num(S.status.data.instructions)}</b> examples are ready.` : "None built yet."}</p><p>On CPU each step takes roughly a minute, so 300 steps is a few hours. On a free Colab/Kaggle GPU it's minutes.</p><p>Then <b>Evaluate</b> it on Analytics and rate answers in Chat.</p>`
      : `<p>A from-scratch model starts from <b>random letters</b> and climbs the 6 quality levels as it reads.</p>${levelBar({ n: 4, label: "Grammatical sentences" })}<p style="margin-top:12px">With the <b>CPU</b> size, expect book-like sentences that drift after a few lines, after about a day. Bigger sizes need a GPU.</p><p>It continues the text you type rather than following instructions.</p>`;
  };
  const setMode = (m) => {
    st.mode = m;
    $$("#modes .opt").forEach((o) => o.classList.toggle("on", o.dataset.v === m));
    $("#sizebox").hidden = m !== "scratch"; $("#basebox").hidden = m === "scratch";
    $("#adv-scratch").hidden = m !== "scratch"; $("#adv-lora").hidden = m === "scratch";
    Object.entries(m === "scratch" ? { steps: 20000, lr: 0.001, batch: 16, accum: 4, name: "my-writer" } : { steps: 300, lr: 0.0002, batch: 2, accum: 8, name: "my-social-writer" })
      .forEach(([k, v]) => ($("#f-" + k).value = v));
    expect();
  };
  $$("#modes .opt").forEach((o) => (o.onclick = () => setMode(o.dataset.v)));
  $$("#presets .opt").forEach((o) => (o.onclick = () => { st.preset = o.dataset.v; $$("#presets .opt").forEach((x) => x.classList.toggle("on", x === o)); }));
  const baseHint = () => { const b = BASES.find((x) => x.id === $("#f-basepick").value); $("#f-basehint").textContent = b ? b.h : "Any causal LM on Hugging Face"; if (b) { $("#f-base").value = b.id; $("#f-lorar").value = b.r; } };
  $("#f-basepick").onchange = baseHint; baseHint();
  api("/api/models").then((ms) => {
    $("#f-initfrom").innerHTML += ms.filter((x) => x.type === "lora").map((x) => `<option value="${esc(x.name)}" data-base="${esc(x.base || "")}">${esc(x.name)}${x.base ? " · " + esc(x.base) : ""}</option>`).join("");
  }).catch(() => {});
  $("#f-initfrom").onchange = () => { const b = $("#f-initfrom").selectedOptions[0].dataset.base; if (b) { $("#f-base").value = b; const known = BASES.some((x) => x.id === b); $("#f-basepick").value = known ? b : ""; } };
  $("#f-bookshare").oninput = (e) => ($("#f-bsv").textContent = e.target.value);
  expect();
  $("#start").onclick = async () => {
    const v = (k) => $("#f-" + k).value;
    try {
      await api("/api/train", { json: { mode: st.mode, name: v("name"), resume: $("#f-resume").checked, preset: st.preset, vocab: +v("vocab"), steps: +v("steps"), batch: +v("batch"), accum: +v("accum"), lr: +v("lr"), base: v("base"), examples: +v("examples"), seq: +v("seq"), instructions: $("#f-inst").checked, book_share: +v("bookshare") / 100, lora_r: +v("lorar"), epochs: +v("epochs"), gguf: $("#f-gguf").checked, warmup: +v("warmup"), min_lr: +v("minlr"), weight_decay: +v("wd"), compile: $("#f-compile").checked, init_from: v("initfrom"), pairs_per_doc: +v("ppd"), min_doc_chars: +v("mindoc") } });
      toast(`Training <b>${esc(v("name"))}</b> started`, "ok");
      location.hash = "#/training/runs";
      setTimeout(() => openRun(v("name")), 400);
    } catch (e) { fail(e); }
  };
}

// ---------------------------------------------------------------- cluster: several PCs train one model
function trainCluster() {
  const d = S.status.data;
  const presetOpts = PRESETS.map((p) => `<option value="${p.id}" ${p.id === "cpu" ? "selected" : ""}>${esc(p.n)} · ${p.p} params</option>`).join("");
  $("#tbody").innerHTML = `<div id="cl-live"></div>
  <div class="grid g2" style="margin-top:16px">
    <div class="card" style="display:flex;flex-direction:column;gap:14px"><div class="card-head"><h2>Start a cluster on this PC</h2><span class="hint">this PC keeps the model and the data</span></div>
      <p class="muted" style="margin:0">Other PCs connect here and train the same from-scratch model with their CPU or GPU. ${d.train_tokens ? `<b>${num(d.train_tokens)}</b> training tokens are ready.` : 'Prepare the data first under <a href="#/training/data"><u>Data</u></a> (steps 1–3).'}</p>
      <div class="grid g3">
        <label class="field">Model name<input class="input" id="c-name" value="cluster-writer"></label>
        <label class="field">Model size<select class="input" id="c-preset">${presetOpts}</select></label>
        <label class="field">Rounds<input class="input" id="c-rounds" type="number" min="1" value="200"><span class="hint">one round = local training + averaging</span></label>
        <label class="field">Minutes per round<input class="input" id="c-minutes" type="number" min="0.2" step="0.5" value="5"><span class="hint">longer = less internet traffic</span></label>
        <label class="field">Learning rate<input class="input" id="c-lr" type="number" step="any" value="0.001"></label>
        <label class="field">Port<input class="input" id="c-port" type="number" value="7861"></label>
      </div>
      <label class="row"><span class="switch"><input type="checkbox" id="c-local" checked><span></span></span> This PC trains too</label>
      <label class="row"><span class="switch"><input type="checkbox" id="c-upnp" checked><span></span></span> Let PCs on other networks join <span class="hint">asks the router to open the port (UPnP); closed again when the cluster stops</span></label>
      <details><summary>Advanced</summary><div class="grid g3" style="margin-top:12px">
        <label class="field">Secret<input class="input mono" id="c-secret" placeholder="made up for you"><span class="hint">workers must type it</span></label>
        <label class="field">Workers needed to start<input class="input" id="c-minw" type="number" min="1" value="1"></label>
        <label class="field">Vocab size<input class="input" id="c-vocab" type="number" value="${d.vocab || 8192}"><span class="hint">must match tokenizer</span></label>
        <label class="field">Final learning rate<input class="input" id="c-minlr" type="number" step="any" min="0" value="0"><span class="hint">0 = learning rate ÷ 10</span></label>
        <label class="field">Warmup steps (per worker)<input class="input" id="c-warmup" type="number" min="0" value="200"></label>
        <label class="field">Weight decay<input class="input" id="c-wd" type="number" step="0.01" min="0" value="0.1"></label>
        <label class="field">Outer learning rate<input class="input" id="c-olr" type="number" step="0.05" value="0.7"><span class="hint">step size of the averaged update</span></label>
        <label class="field">Outer momentum<input class="input" id="c-omom" type="number" step="0.05" min="0" max="0.99" value="0.9"></label>
        <label class="field">Check quality every N rounds<input class="input" id="c-evalevery" type="number" min="1" value="1"></label>
        <label class="field">This PC: batch size<input class="input" id="c-batch" type="number" min="1" value="16"></label>
        <label class="field">This PC: grad accumulation<input class="input" id="c-accum" type="number" min="1" value="4"></label>
        <label class="row" style="align-self:center"><span class="switch"><input type="checkbox" id="c-nesterov" checked><span></span></span> Nesterov momentum</label>
      </div><label class="row" style="margin-top:12px"><span class="switch"><input type="checkbox" id="c-resume"><span></span></span> Resume this model from its last round</label></details>
      <div class="row end"><button class="btn primary" id="c-start">${I.play} Start the cluster</button></div>
    </div>
    <div class="card" style="display:flex;flex-direction:column;gap:14px"><div class="card-head"><h2>Join a cluster</h2><span class="hint">lend this PC to another PC's training</span></div>
      <p class="muted" style="margin:0">Copy the address and secret from the PC that started the cluster. This PC downloads its share of the text once, then trains in rounds on <b>${esc(S.status.device?.name || "its CPU or GPU")}</b>. Pick another device in Settings.</p>
      <div class="grid g2">
        <label class="field">Coordinator address<input class="input mono" id="j-addr" placeholder="http://192.168.1.20:7861"></label>
        <label class="field">Secret<input class="input mono" id="j-secret"></label>
        <label class="field">This PC's name<input class="input" id="j-name" placeholder="shown on the coordinator"></label>
        <label class="field">Text to download (MB)<input class="input" id="j-mb" type="number" min="1" value="200"><span class="hint">100 MB ≈ 50M tokens</span></label>
      </div>
      <details><summary>Advanced</summary><div class="grid g3" style="margin-top:12px">
        <label class="field">Batch size<input class="input" id="j-batch" type="number" min="0" value="0"><span class="hint">0 = 16 on a GPU, 8 on a CPU</span></label>
        <label class="field">Grad accumulation<input class="input" id="j-accum" type="number" min="1" value="4"></label>
        <label class="field">Stop early (seconds)<input class="input" id="j-reserve" type="number" min="0" value="0"><span class="hint">leave time for a slow upload</span></label>
        <label class="row" style="grid-column:1/-1"><span class="switch"><input type="checkbox" id="j-local"><span></span></span> Use this PC's own prepared data instead of downloading (same tokenizer only)</label>
      </div></details>
      <div class="row end"><button class="btn primary" id="j-start">${I.play} Join</button></div>
    </div>
  </div>
  <div class="card" style="margin-top:16px"><h2>How it works</h2><div class="muted">
    <p>Each round, every PC starts from the same model and trains on its own share of the text for the round's minutes. A GPU simply does more steps than a CPU. Then each PC sends how far its copy moved, the coordinator averages the moves (weighted by how much text each PC read) and everyone continues from the result. PCs can join or leave at any time.</p>
    <p>The internet only carries about one copy of the model each way per PC per round (the CPU size is ~${num(14 * 2)} MB), not gradients on every step, so home connections are fine. Expect roughly <b>0.6–0.9× the extra PCs' own speed</b> on top of yours: four similar PCs ≈ 3× faster than one. A slow PC never holds the others up.</p>
    <p><b>PCs on other networks:</b> only the PC that starts the cluster must be reachable; the others just connect out to it, so they need no router setup. When you start a cluster, the Studio asks your router to open the port (UPnP, on by default in most home routers) and shows an <b>Internet</b> address to share. If that fails (UPnP off, a second router, or a provider that shares one public address between customers, common on mobile internet), install the free <b>Tailscale</b> on every PC and use the <b>Tailscale</b> address, or forward TCP port 7861 to this PC in the router's settings. The long secret keeps strangers out; the connection itself is not encrypted, so Tailscale is the most private choice.</p>
    <p>On a PC without the Studio: <span class="mono">run.bat --join http://ADDRESS:7861 --secret SECRET</span> (or <span class="mono">./run.sh</span> on Linux/macOS).</p></div></div>`;
  const v = (k) => $("#" + k).value;
  try { const last = JSON.parse(localStorage.getItem("cluster-join") || "null"); if (last) { $("#j-addr").value = last.address || ""; $("#j-name").value = last.name || ""; } } catch {}
  $("#c-start").onclick = async () => {
    try {
      await api("/api/cluster/start", { json: { name: v("c-name"), preset: v("c-preset"), rounds: +v("c-rounds"), round_minutes: +v("c-minutes"), lr: +v("c-lr"), port: +v("c-port"), local_worker: $("#c-local").checked, upnp: $("#c-upnp").checked, secret: v("c-secret"), min_workers: +v("c-minw"), vocab: +v("c-vocab"), min_lr: +v("c-minlr"), warmup: +v("c-warmup"), weight_decay: +v("c-wd"), outer_lr: +v("c-olr"), outer_momentum: +v("c-omom"), eval_every: +v("c-evalevery"), batch: +v("c-batch"), accum: +v("c-accum"), nesterov: $("#c-nesterov").checked, resume: $("#c-resume").checked } });
      toast("Cluster started. Share the address and secret with the other PCs.", "ok");
      await poll(); drawClusterLive();
    } catch (e) { fail(e); }
  };
  $("#j-start").onclick = async () => {
    try {
      try { localStorage.setItem("cluster-join", JSON.stringify({ address: v("j-addr"), name: v("j-name") })); } catch {}
      await api("/api/cluster/join", { json: { address: v("j-addr"), secret: v("j-secret"), name: v("j-name"), data_mb: +v("j-mb"), batch: +v("j-batch"), accum: +v("j-accum"), reserve: +v("j-reserve"), local_data: $("#j-local").checked } });
      toast("Joining the cluster…", "ok");
      await poll(); drawClusterLive();
    } catch (e) { fail(e); }
  };
  drawClusterLive();
}

const WSTATE = { training: "good", sent: "good", waiting: "", joined: "", "downloading data": "accent", "downloading weights": "accent", uploading: "accent", left: "warn", done: "good" };
async function drawClusterLive() {
  const box = $("#cl-live");
  if (!box) return;
  const st = await api("/api/cluster").catch(() => null);
  if (!st || !$("#cl-live")) return;
  const c = st.coordinator, w = st.worker;
  let html = "";
  if (c && (c.running || c.round)) {
    const now = Date.now() / 1000;
    const left = c.train_until ? Math.max(0, c.train_until - now) : null;
    const phase = { waiting: c.workers?.some((x) => x.state !== "left") ? "Waiting for the next round" : "Waiting for workers to join", training: `Round ${num((c.round || 0) + 1)} training · ${dur(left)} left`, averaging: "Averaging", finished: "Finished", stopped: "Stopped" }[c.phase] || "Starting…";
    const KIND = { internet: "Internet", tailscale: "Tailscale", local: "Same network", loopback: "This PC" };
    const reach = c.reach || {};
    const opts = (reach.addresses || (c.addresses || []).map((ip) => ({ kind: "local", url: `http://${ip}:${c.port}`, note: "" })));
    const addrs = opts.map((o) => o.url);
    const up = reach.upnp;
    const reachNote = !up ? `<span class="hint">Opening the router port automatically is off. PCs on other networks need Tailscale or a forwarded port ${c.port}.</span>`
      : up.pending ? `<span class="hint">Asking the router to open port ${c.port} for PCs on other networks…</span>`
      : up.ok && !up.error ? `<span class="pill good">✓ Reachable from other networks</span> <span class="hint">the router opened port ${c.port} automatically (UPnP); share the Internet address</span>`
      : `<span class="pill warn">Same network only</span> <span class="hint">${esc(up.error || "the router did not open the port")}. For PCs elsewhere: install <b>Tailscale</b> on every PC (works through any router), or forward TCP port ${c.port} to this PC in the router's settings.</span>`;
    const live = (c.workers || []).filter((x) => x.state !== "left");
    html += `<div class="card"><div class="card-head"><h2>Cluster: ${esc(c.run)}</h2><div class="row" style="gap:8px">${c.running ? '<span class="pill good">● running</span>' : '<span class="pill">not running</span>'}<span class="hint">${esc(phase)}</span>${c.running ? `<button class="btn danger sm" id="cl-stop">${I.stop} Stop</button>` : ""}</div></div>
      <div class="stats" style="margin-top:6px">
        <div class="stat"><div class="l">Round</div><div class="v">${num(c.round)} / ${num(c.rounds)}</div></div>
        <div class="stat"><div class="l">PCs training</div><div class="v">${num(live.length)}</div></div>
        <div class="stat"><div class="l">Text read</div><div class="v">${num((c.tokens_total || 0) / 1e6, 1)}M tokens</div></div>
        <div class="stat"><div class="l">Best val loss</div><div class="v">${num(c.best_val, 3)}</div></div>
        <div class="stat"><div class="l">Per transfer</div><div class="v">${num(c.transfer_mb ?? c.model_mb / 2)} MB</div></div>
      </div>
      ${c.running ? `<div style="margin-top:12px"><b>Other PCs join with</b> <div style="margin-top:6px">${reachNote}</div>
        <div class="grid g2" style="margin-top:8px"><label class="field">Address<select class="input mono" id="cl-addr">${opts.map((o) => `<option value="${esc(o.url)}">${esc(KIND[o.kind] || o.kind)}: ${esc(o.url)}</option>`).join("")}</select><span class="hint" id="cl-addr-note">${esc(opts[0]?.note || "")}</span></label><label class="field">Secret<input class="input mono" readonly value="${esc(c.secret)}" id="cl-secret"></label></div>
        <div class="row" style="margin-top:8px"><code class="mono" id="cl-cmd" style="flex:1;overflow:auto;white-space:nowrap">run.bat --join ${esc(addrs[0] || "")} --secret ${esc(c.secret)}</code><button class="btn sm" id="cl-copy">${I.copy} Copy</button></div></div>` : ""}
      <div class="table-scroll" style="margin-top:14px"><table><thead><tr><th>PC</th><th>Device</th><th>State</th><th>Rounds</th><th>Speed</th><th>Loss</th><th>Last seen</th></tr></thead><tbody>
        ${(c.workers || []).map((x) => `<tr><td><b>${esc(x.name)}</b><div class="hint mono">${esc(x.address || "")}${x.note ? " · " + esc(x.note) : ""}</div></td><td>${esc(x.device_name || x.device)}</td>
          <td><span class="pill ${WSTATE[x.state] ?? ""}">${esc(x.state)}</span></td><td>${num(x.rounds)}</td><td>${x.tokens_per_sec ? num(x.tokens_per_sec) + " tok/s" : "—"}</td><td>${num(x.loss, 3)}</td><td class="muted">${ago(x.last_seen)}</td></tr>`).join("") || '<tr><td colspan="7" class="muted">No PCs connected yet.</td></tr>'}
      </tbody></table></div>
      <details style="margin-top:12px"><summary>Log</summary><pre class="console" style="margin-top:8px">${esc(c.log || "")}</pre></details></div>`;
  }
  if (w && (w.running || w.state)) {
    html += `<div class="card" style="margin-top:${html ? 16 : 0}px"><div class="card-head"><h2>This PC is helping ${esc(w.coordinator || w.last?.address || "a cluster")}</h2><div class="row" style="gap:8px">${w.running ? `<span class="pill ${WSTATE[w.state] ?? ""}">● ${esc(w.state || "starting")}</span><button class="btn danger sm" id="wk-stop">${I.stop} Leave</button>` : `<span class="pill">${esc(w.state || "stopped")}</span>`}</div></div>
      <div class="muted">${esc(w.note || "")}${w.state === "downloading data" && w.data_progress != null ? ` <div class="bar" style="margin-top:6px"><i style="width:${(w.data_progress * 100).toFixed(1)}%"></i></div>` : ""}</div>
      <div class="stats" style="margin-top:6px">
        <div class="stat"><div class="l">Device</div><div class="v" style="font-size:14px">${esc(w.device_name || "—")}</div></div>
        <div class="stat"><div class="l">Rounds sent</div><div class="v">${num(w.rounds)}</div></div>
        <div class="stat"><div class="l">Speed</div><div class="v">${w.tokens_per_sec ? num(w.tokens_per_sec) + " tok/s" : "—"}</div></div>
        <div class="stat"><div class="l">Loss</div><div class="v">${num(w.loss, 3)}</div></div>
      </div>
      <details style="margin-top:12px"><summary>Log</summary><pre class="console" style="margin-top:8px">${esc(w.log || "")}</pre></details></div>`;
  }
  S.clusterOpts = c?.reach?.addresses || [];
  const open = $$("#cl-live details").map((x) => x.open);
  const sel = $("#cl-addr")?.value;
  box.innerHTML = html;
  $$("#cl-live details").forEach((x, i) => (x.open = !!open[i]));
  if (sel && $("#cl-addr")) $("#cl-addr").value = sel;
  const cmd = () => `run.bat --join ${$("#cl-addr").value} --secret ${$("#cl-secret").value}`;
  if ($("#cl-addr")) {
    const upd = () => { $("#cl-cmd").textContent = cmd(); const o = (S.clusterOpts || []).find((x) => x.url === $("#cl-addr").value); $("#cl-addr-note").textContent = o ? o.note : ""; };
    upd(); $("#cl-addr").onchange = upd;
  }
  if ($("#cl-copy")) $("#cl-copy").onclick = () => navigator.clipboard.writeText(cmd()).then(() => toast("Copied", "ok"), () => toast("Copy failed; select the text instead", "err"));
  for (const id of ["#cl-stop", "#wk-stop"]) {
    const b = $(id);
    if (b) b.onclick = async () => { b.disabled = true; await api("/api/stop", { method: "POST" }).catch(fail); toast(id === "#cl-stop" ? "Stopping. The last finished round is saved." : "Left the cluster."); poll(); drawClusterLive(); };
  }
}

function trainData() {
  const d = S.status.data;
  $("#tbody").innerHTML = `<div class="grid g3"><div class="card span2"><div class="card-head"><h2>Prepare the training corpus</h2><span class="hint">Run the steps in order</span></div>
    <div class="steps">
      <div class="step ${d.raw_files ? "done" : ""}"><div class="num">${d.raw_files ? "✓" : 1}</div><div><b>Download books and web text</b><div class="hint">${d.raw_files ? `${d.raw_files} files, ${num(d.raw_mb)} MB (${d.custom_files} of your own)` : "Public-domain books from Project Gutenberg plus openly licensed FineWeb-Edu articles. More book sources below."}</div>
        <div class="row" style="margin-top:8px"><label class="field" style="width:140px">Books<input class="input" id="d-books" type="number" value="3000"></label><label class="field" style="width:160px">Web text (GB)<input class="input" id="d-web" type="number" step="0.1" value="0.3"></label></div>
        <details class="more-src" style="margin-top:10px"><summary><b>More book sources</b> <span class="hint">all public domain or openly licensed, 0 = skip</span></summary>
          <div class="row" style="margin-top:8px;flex-wrap:wrap">
            <label class="field" style="width:200px">Gutenberg download<select class="input" id="d-gmethod"><option value="mirror">Mirror (catalog order)</option><option value="harvest">Robot harvest (official)</option></select></label>
            <label class="field" style="width:160px">PG-19 books<input class="input" id="d-pg19" type="number" min="0" value="0"></label>
            <label class="field" style="width:170px">Standard Ebooks<input class="input" id="d-se" type="number" min="0" value="0"></label>
            <label class="field" style="width:190px">Common Corpus (GB)<input class="input" id="d-cc" type="number" min="0" step="0.1" value="0"></label>
            <label class="field" style="width:190px">Internet Archive books<input class="input" id="d-ia" type="number" min="0" value="0"></label>
            <label class="field" style="width:160px">Cosmopedia (GB)<input class="input" id="d-cosmo" type="number" min="0" step="0.1" value="0"></label>
            <label class="field" style="width:220px">Cosmopedia part<select class="input" id="d-cosmosub"><option value="stories">Stories</option><option value="stories,wikihow">Stories + how-to guides</option><option value="openstax,khanacademy,stanford">Textbooks</option><option value="stories,wikihow,openstax,khanacademy,stanford,web_samples_v1,web_samples_v2">Everything except math</option></select></label>
            <label class="field" style="width:160px">Wikipedia (GB)<input class="input" id="d-wiki" type="number" min="0" step="0.1" value="0"></label>
            <label class="field" style="width:130px">Wikipedia language<input class="input" id="d-wikilang" value="en" maxlength="12"></label>
            <label class="field" style="width:170px">Common Pile (GB)<input class="input" id="d-cp" type="number" min="0" step="0.1" value="0"></label>
            <label class="field" style="width:240px">Common Pile part<select class="input" id="d-cpsub"><option value="news,public_domain_review,pressbooks,cccc">Posts, essays, blogs, open books</option><option value="news">CC news articles</option><option value="cccc">CC-licensed web pages and blogs</option><option value="public_domain_review,pressbooks,doab">Essays and open books</option><option value="wikimedia">Wikibooks, Wikivoyage, Wikiquote...</option><option value="youtube,uk_hansard">Spoken style (talks, debates)</option><option value="library_of_congress,pre_1929_books">Public-domain books (OCR)</option></select></label>
          </div>
          <ul class="hint" style="padding-left:18px;margin:6px 0 0">
            <li><b>PG-19</b>: 28,752 Gutenberg books from before 1919 (Apache 2.0). Books Gutenberg already saved are skipped.</li>
            <li><b>Standard Ebooks</b>: about 1,400 carefully proofread classics (CC0), read from their GitHub pages.</li>
            <li><b>Common Corpus</b>: only its "Open Culture" part, public-domain books and newspapers (OCR).</li>
            <li><b>Internet Archive</b>: scanned books published before 1931 (OCR, noisier). Keep it small.</li>
            <li><b>Cosmopedia</b>: synthetic stories and textbooks written by Mixtral (Apache 2.0).</li>
            <li><b>Wikipedia</b>: encyclopedia articles in any language code (en, de, fr, hy...), CC BY-SA 4.0.</li>
            <li><b>Common Pile</b>: EleutherAI's openly licensed collection: CC news sites, Public Domain Review essays, open books, CC-licensed blogs. NC/ND rows are dropped.</li>
            <li>Not offered: BookCorpus, BookCorpusOpen, The Pile and RedPajama books (copyrighted books), Open Library (no book text).</li>
          </ul>
          <div class="row" style="margin-top:8px"><label class="field" style="width:190px">Common Corpus language<input class="input" id="d-cclang" value="en" maxlength="12"><span class="hint">code, or <span class="mono">all</span></span></label></div></details>
        <details style="margin-top:10px"><summary><b>Downloading</b> <span class="hint">for a connection that drops</span></summary>
          <div class="row" style="margin-top:8px;flex-wrap:wrap">
            <label class="field" style="width:260px">Hugging Face sources<select class="input" id="d-hfmode"><option value="stream">Stream (read parts over HTTP)</option><option value="download">Download whole files first (resumable)</option></select></label>
            <label class="field" style="width:150px">Retries per request<input class="input" id="d-retries" type="number" min="0" value="30"><span class="hint">30 ≈ 45 min offline</span></label>
            <label class="field" style="width:150px">Parallel downloads<input class="input" id="d-workers" type="number" min="1" max="64" value="8"></label>
            <label class="row hint" style="align-self:center"><span class="switch"><input type="checkbox" id="d-keepparq"><span></span></span> Keep downloaded parquet files</label>
          </div></details></div>
        <button class="btn" data-step="download">${I.play} Run</button></div>
      <div class="step ${d.tokenizer ? "done" : ""}"><div class="num">${d.tokenizer ? "✓" : 2}</div><div><b>Build the tokenizer</b><div class="hint">${d.tokenizer ? `${num(d.vocab)}-token vocabulary` : "Learns the word pieces the model reads and writes"}</div>
        <div class="row" style="margin-top:8px"><label class="field" style="width:200px">Vocabulary<select class="input" id="d-vocab"><option value="8192">8,192 (CPU)</option><option value="32000">32,000 (GPU)</option></select></label>
          <label class="field" style="width:200px">Files used to learn it<input class="input" id="d-samplefiles" type="number" min="1" value="40"><span class="hint">more = slower, slightly better</span></label></div></div>
        <button class="btn" data-step="tokenizer">${I.play} Run</button></div>
      <div class="step ${d.train_tokens ? "done" : ""}"><div class="num">${d.train_tokens ? "✓" : 3}</div><div><b>Encode the corpus</b><div class="hint">${d.train_tokens ? `${num(d.train_tokens / 1e6, 1)}M tokens ready to train` : "Converts every text into token files for fast training"}</div>
          <div class="row" style="margin-top:8px"><label class="field" style="width:200px">Held out for validation (%)<input class="input" id="d-valfrac" type="number" min="0.01" max="50" step="0.1" value="0.5"></label></div></div>
        <button class="btn" data-step="encode">${I.play} Run</button></div>
      <div class="step ${d.instructions ? "done" : ""}"><div class="num">${d.instructions ? "✓" : 4}</div><div><b>Build writing instructions</b> <span class="pill">for fine-tuning</span><div class="hint">${d.instructions ? `${num(d.instructions)} examples of posts, stories and essays` : "Public-domain stories and essays from Gutenberg, your liked and corrected chat answers, and (optional) OpenAssistant + Dolly"}</div>
        <div class="row" style="margin-top:8px"><label class="row hint"><span class="switch"><input type="checkbox" id="d-hfsets" checked><span></span></span> Include oasst2 (Apache 2.0) and Dolly (CC BY-SA 3.0)</label></div>
        <details style="margin-top:8px"><summary><b>Your own posts</b> <span class="hint" id="d-mpcount"></span></summary>
          <div class="hint" style="margin:6px 0">A <span class="mono">.txt</span> file with posts separated by <span class="mono">---</span> lines, or <span class="mono">.jsonl</span> with <span class="mono">{"format": "linkedin_post", "text": "..."}</span>. Only your own posts.</div>
          <div class="files" id="d-mplist"></div>
          <div class="row" style="margin-top:6px"><label class="btn sm">${I.upload} Add files<input type="file" id="d-mpfile" accept=".txt,.jsonl" multiple hidden></label>
            <label class="row hint"><span class="switch"><input type="checkbox" id="d-mpuse" checked><span></span></span> Use them in this build</label></div></details>
        <details style="margin-top:8px"><summary><b>More options</b> <span class="hint">Gutenberg, synthetic posts</span></summary>
          <div class="row" style="margin-top:8px;flex-wrap:wrap">
            <label class="row hint" style="width:100%"><span class="switch"><input type="checkbox" id="d-igut" checked><span></span></span> Stories and essays from Project Gutenberg</label>
            <label class="field" style="width:130px">Stories<input class="input" id="d-istories" type="number" min="0" value="1200"></label>
            <label class="field" style="width:130px">Essays<input class="input" id="d-iessays" type="number" min="0" value="1200"></label>
            <label class="field" style="width:190px">Books to read per kind<input class="input" id="d-ibooks" type="number" min="1" value="300"></label>
            <label class="field" style="width:110px">Seed<input class="input" id="d-iseed" type="number" value="0"></label>
          </div>
          <div class="row" style="margin-top:8px;flex-wrap:wrap">
            <label class="field" style="width:300px">Model that writes example posts<input class="input" id="d-synth" placeholder="empty = none · e.g. Qwen/Qwen3-8B or C:\\models\\x.gguf"><span class="hint">Apache 2.0 model. Use a GPU; on CPU a 4B GGUF writes ~50 per format in hours.</span></label>
            <label class="field" style="width:160px">Posts per format<input class="input" id="d-synthn" type="number" min="1" value="200"></label>
            <label class="field" style="width:300px">Model that writes the requests<input class="input" id="d-bt" placeholder="empty = simple templates"><span class="hint">Back-translation: writes a better request for each text. GPU recommended.</span></label>
          </div></details></div>
        <button class="btn" data-step="instructions">${I.play} Run</button></div>
    </div></div>
    <div class="card"><h2>Tips</h2><ul class="muted" style="padding-left:18px">
      <li>Only <b>from-scratch</b> training needs steps 2 and 3. Fine-tuning uses step 4 plus a share of the downloaded books.</li>
      <li>Want your own books in the corpus? Add them under <a href="#/training/texts"><u>Your texts</u></a> → "Add to training data", then re-run steps 2–3.</li>
      <li>Progress and logs show at the top of this page while a step runs.</li></ul></div></div>
    <div class="card" style="margin-top:16px"><div class="card-head"><h2>Log</h2></div><pre class="console" id="dlog">${esc(S.status.job.kind === "data" ? S.status.job.log : "No data job yet.")}</pre></div>`;
  $$("[data-step]").forEach((b) => (b.onclick = async () => {
    try {
      await api("/api/data", { json: { fineweb_mode: $("#d-hfmode").value, retries: +$("#d-retries").value, workers: +$("#d-workers").value, keep_parquet: $("#d-keepparq").checked, common_corpus_lang: $("#d-cclang").value, sample_files: +$("#d-samplefiles").value, val_frac: (+$("#d-valfrac").value || 0.5) / 100, inst_gutenberg: $("#d-igut").checked, inst_stories: +$("#d-istories").value, inst_essays: +$("#d-iessays").value, inst_books: +$("#d-ibooks").value, seed: +$("#d-iseed").value, inst_my_posts: $("#d-mpuse").checked, synth_model: $("#d-synth").value, synth_per_format: +$("#d-synthn").value, backtranslate: $("#d-bt").value, step: b.dataset.step, books: +$("#d-books").value, web_gb: +$("#d-web").value, gutenberg_method: $("#d-gmethod").value, pg19_books: +$("#d-pg19").value, standard_ebooks: +$("#d-se").value, common_corpus_gb: +$("#d-cc").value, archive_books: +$("#d-ia").value, cosmopedia_gb: +$("#d-cosmo").value, cosmopedia_subsets: $("#d-cosmosub").value, wikipedia_gb: +$("#d-wiki").value, wikipedia_lang: $("#d-wikilang").value, common_pile_gb: +$("#d-cp").value, common_pile_subsets: $("#d-cpsub").value, vocab: +$("#d-vocab").value, hf_sets: $("#d-hfsets").checked } });
      toast("Started", "ok"); poll();
    } catch (e) { fail(e); }
  }));
  const drawMp = (list) => {
    $("#d-mpcount").textContent = list.length ? `${list.length} file(s)` : "none yet";
    $("#d-mplist").innerHTML = list.map((f) => `<span class="file">📄 ${esc(f.name)} <span class="faint">${num(f.kb)} KB</span><button data-mp="${esc(f.name)}" title="Remove">✕</button></span>`).join("");
    $$("[data-mp]").forEach((x) => (x.onclick = async () => { try { drawMp(await api(`/api/data/my-posts/${encodeURIComponent(x.dataset.mp)}`, { method: "DELETE" })); } catch (e) { fail(e); } }));
  };
  api("/api/data/my-posts").then(drawMp).catch(() => {});
  $("#d-mpfile").onchange = async (e) => {
    const fd = new FormData(); [...e.target.files].forEach((f) => fd.append("files", f));
    try { drawMp(await api("/api/data/my-posts", { method: "POST", body: fd })); toast("Added", "ok"); } catch (x) { fail(x); } finally { e.target.value = ""; }
  };
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
      <details class="span2" style="grid-column:1/-1"><summary>Advanced</summary><div class="grid g2" style="margin-top:8px">
        <label class="field">Old text mixed in, per new token<input class="input" id="x-mix" type="number" min="0" max="10" step="0.5" value="1"><span class="hint">From-scratch models only: keeps them from forgetting. 0 = only the new texts.</span></label>
        <label class="field">Learning rate<input class="input" id="x-lr" type="number" min="0" step="any" value="0"><span class="hint">0 = automatic (0.0003 from scratch, 0.0001 fine-tuned)</span></label></div></details>
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
    fd.append("mix", $("#x-mix").value || "1"); fd.append("lr", $("#x-lr").value || "0");
    try {
      await api("/api/texts", { method: "POST", body: fd });
      toast(action === "learn" ? "Reading your texts and training…" : "Adding texts to the corpus…", "ok");
      files.length = 0; drawFiles(); poll();
      if (action === "learn") setTimeout(() => openRun($("#x-model").value), 500);
    } catch (e) { fail(e); }
  };
}

// ---------------------------------------------------------------- news & web crawler
const KIND_LABEL = { feed: "RSS / Atom feed", page: "Single page", links: "Page of links", mastodon: "Mastodon", bluesky: "Bluesky" };
const SOCIAL = { mastodon: "https://example.social/tags/poetry", bluesky: "https://bsky.app/profile/alice.bsky.social" };
async function trainNews() {
  const [st, models] = await Promise.all([api("/api/crawler"), api("/api/models").catch(() => [])]);
  const L = st.licenses, cs = st.settings, ss = st.stats, LN = st.language_names || {};
  const langName = (c) => (c && c !== "?" ? LN[c] || c : "unknown");
  const licPill = (s) => { const l = L[s.license] || { label: s.license }; return `<span class="pill ${l.ok ? "good" : "warn"}" style="white-space:nowrap" title="${esc(l.note || "")}">${esc(l.label)}</span>`; };
  const okSrc = (s) => (L[s.license] || {}).ok || s.rights_confirmed;
  const osSched = ss.schedule || {};
  const verdictPill = (r) => {
    if (!r.learned) return "";
    const v = r.verdict || "learned";
    const sc = r.score_before != null && r.score_after != null ? ` ${num(r.score_before, 0)} → ${num(r.score_after, 0)}` : "";
    return `<span class="pill ${v === "rolled back" ? "warn" : "good"}" title="Quick-check score before → after learning">${v === "rolled back" ? "undone" : "learned"}${sc}</span>`;
  };
  $("#tbody").innerHTML = `<div class="grid g3"><div class="span2" style="display:flex;flex-direction:column;gap:16px">
    <div class="card"><div class="card-head"><h2>Sources</h2><span class="hint">${num(ss.total)} articles saved · ${num(ss.queued)} links queued · ${num(ss.pending)} not learned yet${ss.skipped_language ? ` · ${num(ss.skipped_language)} skipped (language)` : ""}</span></div>
      <div class="table-scroll"><table><thead><tr><th>On</th><th>Source</th><th>License</th><th>Saved</th><th>Last article</th><th></th></tr></thead><tbody>
      ${st.sources.map((s) => { const n = ss.sources[s.id] || {}; return `<tr data-sid="${esc(s.id)}">
        <td><label class="switch" style="display:inline-block"><input type="checkbox" data-on ${s.enabled ? "checked" : ""}><span></span></label></td>
        <td><b>${esc(s.name)}</b><div class="hint mono" style="max-width:260px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap" title="${esc(s.url)}">${esc(KIND_LABEL[s.kind] || s.kind)} · ${esc(s.url)}</div>${s.enabled && !okSrc(s) ? '<div class="hint" style="color:var(--warn)">Skipped until you confirm the rights</div>' : ""}</td>
        <td>${licPill(s)}</td><td>${num(n.done || 0)}${n.failed ? ` <span class="hint">(${n.failed} failed)</span>` : ""}</td>
        <td class="muted">${n.last ? ago(n.last) : "—"}</td>
        <td><button class="btn" data-del title="Remove source">${I.trash}</button></td></tr>`; }).join("") || '<tr><td colspan="6" class="muted">No sources yet. Add one below.</td></tr>'}
      </tbody></table></div></div>
    <div class="card"><h2>Add a source</h2>
      <div class="grid g2" style="margin-top:10px">
        <label class="field">Link<input class="input" id="n-url" placeholder="https://example.org/feed.xml"></label>
        <label class="field">Name<input class="input" id="n-name" placeholder="Example News"></label>
        <label class="field">What is it?<select class="input" id="n-kind"><option value="feed">RSS / Atom feed (best)</option><option value="links">Page of links: read the articles it links to</option><option value="page">Single page or article</option><option value="mastodon">Mastodon: hashtag, account or server timeline</option><option value="bluesky">Bluesky: account, feed, hashtag or search</option></select></label>
        <label class="field">License of the texts<select class="input" id="n-lic">${Object.entries(L).map(([k, l]) => `<option value="${k}">${esc(l.label)}: ${esc(l.note)}</option>`).join("")}</select></label>
        <label class="field"><span id="n-max-l">Articles per run</span><input class="input" id="n-max" type="number" min="1" max="500" value="30"></label>
      </div>
      <div id="n-social" hidden style="margin-top:10px" class="hint">
        <p style="margin:0 0 8px">Reads public posts through the network's public API. Links like <span class="mono">https://server/tags/poetry</span>, <span class="mono">https://server/@user</span>, <span class="mono">https://bsky.app/profile/handle</span> or <span class="mono">https://bsky.app/hashtag/writing</span>. Replies, boosts, posts with a content warning and authors who opt out (#nobot, #noai, noindex) are skipped. Names, @mentions, links, e-mails and phone numbers are removed from the text. Posts are public but still copyrighted by their authors, so <b>Only openly licensed authors</b> is the safe choice. Reddit, X, Facebook, Instagram, Threads and LinkedIn are not offered: their terms forbid it.</p>
        <label class="row"><span class="switch"><input type="checkbox" id="n-terms"><span></span></span> <span>I checked this server's terms, and they allow downloading public posts for AI training.</span></label></div>
      <div id="n-rights" hidden style="margin-top:10px" class="hint"><label class="row"><span class="switch"><input type="checkbox" id="n-ok"><span></span></span> <span>I own these texts or have the owner's permission to use them for training, and I take responsibility for it.</span></label></div>
      <div class="row end" style="margin-top:12px"><button class="btn primary" id="n-add">${I.plus} Add source</button></div></div>
    <div class="card"><div class="card-head"><h2>Latest articles</h2>${Object.keys(ss.languages || {}).length ? `<span class="hint">${Object.entries(ss.languages).slice(0, 6).map(([k, v]) => `${esc(langName(k))} ${num(v)}`).join(" · ")}</span>` : ""}</div>
      ${ss.recent.length ? `<ul style="padding-left:18px;margin:0">${ss.recent.map((r) => `<li><a href="${esc(r.url)}" target="_blank" rel="noopener noreferrer"><u>${esc(r.title || r.url)}</u></a> <span class="hint">${esc(r.source)} · ${esc(langName(r.lang))} · ${num(r.chars)} chars · ${ago(r.fetched)}</span></li>`).join("")}</ul>` : '<p class="muted">Nothing yet. Press "Run now".</p>'}</div>
    <div class="card"><div class="card-head"><h2>Past runs</h2><span class="hint">score = quick check before → after learning; a drop of more than ${cs.crawl_max_drop} points is undone</span></div>
      ${ss.runs.length ? `<div class="table-scroll"><table><thead><tr><th>Started</th><th>Took</th><th>Added</th><th>Failed</th><th>Learning</th></tr></thead><tbody>${ss.runs.map((r) => `<tr><td>${ago(r.started)}</td><td>${r.ended ? dur(r.ended - r.started) : '<span class="pill good">● running</span>'}</td><td>${num(r.added)}</td><td>${num(r.failed)}</td><td>${r.learned ? `<b>${esc(r.learned.split(" (")[0])}</b> ${verdictPill(r)}` : '<span class="muted">—</span>'}</td></tr>`).join("")}</tbody></table></div>` : '<p class="muted">No runs yet.</p>'}</div>
    <div class="card"><div class="card-head"><h2>Log</h2></div><pre class="console" id="nlog">${esc(st.log || "No crawl yet.")}</pre></div>
  </div>
  <div style="display:flex;flex-direction:column;gap:16px">
    <div class="card"><h2>Schedule</h2>
      <label class="row" style="margin-top:10px"><span class="switch"><input type="checkbox" id="c-on" ${cs.crawl_enabled ? "checked" : ""}><span></span></span> <b>Check sources automatically</b></label>
      <label class="field" style="margin-top:12px">Every (hours)<input class="input" id="c-every" type="number" min="0.25" step="0.25" value="${cs.crawl_every_hours}"></label>
      <label class="field" style="margin-top:10px">Languages to keep<input class="input" id="c-langs" placeholder="all languages" value="${esc(cs.crawl_languages || "")}"><span class="hint">Codes separated by commas, e.g. <span class="mono">en, hy, ru</span>. Empty keeps every language.</span></label>
      <label class="field" style="margin-top:10px">Then teach this model<select class="input" id="c-model"><option value="">None: only save to the training data</option>${models.map((m) => `<option value="${esc(m.name)}" ${m.name === cs.crawl_learn_model ? "selected" : ""}>${esc(m.label)}</option>`).join("")}</select></label>
      <div class="grid g2" style="margin-top:10px"><label class="field">Train after (articles)<input class="input" id="c-min" type="number" min="1" value="${cs.crawl_min_docs}"></label>
        <label class="field">Read each (times)<input class="input" id="c-passes" type="number" min="0.5" max="10" step="0.5" value="${cs.crawl_passes}"></label></div>
      <label class="row" style="margin-top:12px;align-items:flex-start"><span class="switch" style="margin-top:2px"><input type="checkbox" id="c-gate" ${cs.crawl_gate ? "checked" : ""}><span></span></span> <span><b>Check the model after learning</b><div class="hint">Quick check (8 prompts) before and after. If the score drops by more than <input class="input sm" id="c-drop" type="number" min="0" max="100" step="1" value="${cs.crawl_max_drop}" style="width:64px;display:inline-block;padding:2px 6px"> points, the learn step is undone.</div></span></label>
      <div class="hint" style="margin-top:10px">${osSched.installed ? `Runs from ${esc(osSched.kind)}, also when the Studio is closed.` : cs.next ? `Next run ${cs.next < Date.now() / 1000 ? "as soon as no other job is running" : "in " + dur(cs.next - Date.now() / 1000)}. Runs only while the Studio is open.` : "Automatic checks are off."}</div>
      <div class="row end" style="margin-top:12px"><button class="btn" id="c-save">Save</button><button class="btn primary" id="c-run">${I.play} Run now</button></div></div>
    <div class="card"><h2>When the Studio is closed</h2>
      <p class="muted" style="margin:6px 0 10px">${osSched.installed ? `✓ Registered with <b>${esc(osSched.kind)}</b>: the crawl runs every ${Math.max(1, Math.round(cs.crawl_every_hours))} h even when this app isn't open, with the settings above as saved at that time.` : `Register the crawl with ${navigator.platform.startsWith("Win") ? "Windows Task Scheduler" : "cron"} so it runs every ${Math.max(1, Math.round(cs.crawl_every_hours))} h without the Studio. Save the settings first; they are baked into the task.`}</p>
      <div class="row end">${osSched.installed ? `<button class="btn" id="c-os-off">Remove from ${esc(osSched.kind)}</button><button class="btn ghost sm" id="c-os-on" title="Apply the current settings to the task">Update</button>` : `<button class="btn primary" id="c-os-on">${I.clock} Run when closed</button>`}</div></div>
    <div class="card"><h2>Licenses matter</h2><ul class="muted" style="padding-left:18px">
      <li>Most news sites are <b>copyrighted</b>. This project only trains on public-domain or openly licensed text, so the built-in sources are Wikinews and Global Voices (CC BY) and NASA (US government, public domain).</li>
      <li>Sources marked <span class="pill warn">Copyrighted / unknown</span> or <span class="pill warn">I have the rights</span> are skipped until you confirm the rights. You are responsible for them.</li>
      <li>Not allowed: CC BY-ND or NC-ND sites, and sites whose terms forbid text and data mining.</li>
      <li>Social posts are public but <b>copyrighted by their authors</b>. With <span class="pill good">Only openly licensed authors</span> a Mastodon or Bluesky post is kept only if its author's profile says CC0, CC BY or CC BY-SA. mastodon.social forbids collecting posts for AI training, so it is refused.</li>
      <li>Every article's link, title and license is kept in <span class="mono">data/raw/news/attribution.jsonl</span>, as CC BY requires.</li></ul></div>
    <div class="card"><h2>How it reads the web</h2><ul class="muted" style="padding-left:18px">
      <li>Obeys robots.txt and waits at least 3 seconds between requests to a site.</li>
      <li>Skips duplicates, pages with almost no text, and other languages when a filter is set.</li>
      <li>If the connection drops it resumes the download, and links it found stay queued for the next run.</li>
      <li>New articles also go into <span class="mono">data/raw/news/</span> for your next full training. Ask the model about them in Chat with the format <b>About today's news</b>.</li></ul></div>
  </div></div>`;
  const refresh = () => trainNews();
  const toggleRights = () => ($("#n-rights").hidden = !!(L[$("#n-lic").value] || {}).ok);
  const toggleKind = () => {
    const k = $("#n-kind").value, soc = k in SOCIAL;
    $("#n-social").hidden = !soc;
    $("#n-url").placeholder = SOCIAL[k] || "https://example.org/feed.xml";
    $("#n-max-l").textContent = soc ? "Posts per run" : "Articles per run";
    $$("#n-lic option").forEach((o) => { const forSocial = ["cc-stated", "own-rights", "unknown"].includes(o.value); o.hidden = soc ? !forSocial : !!(L[o.value] || {}).social; o.disabled = o.hidden; });
    if ($("#n-lic").selectedOptions[0].hidden) $("#n-lic").value = soc ? "cc-stated" : "cc-by";
    toggleRights();
  };
  $("#n-lic").onchange = toggleRights; $("#n-kind").onchange = toggleKind; toggleKind();
  $("#n-add").onclick = async () => {
    try {
      await api("/api/crawler/sources", { json: { url: $("#n-url").value, name: $("#n-name").value, kind: $("#n-kind").value, license: $("#n-lic").value, max_items: +$("#n-max").value, rights_confirmed: $("#n-ok").checked, terms_confirmed: $("#n-terms").checked } });
      toast("Source added", "ok"); refresh();
    } catch (e) { fail(e); }
  };
  $$("tr[data-sid]").forEach((tr) => {
    const sid = encodeURIComponent(tr.dataset.sid);
    $("[data-on]", tr).onchange = async (e) => { try { await api(`/api/crawler/sources/${sid}`, { method: "PATCH", json: { enabled: e.target.checked } }); } catch (x) { fail(x); refresh(); } };
    $("[data-del]", tr).onclick = async () => { if (!confirm("Remove this source? Articles already saved stay in the training data.")) return; try { await api(`/api/crawler/sources/${sid}`, { method: "DELETE" }); refresh(); } catch (x) { fail(x); } };
  });
  const settings = () => ({ enabled: $("#c-on").checked, every_hours: +$("#c-every").value, learn_model: $("#c-model").value, min_docs: +$("#c-min").value, passes: +$("#c-passes").value, languages: $("#c-langs").value, gate: $("#c-gate").checked, max_drop: +$("#c-drop").value });
  $("#c-save").onclick = async () => { try { await api("/api/crawler/settings", { json: settings() }); toast("Saved", "ok"); refresh(); } catch (e) { fail(e); } };
  $("#c-run").onclick = async () => { try { await api("/api/crawler/settings", { json: settings() }); await api("/api/crawler/run", { method: "POST" }); toast("Reading the sources…", "ok"); poll(); } catch (e) { fail(e); } };
  const os = (install) => async () => { try { await api("/api/crawler/settings", { json: settings() }); const r = await api(`/api/crawler/schedule?install=${install}`, { method: "POST" }); toast(esc(r.message), "ok"); refresh(); } catch (e) { fail(e); } };
  if ($("#c-os-on")) $("#c-os-on").onclick = os(true);
  if ($("#c-os-off")) $("#c-os-off").onclick = os(false);
  const prev = S.onStatus;
  S.onStatus = (s) => { prev && prev(s); const l = $("#nlog"); if (l && s.job && s.job.kind === "crawl") { l.textContent = s.job.log; l.scrollTop = 1e9; } };
}

// ---------------------------------------------------------------- style profiles
const STYLE_SRC = {
  "x-archive": { file: true, help: "X → Settings → Your account → <b>Download an archive of your data</b>. Upload the .zip as it is; only <span class=\"mono\">data/tweets.js</span> is read and the file is deleted right after." },
  "facebook-export": { file: true, help: "Facebook → Settings → <b>Download your information</b> → Format <b>JSON</b>, at least <i>Posts</i>. The .zip is deleted right after the import." },
  "instagram-export": { file: true, help: "Instagram → Accounts Center → Your information → <b>Download your information</b> → Format <b>JSON</b>. Captions of your posts and reels are used." },
  "mastodon-archive": { file: true, help: "Mastodon → Preferences → Import and export → <b>Request your archive</b>. Public posts from <span class=\"mono\">outbox.json</span> are used." },
  "meta-page": { id: "Page ID", token: true, help: "A Facebook Page <b>you manage</b>. Get a Page access token (permission <span class=\"mono\">pages_read_engagement</span>) in Meta's Graph API Explorer. The token is used once and not stored." },
  "instagram-account": { id: "Instagram account ID", token: true, help: "An Instagram professional account linked to a Page you manage, read through the Graph API (<span class=\"mono\">instagram_basic</span>). The token is used once and not stored." },
  "profile-link": { url: true, help: "A Mastodon or Bluesky profile, e.g. <span class=\"mono\">https://bsky.app/profile/you.bsky.social</span>. Your own (tick below), or someone whose profile states CC0, CC BY or CC BY-SA." },
};
async function trainStyle() {
  const [st, models] = await Promise.all([api("/api/styles"), api("/api/models").catch(() => [])]);
  const lora = models.filter((m) => m.type === "lora" && !m.style);
  const srcLine = (p) => p.sources.map((x) => `${esc(x.label)} +${num(x.added)}`).join(" · ");
  $("#tbody").innerHTML = `<div class="grid g3"><div class="span2" style="display:flex;flex-direction:column;gap:16px">
    <div class="card"><div class="card-head"><h2>Style profiles</h2><span class="hint">one per account · a model learns to write like it</span></div>
      ${st.profiles.length ? `<div class="table-scroll"><table><thead><tr><th>Profile</th><th>Posts</th><th>Model</th><th></th></tr></thead><tbody>
      ${st.profiles.map((p) => `<tr data-pid="${esc(p.id)}"><td><b>${esc(p.name)}</b><div class="hint">${srcLine(p)}</div></td><td>${num(p.posts || 0)}</td>
        <td>${p.training ? '<span class="pill good">● training</span>' : p.trained ? `<span class="pill good">${esc(p.run)}</span>` : '<span class="muted">not trained</span>'}</td>
        <td class="row" style="gap:6px;justify-content:flex-end"><button class="btn sm" data-sample>Examples</button>${p.trained ? `<a class="btn sm" href="#/chat" data-chat="${esc(p.run)}">${I.chat} Write</a>` : ""}<button class="btn" data-del title="Delete profile">${I.trash}</button></td></tr>
        <tr data-sample-row="${esc(p.id)}" hidden><td colspan="4"></td></tr>`).join("")}</tbody></table></div>` : '<p class="muted">No profiles yet. Import your posts below.</p>'}</div>
    <div class="card"><h2>Import posts</h2>
      <div class="grid g2" style="margin-top:10px">
        <label class="field">Profile name<input class="input" id="s-name" placeholder="e.g. Hovhannes" list="s-names"><datalist id="s-names">${st.profiles.map((p) => `<option value="${esc(p.name)}">`).join("")}</datalist><span class="hint">Same name = add to that profile</span></label>
        <label class="field">From<select class="input" id="s-kind">${Object.entries(st.kinds).map(([k, l]) => `<option value="${k}">${esc(l)}</option>`).join("")}</select></label>
      </div>
      <p class="hint" id="s-help" style="margin:10px 0"></p>
      <div class="grid g2">
        <label class="field" id="s-file-f">Export file (.zip)<input class="input" id="s-file" type="file" accept=".zip"></label>
        <label class="field" id="s-url-f">Profile link<input class="input" id="s-url" placeholder="https://bsky.app/profile/you.bsky.social"></label>
        <label class="field" id="s-id-f"><span id="s-id-l">Page ID</span><input class="input" id="s-id"></label>
        <label class="field" id="s-token-f">Access token<input class="input" id="s-token" type="password" autocomplete="off"></label>
      </div>
      <label class="row" style="margin-top:10px"><span class="switch"><input type="checkbox" id="s-mine"><span></span></span> <span>These are <b>my own posts</b> (or I have the author's written permission).</span></label>
      <label class="row" style="margin-top:6px"><span class="switch"><input type="checkbox" id="s-replies"><span></span></span> <span class="muted">Keep replies too</span></label>
      <div class="row end" style="margin-top:12px"><button class="btn primary" id="s-import">${I.plus} Import</button></div></div>
    <div class="card"><h2>Teach a model the style</h2>
      <div class="grid g2" style="margin-top:10px">
        <label class="field">Profile<select class="input" id="t-profile">${st.profiles.map((p) => `<option value="${esc(p.id)}" ${p.posts < st.min_posts ? "disabled" : ""}>${esc(p.name)} (${num(p.posts || 0)} posts${p.posts < st.min_posts ? `, need ${st.min_posts}` : ""})</option>`).join("") || "<option value=''>Import posts first</option>"}</select></label>
        <label class="field">Start from<select class="input" id="t-from"><option value="">Base model (Qwen3 0.6B)</option>${lora.map((m) => `<option value="${esc(m.name)}">${esc(m.name)}: keeps its writing skills</option>`).join("")}</select></label>
        <label class="field">Read each post (times)<input class="input" id="t-epochs" type="number" min="0.5" max="10" step="0.5" value="3"></label>
      </div>
      <p class="hint" style="margin-top:8px">Makes the model <span class="mono">style-&lt;profile&gt;</span>. Then open Chat, pick it, and the format switches to <b>Write like a style profile</b>. Give it a topic, e.g. “our new bakery opening on Saturday”. On CPU, a few hundred posts take roughly an hour.</p>
      <div class="row end" style="margin-top:12px"><button class="btn primary" id="t-go" ${st.profiles.some((p) => p.posts >= st.min_posts) ? "" : "disabled"}>${I.play} Train</button></div></div>
  </div>
  <div style="display:flex;flex-direction:column;gap:16px">
    <div class="card"><h2>Why not any public page?</h2><p class="muted" style="margin-top:6px">${esc(st.not_possible)}</p>
      <ul class="muted" style="padding-left:18px">
        <li><b>X / Twitter</b>: only your own archive. The API is paid and its terms forbid training models on X content.</li>
        <li><b>Facebook, Instagram, Threads</b>: your own export, or Pages and professional accounts you manage. There is no API for other people's posts, and scraping breaks Meta's terms.</li>
        <li><b>Mastodon, Bluesky</b>: your own profile, or any author whose profile states CC0, CC BY or CC BY-SA.</li>
        <li>Copying a living person's voice without asking can mislead readers. Use other people's styles only with their permission.</li></ul></div>
    <div class="card"><h2>What is kept</h2><ul class="muted" style="padding-left:18px">
      <li>Only your original posts with text. Reposts, retweets, replies and photo-only posts are dropped.</li>
      <li>Links, @mentions, e-mail addresses and phone numbers are removed.</li>
      <li>Uploaded exports are deleted as soon as the posts are read. Posts stay in <span class="mono">runs/_styles/</span> until you delete the profile.</li></ul></div>
  </div></div>`;
  const refresh = () => trainStyle();
  const show = () => {
    const c = STYLE_SRC[$("#s-kind").value] || {};
    $("#s-help").innerHTML = c.help || "";
    $("#s-file-f").hidden = !c.file; $("#s-url-f").hidden = !c.url; $("#s-id-f").hidden = !c.id; $("#s-token-f").hidden = !c.token;
    $("#s-id-l").textContent = c.id || "";
  };
  $("#s-kind").onchange = show; show();
  $("#s-import").onclick = async () => {
    const fd = new FormData();
    const name = $("#s-name").value.trim();
    if (!name) return fail(new Error("Give the profile a name."));
    const known = st.profiles.find((p) => p.name.toLowerCase() === name.toLowerCase());
    fd.append("name", name); fd.append("profile", known ? known.id : name); fd.append("kind", $("#s-kind").value);
    fd.append("mine", $("#s-mine").checked); fd.append("keep_replies", $("#s-replies").checked);
    fd.append("url", $("#s-url").value); fd.append("page_id", $("#s-id").value); fd.append("token", $("#s-token").value);
    if ($("#s-file").files[0]) fd.append("file", $("#s-file").files[0]);
    $("#s-import").disabled = true; $("#s-import").textContent = "Importing…";
    try {
      const r = await api("/api/styles/import", { method: "POST", body: fd });
      const dr = Object.entries(r.dropped).filter(([, v]) => v).map(([k, v]) => `${v} ${k}`).join(", ");
      toast(`${num(r.added)} posts added to ${esc(r.profile.name)}${dr ? ` (skipped: ${esc(dr)})` : ""}`, "ok");
      refresh();
    } catch (e) { fail(e); $("#s-import").disabled = false; $("#s-import").textContent = "Import"; }
  };
  $("#t-go").onclick = async () => {
    try {
      await api(`/api/styles/${encodeURIComponent($("#t-profile").value)}/train`, { json: { start_from: $("#t-from").value, epochs: +$("#t-epochs").value } });
      toast("Training started", "ok"); location.hash = "#/training/runs";
    } catch (e) { fail(e); }
  };
  $$("tr[data-pid]").forEach((tr) => {
    const pid = tr.dataset.pid;
    $("[data-del]", tr).onclick = async () => {
      if (!confirm("Delete this profile and its imported posts?")) return;
      const alsoModel = tr.innerHTML.includes("style-") && confirm("Also delete the model trained on it?");
      try { await api(`/api/styles/${encodeURIComponent(pid)}?model=${alsoModel}`, { method: "DELETE" }); refresh(); } catch (e) { fail(e); }
    };
    $("[data-sample]", tr).onclick = async () => {
      const row = $(`tr[data-sample-row="${CSS.escape(pid)}"]`);
      if (!row.hidden) return void (row.hidden = true);
      try { const ps = await api(`/api/styles/${encodeURIComponent(pid)}/sample`); row.firstElementChild.innerHTML = ps.map((p) => `<div class="hint" style="white-space:pre-wrap;border-left:3px solid var(--border);padding:4px 10px;margin:6px 0">${esc(p.text)}</div>`).join("") || '<span class="muted">No posts</span>'; row.hidden = false; } catch (e) { fail(e); }
    };
    const w = $("[data-chat]", tr);
    if (w) w.onclick = () => { try { localStorage.setItem("chatModel", w.dataset.chat); localStorage.setItem("chatFormat", "style"); } catch {} };
  });
}

// ---------------------------------------------------------------- corpus
async function trainCorpus() {
  let c;
  try { c = await api("/api/corpus"); } catch (e) { return fail(e); }
  $("#tbody").innerHTML = `<div class="grid g3"><div class="card span2"><div class="card-head"><h2>What the models learn from</h2><span class="hint">${num(c.total_mb)} MB of text in data/raw · ${c.encoded ? num(c.encoded / 1e6, 1) + "M tokens encoded" : "not encoded yet"}</span></div>
    ${c.sources.length ? `<div class="table-scroll"><table><thead><tr><th>Source</th><th>Files</th><th>Size</th><th>Documents</th><th>≈ Tokens</th><th>Updated</th><th></th></tr></thead><tbody>
      ${c.sources.map((s) => `<tr data-src="${esc(s.source)}"><td><b>${esc(s.source)}</b>${s.note ? `<div class="hint">${esc(s.note)}</div>` : ""}</td><td>${num(s.files)}</td><td>${num(s.mb, 1)} MB</td><td>${s.docs_approx ? "≈ " : ""}${num(s.docs)}</td><td>${num(s.tokens_est / 1e6, 1)}M</td><td class="muted">${ago(s.updated)}</td>
        <td style="white-space:nowrap"><button class="btn sm" data-sample title="Show a few random documents">Sample</button> <button class="btn sm ghost danger" data-del title="Delete this folder">${I.trash}</button></td></tr>`).join("")}
    </tbody></table></div>` : '<p class="muted">No text yet. Download books under Data, add your own under Your texts, or collect articles under News & web.</p>'}
    <div id="samples" style="margin-top:12px"></div></div>
    <div class="card"><h2>Good to know</h2><ul class="muted" style="padding-left:18px">
      <li>Each source is a folder under <span class="mono">data/raw/</span>. Training reads all of them.</li>
      <li>After deleting a source, run <b>Build the tokenizer</b> and <b>Encode the corpus</b> again (Data tab) so the change reaches from-scratch training. Fine-tuning reads the folders directly.</li>
      <li>Deleting a news source's folder also lets the crawler fetch those articles again.</li>
      <li>Models already trained keep what they learned; deleting text does not untrain them.</li></ul></div></div>`;
  $$("tr[data-src]").forEach((tr) => {
    const src = tr.dataset.src;
    $("[data-sample]", tr).onclick = async () => {
      const box = $("#samples");
      box.innerHTML = '<div class="hint">Reading…</div>';
      try {
        const r = await api(`/api/corpus/sample?source=${encodeURIComponent(src)}&n=3`);
        box.innerHTML = `<div class="card-head"><h2 style="font-size:15px">Samples from ${esc(src)}</h2><button class="btn ghost sm" id="samples-close">✕</button></div>` + (r.samples.length ? r.samples.map((s) => `<div class="hint mono" style="margin-top:8px">${esc(s.file)} · ${num(s.chars)} chars</div><pre class="console" style="max-height:200px;white-space:pre-wrap">${esc(s.text)}${s.chars > s.text.length ? " …" : ""}</pre>`).join("") : '<p class="muted">Empty.</p>');
        $("#samples-close").onclick = () => (box.innerHTML = "");
      } catch (e) { box.innerHTML = ""; fail(e); }
    };
    $("[data-del]", tr).onclick = async () => {
      if (!confirm(`Delete the folder data/raw/${src} and all its text?\n\nThis cannot be undone. Models already trained keep what they learned.`)) return;
      try { const r = await api(`/api/corpus/${src}`, { method: "DELETE" }); toast(esc(r.note), "ok"); trainCorpus(); } catch (e) { fail(e); }
    };
  });
}

const IMPORT_BAR = `<div class="row" style="justify-content:space-between;margin-bottom:12px"><span class="hint">Moving from another PC? Export the model there (open it → Export), then import the .zip here and press Resume.</span>
  <label class="btn sm">${I.upload} Import a model (.zip)<input type="file" id="imp-file" accept=".zip" hidden></label></div>`;

function wireImport() {
  const inp = $("#imp-file");
  if (!inp) return;
  inp.onchange = async () => {
    const f = inp.files[0];
    if (!f) return;
    const fd = new FormData(); fd.append("file", f);
    toast(`Importing <b>${esc(f.name)}</b>…`);
    try {
      let r;
      try { r = await api("/api/runs/import", { method: "POST", body: fd }); }
      catch (e) {
        if (!/already exists/.test(e.message)) throw e;
        const nn = prompt(`${e.message}\n\nType another name to keep both, or leave it empty to replace the existing one:`, "");
        if (nn === null) return;
        const fd2 = new FormData(); fd2.append("file", f);
        if (nn.trim()) fd2.append("name", nn.trim()); else fd2.append("force", "true");
        r = await api("/api/runs/import", { method: "POST", body: fd2 });
      }
      const warn = (r.log || "").split("\n").filter((l) => /^(Note|Warning)/.test(l));
      toast(`Imported <b>${esc(r.name)}</b>. Open it and press Resume to keep training.${warn.map((w) => "<br>" + esc(w)).join("")}`, warn.length ? "" : "ok");
      route(); setTimeout(() => openRun(r.name), 400);
    } catch (e) { fail(e); } finally { inp.value = ""; }
  };
}

function drawRuns(rs) {
  const body = $("#tbody");
  if (!body) return;
  if (!rs.length) { body.innerHTML = IMPORT_BAR + `<div class="card empty"><div class="big">🏁</div>No runs yet.<br><a class="btn primary" style="margin-top:14px" href="#/training/new">Start a training</a></div>`; wireImport(); return; }
  body.innerHTML = IMPORT_BAR + `<div class="card"><table><thead><tr><th>Model</th><th>Type</th><th>Progress</th><th>Level</th><th>Val loss</th><th>Updated</th></tr></thead><tbody>
    ${rs.map((r) => `<tr class="click" data-run="${esc(r.name)}"><td><b>${esc(r.name)}</b> ${r.running ? '<span class="pill good">● running</span>' : r.ready ? "" : '<span class="pill">no weights yet</span>'}</td>
      <td>${typePill(r.type)}</td><td style="min-width:160px"><div class="bar"><i style="width:${(r.progress * 100).toFixed(1)}%"></i></div><div class="hint">${num(r.step)} / ${num(r.max_steps)}${r.eta_sec ? " · ETA " + dur(r.eta_sec) : ""}</div></td>
      <td>${levelBar(r.level)}</td><td>${num(r.val_loss, 3)}</td><td class="muted">${ago(r.updated)}</td></tr>`).join("")}</tbody></table></div>`;
  $$("tr[data-run]").forEach((t) => (t.onclick = () => openRun(t.dataset.run)));
  wireImport();
}

async function openRun(name) {
  let timer;
  const close = modal(`<div class="row" style="justify-content:space-between"><div><h1 style="font-size:22px" id="r-title"></h1><div class="muted" id="r-sub"></div></div><div class="row" id="r-btns"></div></div>
    <div class="stats" id="r-stats"></div><div id="r-opts"></div><div class="card"><div class="card-head"><h2>Loss</h2><span class="hint">training · validation · learning from ratings</span></div><div id="r-chart"></div></div>
    <div><div class="card-head"><h2>Log</h2></div><pre class="console" id="r-log"></pre></div>`, (m) => {
    let optsDrawn = false;
    const drawOpts = async (s) => {
      optsDrawn = true;
      const fm = Object.fromEntries((await api("/api/formats").catch(() => [])).map((f) => [f.id, f.label]));
      $("#r-opts", m).innerHTML = `<div class="card">${evalOptsHtml(fm)}${s.type === "lora" ? `<details style="margin-top:8px"><summary>Fast CPU copy options</summary><div class="row" style="margin-top:8px"><label class="field" style="width:260px">Quality<select class="input" id="r-quant"><option value="Q4_K_M">Q4_K_M · smallest, fastest (default)</option><option value="Q5_K_M">Q5_K_M · a bit better</option><option value="Q6_K">Q6_K · close to the original</option><option value="Q8_0">Q8_0 · near lossless, biggest</option></select></label></div><div class="hint">Making the copy again replaces the old one.</div></details>` : ""}</div>`;
    };
    const load = async () => {
      if (!document.body.contains(m)) return clearInterval(timer);
      let d;
      try { d = await api(`/api/runs/${encodeURIComponent(name)}`); } catch (e) { clearInterval(timer); return fail(e); }
      const s = d.summary;
      if (!optsDrawn && s.ready) drawOpts(s);
      $("#r-title", m).textContent = s.name;
      $("#r-sub", m).innerHTML = `${typePill(s.type)} ${s.base ? esc(s.base) : s.preset ? esc(s.preset) + " size" : ""} ${s.running ? '<span class="pill good">● training</span>' : ""}`;
      $("#r-btns", m).innerHTML = `${s.ready ? `<button class="btn sm primary" id="r-chat">${I.chat} Chat</button>` : ""}${s.ready && !s.running ? `<button class="btn sm" id="r-eval" title="Score it on the 60 fixed prompts">Evaluate</button><button class="btn sm ghost" id="r-quick" title="8 prompts, a few minutes">Quick check</button>` : ""}${s.type === "lora" && s.ready && !s.running ? `<button class="btn sm ${s.fast ? "ghost" : ""}" id="r-gguf" title="Merge, convert to GGUF and quantize (quality under Fast CPU copy options)">${s.fast ? "⚡ Fast copy ✓ (redo)" : "⚡ Make fast CPU copy"}</button>` : ""}${s.ready && !s.running ? `<button class="btn sm" id="r-resume" title="Continue training from the last checkpoint">${I.play} Resume</button><button class="btn sm ghost" id="r-export" title="Download as one .zip to continue on another PC">${I.download} Export</button>` : ""}${s.running ? `<button class="btn sm danger" id="r-stop">${I.stop} Stop</button>` : `<button class="btn sm ghost" id="r-del">${I.trash}</button>`}`;
      $("#r-stats", m).innerHTML = [
        ["Progress", `${(s.progress * 100).toFixed(0)}%`], ["Step", `${num(s.step)} / ${num(s.max_steps)}`], ["ETA", s.running ? dur(s.eta_sec) : "—"],
        ["Train loss", num(s.train_loss, 3)], ["Val loss", num(s.val_loss, 3)], ["Perplexity", num(s.perplexity, 1)],
        ["Speed", s.tokens_per_sec ? num(s.tokens_per_sec) + " tok/s" : "—"], ["Trained for", dur(s.train_seconds)],
        ["Eval score", s.eval_score != null ? `${num(s.eval_score, 1)}${s.eval_quick ? " (quick)" : ""}` : "—"],
      ].map(([l, v]) => `<div class="stat"><div class="l">${l}</div><div class="v">${v}</div></div>`).join("") + `<div class="stat" style="grid-column:span 2"><div class="l">Level</div><div style="margin-top:6px">${levelBar(s.level)}</div></div>`;
      const kinds = [["train", "Training", Charts.PALETTE[0]], ["val", "Validation", Charts.PALETTE[1]], ["feedback", "From ratings", Charts.PALETTE[2]]];
      Charts.line($("#r-chart", m), kinds.map(([k, n, c]) => ({ name: n, color: c, area: k === "train", points: d.metrics.filter((x) => x.kind === k).map((x) => ({ x: x.step, y: x.loss })) })).filter((x) => x.points.length), { xLabel: "step", empty: "Waiting for the first steps…" });
      const log = $("#r-log", m); const atEnd = log.scrollTop + log.clientHeight >= log.scrollHeight - 30;
      log.textContent = d.log || "No log yet."; if (atEnd) log.scrollTop = 1e9;
      const on = (id, f) => { const b = $(id, m); if (b) b.onclick = f; };
      on("#r-chat", () => { close(); localStorage.setItem("chatModel", s.name); location.hash = "#/chat"; });
      on("#r-eval", () => startEval(s.name, false, readEvalOpts(m)).then(load));
      on("#r-quick", () => startEval(s.name, true, readEvalOpts(m)).then(load));
      on("#r-gguf", async () => { try { await api(`/api/runs/${encodeURIComponent(s.name)}/gguf?quant=${encodeURIComponent($("#r-quant", m)?.value || "Q4_K_M")}`, { method: "POST" }); toast("Making the fast CPU copy…", "ok"); poll(); load(); } catch (e) { fail(e); } });
      on("#r-stop", async () => { await api("/api/stop", { method: "POST" }).catch(fail); toast("Stopped. The last checkpoint is kept."); load(); });
      on("#r-resume", async () => {
        try { await api("/api/train", { json: { mode: s.type, name: s.name, resume: true, preset: s.preset || "cpu", base: s.base || undefined, steps: s.max_steps, vocab: S.status?.data?.vocab || 8192 } }); toast("Resumed", "ok"); load(); } catch (e) { fail(e); }
      });
      on("#r-export", () => {
        const data = s.type === "scratch" && confirm(`Export ${s.name}.\n\nOK = also include the training data (bigger file; the other PC continues on exactly the same text).\nCancel = model only.`);
        const link = document.createElement("a");
        link.href = `/api/runs/${encodeURIComponent(s.name)}/export${data ? "?data=true" : ""}`;
        link.download = ""; document.body.appendChild(link); link.click(); link.parentNode.removeChild(link);
        toast("Packing the model… the download starts in a moment.", "ok");
      });
      on("#r-del", async () => { if (!confirm(`Delete ${s.name} and its weights?`)) return; await api(`/api/runs/${encodeURIComponent(s.name)}`, { method: "DELETE" }).catch(fail); close(); route(); });
    };
    load();
    timer = setInterval(load, 4000);
  });
}

// Evaluate options (evaluate.py --formats/--limit/--max-tokens). A subset run is not comparable,
// so the promotion gate ignores it.
const evalOptsHtml = (fmts) => `<details class="evalopts"><summary>Evaluation options</summary><div class="row" style="margin-top:8px;flex-wrap:wrap;align-items:flex-start">
  <label class="field" style="width:240px">Only these formats<select class="input" data-eo="formats" multiple size="4">${Object.entries(fmts).filter(([k]) => k !== "free").map(([k, l]) => `<option value="${esc(k)}">${esc(l)}</option>`).join("")}</select><span class="hint">None selected = all. Ctrl/⌘-click for several.</span></label>
  <label class="field" style="width:150px">At most (prompts)<input class="input" data-eo="limit" type="number" min="0" value="0"><span class="hint">0 = all</span></label>
  <label class="field" style="width:170px">Max answer length<input class="input" data-eo="max" type="number" min="0" step="16" value="0"><span class="hint">tokens · 0 = format default · lower is faster</span></label></div>
  <div class="hint">Runs with only some formats or prompts are saved but not used by the promotion gate.</div></details>`;
const readEvalOpts = (root) => {
  const box = root && $(".evalopts", root);
  if (!box) return {};
  return { formats: [...$('[data-eo="formats"]', box).selectedOptions].map((o) => o.value).join(","), limit: +$('[data-eo="limit"]', box).value || 0, max_tokens: +$('[data-eo="max"]', box).value || 0 };
};
async function startEval(name, quick, opts = {}) {
  try { await api("/api/eval", { json: { model: name, quick, ...opts } }); toast(`${quick ? "Quick check" : "Evaluation"} of <b>${esc(name)}</b> started`, "ok"); poll(); }
  catch (e) { fail(e); }
}

// ================================================================ CHAT
const SUGGEST = [
  ["Short story", "A lighthouse keeper finds a message in a bottle", "story"],
  ["LinkedIn post", "I trained my own language model on public-domain books", "linkedin_post"],
  ["Essay", "Why reading old books still matters", "essay"],
  ["X post", "Our small bakery now opens at 7am on weekdays", "x_post"],
  ["Today's news", "What happened in space exploration this week?", "news"],
];
// temp/max: null = use the chosen format's defaults (writing/formats.json)
const C = { conv: null, models: [], busy: false, temp: null, max: null, prefsFor: null, formats: [], format: "free", fields: {} };
const fmtLabel = (id) => (C.formats.find((f) => f.id === id) || {}).label || id;

pages.chat = async () => {
  main.innerHTML = `<div class="chat">
    <div class="convs"><div class="convs-head"><button class="btn primary" style="width:100%" id="newchat">${I.plus} New chat</button></div><div class="convs-list" id="convs"></div></div>
    <div class="thread">
      <div class="thread-head"><select class="input" id="model"></select><select class="input" id="format" title="What to write" style="max-width:190px"></select><span id="model-info" class="row" style="gap:6px"></span><span class="spacer"></span>
        <div style="position:relative"><button class="iconbtn" id="settings-btn" title="Settings">${I.sliders}</button><div class="popover" id="settings" hidden>
          <label class="field"><span>Creativity <b id="tv">auto</b></span><input type="range" id="temp" min="0" max="1.5" step="0.05" value="0.8"><span class="hint">Low = focused and predictable · high = surprising</span></label>
          <label class="field"><span>Max length <b id="mv">auto</b> tokens</span><input type="range" id="max" min="32" max="1024" step="16" value="400"></label>
          <button class="btn sm ghost" id="samp-reset">Use the format's defaults</button></div></div></div>
      <div class="msgs" id="msgs"><div class="msgs-inner" id="inner"></div></div>
      <div class="composer"><div class="fmt-fields" id="fmt-fields"></div><div class="composer-box"><textarea id="input" rows="1" placeholder="Ask for a story, a post, an article…"></textarea><button class="send" id="send" title="Send">${I.send}</button></div>
        <div class="note">Pick a format for posts, stories and essays · rate answers with ★ — each rating teaches the model what you like.</div></div>
    </div></div>`;
  [C.models, C.formats] = await Promise.all([api("/api/models").catch(() => []), api("/api/formats").catch(() => [])]);
  const pref = (() => { try { return localStorage.getItem("chatModel"); } catch { return null; } })() || S.user.prefs.model || (C.models.find((m) => m.default) || {}).name;
  $("#model").innerHTML = C.models.length ? C.models.map((m) => `<option value="${esc(m.name)}" ${m.name === pref ? "selected" : ""}>${esc(m.name)} · ${m.style ? `writes like ${esc(m.style.name)}` : m.type === "lora" ? "fine-tuned" : "from scratch"}${m.default ? " · default" : ""}</option>`).join("") : `<option value="">No trained models yet</option>`;
  const modelInfo = () => {
    const m = C.models.find((x) => x.name === $("#model").value);
    $("#model-info").innerHTML = m ? `${m.type === "lora" ? "" : levelBar(m.level)} <span class="pill">${starsText(m.avg_rating)} · ${m.ratings} ratings</span>` : "";
  };
  const styleFormat = () => {   // a style model writes best with the format it was trained with
    const m = C.models.find((x) => x.name === $("#model").value);
    if (m && m.style && C.formats.some((f) => f.id === "style") && $("#format")) { C.format = "style"; $("#format").value = "style"; drawFields(); drawSampling(); }
  };
  $("#model").onchange = () => { try { localStorage.setItem("chatModel", $("#model").value); } catch {} modelInfo(); styleFormat(); };
  modelInfo();
  $("#settings-btn").onclick = (e) => { e.stopPropagation(); $("#settings").hidden = !$("#settings").hidden; };
  document.addEventListener("click", (e) => { const p = $("#settings"); if (p && !p.contains(e.target)) p.hidden = true; }, { once: false });
  $("#temp").oninput = (e) => ($("#tv").textContent = C.temp = +e.target.value);
  $("#max").oninput = (e) => ($("#mv").textContent = C.max = +e.target.value);
  $("#samp-reset").onclick = () => { C.temp = C.max = null; drawSampling(); };
  const groups = [...new Set(C.formats.map((f) => f.group))];
  $("#format").innerHTML = groups.map((g) => `<optgroup label="${esc(g)}">${C.formats.filter((f) => f.group === g).map((f) => `<option value="${f.id}">${esc(f.label)}</option>`).join("")}</optgroup>`).join("");
  try { C.format = localStorage.getItem("chatFormat") || C.format; } catch {}
  if (!C.formats.some((f) => f.id === C.format)) C.format = "free";
  $("#format").value = C.format;
  $("#format").onchange = () => { C.format = $("#format").value; try { localStorage.setItem("chatFormat", C.format); } catch {} drawFields(); drawSampling(); };
  drawFields(); drawSampling(); styleFormat();
  const input = $("#input");
  input.oninput = () => { input.style.height = "auto"; input.style.height = Math.min(input.scrollHeight, 200) + "px"; };
  input.onkeydown = (e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); } };
  $("#send").onclick = () => send();
  $("#newchat").onclick = () => { C.conv = null; drawConvs(); drawThread([]); input.focus(); };
  await drawConvs();
  if (C.conv) openConv(C.conv); else drawThread([]);
};

function drawSampling() {
  const f = C.formats.find((x) => x.id === C.format);
  const s = f ? f.sampling : { temperature: 0.8, max_tokens: 400 };
  $("#temp").value = C.temp ?? s.temperature; $("#tv").textContent = C.temp ?? `auto (${s.temperature})`;
  $("#max").value = C.max ?? s.max_tokens; $("#mv").textContent = C.max ?? `auto (${s.max_tokens})`;
}
function drawFields() {
  const f = C.formats.find((x) => x.id === C.format);
  const box = $("#fmt-fields");
  if (!f || !f.fields.length) { box.innerHTML = ""; return; }
  const lim = f.chars ? `${num(f.chars)} characters max` : f.words ? `${f.words[0]}–${f.words[1]} words` : "";
  box.innerHTML = `<span class="pill">${esc(f.label)}${lim ? " · " + lim : ""}</span>` + f.fields.map((x) => `<input class="input sm" data-field="${x.id}" placeholder="${esc(x.label)}" value="${esc(C.fields[x.id] || "")}">`).join("");
  $$("[data-field]", box).forEach((i) => (i.oninput = () => (C.fields[i.dataset.field] = i.value)));
}

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
      <div class="chips">${SUGGEST.map(([t, s, f]) => `<div class="chip" data-p="${esc(s)}" data-f="${f}"><b>${t}</b><span>${esc(s)}</span></div>`).join("")}</div></div>`;
    $$(".chip").forEach((c) => (c.onclick = () => { C.format = c.dataset.f; $("#format").value = C.format; drawFields(); drawSampling(); $("#input").value = c.dataset.p; send(); }));
    return;
  }
  const answers = {};
  messages.filter((m) => m.role === "assistant").forEach((m) => (answers[m.prompt_id] = answers[m.prompt_id] || []).push(m));
  inner.innerHTML = messages.filter((m) => m.role === "user").map((u) => userHtml(u) + (answers[u.id] || []).map((a, i, arr) => answerHtml(a, arr.length > 1 ? `Answer ${i + 1} of ${arr.length}` : "")).join("")).join("");
  bindAnswers();
  $("#msgs").scrollTop = 1e9;
}
const userHtml = (u) => {
  const fl = u.fields ? (() => { try { return JSON.parse(u.fields); } catch { return {}; } })() : {};
  const tag = u.format && u.format !== "free" ? `<div class="fmt-tag">${esc(fmtLabel(u.format))}${Object.values(fl).length ? " · " + esc(Object.values(fl).join(" · ")) : ""}</div>` : "";
  return `<div class="msg user" data-uid="${u.id}"><div class="bubble">${tag}${esc(u.content)}</div></div>`;
};
const checkPill = (score) => score == null ? "" : `<span class="pill ${score >= 100 ? "good" : score >= 70 ? "" : "warn"}" title="Automatic format checks: length, hashtags, hook, repetition">checks ${Math.round(score)}%</span>`;
const fmtBody = (t) => esc(t).split(/\n{2,}/).map((p) => `<p>${p}</p>`).join("");
const sourcesHtml = (list) => !list ? "" : list.length ? `<div class="hint" style="margin-top:6px">Based on: ${list.map((x) => `<a href="${esc(x.url)}" target="_blank" rel="noopener noreferrer" title="${esc(x.source)}"><u>${esc(x.title)}</u></a>`).join(" · ")}</div>` : '<div class="hint" style="margin-top:6px">No recent articles collected yet · <a href="#/training/news"><u>News & web</u></a></div>';
function answerHtml(a, alt = "") {
  const fl = a.fields ? (() => { try { return JSON.parse(a.fields); } catch { return {}; } })() : {};
  return `<div class="msg" data-mid="${a.id}" data-uid="${a.prompt_id}"><div class="avatar">${I.logo}</div><div class="answer">
    ${alt ? `<div class="alt">${alt}</div>` : ""}<div class="body">${fmtBody(a.content)}</div>${sourcesHtml(fl.sources)}
    ${a.correction ? `<div class="correction"><b>Your better answer · the model learns from this</b>${esc(a.correction)}</div>` : ""}
    <div class="actions"><span class="stars" title="Rate this answer">${[1, 2, 3, 4, 5].map((n) => `<button data-r="${n}" class="${a.rating >= n ? "on" : ""}" title="${["Bad", "Weak", "OK", "Good", "Great"][n - 1]}">${I.star}</button>`).join("")}</span>
      <button class="act" data-act="improve">${I.edit} Improve</button><button class="act" data-act="regen">${I.redo} Try again</button><button class="act" data-act="copy">${I.copy} Copy</button>
      ${checkPill(a.check_score)}<span class="hint" data-status>${a.trained_in ? (a.trained_in.startsWith("gate-rejected") ? "held back by the quality gate" : "✓ learned") : a.rating ? "Rated · will be learned" : ""}</span></div>
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
    inner.insertAdjacentHTML("beforeend", userHtml({ content: text, format: C.format, fields: JSON.stringify(C.fields) }));
    input.value = ""; input.style.height = "auto";
  }
  const holder = document.createElement("div");
  holder.className = "msg";
  holder.innerHTML = `<div class="avatar">${I.logo}</div><div class="answer"><div class="hint" data-live-status></div><details class="outline" hidden><summary>Plan</summary><pre></pre></details><div class="body cursor"></div><div class="sources"></div></div>`;
  if (regenUid) {
    const sibs = $$(`.msg[data-uid="${regenUid}"]`);
    sibs[sibs.length - 1].after(holder);
  } else inner.appendChild(holder);
  const body = $(".body", holder);
  $("#msgs").scrollTop = 1e9;
  let full = "", done = null;
  try {
    const r = await fetch("/api/chat", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ model, message: text, conversation_id: C.conv, regenerate: regenUid || null, format: C.format, fields: C.fields, temperature: C.temp, max_tokens: C.max }) });
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
        if (ev.type === "text") { full = ev.text; body.innerHTML = fmtBody(full); const m = $("#msgs"); if (m.scrollHeight - m.scrollTop - m.clientHeight < 120) m.scrollTop = 1e9; }
        if (ev.type === "status") $("[data-live-status]", holder).textContent = ev.text;
        if (ev.type === "sources") $(".sources", holder).innerHTML = sourcesHtml(ev.sources);
        if (ev.type === "outline") { const o = $(".outline", holder); o.hidden = false; $("pre", o).textContent = ev.text; $("[data-live-status]", holder).textContent = "Writing from the plan…"; }
        if (ev.type === "check") $("[data-live-status]", holder).innerHTML = checkPill(ev.check.score);
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
  let a, ev;
  try { [a, ev] = await Promise.all([api("/api/analytics"), api("/api/eval")]); } catch (e) { return fail(e); }
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
    ${evalSection(a, ev)}
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
  bindEval(a, ev);
};

// ---------------------------------------------------------------- evaluation, blind A/B, promotion
function evalSection(a, ev) {
  const fmts = Object.keys(ev.formats).filter((f) => f !== "free");
  const ready = a.runs.filter((r) => r.ready);
  const opts = (sel) => ready.map((r) => `<option value="${esc(r.name)}" ${r.name === sel ? "selected" : ""}>${esc(r.name)}${r.name === ev.default_model ? " (default)" : ""}</option>`).join("");
  const byModel = Object.fromEntries(ev.evals.map((e) => [e.model, e.latest]));
  const cell = (v) => v == null ? '<td class="faint">—</td>' : `<td><span class="pill ${v >= 90 ? "good" : v >= 70 ? "" : "warn"}">${num(v, 0)}</span></td>`;
  const pf = a.per_format || [];
  return `<div class="card"><div class="card-head"><h2>Writing quality</h2><span class="hint">automatic checks on the fixed prompt set · 0–100 per format</span></div>
    <div class="table-scroll"><table><thead><tr><th>Model</th><th>Overall</th>${fmts.map((f) => `<th>${esc(ev.formats[f])}</th>`).join("")}<th>Evaluated</th><th></th></tr></thead><tbody>
      ${ready.map((r) => { const l = byModel[r.name]; return `<tr><td><b>${esc(r.name)}</b> ${r.name === ev.default_model ? '<span class="pill good">default</span>' : ""} ${r.fast ? '<span class="pill" title="Chat uses the quantized GGUF copy">⚡ fast</span>' : ""}</td>${cell(l?.score)}${fmts.map((f) => cell(l?.per_format?.[f])).join("")}
        <td class="muted">${l ? `${ago(l.time)}${l.quick ? " · quick" : ""}` : "never"}</td><td style="text-align:right;white-space:nowrap"><button class="btn sm" data-eval="${esc(r.name)}">Evaluate</button> <button class="btn sm ghost" data-quick="${esc(r.name)}">Quick</button></td></tr>`; }).join("") || `<tr><td colspan="${fmts.length + 4}" class="empty">No trained models yet</td></tr>`}
    </tbody></table></div>${evalOptsHtml(ev.formats)}</div>
  <div class="grid g2">
    <div class="card" style="display:flex;flex-direction:column;gap:12px"><div class="card-head"><h2>Compare and promote</h2></div>
      <p class="hint" style="margin:0">A new model becomes the default only if it wins at least 55% of blind votes and no format's checks drop by more than 5 points.</p>
      <div class="grid g2"><label class="field">New model<select class="input" id="g-cand">${opts(ready[0]?.name)}</select></label>
        <label class="field">Current model<select class="input" id="g-cur">${opts(ev.default_model || ready[1]?.name)}</select></label></div>
      <label class="field">Judge model (optional)<input class="input" id="g-judge" placeholder="e.g. Qwen/Qwen3-4B or /path/to/judge.gguf · empty = you vote"><span class="hint">A bigger Apache 2.0 model votes on every pair, in both orders, to cancel position bias.</span></label>
      <div class="row"><button class="btn" id="g-pairs">Make A/B pairs</button><button class="btn" id="g-check">Check the gate</button><span class="spacer"></span><button class="btn primary" id="g-promote">Make default</button></div>
      <div id="g-result" class="muted"></div></div>
    <div class="card" style="display:flex;flex-direction:column;gap:12px"><div class="card-head"><h2>Blind vote</h2><span class="hint" id="v-left"></span></div><div id="vote"></div>
      ${ev.pairs.length ? `<table><thead><tr><th>New vs current</th><th>Votes</th><th>New wins</th></tr></thead><tbody>${ev.pairs.map((p) => `<tr><td>${esc(p.run_a)} vs ${esc(p.run_b)}</td><td>${p.votes} <span class="faint">(${p.open} open)</span></td><td>${p.win_rate == null ? "—" : Math.round(p.win_rate * 100) + "%"}</td></tr>`).join("")}</tbody></table>` : ""}</div>
  </div>
  <div class="card"><div class="card-head"><h2>Ratings by format</h2><span class="hint">from Chat</span></div><table><thead><tr><th>Format</th><th>Answers</th><th>Rated</th><th>Avg rating</th><th>Avg checks</th><th>Corrected</th></tr></thead><tbody>
    ${pf.map((r) => `<tr><td>${esc(ev.formats[r.format] || r.format)}</td><td>${num(r.answers)}</td><td>${num(r.rated)}</td><td>${starsText(r.avg)}</td><td>${r.checks == null ? "—" : num(r.checks, 0) + "%"}</td><td>${num(r.corrected)} ${r.answers ? `<span class="faint">(${Math.round((r.corrected / r.answers) * 100)}%)</span>` : ""}</td></tr>`).join("") || `<tr><td colspan="6" class="empty">No answers yet</td></tr>`}
  </tbody></table></div>`;
}
function bindEval(a, ev) {
  $$("[data-eval]").forEach((b) => (b.onclick = () => startEval(b.dataset.eval, false, readEvalOpts(b.closest(".card")))));
  $$("[data-quick]").forEach((b) => (b.onclick = () => startEval(b.dataset.quick, true, readEvalOpts(b.closest(".card")))));
  const cand = () => $("#g-cand").value, cur = () => $("#g-cur").value;
  const showGate = (g) => {
    const verdict = g.pass === true ? '<span class="pill good">passes</span>' : g.pass === false ? '<span class="pill bad">does not pass</span>' : '<span class="pill warn">not enough evidence</span>';
    $("#g-result").innerHTML = `${verdict}<ul style="padding-left:18px;margin:8px 0 0">${g.reasons.map((r) => `<li>${esc(r)}</li>`).join("")}</ul>`;
  };
  $("#g-pairs").onclick = async () => {
    if (!cand() || cand() === cur()) return toast("Pick two different models.", "err");
    try { await api("/api/pairs", { json: { candidate: cand(), current: cur(), judge: $("#g-judge").value } }); toast("Making pairs (evaluates any model without results first)…", "ok"); poll(); } catch (e) { fail(e); }
  };
  $("#g-check").onclick = async () => { try { showGate(await api(`/api/gate?candidate=${encodeURIComponent(cand())}&current=${encodeURIComponent(cur())}`)); } catch (e) { fail(e); } };
  $("#g-promote").onclick = async () => {
    try { await api("/api/promote", { json: { model: cand() } }); toast(`<b>${esc(cand())}</b> is now the default model`, "ok"); route(); }
    catch (e) {
      if (confirm(`${e.message}\n\nMake it the default anyway?`)) { await api("/api/promote", { json: { model: cand(), force: true } }).catch(fail); route(); }
    }
  };
  const drawVote = (r) => {
    $("#v-left").textContent = r.left ? `${r.left} pairs left` : "";
    const p = r.pair;
    if (!p) { $("#vote").innerHTML = `<div class="empty" style="padding:18px">No pairs to vote on. Use <b>Make A/B pairs</b>.</div>`; return; }
    $("#vote").innerHTML = `<div class="hint"><b>${esc(ev.formats[p.format] || p.format)}</b> · ${esc(p.prompt)}</div>
      <div class="grid g2"><div class="ab"><div class="body">${fmtBody(p.left)}</div></div><div class="ab"><div class="body">${fmtBody(p.right)}</div></div></div>
      <div class="row" style="justify-content:center"><button class="btn" data-v="left">Left is better</button><button class="btn ghost" data-v="tie">Tie</button><button class="btn" data-v="right">Right is better</button></div>`;
    $$("[data-v]", $("#vote")).forEach((b) => (b.onclick = async () => { try { drawVote(await api(`/api/pairs/${p.id}/vote`, { json: { choice: b.dataset.v } })); } catch (e) { fail(e); } }));
  };
  api("/api/pairs/next").then(drawVote).catch(() => {});
}

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
        <label class="row"><span class="switch"><input type="checkbox" id="gatefb" ${st.gate_on_feedback !== false ? "checked" : ""}><span></span></span> Quality gate: undo if quick checks get worse</label>
        <details><summary>Training options</summary><div class="grid g3" style="margin-top:8px">
          <label class="field">Passes<input class="input" type="number" id="fb-ep" min="0.5" max="20" step="0.5" value="${st.feedback_epochs ?? 3}"></label>
          <label class="field">Learning rate<input class="input" type="number" id="fb-lr" min="0" step="any" value="${st.feedback_lr ?? 0.00005}"></label>
          <label class="field">Preference strength<input class="input" type="number" id="fb-beta" min="0.01" max="2" step="0.01" value="${st.feedback_beta ?? 0.1}"><span class="hint">DPO beta</span></label></div>
          <div class="hint">Used by "Train now" and by automatic learning.</div></details>
        <p class="hint" style="margin:0">Starts when the app is idle. Fine-tuned models use supervised + preference (DPO) training; from-scratch models re-read liked answers. The gate runs a quick check before and after (a few minutes on CPU).</p></div>
    </div>
    <div class="card"><div class="card-head"><h2>All rated answers</h2><div class="tabs" id="ff"><button class="on" data-f="all">All</button><button data-f="liked">Liked</button><button data-f="disliked">Disliked</button><button data-f="corrected">Corrected</button><button data-f="pending">Not learned yet</button></div></div>
      <div id="fb-table"></div></div></div>`;
  const draw = (f) => {
    const rows = list.filter((r) => f === "all" || (f === "liked" && r.rating >= 4) || (f === "disliked" && r.rating <= 2) || (f === "corrected" && r.correction) || (f === "pending" && !r.trained_in));
    $("#fb-table").innerHTML = rows.length ? `<table><thead><tr><th style="width:24%">Prompt</th><th>Answer</th><th>Rating</th><th>Model</th><th>From</th><th>Status</th></tr></thead><tbody>
      ${rows.map((r) => `<tr class="click" data-conv="${r.conversation_id}"><td><div class="clamp">${esc(r.prompt)}</div></td><td><div class="clamp">${esc(r.content)}</div>${r.correction ? `<div class="hint" style="color:var(--good)">✎ corrected</div>` : ""}${r.comment ? `<div class="hint">“${esc(r.comment)}”</div>` : ""}</td>
        <td style="white-space:nowrap;color:var(--star)">${r.rating ? "★".repeat(r.rating) + '<span class="faint">' + "★".repeat(5 - r.rating) + "</span>" : "—"}</td><td>${esc(r.model)}</td><td class="muted">${esc(r.author || "—")}</td>
        <td>${r.trained_in ? (r.trained_in.startsWith("gate-rejected") ? `<span class="pill bad" title="Training on these made the quick checks worse, so it was undone. Rate again to retry.">held back</span>` : `<span class="pill good" title="${esc(r.trained_in)}">learned</span>`) : '<span class="pill warn">waiting</span>'}</td></tr>`).join("")}</tbody></table>`
      : `<div class="empty"><div class="big">⭐</div>Nothing here yet. Rate answers in <a href="#/chat"><u>Chat</u></a>.</div>`;
    $$("tr[data-conv]").forEach((t) => (t.onclick = () => { C.conv = +t.dataset.conv; location.hash = "#/chat"; }));
  };
  draw("all");
  $$("#ff button").forEach((b) => (b.onclick = () => { $$("#ff button").forEach((x) => x.classList.toggle("on", x === b)); draw(b.dataset.f); }));
  const save = async () => { try { S.status.settings = await api("/api/settings", { json: { auto_train: $("#auto").checked, auto_threshold: Math.max(1, +$("#thr").value), gate_on_feedback: $("#gatefb").checked, feedback_epochs: +$("#fb-ep").value || 3, feedback_lr: +$("#fb-lr").value || 0.00005, feedback_beta: +$("#fb-beta").value || 0.1 } }); toast("Saved", "ok"); } catch (e) { fail(e); } };
  ["#auto", "#thr", "#gatefb", "#fb-ep", "#fb-lr", "#fb-beta"].forEach((id) => ($(id).onchange = save));
  $$("[data-train]").forEach((b) => (b.onclick = async () => {
    try { await api("/api/feedback/train", { json: { model: b.dataset.train } }); toast(`🎓 <b>${esc(b.dataset.train)}</b> is learning from your ratings`, "ok"); openRun(b.dataset.train); poll(); } catch (e) { fail(e); }
  }));
};

// ================================================================ SETTINGS (everyone)
pages.settings = async () => {
  const u = S.user, p = u.prefs;
  const models = await api("/api/models").catch(() => []);
  main.innerHTML = `<div class="page" style="max-width:980px">
    <div class="page-head"><div><h1>Settings</h1><p>Your account and how Chat behaves for you.</p></div></div>
    <div class="settings-grid">
      <div class="card"><h2>Profile</h2>
        <div class="row"><div class="avatar-sm" style="width:44px;height:44px;font-size:16px">${esc(initial(u))}</div>
          <div><b>${esc(u.username)}</b><div class="hint">${u.role === "admin" ? "Admin · full access" : "User · Chat and Settings"}</div></div></div>
        <label class="field">Display name<input class="input" id="s-name" value="${esc(u.display_name)}" maxlength="60"></label>
        <div class="row end"><button class="btn primary sm" id="s-save-name">Save</button></div></div>
      <div class="card"><h2>Chat defaults</h2>
        <label class="field">Model<select class="input" id="s-model"><option value="">First available</option>${models.map((m) => `<option value="${esc(m.name)}" ${m.name === p.model ? "selected" : ""}>${esc(m.name)}</option>`).join("")}</select></label>
        <label class="field"><span>Creativity <b id="s-tv">${p.temperature}</b></span><input type="range" id="s-temp" min="0" max="1.5" step="0.05" value="${p.temperature}"><span class="hint">Low = focused and predictable · high = surprising</span></label>
        <label class="field"><span>Max length <b id="s-mv">${p.max_tokens}</b> tokens</span><input type="range" id="s-max" min="32" max="1024" step="16" value="${p.max_tokens}"></label>
        <div class="row end"><button class="btn primary sm" id="s-save-chat">Save</button></div></div>
      ${isAdmin() ? `<div class="card"><h2>Hardware <span class="pill">admin</span></h2>
        <label class="field">Train and chat on<select class="input" id="s-device"></select><span class="hint" id="s-devhint"></span></label>
        <div class="row end"><button class="btn primary sm" id="s-save-dev">Save</button></div></div>` : ""}
      <div class="card"><h2>Password</h2>
        <label class="field">Current password<input class="input" type="password" id="s-cur" autocomplete="current-password"></label>
        <label class="field">New password<input class="input" type="password" id="s-new" autocomplete="new-password"><span class="hint">At least 6 characters. Other devices get signed out.</span></label>
        <label class="field">Repeat new password<input class="input" type="password" id="s-new2" autocomplete="new-password"></label>
        <div class="row end"><button class="btn primary sm" id="s-save-pw">Change password</button></div></div>
    </div></div>`;
  if (isAdmin()) {
    const drawDev = async () => {
      await poll();
      const dv = S.status.device || {};
      const ch = dv.choices || [{ id: "auto", name: "Automatic" }, { id: "cpu", name: "CPU" }];
      if (dv.choice && !ch.some((c) => c.id === dv.choice)) ch.push({ id: dv.choice, name: `${dv.choice} (not found on this PC)` });
      $("#s-device").innerHTML = ch.map((c) => `<option value="${esc(c.id)}" ${c.id === (dv.choice || "auto") ? "selected" : ""}>${esc(c.name)}</option>`).join("");
      $("#s-devhint").innerHTML = `Now: <b>${esc(dv.name || "?")}</b>. Applies to the next training job and reloads chat models.${dv.hint ? "<br>" + esc(dv.hint) : ""}${!dv.cuda_build ? "<br>For an NVIDIA GPU this PyTorch has to be the CUDA build: close the Studio and run <span class=\"mono\">run.bat --gpu</span> (or <span class=\"mono\">./run.sh --gpu</span>)." : ""}`;
    };
    drawDev();
    $("#s-save-dev").onclick = async () => { try { await api("/api/settings", { json: { device: $("#s-device").value } }); toast("Saved", "ok"); drawDev(); } catch (e) { fail(e); } };
  }
  $("#s-temp").oninput = (e) => ($("#s-tv").textContent = e.target.value);
  $("#s-max").oninput = (e) => ($("#s-mv").textContent = e.target.value);
  const save = async (body, msg) => { try { setUser(await api("/api/me", { json: body })); toast(msg, "ok"); return true; } catch (e) { fail(e); } };
  $("#s-save-name").onclick = () => save({ display_name: $("#s-name").value }, "Saved");
  $("#s-save-chat").onclick = async () => {
    if (await save({ prefs: { model: $("#s-model").value, temperature: +$("#s-temp").value, max_tokens: +$("#s-max").value } }, "Chat defaults saved")) {
      C.temp = S.user.prefs.temperature; C.max = S.user.prefs.max_tokens;
      try { localStorage.removeItem("chatModel"); } catch {}
    }
  };
  $("#s-save-pw").onclick = async () => {
    if ($("#s-new").value !== $("#s-new2").value) return toast("The new passwords don't match.", "err");
    if (await save({ current_password: $("#s-cur").value, new_password: $("#s-new").value }, "Password changed")) $$("input[type=password]").forEach((i) => (i.value = ""));
  };
};

// ================================================================ USERS (admin)
pages.users = async () => {
  let list;
  try { list = await api("/api/users"); } catch (e) { return fail(e); }
  main.innerHTML = `<div class="page">
    <div class="page-head"><div><h1>Users</h1><p>Admins see everything. Users only see Chat and their own Settings; their ratings still teach the models.</p></div>
      <button class="btn primary" id="u-add">${I.plus} Add user</button></div>
    <div class="card"><table><thead><tr><th>User</th><th>Role</th><th>Chats</th><th>Ratings</th><th>Last sign-in</th><th></th></tr></thead><tbody>
      ${list.map((u) => `<tr data-uid="${u.id}"><td><div class="row" style="gap:10px;flex-wrap:nowrap"><div class="avatar-sm">${esc(initial(u))}</div><div><b>${esc(u.display_name)}</b><div class="hint">${esc(u.username)}${u.id === S.user.id ? " · you" : ""}</div></div></div></td>
        <td><select class="input role-select" data-role ${u.id === S.user.id ? "disabled" : ""}><option value="user" ${u.role === "user" ? "selected" : ""}>User</option><option value="admin" ${u.role === "admin" ? "selected" : ""}>Admin</option></select></td>
        <td>${num(u.chats)}</td><td>${num(u.ratings)}</td><td class="muted">${u.last_login ? ago(u.last_login) : "never"}</td>
        <td style="text-align:right;white-space:nowrap"><button class="btn sm" data-pw>Reset password</button> ${u.id === S.user.id ? "" : `<button class="btn sm danger" data-del>${I.trash}</button>`}</td></tr>`).join("")}
    </tbody></table></div></div>`;
  const row = (el) => list.find((u) => u.id === +el.closest("tr").dataset.uid);
  $$("[data-role]").forEach((s) => (s.onchange = async () => {
    const u = row(s);
    try { await api(`/api/users/${u.id}`, { method: "PATCH", json: { role: s.value } }); toast(`${esc(u.display_name)} is now ${s.value === "admin" ? "an admin" : "a user"}`, "ok"); } catch (e) { fail(e); }
    pages.users();
  }));
  $$("[data-del]").forEach((b) => (b.onclick = async () => {
    const u = row(b);
    if (!confirm(`Delete ${u.username} and their ${u.chats} chats?`)) return;
    try { await api(`/api/users/${u.id}`, { method: "DELETE" }); toast("Deleted", "ok"); } catch (e) { fail(e); }
    pages.users();
  }));
  $$("[data-pw]").forEach((b) => (b.onclick = () => {
    const u = row(b);
    modal(`<h2>New password for ${esc(u.username)}</h2><label class="field">Password<input class="input" type="password" id="m-pw" autocomplete="new-password"><span class="hint">They'll be signed out and can sign in with this password.</span></label>
      <div class="row end"><button class="btn ghost" id="m-cancel">Cancel</button><button class="btn primary" id="m-ok">Set password</button></div>`, (m, close) => {
      $("#m-pw", m).focus();
      $("#m-cancel", m).onclick = close;
      $("#m-ok", m).onclick = async () => { try { await api(`/api/users/${u.id}`, { method: "PATCH", json: { password: $("#m-pw", m).value } }); toast("Password set", "ok"); close(); } catch (e) { fail(e); } };
    });
  }));
  $("#u-add").onclick = () => modal(`<h2>Add user</h2>
      <label class="field">Username<input class="input" id="m-user" autocomplete="off"></label>
      <label class="field">Display name<input class="input" id="m-name" placeholder="optional"></label>
      <label class="field">Password<input class="input" type="password" id="m-pw" autocomplete="new-password"><span class="hint">At least 6 characters. They can change it in Settings.</span></label>
      <label class="field">Role<select class="input" id="m-role"><option value="user">User · Chat and Settings only</option><option value="admin">Admin · everything</option></select></label>
      <div class="row end"><button class="btn ghost" id="m-cancel">Cancel</button><button class="btn primary" id="m-ok">Add user</button></div>`, (m, close) => {
    $("#m-user", m).focus();
    $("#m-cancel", m).onclick = close;
    $("#m-ok", m).onclick = async () => {
      try { await api("/api/users", { json: { username: $("#m-user", m).value, display_name: $("#m-name", m).value, password: $("#m-pw", m).value, role: $("#m-role", m).value } }); toast("User added", "ok"); close(); pages.users(); } catch (e) { fail(e); }
    };
  });
};

// ================================================================ SIGN IN
const initial = (u) => (u.display_name || u.username || "?").trim().charAt(0);
function setUser(u) {
  S.user = u;
  document.body.classList.toggle("role-user", u.role !== "admin");
  $("#user-initial").textContent = initial(u);
  $("#user-name").textContent = u.display_name;
  $("#user-role").textContent = u.role;
}
function startApp(u) {
  setUser(u);
  if (C.prefsFor !== u.id) { C.prefsFor = u.id; C.conv = null; C.temp = u.prefs.temperature; C.max = u.prefs.max_tokens; }
  $("#auth-root").innerHTML = "";
  $("#shell").hidden = false;
  clearInterval(S.timer);
  S.timer = setInterval(poll, 3000);
  poll().then(route);
}
async function showAuth(err = "") {
  S.user = null;
  clearInterval(S.timer);
  $("#shell").hidden = true;
  $("#modal-root").innerHTML = "";
  let st = { needs_setup: false };
  try { st = await (await fetch("/api/auth/state")).json(); } catch {}
  const setup = st.needs_setup;
  $("#auth-root").innerHTML = `<div class="auth"><form class="card auth-card" id="auth-form">
    <div class="logo">${I.logo}</div>
    <h1>${setup ? "Create the admin account" : "Sign in to Text LLM Studio"}</h1>
    <p>${setup ? "This is the first account. It can train models, see analytics and add other people." : "Use the account your admin gave you."}</p>
    <label class="field">Username<input class="input" id="a-user" autocomplete="username" required></label>
    ${setup ? '<label class="field">Display name<input class="input" id="a-name" placeholder="optional"></label>' : ""}
    <label class="field">Password<input class="input" type="password" id="a-pw" autocomplete="${setup ? "new-password" : "current-password"}" required>${setup ? '<span class="hint">At least 6 characters.</span>' : ""}</label>
    <div class="auth-err" id="a-err">${esc(err)}</div>
    <button class="btn primary" type="submit">${setup ? "Create account" : "Sign in"}</button>
  </form></div>`;
  $("#a-user").focus();
  $("#auth-form").onsubmit = async (e) => {
    e.preventDefault();
    const body = { username: $("#a-user").value, password: $("#a-pw").value };
    if (setup) body.display_name = $("#a-name").value;
    try {
      const u = await api(setup ? "/api/auth/setup" : "/api/auth/login", { json: body });
      if (!location.hash) location.hash = u.role === "admin" ? "#/dashboard" : "#/chat";
      startApp(u);
    } catch (e) { $("#a-err").textContent = e.message; }
  };
}
$("#logout").onclick = async () => { await fetch("/api/auth/logout", { method: "POST" }).catch(() => {}); C.prefsFor = null; showAuth(); };

fetch("/api/auth/state").then((r) => r.json()).then((st) => (st.user ? startApp(st.user) : showAuth())).catch(() => showAuth("Can't reach the server."));
