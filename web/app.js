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
    $(".jobtext", pill).textContent = j.running ? `${labelKind(j.kind)}: ${j.name}` : "Idle · no job running";
    const b = $("#nav-pending");
    b.hidden = !S.status.pending_feedback;
    b.textContent = S.status.pending_feedback;
    S.onStatus && S.onStatus(S.status);
  } catch {}
}
const labelKind = (k) => ({ training: "Training", data: "Preparing data", texts: "Learning texts", feedback: "Learning from ratings", eval: "Evaluating", export: "Making fast CPU copy" }[k] || k);
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
const deviceNote = (dv) => !dv ? "" : `<div class="hint">${dv.gpu ? "⚡ Training and chat run on the GPU" : "Training and chat run on the CPU"}: <b>${esc(dv.name)}</b>${dv.hint ? ` · ${esc(dv.hint)}` : ""}</div>`;
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
    Object.entries(m === "scratch" ? { steps: 20000, lr: 0.001, batch: 16, accum: 4, name: "my-writer" } : { steps: 300, lr: 0.0002, batch: 2, accum: 8, name: "my-social-writer" })
      .forEach(([k, v]) => ($("#f-" + k).value = v));
    expect();
  };
  $$("#modes .opt").forEach((o) => (o.onclick = () => setMode(o.dataset.v)));
  $$("#presets .opt").forEach((o) => (o.onclick = () => { st.preset = o.dataset.v; $$("#presets .opt").forEach((x) => x.classList.toggle("on", x === o)); }));
  const baseHint = () => { const b = BASES.find((x) => x.id === $("#f-basepick").value); $("#f-basehint").textContent = b ? b.h : "Any causal LM on Hugging Face"; if (b) { $("#f-base").value = b.id; $("#f-lorar").value = b.r; } };
  $("#f-basepick").onchange = baseHint; baseHint();
  $("#f-bookshare").oninput = (e) => ($("#f-bsv").textContent = e.target.value);
  expect();
  $("#start").onclick = async () => {
    const v = (k) => $("#f-" + k).value;
    try {
      await api("/api/train", { json: { mode: st.mode, name: v("name"), resume: $("#f-resume").checked, preset: st.preset, vocab: +v("vocab"), steps: +v("steps"), batch: +v("batch"), accum: +v("accum"), lr: +v("lr"), base: v("base"), examples: +v("examples"), seq: +v("seq"), instructions: $("#f-inst").checked, book_share: +v("bookshare") / 100, lora_r: +v("lorar"), epochs: +v("epochs"), gguf: $("#f-gguf").checked } });
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
          </div>
          <ul class="hint" style="padding-left:18px;margin:6px 0 0">
            <li><b>PG-19</b>: 28,752 Gutenberg books from before 1919 (Apache 2.0). Books Gutenberg already saved are skipped.</li>
            <li><b>Standard Ebooks</b>: about 1,400 carefully proofread classics (CC0), read from their GitHub pages.</li>
            <li><b>Common Corpus</b>: only its "Open Culture" part, public-domain books and newspapers (OCR).</li>
            <li><b>Internet Archive</b>: scanned books published before 1931 (OCR, noisier). Keep it small.</li>
            <li><b>Cosmopedia</b>: synthetic stories and textbooks written by Mixtral (Apache 2.0).</li>
            <li>Not offered: BookCorpus, BookCorpusOpen, The Pile and RedPajama books (copyrighted books), Open Library (no book text).</li>
          </ul></details></div>
        <button class="btn" data-step="download">${I.play} Run</button></div>
      <div class="step ${d.tokenizer ? "done" : ""}"><div class="num">${d.tokenizer ? "✓" : 2}</div><div><b>Build the tokenizer</b><div class="hint">${d.tokenizer ? `${num(d.vocab)}-token vocabulary` : "Learns the word pieces the model reads and writes"}</div>
        <div class="row" style="margin-top:8px"><label class="field" style="width:200px">Vocabulary<select class="input" id="d-vocab"><option value="8192">8,192 (CPU)</option><option value="32000">32,000 (GPU)</option></select></label></div></div>
        <button class="btn" data-step="tokenizer">${I.play} Run</button></div>
      <div class="step ${d.train_tokens ? "done" : ""}"><div class="num">${d.train_tokens ? "✓" : 3}</div><div><b>Encode the corpus</b><div class="hint">${d.train_tokens ? `${num(d.train_tokens / 1e6, 1)}M tokens ready to train` : "Converts every text into token files for fast training"}</div></div>
        <button class="btn" data-step="encode">${I.play} Run</button></div>
      <div class="step ${d.instructions ? "done" : ""}"><div class="num">${d.instructions ? "✓" : 4}</div><div><b>Build writing instructions</b> <span class="pill">for fine-tuning</span><div class="hint">${d.instructions ? `${num(d.instructions)} examples of posts, stories and essays` : "Public-domain stories and essays from Gutenberg, your liked and corrected chat answers, and (optional) OpenAssistant + Dolly"}</div>
        <div class="row" style="margin-top:8px"><label class="row hint"><span class="switch"><input type="checkbox" id="d-hfsets" checked><span></span></span> Include oasst2 (Apache 2.0) and Dolly (CC BY-SA 3.0)</label></div>
        <div class="hint" style="margin-top:6px">Your own posts and synthetic posts from a bigger model: see <code>data/build_instructions.py --help</code>.</div></div>
        <button class="btn" data-step="instructions">${I.play} Run</button></div>
    </div></div>
    <div class="card"><h2>Tips</h2><ul class="muted" style="padding-left:18px">
      <li>Only <b>from-scratch</b> training needs steps 2 and 3. Fine-tuning uses step 4 plus a share of the downloaded books.</li>
      <li>Want your own books in the corpus? Add them under <a href="#/training/texts"><u>Your texts</u></a> → "Add to training data", then re-run steps 2–3.</li>
      <li>Progress and logs show at the top of this page while a step runs.</li></ul></div></div>
    <div class="card" style="margin-top:16px"><div class="card-head"><h2>Log</h2></div><pre class="console" id="dlog">${esc(S.status.job.kind === "data" ? S.status.job.log : "No data job yet.")}</pre></div>`;
  $$("[data-step]").forEach((b) => (b.onclick = async () => {
    try {
      await api("/api/data", { json: { step: b.dataset.step, books: +$("#d-books").value, web_gb: +$("#d-web").value, gutenberg_method: $("#d-gmethod").value, pg19_books: +$("#d-pg19").value, standard_ebooks: +$("#d-se").value, common_corpus_gb: +$("#d-cc").value, archive_books: +$("#d-ia").value, cosmopedia_gb: +$("#d-cosmo").value, cosmopedia_subsets: $("#d-cosmosub").value, vocab: +$("#d-vocab").value, hf_sets: $("#d-hfsets").checked } });
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
    <div class="stats" id="r-stats"></div><div class="card"><div class="card-head"><h2>Loss</h2><span class="hint">training · validation · learning from ratings</span></div><div id="r-chart"></div></div>
    <div><div class="card-head"><h2>Log</h2></div><pre class="console" id="r-log"></pre></div>`, (m) => {
    const load = async () => {
      if (!document.body.contains(m)) return clearInterval(timer);
      let d;
      try { d = await api(`/api/runs/${encodeURIComponent(name)}`); } catch (e) { clearInterval(timer); return fail(e); }
      const s = d.summary;
      $("#r-title", m).textContent = s.name;
      $("#r-sub", m).innerHTML = `${typePill(s.type)} ${s.base ? esc(s.base) : s.preset ? esc(s.preset) + " size" : ""} ${s.running ? '<span class="pill good">● training</span>' : ""}`;
      $("#r-btns", m).innerHTML = `${s.ready ? `<button class="btn sm primary" id="r-chat">${I.chat} Chat</button>` : ""}${s.ready && !s.running ? `<button class="btn sm" id="r-eval" title="Score it on the 60 fixed prompts">Evaluate</button><button class="btn sm ghost" id="r-quick" title="8 prompts, a few minutes">Quick check</button>` : ""}${s.type === "lora" && s.ready && !s.running ? `<button class="btn sm ${s.fast ? "ghost" : ""}" id="r-gguf" title="Merge, convert to GGUF and quantize to 4-bit">${s.fast ? "⚡ Fast copy ✓" : "⚡ Make fast CPU copy"}</button>` : ""}${s.ready && !s.running ? `<button class="btn sm" id="r-resume" title="Continue training from the last checkpoint">${I.play} Resume</button><button class="btn sm ghost" id="r-export" title="Download as one .zip to continue on another PC">${I.download} Export</button>` : ""}${s.running ? `<button class="btn sm danger" id="r-stop">${I.stop} Stop</button>` : `<button class="btn sm ghost" id="r-del">${I.trash}</button>`}`;
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
      on("#r-eval", () => startEval(s.name, false).then(load));
      on("#r-quick", () => startEval(s.name, true).then(load));
      on("#r-gguf", async () => { try { await api(`/api/runs/${encodeURIComponent(s.name)}/gguf`, { method: "POST" }); toast("Making the fast CPU copy…", "ok"); poll(); load(); } catch (e) { fail(e); } });
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

async function startEval(name, quick) {
  try { await api("/api/eval", { json: { model: name, quick } }); toast(`${quick ? "Quick check" : "Evaluation"} of <b>${esc(name)}</b> started`, "ok"); poll(); }
  catch (e) { fail(e); }
}

// ================================================================ CHAT
const SUGGEST = [
  ["Short story", "A lighthouse keeper finds a message in a bottle", "story"],
  ["LinkedIn post", "I trained my own language model on public-domain books", "linkedin_post"],
  ["Essay", "Why reading old books still matters", "essay"],
  ["X post", "Our small bakery now opens at 7am on weekdays", "x_post"],
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
  $("#model").innerHTML = C.models.length ? C.models.map((m) => `<option value="${esc(m.name)}" ${m.name === pref ? "selected" : ""}>${esc(m.name)} · ${m.type === "lora" ? "fine-tuned" : "from scratch"}${m.default ? " · default" : ""}</option>`).join("") : `<option value="">No trained models yet</option>`;
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
  $("#samp-reset").onclick = () => { C.temp = C.max = null; drawSampling(); };
  const groups = [...new Set(C.formats.map((f) => f.group))];
  $("#format").innerHTML = groups.map((g) => `<optgroup label="${esc(g)}">${C.formats.filter((f) => f.group === g).map((f) => `<option value="${f.id}">${esc(f.label)}</option>`).join("")}</optgroup>`).join("");
  try { C.format = localStorage.getItem("chatFormat") || C.format; } catch {}
  if (!C.formats.some((f) => f.id === C.format)) C.format = "free";
  $("#format").value = C.format;
  $("#format").onchange = () => { C.format = $("#format").value; try { localStorage.setItem("chatFormat", C.format); } catch {} drawFields(); drawSampling(); };
  drawFields(); drawSampling();
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
function answerHtml(a, alt = "") {
  return `<div class="msg" data-mid="${a.id}" data-uid="${a.prompt_id}"><div class="avatar">${I.logo}</div><div class="answer">
    ${alt ? `<div class="alt">${alt}</div>` : ""}<div class="body">${fmtBody(a.content)}</div>
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
  holder.innerHTML = `<div class="avatar">${I.logo}</div><div class="answer"><div class="hint" data-live-status></div><details class="outline" hidden><summary>Plan</summary><pre></pre></details><div class="body cursor"></div></div>`;
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
    </tbody></table></div></div>
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
  $$("[data-eval]").forEach((b) => (b.onclick = () => startEval(b.dataset.eval, false)));
  $$("[data-quick]").forEach((b) => (b.onclick = () => startEval(b.dataset.quick, true)));
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
  const save = async () => { try { S.status.settings = await api("/api/settings", { json: { auto_train: $("#auto").checked, auto_threshold: Math.max(1, +$("#thr").value), gate_on_feedback: $("#gatefb").checked } }); toast("Saved", "ok"); } catch (e) { fail(e); } };
  $("#auto").onchange = save; $("#thr").onchange = save; $("#gatefb").onchange = save;
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
      <div class="card"><h2>Password</h2>
        <label class="field">Current password<input class="input" type="password" id="s-cur" autocomplete="current-password"></label>
        <label class="field">New password<input class="input" type="password" id="s-new" autocomplete="new-password"><span class="hint">At least 6 characters. Other devices get signed out.</span></label>
        <label class="field">Repeat new password<input class="input" type="password" id="s-new2" autocomplete="new-password"></label>
        <div class="row end"><button class="btn primary sm" id="s-save-pw">Change password</button></div></div>
    </div></div>`;
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
