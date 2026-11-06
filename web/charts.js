// Tiny dependency-free SVG charts (works offline).
const Charts = (() => {
  const PALETTE = ["#7c5cff", "#22d3ee", "#34d399", "#fbbf24", "#f472b6", "#f87171", "#a3e635", "#60a5fa"];
  const ns = "http://www.w3.org/2000/svg";
  const el = (tag, attrs = {}, parent) => {
    const e = document.createElementNS(ns, tag);
    for (const k in attrs) e.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(e);
    return e;
  };
  const nice = (v) => (Math.abs(v) >= 1000 ? (v / 1000).toFixed(v >= 10000 ? 0 : 1) + "k" : +v.toFixed(Math.abs(v) < 10 ? 2 : 0));

  // series: [{name, color?, points: [{x, y}], dashed?}]
  function line(container, series, { height = 220, yLabel = "", xLabel = "", empty = "No data yet" } = {}) {
    container.innerHTML = "";
    container.classList.add("chart");
    const pts = series.flatMap((s) => s.points);
    if (!pts.length) {
      container.innerHTML = `<div class="empty" style="padding:30px">${empty}</div>`;
      return;
    }
    const W = Math.max(320, container.clientWidth || 600), H = height, P = { l: 42, r: 12, t: 10, b: 26 };
    let x0 = Math.min(...pts.map((p) => p.x)), x1 = Math.max(...pts.map((p) => p.x));
    let y0 = Math.min(...pts.map((p) => p.y)), y1 = Math.max(...pts.map((p) => p.y));
    if (x0 === x1) x1 = x0 + 1;
    const pad = (y1 - y0) * 0.08 || 0.5;
    y0 -= pad; y1 += pad;
    const sx = (x) => P.l + ((x - x0) / (x1 - x0)) * (W - P.l - P.r);
    const sy = (y) => H - P.b - ((y - y0) / (y1 - y0)) * (H - P.t - P.b);
    const svg = el("svg", { viewBox: `0 0 ${W} ${H}` }, container);
    const defs = el("defs", {}, svg);
    for (let i = 0; i <= 4; i++) {
      const y = y0 + ((y1 - y0) * i) / 4;
      el("line", { x1: P.l, x2: W - P.r, y1: sy(y), y2: sy(y), class: "axis", "stroke-dasharray": i ? "3 4" : "" }, svg);
      el("text", { x: P.l - 8, y: sy(y) + 3, "text-anchor": "end" }, svg).textContent = nice(y);
    }
    for (let i = 0; i <= 5; i++) {
      const x = x0 + ((x1 - x0) * i) / 5;
      el("text", { x: sx(x), y: H - 8, "text-anchor": "middle" }, svg).textContent = nice(Math.round(x));
    }
    series.forEach((s, i) => {
      const c = s.color || PALETTE[i % PALETTE.length];
      s._c = c;
      const p = [...s.points].sort((a, b) => a.x - b.x);
      if (!p.length) return;
      const d = p.map((q, j) => `${j ? "L" : "M"}${sx(q.x).toFixed(1)},${sy(q.y).toFixed(1)}`).join("");
      if (s.area) {
        const gid = "g" + Math.random().toString(36).slice(2);
        const g = el("linearGradient", { id: gid, x1: 0, x2: 0, y1: 0, y2: 1 }, defs);
        el("stop", { offset: "0", "stop-color": c, "stop-opacity": ".28" }, g);
        el("stop", { offset: "1", "stop-color": c, "stop-opacity": "0" }, g);
        el("path", { d: `${d}L${sx(p[p.length - 1].x)},${H - P.b}L${sx(p[0].x)},${H - P.b}Z`, fill: `url(#${gid})` }, svg);
      }
      el("path", { d, class: "line", stroke: c, "stroke-dasharray": s.dashed ? "5 5" : "" }, svg);
      if (p.length < 40) p.forEach((q) => el("circle", { cx: sx(q.x), cy: sy(q.y), r: 2.6, fill: c }, svg));
    });
    // hover tooltip
    const tip = document.createElement("div");
    tip.className = "tip"; tip.hidden = true;
    container.appendChild(tip);
    const guide = el("line", { y1: P.t, y2: H - P.b, class: "axis", visibility: "hidden" }, svg);
    svg.addEventListener("mousemove", (ev) => {
      const r = svg.getBoundingClientRect();
      const mx = ((ev.clientX - r.left) / r.width) * W;
      const xv = x0 + ((mx - P.l) / (W - P.l - P.r)) * (x1 - x0);
      const rows = series.map((s) => {
        if (!s.points.length) return null;
        const q = s.points.reduce((a, b) => (Math.abs(b.x - xv) < Math.abs(a.x - xv) ? b : a));
        return { s, q };
      }).filter(Boolean);
      if (!rows.length) return;
      const gx = sx(rows[0].q.x);
      guide.setAttribute("x1", gx); guide.setAttribute("x2", gx); guide.setAttribute("visibility", "visible");
      tip.hidden = false;
      tip.style.left = (gx / W) * r.width + "px";
      tip.style.top = (Math.min(...rows.map((o) => sy(o.q.y))) / H) * r.height + "px";
      tip.innerHTML = `<b>${xLabel || "x"} ${Math.round(rows[0].q.x)}</b><br>` +
        rows.map((o) => `<span style="color:${o.s._c}">●</span> ${o.s.name}: ${(+o.q.y).toFixed(3)}`).join("<br>");
    });
    svg.addEventListener("mouseleave", () => { tip.hidden = true; guide.setAttribute("visibility", "hidden"); });
    if (series.length > 1 || series[0].name) {
      const lg = document.createElement("div");
      lg.className = "legend";
      lg.innerHTML = series.map((s) => `<span><i style="background:${s._c}"></i>${s.name}</span>`).join("");
      container.appendChild(lg);
    }
  }

  // bars: [{label, value, color?, note?}]
  function bars(container, data, { height = 200, max = null, fmt = (v) => v, empty = "No data yet" } = {}) {
    container.innerHTML = "";
    container.classList.add("chart");
    if (!data.length || data.every((d) => !d.value)) {
      container.innerHTML = `<div class="empty" style="padding:30px">${empty}</div>`;
      return;
    }
    const W = Math.max(320, container.clientWidth || 600), H = height, P = { l: 8, r: 8, t: 18, b: 26 };
    const m = max || Math.max(...data.map((d) => d.value)) || 1;
    const bw = Math.min((W - P.l - P.r) / data.length, 140);
    P.l += ((W - P.l - P.r) - bw * data.length) / 2;
    const svg = el("svg", { viewBox: `0 0 ${W} ${H}` }, container);
    el("line", { x1: P.l, x2: W - P.r, y1: H - P.b, y2: H - P.b, class: "axis" }, svg);
    data.forEach((d, i) => {
      const h = (d.value / m) * (H - P.t - P.b);
      const x = P.l + i * bw + bw * 0.18, w = bw * 0.64;
      const r = el("rect", { x, y: H - P.b - h, width: w, height: Math.max(h, 0), rx: 5, fill: d.color || PALETTE[0] }, svg);
      el("title", {}, r).textContent = `${d.label}: ${fmt(d.value)}${d.note ? " · " + d.note : ""}`;
      if (d.value) el("text", { x: x + w / 2, y: H - P.b - h - 5, "text-anchor": "middle" }, svg).textContent = fmt(d.value);
      el("text", { x: x + w / 2, y: H - 8, "text-anchor": "middle" }, svg).textContent = d.label;
    });
  }

  function ring(pct) {
    const r = 26, c = 2 * Math.PI * r;
    return `<svg class="ring" viewBox="0 0 64 64"><defs><linearGradient id="ringgrad"><stop offset="0" stop-color="#7c5cff"/><stop offset="1" stop-color="#22d3ee"/></linearGradient></defs>
      <circle class="bg" cx="32" cy="32" r="${r}"/><circle class="fg" cx="32" cy="32" r="${r}" stroke-dasharray="${c}" stroke-dashoffset="${c * (1 - pct)}" transform="rotate(-90 32 32)"/>
      <text x="32" y="36" text-anchor="middle" style="font-size:13px;font-weight:700;fill:var(--text);stroke:none">${Math.round(pct * 100)}%</text></svg>`;
  }

  return { line, bars, ring, PALETTE };
})();
