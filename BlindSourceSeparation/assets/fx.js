/*
  DSP Lab visual effects (presentation only).

  Injected once per session from app.py through a zero-height component
  iframe. It reaches into the parent Streamlit document to add:
    - a flowing sine-wave background (SVG + CSS transforms, no per-frame JS)
    - hover balloons on the hero title
    - IntersectionObserver scroll reveals
    - cursor-reactive card highlight + Theory card tilt
    - smooth scrolling for the Theory section index

  It never touches app state, widgets or data. Idempotent: safe to run again.
*/
(function () {
  var win = window.parent;
  var doc;
  try { doc = win.document; } catch (e) { return; }
  if (win.__dspFx) return;
  win.__dspFx = true;

  var root = doc.documentElement;
  var reduce = win.matchMedia && win.matchMedia("(prefers-reduced-motion: reduce)").matches;
  var coarse = win.matchMedia && win.matchMedia("(pointer: coarse)").matches;
  root.classList.add("dsp-fx");
  if (reduce) root.classList.add("dsp-reduce");

  /* ---------------- Flowing sine waves ---------------- */
  var NS = "http://www.w3.org/2000/svg";
  var WAVES = [
    { amp: 46, period: 620, y: 0.22, dur: 46, op: 0.16, w: 1.1, c: "#c9ced8" },
    { amp: 30, period: 410, y: 0.34, dur: 34, op: 0.12, w: 0.9, c: "#8f97a6" },
    { amp: 62, period: 900, y: 0.5,  dur: 62, op: 0.14, w: 1.2, c: "#e4e7ee" },
    { amp: 24, period: 300, y: 0.62, dur: 28, op: 0.10, w: 0.8, c: "#7fb4c4" },
    { amp: 40, period: 540, y: 0.74, dur: 52, op: 0.12, w: 1.0, c: "#aab0bd" },
    { amp: 18, period: 230, y: 0.86, dur: 24, op: 0.09, w: 0.8, c: "#98a0ae" }
  ];

  function buildWaves() {
    var old = doc.getElementById("dsp-waves");
    if (old) old.remove();
    var W = Math.max(win.innerWidth, 320);
    var H = Math.max(win.innerHeight, 480);
    var wrap = doc.createElement("div");
    wrap.id = "dsp-waves";
    wrap.setAttribute("aria-hidden", "true");
    var count = W < 700 ? 4 : WAVES.length;
    WAVES.slice(0, count).forEach(function (w, i) {
      var svg = doc.createElementNS(NS, "svg");
      var total = W + w.period * 2;
      var h = w.amp * 2 + 8;
      svg.setAttribute("width", total);
      svg.setAttribute("height", h);
      svg.setAttribute("viewBox", "0 0 " + total + " " + h);
      svg.style.top = Math.round(H * w.y - h / 2) + "px";
      svg.style.opacity = w.op;
      svg.style.setProperty("--period", w.period + "px");
      svg.style.setProperty("--dur", w.dur + "s");
      svg.style.setProperty("--bob", 9 + i * 2 + "s");
      var d = "M0 " + h / 2;
      var step = w.period / 8;
      for (var x = 0; x <= total; x += step) {
        var y = h / 2 - Math.sin((x / w.period) * Math.PI * 2 + i) * w.amp;
        d += " L" + x.toFixed(1) + " " + y.toFixed(1);
      }
      var p = doc.createElementNS(NS, "path");
      p.setAttribute("d", d);
      p.setAttribute("fill", "none");
      p.setAttribute("stroke", w.c);
      p.setAttribute("stroke-width", w.w);
      p.setAttribute("stroke-linejoin", "round");
      p.setAttribute("vector-effect", "non-scaling-stroke");
      svg.appendChild(p);
      wrap.appendChild(svg);
    });
    doc.body.appendChild(wrap);
  }
  buildWaves();
  var rz;
  win.addEventListener("resize", function () {
    clearTimeout(rz);
    rz = setTimeout(buildWaves, 250);
  });

  /* ---------------- Balloons on the hero title ---------------- */
  var MAX_BALLOONS = 28;
  var live = 0;
  var lastBurst = 0;
  var TINTS = ["#e8eaf0", "#c4c9d4", "#9aa2b1", "#f5f6f9", "#8ecad8", "#b9a8e6"];

  function burst(host, ev) {
    var now = Date.now();
    if (now - lastBurst < 140) return;
    lastBurst = now;
    var r = host.getBoundingClientRect();
    var n = reduce ? 2 : 3 + Math.floor(Math.random() * 3);
    for (var i = 0; i < n && live < MAX_BALLOONS; i++) {
      var b = doc.createElement("span");
      b.className = "dsp-balloon";
      var size = 12 + Math.random() * 14;
      var x = ev && ev.clientX != null ? ev.clientX : r.left + Math.random() * r.width;
      x += (Math.random() - 0.5) * 60;
      var y = r.top + r.height * (0.2 + Math.random() * 0.6);
      var tint = TINTS[Math.floor(Math.random() * TINTS.length)];
      b.style.cssText =
        "left:" + x + "px;top:" + y + "px;width:" + size + "px;height:" + size * 1.22 + "px;" +
        "--tint:" + tint + ";--dx:" + (Math.random() - 0.5) * 90 + "px;" +
        "--rise:" + (120 + Math.random() * 140) + "px;--dur:" + (2.2 + Math.random() * 1.6) + "s;" +
        "--sway:" + (10 + Math.random() * 16) + "px;";
      live++;
      b.addEventListener("animationend", function (e) {
        if (e.animationName !== "dsp-rise") return;
        e.currentTarget.remove();
        live--;
      });
      doc.body.appendChild(b);
    }
  }

  function titleFrom(t) {
    return t && t.closest ? t.closest(".dsp-title-sub") : null;
  }
  doc.addEventListener("pointerover", function (e) {
    var h = titleFrom(e.target);
    if (h && !h.contains(e.relatedTarget)) burst(h, e);
  }, true);
  doc.addEventListener("pointermove", function (e) {
    var h = titleFrom(e.target);
    if (h && Math.random() < 0.12) burst(h, e);
  }, true);
  // touch fallback: tap the title
  doc.addEventListener("click", function (e) {
    var h = titleFrom(e.target);
    if (h && coarse) burst(h, e);
  }, true);

  /* ---------------- Scroll reveals ---------------- */
  var REVEAL = '[class*="st-key-rv_"],[class*="st-key-rvl_"],[class*="st-key-rvr_"],[class*="st-key-featl_"],[class*="st-key-featr_"]';
  var io = null;
  if ("IntersectionObserver" in win) {
    io = new win.IntersectionObserver(function (entries) {
      entries.forEach(function (en) {
        if (en.isIntersecting) {
          en.target.setAttribute("data-in", "1");
          io.unobserve(en.target);
        }
      });
    }, { threshold: 0.12, rootMargin: "0px 0px -6% 0px" });
  }
  function scan() {
    doc.querySelectorAll(REVEAL).forEach(function (el) {
      if (el.__dspSeen) return;
      el.__dspSeen = true;
      if (io && !reduce) io.observe(el);
      else el.setAttribute("data-in", "1");
    });
  }
  var pending = false;
  new win.MutationObserver(function () {
    if (pending) return;
    pending = true;
    win.requestAnimationFrame(function () { pending = false; scan(); });
  }).observe(doc.body, { childList: true, subtree: true });
  scan();

  /* ---------------- Cursor-reactive highlight + tilt ---------------- */
  var GLOW = '.st-key-upload_card,.st-key-theory_card,[class*="st-key-featl_"],[class*="st-key-featr_"],[class*="st-key-rv_"],[data-testid="stVerticalBlockBorderWrapper"]';
  var raf = 0, lastEv = null;
  doc.addEventListener("pointermove", function (e) {
    if (coarse || reduce) return;
    lastEv = e;
    if (raf) return;
    raf = win.requestAnimationFrame(function () {
      raf = 0;
      var t = lastEv.target;
      var el = t && t.closest ? t.closest(GLOW) : null;
      if (!el) return;
      var r = el.getBoundingClientRect();
      var px = (lastEv.clientX - r.left) / r.width;
      var py = (lastEv.clientY - r.top) / r.height;
      el.style.setProperty("--mx", (px * 100).toFixed(1) + "%");
      el.style.setProperty("--my", (py * 100).toFixed(1) + "%");
      if (el.classList.contains("st-key-theory_card")) {
        el.style.setProperty("--ry", ((px - 0.5) * 7).toFixed(2) + "deg");
        el.style.setProperty("--rx", ((0.5 - py) * 7).toFixed(2) + "deg");
      }
    });
  }, { passive: true });
  doc.addEventListener("pointerout", function (e) {
    var t = e.target;
    var el = t && t.closest ? t.closest(".st-key-theory_card") : null;
    if (el && !el.contains(e.relatedTarget)) {
      el.style.setProperty("--rx", "0deg");
      el.style.setProperty("--ry", "0deg");
    }
  }, true);

  /* ---------------- Drag-over feedback for the upload dropzone ---------------- */
  var dragDepth = 0;
  function isFiles(e) {
    return e.dataTransfer && Array.prototype.indexOf.call(e.dataTransfer.types || [], "Files") !== -1;
  }
  doc.addEventListener("dragenter", function (e) {
    if (!isFiles(e)) return;
    dragDepth++;
    root.classList.add("dsp-drag");
  }, true);
  doc.addEventListener("dragleave", function (e) {
    if (!isFiles(e)) return;
    dragDepth = Math.max(0, dragDepth - 1);
    if (!dragDepth) root.classList.remove("dsp-drag");
  }, true);
  doc.addEventListener("drop", function () {
    dragDepth = 0;
    root.classList.remove("dsp-drag");
  }, true);

  /* ---------------- Theory section index (smooth scroll + scrollspy) ---------------- */
  doc.addEventListener("click", function (e) {
    var a = e.target.closest ? e.target.closest("[data-scroll]") : null;
    if (!a) return;
    var target = doc.querySelector(".st-key-" + a.getAttribute("data-scroll"));
    if (!target) return;
    e.preventDefault();
    target.scrollIntoView({ behavior: reduce ? "auto" : "smooth", block: "start" });
  });

  var spy = 0;
  function scrollspy() {
    spy = 0;
    var links = doc.querySelectorAll(".dsp-index [data-scroll]");
    if (!links.length) return;
    var best = null, bestTop = -1e9;
    links.forEach(function (a) {
      var t = doc.querySelector(".st-key-" + a.getAttribute("data-scroll"));
      if (!t) return;
      var top = t.getBoundingClientRect().top - 140;
      if (top <= 0 && top > bestTop) { bestTop = top; best = a; }
    });
    links.forEach(function (a) { a.classList.toggle("on", a === best); });
  }
  doc.addEventListener("scroll", function () {
    if (!spy) spy = win.requestAnimationFrame(scrollspy);
  }, true);
})();
