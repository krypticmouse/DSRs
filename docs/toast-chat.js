// DSRs docs AI chat, powered by Mixedbread toast-1.
// Mintlify auto-injects this file into every page. The widget streams
// answers from the docs-chat-worker proxy (see /docs-chat-worker), which
// holds the MXBAI_API_KEY and grounds toast-1 in the dsrs-docs store.
(function () {
  "use strict";

  var ENDPOINT =
    window.TOAST_CHAT_ENDPOINT ||
    (location.hostname === "localhost"
      ? "http://localhost:8787" // `npx wrangler dev` in docs-chat-worker
      : "https://dsrs-toast-chat.herumbshandilya123.workers.dev");

  var SUGGESTIONS = [
    "Walk through Predict::forward from signature to adapter to LM",
    "How does GEPA use traces to evolve better instructions?",
    "How do optimizers mutate holes in the IR?",
    "What happens when a .dsrs file is loaded back into a program?",
  ];

  // Logos inlined as data URIs (the page CSP blocks external assets).
  // DSRs mark: downscaled from docs/logo/logo.png.
  var DSRS_LOGO = "data:image/png;base64,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";
  // Mixedbread mark: regenerate from https://www.mixedbread.com if the
  // brand changes.
  var MXB_LOGO = "data:image/png;base64,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";

  var history = [];

  var host = document.createElement("div");
  host.id = "toast-chat";
  var root = host.attachShadow({ mode: "open" });
  root.innerHTML =
    "<style>" +
    ":host{all:initial}" +
    "*{box-sizing:border-box;margin:0;font-family:ui-sans-serif,system-ui,-apple-system,'Segoe UI',sans-serif}" +
    ".wrap{--accent:#ed6c13;--accent-soft:#fff1e7;--accent-text:#b34e07;" +
    "--bg:#ffffff;--fg:#18181b;--muted:#71717a;--card:#fafafa;--line:#e9e9ec;--code:#f4f4f5;" +
    "position:fixed;bottom:0;right:0;z-index:2147483000;font-size:15px}" +
    ".wrap.dark{--accent-soft:#3a2314;--accent-text:#ffb27d;" +
    "--bg:#131417;--fg:#ededf0;--muted:#8f8f98;--card:#1b1c21;--line:#2a2b32;--code:#22232a}" +

    ".fab{position:fixed;bottom:18px;right:18px;display:flex;align-items:center;gap:6px;" +
    "cursor:pointer;background:var(--bg);color:var(--muted);border:1px solid var(--line);" +
    "border-radius:999px;padding:6px 13px;font-size:12.5px;font-weight:500;" +
    "box-shadow:0 1px 3px rgba(0,0,0,.07);opacity:.85;" +
    "transition:color .15s ease,border-color .15s ease,opacity .15s ease," +
    "transform .15s ease,box-shadow .15s ease}" +
    ".fab:hover{opacity:1;color:var(--accent-text);border-color:var(--accent);" +
    "transform:translateY(-1px);box-shadow:0 4px 12px rgba(0,0,0,.10)}" +
    ".wrap.open .fab{opacity:0;pointer-events:none}" +

    ".veil{position:fixed;inset:0;background:rgba(12,12,16,.45);opacity:0;pointer-events:none;" +
    "backdrop-filter:blur(3px);-webkit-backdrop-filter:blur(3px);transition:opacity .22s ease}" +
    ".veil.open{opacity:1;pointer-events:auto}" +

    ".panel{position:fixed;left:50%;top:50%;width:min(760px,calc(100vw - 32px));" +
    "max-height:min(78vh,680px);display:flex;flex-direction:column;" +
    "background:var(--bg);color:var(--fg);border:1px solid var(--line);border-radius:16px;" +
    "box-shadow:0 24px 70px rgba(0,0,0,.22),0 4px 14px rgba(0,0,0,.08);" +
    "opacity:0;transform:translate(-50%,calc(-50% + 12px)) scale(.98);pointer-events:none;" +
    "transition:opacity .22s ease,transform .22s cubic-bezier(.32,.72,.24,1);overflow:hidden}" +
    ".panel.open{opacity:1;transform:translate(-50%,-50%) scale(1);pointer-events:auto}" +
    ".wrap.dark .panel{box-shadow:0 24px 80px rgba(0,0,0,.6),0 4px 14px rgba(0,0,0,.4)}" +
    "@media (max-width:520px){.panel{left:0;top:0;width:100vw;height:100%;max-height:none;" +
    "transform:translate(0,10px) scale(.98);border-radius:0;border:none}" +
    ".panel.open{transform:none}}" +

    ".head{display:flex;align-items:center;gap:10px;padding:14px 16px;border-bottom:1px solid var(--line)}" +
    ".mark{width:26px;height:26px;display:flex;align-items:center;justify-content:center}" +
    ".mark img{width:100%;height:100%;display:block}" +
    ".head b{font-size:14px;font-weight:600}" +
    ".pill{display:inline-flex;align-items:center;gap:5px;font-size:11px;font-weight:500;" +
    "color:var(--accent-text);background:var(--accent-soft);text-decoration:none;" +
    "border-radius:999px;padding:3px 9px;transition:box-shadow .15s ease}" +
    ".pill:hover{box-shadow:inset 0 0 0 1px var(--accent)}" +
    ".pill img{height:11px;width:auto;display:block}" +
    ".x{width:28px;height:28px;border:none;background:none;color:var(--muted);" +
    "cursor:pointer;font-size:14px;border-radius:8px;display:flex;align-items:center;justify-content:center}" +
    ".x:hover{background:var(--card);color:var(--fg)}" +
    ".x.new{margin-left:auto;font-size:16px}" +

    ".msgs{flex:1;min-height:0;overflow-y:auto;padding:16px;display:flex;flex-direction:column;gap:12px;" +
    "scrollbar-width:thin;scrollbar-color:var(--line) transparent}" +
    ".msgs::-webkit-scrollbar{width:6px}" +
    ".msgs::-webkit-scrollbar-thumb{background:var(--line);border-radius:3px}" +
    ".msgs::-webkit-scrollbar-track{background:transparent}" +

    ".hello{margin:auto 0;display:flex;flex-direction:column;gap:14px;padding:10px 6px}" +
    ".hello h3{font-size:16px;font-weight:600}" +
    ".hello p{font-size:13px;color:var(--muted);line-height:1.6}" +
    ".qlabel{font-size:10.5px;font-weight:600;letter-spacing:.09em;color:var(--muted);margin-top:2px}" +
    ".chips{display:flex;flex-direction:column;gap:8px;align-items:flex-start}" +
    ".chip{position:relative;overflow:hidden;" +
    "border:1px solid var(--line);background:var(--card);color:var(--fg);cursor:pointer;" +
    "border-radius:999px;padding:7px 14px;font-size:13px;text-align:left;" +
    "transition:border-color .15s ease,background .15s ease,transform .15s ease,box-shadow .15s ease}" +
    ".chip:hover{border-color:var(--accent);color:var(--accent-text);background:var(--accent-soft);" +
    "transform:translateY(-1px);box-shadow:0 3px 10px rgba(0,0,0,.07)}" +
    // border beam: a light with a tail orbits the outline of the lit chip.
    // The span is masked down to a 2px ring; its ::before is an oversized
    // rotating conic gradient whose bright head traces the border.
    ".beam{position:absolute;inset:0;border-radius:inherit;padding:2px;display:none;" +
    "pointer-events:none;" +
    "-webkit-mask:linear-gradient(#000 0 0) content-box,linear-gradient(#000 0 0);" +
    "-webkit-mask-composite:xor;" +
    "mask:linear-gradient(#000 0 0) content-box,linear-gradient(#000 0 0);" +
    "mask-composite:exclude}" +
    ".chip.glint .beam{display:block}" +
    ".beam::before{content:'';position:absolute;left:50%;top:50%;width:220%;aspect-ratio:1;" +
    "transform:translate(-50%,-50%) rotate(0deg);" +
    "background:conic-gradient(transparent 0 265deg," +
    "color-mix(in srgb,var(--accent) 50%,transparent) 330deg,var(--accent) 356deg," +
    "transparent 360deg);" +
    "animation:orbit 3.2s linear infinite}" +
    "@keyframes orbit{to{transform:translate(-50%,-50%) rotate(360deg)}}" +

    ".m{max-width:88%;font-size:13.5px;line-height:1.6;overflow-wrap:break-word;" +
    "animation:rise .22s ease}" +
    "@keyframes rise{from{opacity:0;transform:translateY(6px)}to{opacity:1;transform:none}}" +
    ".m.user{align-self:flex-end;background:var(--accent-soft);color:var(--fg);" +
    "border-radius:14px 14px 4px 14px;padding:9px 14px;white-space:pre-wrap}" +
    ".m.bot{align-self:stretch;max-width:100%;padding:2px 2px 6px}" +
    ".m.bot p{margin:0 0 10px}.m.bot p:last-child,.m.bot ul:last-child,.m.bot ol:last-child{margin-bottom:0}" +
    ".m.bot ul,.m.bot ol{margin:0 0 10px;padding-left:20px}.m.bot li{margin:3px 0}" +
    ".m.bot li>ul,.m.bot li>ol{margin:3px 0 0}" +
    ".m.bot h3{font-size:13.5px;font-weight:650;margin:12px 0 4px}" +
    ".m.bot h4{font-size:13px;font-weight:600;margin:10px 0 4px}" +
    ".m.bot h3:first-child,.m.bot h4:first-child{margin-top:0}" +
    ".m.bot hr{border:none;border-top:1px solid var(--line);margin:10px 0}" +
    ".m.bot .tblwrap{overflow-x:auto;max-width:100%;margin:8px 0}" +
    ".m.bot table{border-collapse:collapse;font-size:12px;line-height:1.45}" +
    ".m.bot th,.m.bot td{border:1px solid var(--line);padding:4px 8px;text-align:left;vertical-align:top}" +
    ".m.bot th{background:var(--card);font-weight:600}" +
    ".m.bot code{background:var(--code);padding:1px 5px;border-radius:5px;" +
    "font-size:12px;font-family:ui-monospace,SFMono-Regular,Menlo,monospace}" +
    ".m.bot pre{background:#16181d;color:#e6e6ea;padding:11px 13px;border-radius:10px;" +
    "overflow-x:auto;max-width:100%;margin:8px 0;font-size:12px;line-height:1.55;" +
    "scrollbar-width:thin;scrollbar-color:rgba(255,255,255,.25) transparent}" +
    ".m.bot pre::-webkit-scrollbar{height:6px}" +
    ".m.bot pre::-webkit-scrollbar-thumb{background:rgba(255,255,255,.22);border-radius:3px}" +
    ".m.bot pre::-webkit-scrollbar-track{background:transparent}" +
    ".m.bot pre code{background:none;border:none;padding:0;color:inherit;font-size:inherit}" +
    ".c-kw{color:#c792ea}.c-ty{color:#ffcb6b}.c-str{color:#c3e88d}" +
    ".c-com{color:#7a8494;font-style:italic}.c-num{color:#f78c6c}.c-mac{color:#89ddff}" +
    ".m.bot a{color:var(--accent-text);text-decoration:underline;text-underline-offset:2px}" +

    ".status{align-self:flex-start;display:flex;align-items:center;gap:8px;font-size:12.5px;" +
    "color:var(--muted);padding:2px 4px;max-width:100%}" +
    ".status .stxt{max-width:290px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;" +
    "font-style:italic}" +
    ".trace{margin-top:9px;border-top:1px dashed var(--line);padding-top:7px}" +
    ".trace summary{cursor:pointer;font-size:11px;color:var(--muted);user-select:none;" +
    "list-style:none;display:flex;align-items:center;gap:5px}" +
    ".trace summary::before{content:'▸';font-size:9px;transition:transform .15s ease}" +
    ".trace[open] summary::before{transform:rotate(90deg)}" +
    ".trace-body{margin-top:6px;font-size:11.5px;color:var(--muted);line-height:1.55;" +
    "white-space:pre-wrap}" +
    ".trace-q{margin-top:6px;display:flex;gap:6px;align-items:baseline;font-size:11px;" +
    "color:var(--accent-text);font-family:ui-monospace,SFMono-Regular,Menlo,monospace}" +
    ".trace-q::before{content:'⌕';font-size:12px}" +
    ".dots{display:inline-flex;gap:3px}" +
    ".dots i{width:4px;height:4px;border-radius:50%;background:var(--accent);opacity:.4;" +
    "animation:blink 1.2s infinite}" +
    ".dots i:nth-child(2){animation-delay:.2s}.dots i:nth-child(3){animation-delay:.4s}" +
    "@keyframes blink{0%,80%,100%{opacity:.35}40%{opacity:1}}" +

    ".foot{display:flex;gap:8px;padding:12px 14px 8px;border-top:1px solid var(--line);background:var(--bg)}" +
    "textarea{flex:1;resize:none;border:1px solid var(--line);background:var(--card);color:var(--fg);" +
    "border-radius:12px;padding:9px 13px;font-size:13.5px;line-height:1.4;height:40px;" +
    "max-height:120px;outline:none;" +
    "transition:border-color .15s ease,box-shadow .15s ease}" +
    "textarea::placeholder{color:var(--muted)}" +
    "textarea:focus{border-color:var(--accent);" +
    "box-shadow:0 0 0 3px color-mix(in srgb,var(--accent) 14%,transparent)}" +
    ".send{width:40px;height:40px;flex:none;border:1px solid var(--line);background:var(--card);" +
    "color:var(--muted);border-radius:12px;cursor:pointer;display:flex;align-items:center;" +
    "justify-content:center;align-self:flex-end;" +
    "transition:color .15s ease,border-color .15s ease,background .15s ease}" +
    ".send:hover{color:var(--accent-text);border-color:var(--accent)}" +
    ".send.ready:not(:disabled){background:var(--accent);border-color:var(--accent);color:#fff}" +
    ".send.ready:not(:disabled):hover{filter:brightness(1.07)}" +
    ".send:disabled{opacity:.45;cursor:default}" +
    ".send svg{width:16px;height:16px}" +

    ".brand{display:flex;justify-content:center;padding:0 14px 9px;background:var(--bg)}" +
    ".brand a{display:inline-flex;align-items:center;gap:5px;font-size:11px;" +
    "color:var(--muted);text-decoration:none;transition:color .15s ease}" +
    ".brand a:hover{color:var(--accent-text)}" +
    ".brand img{height:12px;width:auto;display:block}" +

    "@media (prefers-reduced-motion: reduce){*{animation:none!important;transition:none!important}" +
    ".beam{display:none!important}}" +
    "</style>" +
    '<div class="wrap">' +
    '<button class="fab">✦ Ask AI</button>' +
    '<div class="veil"></div>' +
    '<div class="panel">' +
    '<div class="head"><div class="mark"><img src="' + DSRS_LOGO + '" alt="DSRs"></div>' +
    "<b>Ask DSRs</b>" +
    '<a class="pill" href="https://www.mixedbread.com/blog/toast-1" target="_blank" ' +
    'rel="noopener" title="toast-1 by Mixedbread">' +
    '<img src="' + MXB_LOGO + '" alt="Mixedbread"> toast-1</a>' +
    '<button class="x new" title="New chat">＋</button>' +
    '<button class="x close" title="Close (esc)">✕</button></div>' +
    '<div class="msgs"></div>' +
    '<div class="foot"><textarea rows="1" placeholder="Ask about DSRs…"></textarea>' +
    '<button class="send" title="Send">' +
    '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" ' +
    'stroke-linecap="round" stroke-linejoin="round"><path d="M12 19V5M5 12l7-7 7 7"/></svg>' +
    "</button></div>" +
    '<div class="brand"><a href="https://www.mixedbread.com" target="_blank" rel="noopener">' +
    'Powered by <img src="' + MXB_LOGO + '" alt=""> Mixedbread</a></div>' +
    "</div></div>";

  var wrap = root.querySelector(".wrap");
  var fab = root.querySelector(".fab");
  var panel = root.querySelector(".panel");
  var msgs = root.querySelector(".msgs");
  var input = root.querySelector("textarea");
  var send = root.querySelector(".send");

  function makeHello() {
    var h = document.createElement("div");
    h.className = "hello";
    h.innerHTML =
      "<h3>Hi, ask me about DSRs</h3>" +
      "<p>Answers come straight from the documentation and source code, " +
      "searched and composed by toast-1.</p>" +
      '<div class="qlabel">EXAMPLE QUESTIONS</div>' +
      '<div class="chips"></div>';
    var chips = h.querySelector(".chips");
    SUGGESTIONS.forEach(function (q) {
      var c = document.createElement("button");
      c.className = "chip";
      c.textContent = q;
      var b = document.createElement("span");
      b.className = "beam";
      c.appendChild(b);
      c.addEventListener("click", function () {
        input.value = q;
        askToast();
      });
      chips.appendChild(c);
    });
    return h;
  }

  function newChat() {
    if (send.disabled) return; // don't clear mid-answer
    history = [];
    msgs.innerHTML = "";
    msgs.appendChild(makeHello());
    resetInput();
    input.focus();
    glintTick();
  }
  msgs.appendChild(makeHello());

  // Follow Mintlify's explicit theme class; fall back to the OS preference
  // only when the page hasn't declared one.
  function theme() {
    var cl = document.documentElement.classList;
    var dark = cl.contains("dark")
      ? true
      : cl.contains("light")
        ? false
        : window.matchMedia && matchMedia("(prefers-color-scheme: dark)").matches;
    wrap.classList.toggle("dark", dark);
  }
  new MutationObserver(theme).observe(document.documentElement, {
    attributes: true,
    attributeFilter: ["class", "data-theme"],
  });
  theme();

  // the beam orbits one chip's border; after each full lap it hops to the
  // next chip, so the light never stops moving while suggestions are shown
  var glintI = 0;
  function glintTick() {
    var chips = msgs.querySelectorAll(".chip");
    if (!chips.length) return;
    chips.forEach(function (c) {
      c.classList.remove("glint");
    });
    var chip = chips[glintI % chips.length];
    void chip.offsetWidth; // restart the lap even on a repeat visit
    chip.classList.add("glint");
    glintI++;
  }
  setInterval(function () {
    if (panel.classList.contains("open")) glintTick(); // paused while closed
  }, 3200);

  // textarea grows with input (up to max-height); send lights up when ready
  function syncInput() {
    input.style.height = "auto";
    input.style.height = Math.min(input.scrollHeight, 120) + "px";
    send.classList.toggle("ready", !!input.value.trim());
  }
  function resetInput() {
    input.value = "";
    input.style.height = "";
    send.classList.remove("ready");
  }
  input.addEventListener("input", syncInput);

  function setOpen(open) {
    panel.classList.toggle("open", open);
    wrap.classList.toggle("open", open);
    root.querySelector(".veil").classList.toggle("open", open);
    if (open) {
      input.focus();
      glintTick(); // light a chip right away instead of waiting for a tick
    }
  }
  fab.addEventListener("click", function () {
    setOpen(true);
  });
  root.querySelector(".x.close").addEventListener("click", function () {
    setOpen(false);
  });
  root.querySelector(".x.new").addEventListener("click", newChat);
  document.addEventListener("keydown", function (e) {
    if (e.key === "Escape") setOpen(false);
  });
  root.querySelector(".veil").addEventListener("click", function () {
    setOpen(false);
  });

  function esc(s) {
    return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  }

  // tiny Rust-leaning highlighter (CSP forbids external libs); runs on
  // escaped text, so entities like &lt; pass through untouched
  var HL_RE = new RegExp(
    "(\\/\\/[^\\n]*)" + // comment
      '|("(?:[^"\\\\\\n]|\\\\.)*")' + // string
      "|\\b(fn|let|mut|pub|impl|trait|struct|enum|match|if|else|for|while|loop|" +
      "return|use|mod|where|async|await|move|ref|type|const|static|dyn|self|" +
      "Self|super|crate|in|as|break|continue|unsafe)\\b" + // keyword
      "|\\b([A-Z][A-Za-z0-9_]*)\\b" + // type
      "|\\b(\\d[\\d_]*(?:\\.\\d+)?)\\b" + // number
      "|\\b([a-z_][a-z0-9_]*!)", // macro
    "g"
  );

  function hl(src) {
    return esc(src).replace(HL_RE, function (m, com, str, kw, ty, num, mac) {
      var cls = com ? "c-com" : str ? "c-str" : kw ? "c-kw" : ty ? "c-ty" : num ? "c-num" : "c-mac";
      return '<span class="' + cls + '">' + m + "</span>";
    });
  }

  // minimal markdown, parsed line by line so partial streamed input and
  // backticks quoted mid-sentence (e.g. .dsrs js``` syntax) can't derail it:
  // fences open only at line starts, ordered lists keep the author's
  // numbering via <li value>, plus nested lists, tables, headings, hr,
  // inline code / bold / italic / links
  function inline(s) {
    return esc(s)
      .replace(/`([^`\n]+)`/g, "<code>$1</code>")
      .replace(/\*\*([^*]+)\*\*/g, "<b>$1</b>")
      .replace(/(^|[\s(])\*([^*\n]+)\*(?=$|[\s).,;:!?])/g, "$1<i>$2</i>")
      .replace(/\[([^\]]+)\]\((https?:[^)\s]+)\)/g, '<a href="$2" target="_blank">$1</a>');
  }

  function md(s) {
    var html = "";
    var para = []; // pending paragraph lines (already inlined)
    var tbl = null; // pending table lines (raw)
    var code = null; // pending fence body lines while inside ```
    var lists = []; // open lists: {tag, indent, liOpen}

    function flushPara() {
      if (para.length) {
        html += "<p>" + para.join("<br>") + "</p>";
        para = [];
      }
    }
    function cells(row) {
      return row
        .replace(/^\s*\|/, "")
        .replace(/\|\s*$/, "")
        .split("|")
        .map(function (c) { return c.trim(); });
    }
    function flushTable() {
      if (!tbl) return;
      var rows = tbl;
      tbl = null;
      if (rows.length >= 2 && /^[\s:|-]+$/.test(rows[1]) && rows[1].indexOf("-") !== -1) {
        html += '<div class="tblwrap"><table><tr>' +
          cells(rows[0]).map(function (c) { return "<th>" + inline(c) + "</th>"; }).join("") +
          "</tr>" +
          rows.slice(2).map(function (r) {
            return "<tr>" +
              cells(r).map(function (c) { return "<td>" + inline(c) + "</td>"; }).join("") +
              "</tr>";
          }).join("") +
          "</table></div>";
      } else {
        // no separator row (yet) — fall back to plain lines
        para = para.concat(rows.map(inline));
        flushPara();
      }
    }
    function closeList() {
      var l = lists.pop();
      html += (l.liOpen ? "</li>" : "") + "</" + l.tag + ">";
    }
    function closeAllLists() {
      while (lists.length) closeList();
    }

    var lines = s.split("\n");
    for (var i = 0; i < lines.length; i++) {
      var line = lines[i];

      if (code) {
        if (/^\s*```/.test(line)) {
          html += "<pre><code>" + hl(code.join("\n")) + "</code></pre>";
          code = null;
        } else {
          code.push(line);
        }
        continue;
      }
      if (/^\s*```/.test(line)) {
        flushPara();
        flushTable();
        closeAllLists();
        code = [];
        continue;
      }

      if (/^\s*\|.*\|\s*$/.test(line)) {
        flushPara();
        (tbl = tbl || []).push(line);
        continue;
      }
      flushTable();

      if (!line.trim()) {
        flushPara(); // blank lines end paragraphs but not lists
        continue;
      }

      var li = /^(\s*)([-*]|\d+[.)])\s+(.*)$/.exec(line);
      if (li) {
        flushPara();
        var indent = li[1].replace(/\t/g, "    ").length;
        var tag = /\d/.test(li[2]) ? "ol" : "ul";
        var top = lists[lists.length - 1];
        while (top && indent < top.indent - 1) {
          closeList();
          top = lists[lists.length - 1];
        }
        if (top && indent <= top.indent + 1) {
          if (top.liOpen) {
            html += "</li>";
            top.liOpen = false;
          }
          if (top.tag !== tag) {
            closeList();
            top = lists[lists.length - 1];
          }
        }
        top = lists[lists.length - 1];
        if (!top || indent > top.indent + 1) {
          html += "<" + tag + ">"; // nested lists live inside the open <li>
          lists.push({ tag: tag, indent: indent, liOpen: false });
          top = lists[lists.length - 1];
        }
        html += "<li" +
          (tag === "ol" ? ' value="' + parseInt(li[2], 10) + '"' : "") +
          ">" + inline(li[3]);
        top.liOpen = true;
        continue;
      }

      var h = /^\s*(#{1,6})\s+(.*)$/.exec(line);
      if (h) {
        flushPara();
        closeAllLists();
        var hTag = h[1].length <= 2 ? "h3" : "h4";
        html += "<" + hTag + ">" + inline(h[2]) + "</" + hTag + ">";
        continue;
      }
      if (/^\s*([-*_])\s*(\1\s*){2,}$/.test(line)) {
        flushPara();
        closeAllLists();
        html += "<hr>";
        continue;
      }

      closeAllLists();
      para.push(inline(line));
    }

    if (code) html += "<pre><code>" + hl(code.join("\n")) + "</code></pre>"; // fence still streaming
    flushPara();
    flushTable();
    closeAllLists();
    return html;
  }

  function bubble(cls, text) {
    var el = document.createElement("div");
    el.className = "m " + cls;
    if (cls === "user") el.textContent = text;
    else el.innerHTML = md(text);
    msgs.appendChild(el);
    msgs.scrollTop = msgs.scrollHeight;
    return el;
  }

  function askToast() {
    var q = input.value.trim();
    if (!q || send.disabled) return;
    if (ENDPOINT.indexOf("YOUR-SUBDOMAIN") !== -1) {
      var h0 = msgs.querySelector(".hello");
      if (h0) h0.remove();
      bubble("user", q);
      resetInput();
      bubble(
        "bot",
        "**The chat backend isn't deployed yet.** Deploy `docs-chat-worker/` " +
          "with wrangler and set its URL in `toast-chat.js` (see docs/README.md)."
      );
      return;
    }
    resetInput();
    send.disabled = true;
    var h = msgs.querySelector(".hello");
    if (h) h.remove();
    bubble("user", q);
    history.push({ role: "user", content: q });

    var status = document.createElement("div");
    status.className = "status";
    status.innerHTML =
      '<span class="dots"><i></i><i></i><i></i></span><span class="stxt">Searching the docs…</span>';
    var stxt = status.querySelector(".stxt");
    msgs.appendChild(status);
    msgs.scrollTop = msgs.scrollHeight;

    var answer = "";
    var el = null;
    // ordered trace: toast's reasoning text interleaved with its searches
    var events = [];
    var seenCalls = {};

    function setStatus(text) {
      if (status.isConnected) {
        stxt.textContent = text.length > 90 ? "…" + text.slice(-90) : text;
      }
    }

    function progress(text) {
      var last = events[events.length - 1];
      if (last && last.t === "txt") {
        last.s += text;
      } else {
        last = { t: "txt", s: text };
        events.push(last);
      }
      var lines = last.s.trim().split(/\n+/);
      var line = lines[lines.length - 1].trim();
      if (line) setStatus(line);
    }

    function onSearch(call) {
      if (!call || !call.id || seenCalls[call.id]) return;
      seenCalls[call.id] = true;
      var q, label;
      if (call.type === "store_search_call") {
        q = (call.queries || []).join(" · ");
        label = "Searching: “" + q + "”";
      } else if (call.type === "store_grep_call") {
        q = "grep " + (call.pattern || "");
        label = "Grepping: " + (call.pattern || "");
      }
      if (!q) return;
      events.push({ t: "q", s: q });
      setStatus(label);
    }

    fetch(ENDPOINT, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ messages: history }),
    })
      .then(function (resp) {
        if (!resp.ok) throw new Error("proxy returned " + resp.status);
        var reader = resp.body.getReader();
        var dec = new TextDecoder();
        var buf = "";
        function pump() {
          return reader.read().then(function (r) {
            if (r.done) return;
            buf += dec.decode(r.value, { stream: true });
            var lines = buf.split("\n");
            buf = lines.pop();
            lines.forEach(function (line) {
              if (line.indexOf("data: ") !== 0) return;
              var payload = line.slice(6);
              if (payload === "[DONE]") return;
              var chunk, delta;
              try {
                chunk = JSON.parse(payload);
              } catch (e) {
                return;
              }
              delta = (chunk.choices && chunk.choices[0] && chunk.choices[0].delta) || null;
              (chunk.hosted_tool_calls || []).forEach(onSearch);
              if (delta && delta.reasoning_content) progress(delta.reasoning_content);
              if (delta && delta.content) {
                if (!el) {
                  status.remove();
                  el = bubble("bot", "");
                }
                answer += delta.content;
                el.innerHTML = md(answer);
                msgs.scrollTop = msgs.scrollHeight;
              }
            });
            return pump();
          });
        }
        return pump();
      })
      .then(function () {
        if (answer) {
          history.push({ role: "assistant", content: answer });
          if (el && events.length) {
            var trace = document.createElement("details");
            trace.className = "trace";
            var sum = document.createElement("summary");
            var n = events.filter(function (ev) { return ev.t === "q"; }).length;
            sum.textContent = "trace" + (n ? " · " + n + " search" + (n > 1 ? "es" : "") : "");
            trace.appendChild(sum);
            events.forEach(function (ev) {
              var row = document.createElement("div");
              if (ev.t === "q") {
                row.className = "trace-q";
                row.textContent = ev.s;
              } else {
                var text = ev.s.trim().replace(/\n{3,}/g, "\n\n");
                if (!text) return;
                row.className = "trace-body";
                row.textContent = text;
              }
              trace.appendChild(row);
            });
            el.appendChild(trace);
          }
        } else {
          status.remove();
          bubble("bot", "No answer came back — try again.");
        }
      })
      .catch(function (err) {
        status.remove();
        bubble("bot", "**Chat is unavailable:** " + err.message);
      })
      .finally(function () {
        send.disabled = false;
        input.focus();
      });
  }

  send.addEventListener("click", askToast);
  input.addEventListener("keydown", function (e) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      askToast();
    }
  });

  document.body.appendChild(host);
})();
