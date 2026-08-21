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
    "What is a signature?",
    "How do optimizers mutate holes?",
    "How is Predict::forward implemented?",
  ];

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
    "transition:color .15s ease,border-color .15s ease,opacity .15s ease}" +
    ".fab:hover{opacity:1;color:var(--accent-text);border-color:var(--accent)}" +
    ".wrap.open .fab{opacity:0;pointer-events:none}" +

    ".panel{position:fixed;top:0;right:0;bottom:0;width:min(400px,100vw);" +
    "display:flex;flex-direction:column;background:var(--bg);color:var(--fg);" +
    "border-left:1px solid var(--line);box-shadow:-8px 0 32px rgba(0,0,0,.08);" +
    "transform:translateX(103%);pointer-events:none;" +
    "transition:transform .24s cubic-bezier(.32,.72,.24,1);overflow:hidden}" +
    ".panel.open{transform:none;pointer-events:auto}" +

    ".head{display:flex;align-items:center;gap:10px;padding:14px 16px;border-bottom:1px solid var(--line)}" +
    ".mark{width:28px;height:28px;border-radius:8px;background:var(--accent-soft);color:var(--accent);" +
    "display:flex;align-items:center;justify-content:center;font-size:15px}" +
    ".head b{font-size:14px;font-weight:600}" +
    ".pill{font-size:11px;font-weight:500;color:var(--accent-text);background:var(--accent-soft);" +
    "border-radius:999px;padding:3px 9px}" +
    ".x{width:28px;height:28px;border:none;background:none;color:var(--muted);" +
    "cursor:pointer;font-size:14px;border-radius:8px;display:flex;align-items:center;justify-content:center}" +
    ".x:hover{background:var(--card);color:var(--fg)}" +
    ".x.new{margin-left:auto;font-size:16px}" +

    ".msgs{flex:1;overflow-y:auto;padding:16px;display:flex;flex-direction:column;gap:12px}" +

    ".hello{margin:auto 0;display:flex;flex-direction:column;gap:14px;padding:8px 4px}" +
    ".hello h3{font-size:15px;font-weight:600}" +
    ".hello p{font-size:13px;color:var(--muted);line-height:1.6}" +
    ".chips{display:flex;flex-direction:column;gap:8px;align-items:flex-start}" +
    ".chip{border:1px solid var(--line);background:var(--card);color:var(--fg);cursor:pointer;" +
    "border-radius:999px;padding:7px 14px;font-size:13px;text-align:left;" +
    "transition:border-color .15s ease,background .15s ease}" +
    ".chip:hover{border-color:var(--accent);color:var(--accent-text);background:var(--accent-soft)}" +

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

    ".foot{display:flex;gap:8px;padding:12px 14px;border-top:1px solid var(--line);background:var(--bg)}" +
    "textarea{flex:1;resize:none;border:1px solid var(--line);background:var(--card);color:var(--fg);" +
    "border-radius:12px;padding:9px 13px;font-size:13.5px;line-height:1.4;height:40px;outline:none;" +
    "transition:border-color .15s ease,box-shadow .15s ease}" +
    "textarea::placeholder{color:var(--muted)}" +
    "textarea:focus{border-color:var(--accent)}" +
    ".send{width:40px;height:40px;flex:none;border:1px solid var(--line);background:var(--card);" +
    "color:var(--muted);border-radius:12px;cursor:pointer;display:flex;align-items:center;" +
    "justify-content:center;transition:color .15s ease,border-color .15s ease}" +
    ".send:hover{color:var(--accent-text);border-color:var(--accent)}" +
    ".send:disabled{opacity:.45;cursor:default}" +
    ".send svg{width:16px;height:16px}" +

    "@media (prefers-reduced-motion: reduce){*{animation:none!important;transition:none!important}}" +
    "</style>" +
    '<div class="wrap">' +
    '<button class="fab">✦ Ask AI</button>' +
    '<div class="panel">' +
    '<div class="head"><div class="mark">✦</div><b>Ask DSRs</b>' +
    '<span class="pill">toast-1</span>' +
    '<button class="x new" title="New chat">＋</button>' +
    '<button class="x close" title="Close (esc)">✕</button></div>' +
    '<div class="msgs"></div>' +
    '<div class="foot"><textarea rows="1" placeholder="Ask about DSRs…"></textarea>' +
    '<button class="send" title="Send">' +
    '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" ' +
    'stroke-linecap="round" stroke-linejoin="round"><path d="M12 19V5M5 12l7-7 7 7"/></svg>' +
    "</button></div></div></div>";

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
      "<h3>Hi — ask me about DSRs</h3>" +
      "<p>Answers come straight from the documentation and source code, " +
      "searched and composed by toast-1.</p>" +
      '<div class="chips"></div>';
    var chips = h.querySelector(".chips");
    SUGGESTIONS.forEach(function (q) {
      var c = document.createElement("button");
      c.className = "chip";
      c.textContent = q;
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
    input.value = "";
    input.focus();
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

  function setOpen(open) {
    panel.classList.toggle("open", open);
    wrap.classList.toggle("open", open);
    if (open) input.focus();
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
  document.addEventListener("click", function (e) {
    if (wrap.classList.contains("open") && e.composedPath().indexOf(host) === -1) {
      setOpen(false);
    }
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
      input.value = "";
      bubble(
        "bot",
        "**The chat backend isn't deployed yet.** Deploy `docs-chat-worker/` " +
          "with wrangler and set its URL in `toast-chat.js` (see docs/README.md)."
      );
      return;
    }
    input.value = "";
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
