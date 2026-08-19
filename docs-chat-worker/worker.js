// Cloudflare Worker: proxies the docs chat widget to Mixedbread toast-1,
// keeping MXBAI_API_KEY server-side. Deploy from this directory:
//
//     npx wrangler deploy
//     npx wrangler secret put MXBAI_API_KEY
//
// Optional vars (wrangler.toml): STORE (default dsrs-docs),
// ALLOWED_ORIGINS (comma-separated; default *).

const SYSTEM = `You are the DSRs docs assistant. DSRs (dspy-rs) is a Rust \
framework for building and optimizing LM pipelines. Answer only from the \
attached stores: dsrs-docs (published documentation) and dsrs-code (the \
Rust sources under crates/). Search before answering; use grep for exact \
symbol or signature lookups. When an answer touches implementation, quote \
the real code with its file path. If the stores don't cover the question, \
say so instead of guessing. Answer in concise markdown.`;

export default {
  async fetch(request, env) {
    const origin = request.headers.get("Origin") || "";
    const allowed = (env.ALLOWED_ORIGINS || "*").split(",").map((s) => s.trim());
    const cors = {
      "Access-Control-Allow-Origin":
        allowed.includes("*") || allowed.includes(origin) ? origin || "*" : "null",
      "Access-Control-Allow-Methods": "POST, OPTIONS",
      "Access-Control-Allow-Headers": "Content-Type",
    };
    if (request.method === "OPTIONS") return new Response(null, { headers: cors });
    if (request.method !== "POST")
      return new Response("POST only", { status: 405, headers: cors });

    let messages;
    try {
      const raw = await request.text();
      if (raw.length > 32_000) throw new Error(); // bound input-token spend
      ({ messages } = JSON.parse(raw));
      if (!Array.isArray(messages) || !messages.length) throw new Error();
      if (messages.some((m) => typeof m.content !== "string" || m.content.length > 4_000))
        throw new Error();
    } catch {
      return new Response("body must be {messages: [...]} within size limits", {
        status: 400,
        headers: cors,
      });
    }

    const allStores = (env.STORES || "dsrs-docs,dsrs-code")
      .split(",")
      .map((s) => s.trim());

    const call = (stores) =>
      fetch("https://api.mixedbread.com/v1/chat/completions", {
        method: "POST",
        headers: {
          Authorization: `Bearer ${env.MXBAI_API_KEY}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          model: "toast-1",
          stream: true,
          max_tokens: 2048,
          messages: [
            { role: "system", content: SYSTEM },
            // keep history bounded; roles/content only, drop anything else
            ...messages.slice(-12).map(({ role, content }) => ({ role, content })),
          ],
          tools: [
            { type: "store_search", store_identifiers: stores },
            { type: "store_grep", store_identifiers: stores },
          ],
        }),
      });

    let upstream = await call(allStores);
    if (!upstream.ok && allStores.length > 1) {
      // a store may not exist yet (e.g. code not synced) — degrade to the first
      upstream = await call(allStores.slice(0, 1));
    }
    if (!upstream.ok) {
      const detail = await upstream.text();
      return new Response(`upstream ${upstream.status}: ${detail.slice(0, 300)}`, {
        status: 502,
        headers: cors,
      });
    }
    return new Response(upstream.body, {
      headers: { ...cors, "Content-Type": "text/event-stream" },
    });
  },
};
