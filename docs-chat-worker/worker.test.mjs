import assert from "node:assert/strict";
import { registerHooks } from "node:module";
import test from "node:test";

const hooks = registerHooks({
  resolve(specifier, context, next) {
    if (specifier === "cloudflare:workers") {
      return {
        url: "data:text/javascript,export class DurableObject { constructor(ctx,env) { this.ctx=ctx; this.env=env; } }",
        shortCircuit: true,
      };
    }
    return next(specifier, context);
  },
});
const { default: worker, ChatQuota } = await import("./worker.js");
hooks.deregister();

test("the Worker charges fallback and preserves capacity for another client", async (t) => {
  t.mock.timers.enable({ apis: ["Date"], now: 1_000 });
  let attempts = 0;
  const calls = t.mock.method(globalThis, "fetch", async () => (
    ++attempts === 1
      ? new Response("store unavailable", { status: 502 })
      : new Response("event", { status: 200 })
  ));
  const state = new Map();
  const env = {
    ALLOWED_ORIGINS: "https://docs.example",
    CHAT_DAILY_BUDGET: "3",
    CHAT_CLIENT_DAILY_BUDGET: "2",
    MXBAI_API_KEY: "test-only",
    CHAT_QUOTA: { getByName(name) {
      assert.equal(name, "global");
      return quota;
    } },
  };
  const quota = new ChatQuota({ storage: {
    kv: { get: (key) => state.get(key), put: (key, value) => state.set(key, value) },
    transactionSync: (fn) => fn(),
  } }, env);
  const chat = (ip) => worker.fetch(new Request("https://worker.example", {
    method: "POST",
    headers: {
      Origin: "https://docs.example",
      ...(ip ? { "CF-Connecting-IP": ip } : {}),
    },
    body: JSON.stringify({ messages: [{ role: "user", content: "Hello" }] }),
  }), env);

  assert.equal((await chat()).status, 503);
  assert.equal(calls.mock.callCount(), 0);
  const first = await chat("192.0.2.1");
  assert.equal(first.status, 200);
  assert.equal(first.headers.get("Content-Type"), "text/event-stream");
  assert.equal(await first.text(), "event");
  assert.equal(calls.mock.callCount(), 2);
  const blocked = await chat("192.0.2.1");
  assert.equal(blocked.status, 429);
  assert.equal(blocked.headers.get("Retry-After"), "86399");
  assert.equal(calls.mock.callCount(), 2);
  assert.equal((await chat("192.0.2.2")).status, 200);
  assert.equal((await chat("192.0.2.3")).status, 429);
  assert.equal(calls.mock.callCount(), 3);
});
