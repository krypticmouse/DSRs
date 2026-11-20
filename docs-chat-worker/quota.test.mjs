import assert from "node:assert/strict";
import test from "node:test";
import { quotaClientKey, takeDailyBudget } from "./quota.mjs";

const first = "a".repeat(64);
const second = "b".repeat(64);

function storage() {
  const values = new Map();
  return {
    kv: {
      get: (key) => structuredClone(values.get(key)),
      put: (key, value) => values.set(key, structuredClone(value)),
    },
    transactionSync: (fn) => fn(),
  };
}

test("a saturated client leaves capacity for others and both limits reset", () => {
  const state = storage();
  assert.equal(takeDailyBudget(state, 3, 2, first, 0), null);
  assert.equal(takeDailyBudget(state, 3, 2, first, 1), null);
  assert.deepEqual(takeDailyBudget(state, 3, 2, first, 2), { retryAfter: 86_400 });
  assert.equal(takeDailyBudget(state, 3, 2, second, 3), null);
  assert.deepEqual(takeDailyBudget(state, 3, 2, second, 86_399_999), { retryAfter: 1 });
  assert.equal(takeDailyBudget(state, 3, 2, first, 86_400_000), null);
  assert.deepEqual(state.kv.get("daily"), { day: 1, count: 1, clients: { [first]: 1 } });
});

test("the rollout preserves the active day’s aggregate usage", () => {
  const state = storage();
  state.kv.put("daily", { day: 0, count: 2 });
  assert.equal(takeDailyBudget(state, 3, 1, first, 0), null);
  assert.deepEqual(takeDailyBudget(state, 3, 1, second, 0), { retryAfter: 86_400 });
});

test("invalid limits and stored counters fail closed", () => {
  const state = storage();
  assert.throws(() => takeDailyBudget(state, 3, 3, first, 0));
  assert.throws(() => takeDailyBudget(state, 3, 1, "not-a-key", 0));
  state.kv.put("daily", { day: 0, count: NaN });
  assert.throws(() => takeDailyBudget(state, 3, 1, first, 0));
  state.kv.put("daily", { day: 0, count: 1, clients: { [first]: -1 } });
  assert.throws(() => takeDailyBudget(state, 3, 1, first, 0));
});

test("the edge address determines the bucket across equivalent network forms", async () => {
  const key = (address) => quotaClientKey({ headers: new Headers({
    "CF-Connecting-IP": address, "X-Forwarded-For": "192.0.2.200",
  }) });
  const ipv4 = await key("192.0.2.1");
  assert.match(ipv4, /^[0-9a-f]{64}$/);
  assert.equal(await key("::ffff:192.0.2.1"), ipv4);
  assert.notEqual(await key("192.0.2.2"), ipv4);
  assert.equal(await key("2001:db8:0:0::1"), await key("2001:0DB8::ffff"));
  assert.notEqual(await key("2001:db8:0:1::1"), await key("2001:db8::1"));
  for (const invalid of ["", "192.0.2.1, 192.0.2.2", "192.0.2.256", "2001:::1"]) {
    assert.equal(await key(invalid), null);
  }
});
