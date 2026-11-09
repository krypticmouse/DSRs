import assert from "node:assert/strict";
import test from "node:test";
import { takeDailyBudget } from "./quota.mjs";

function storage() {
  const values = new Map();
  return {
    kv: {
      get: (key) => values.get(key),
      put: (key, value) => values.set(key, value),
    },
    transactionSync: (fn) => fn(),
  };
}

test("the daily budget blocks further upstream calls and resets the next day", () => {
  const state = storage();
  assert.equal(takeDailyBudget(state, 2, 0), true);
  assert.equal(takeDailyBudget(state, 2, 1), true);
  assert.equal(takeDailyBudget(state, 2, 2), false);
  assert.equal(takeDailyBudget(state, 2, 86_400_000), true);
});

test("invalid limits and stored counters fail closed", () => {
  const state = storage();
  assert.throws(() => takeDailyBudget(state, 0, 0));
  state.kv.put("daily", { day: 0, count: NaN });
  assert.throws(() => takeDailyBudget(state, 2, 0));
});
