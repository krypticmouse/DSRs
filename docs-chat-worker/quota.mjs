const DAY_MS = 86_400_000;

export function takeDailyBudget(storage, limit, now) {
  if (!Number.isSafeInteger(limit) || limit <= 0) {
    throw new Error("CHAT_DAILY_BUDGET must be a positive integer");
  }

  const day = Math.floor(now / DAY_MS);
  return storage.transactionSync(() => {
    const previous = storage.kv.get("daily");
    const count = previous?.day === day ? previous.count : 0;
    if (!Number.isSafeInteger(count) || count < 0) {
      throw new Error("Invalid stored chat quota");
    }
    if (count >= limit) return false;
    storage.kv.put("daily", { day, count: count + 1 });
    return true;
  });
}
