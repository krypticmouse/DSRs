const DAY_MS = 86_400_000;

function clientNetwork(ip) {
  if (!ip || !/^[\da-f:.]+$/i.test(ip)) return null;
  if (!ip.includes(":")) {
    const parts = ip.split(".");
    return parts.length === 4 && parts.every(
      (part) => /^(0|[1-9]\d{0,2})$/.test(part) && Number(part) <= 255,
    ) ? ip : null;
  }

  // The URL parser validates IPv6 and canonicalizes equivalent spellings.
  let host;
  try {
    host = new URL(`http://[${ip}]/`).hostname.slice(1, -1);
  } catch {
    return null;
  }
  const [first, last] = host.split("::");
  const head = first ? first.split(":") : [];
  const tail = last ? last.split(":") : [];
  const groups = last === undefined
    ? head
    : [...head, ...Array(8 - head.length - tail.length).fill("0"), ...tail];
  if (groups.slice(0, 5).every((group) => group === "0") && groups[5] === "ffff") {
    const high = parseInt(groups[6], 16);
    const low = parseInt(groups[7], 16);
    return [high >> 8, high & 255, low >> 8, low & 255].join(".");
  }
  return groups.slice(0, 4).join(":") + "::/64";
}

export async function quotaClientKey(request) {
  // Use the edge-provided address, never caller-supplied forwarding headers.
  const network = clientNetwork(request.headers.get("CF-Connecting-IP"));
  if (!network) return null;
  const digest = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(network));
  return Array.from(new Uint8Array(digest), (byte) => byte.toString(16).padStart(2, "0")).join("");
}

export function takeDailyBudget(storage, globalLimit, clientLimit, clientKey, now) {
  if (!Number.isSafeInteger(globalLimit) || globalLimit <= 1 ||
      !Number.isSafeInteger(clientLimit) || clientLimit <= 0 || clientLimit >= globalLimit ||
      !/^[a-f0-9]{64}$/.test(clientKey) || !Number.isSafeInteger(now) || now < 0) {
    throw new Error("Invalid chat quota configuration or identity");
  }
  const day = Math.floor(now / DAY_MS);
  return storage.transactionSync(() => {
    const previous = storage.kv.get("daily");
    const current = previous?.day === day;
    const count = current ? previous.count : 0;
    // Older deployments stored only the day and aggregate count.
    const clients = current && Object.hasOwn(previous, "clients") ? previous.clients : {};
    if (!Number.isSafeInteger(count) || count < 0 ||
        !clients || typeof clients !== "object" || Array.isArray(clients)) {
      throw new Error("Invalid stored chat quota");
    }
    const clientCount = Object.hasOwn(clients, clientKey) ? clients[clientKey] : 0;
    if (!Number.isSafeInteger(clientCount) || clientCount < 0 || clientCount > count) {
      throw new Error("Invalid stored client quota");
    }
    if (count >= globalLimit || clientCount >= clientLimit) {
      return { retryAfter: Math.ceil(((day + 1) * DAY_MS - now) / 1000) };
    }
    clients[clientKey] = clientCount + 1;
    storage.kv.put("daily", { day, count: count + 1, clients });
    return null;
  });
}
