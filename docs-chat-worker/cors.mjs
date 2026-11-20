export function corsHeaders(origin, configuredOrigins) {
  const allowed = (configuredOrigins || "*").split(",").map((value) => value.trim());
  if (origin && !allowed.includes("*") && !allowed.includes(origin)) return null;

  return {
    "Access-Control-Allow-Origin": origin || "*",
    "Access-Control-Allow-Methods": "POST, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type",
  };
}
