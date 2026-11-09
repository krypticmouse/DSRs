import assert from "node:assert/strict";
import test from "node:test";
import { corsHeaders } from "./cors.mjs";

test("only configured browser origins receive CORS access", () => {
  const configured = "https://dsrs.herumbshandilya.com";
  assert.equal(
    corsHeaders("https://dsrs.herumbshandilya.com", configured)[
      "Access-Control-Allow-Origin"
    ],
    "https://dsrs.herumbshandilya.com",
  );
  assert.equal(corsHeaders("null", configured), null);
  assert.equal(corsHeaders("https://unlisted.example", configured), null);
  assert.ok(corsHeaders("", configured)); // direct clients have no browser origin
});

test("an explicit wildcard permits any origin", () => {
  assert.equal(
    corsHeaders("https://other.example", "*")["Access-Control-Allow-Origin"],
    "https://other.example",
  );
});
