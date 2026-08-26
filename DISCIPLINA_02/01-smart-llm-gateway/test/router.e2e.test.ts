import test from "node:test";
import assert from "node:assert/strict";
import { createServer } from "../src/server.ts";
import { config } from "../src/config.ts";
import { type LLMResponse, OpenRouterService } from "../src/openRouterService.ts";

console.assert(
  process.env.API_KEY_OPENROUTER,
  "API_KEY_OPENROUTER environment variable is not set",
);

// test.todo("Sla, escreve um caso de teste ai");
test("Im poor, use the cheapest.", async () => {
  const customConfig = {
    ...config,
    provider: {
      ...config.provider,
      sort: {
        ...config.provider.sort,
        by: "price",
      },
    },
  };
  const openRouterService = new OpenRouterService(customConfig);
  const app = createServer(openRouterService);

  const response = await app.inject({
    method: "POST",
    url: "/chat",
    body: { message: "What is the cheapest model?" },
  });

  assert.equal(response.statusCode, 200);
  const body = response.json() as LLMResponse;
});

test("Im lazy, use the fastest throughput.", async () => {
  const customConfig = {
    ...config,
    provider: {
      ...config.provider,
      sort: {
        ...config.provider.sort,
        by: "throughput",
      },
    },
  };
  const openRouterService = new OpenRouterService(customConfig);
  const app = createServer(openRouterService);

  const response = await app.inject({
    method: "POST",
    url: "/chat",
    body: { message: "What is the fastest model?" },
  });

  assert.equal(response.statusCode, 200);
  const body = response.json() as LLMResponse;
});