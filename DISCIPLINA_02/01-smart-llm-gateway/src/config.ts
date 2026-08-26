console.assert(
  process.env.API_KEY_OPENROUTER,
  "API_KEY_OPENROUTER environment variable is not set",
);

export type ModelConfig = {
  apiKeyOpenRouter: string;
  httpReferer: string;
  xTitle: string;
  port: number;
  models: string[];
  temperature: number;
  maxTokens: number;
  systemPrompt: string;
  provider: {
    sort: {
      by: string;
      partition: string;
    };
  };
};

export const config: ModelConfig = {
  apiKeyOpenRouter: process.env.API_KEY_OPENROUTER!,
  httpReferer: process.env.HTTP_REFERER || "http://localhost:3000",
  xTitle: "Smart LLM Gateway",
  port: 3000,
  models: [
    // 'deepseek/deepseek-v4-flash:free',
    'nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free',
    // 'poolside/laguna-xs.2:free',
    'black-forest-labs/flux.2-klein-4b',
    'liquid/lfm-2.5-1.2b-instruct:free'
  ],
  temperature: 0.2,
  maxTokens: 2048,
  systemPrompt: "You are a helpful assistant.",
  provider: {
    sort: { 
      by: "price",
      // by: "throughput", 
      // by: "latency",  
      partition: "none" },
  },
};
