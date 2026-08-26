import Fastify from "fastify";
import { OpenRouterService } from "./openRouterService.ts";

export const createServer = (openRouterService: OpenRouterService) => {
  const server = Fastify({ logger: false });

  server.post(
    "/chat",
    {
      schema: {
        body: {
          type: "object",
          required: ["message"],
          properties: {
            message: { type: "string", minLength: 10 },
          },
        },
      },
    },
    async (request, reply) => {
      try {
        const { message } = request.body as { message: string };
        const response = await openRouterService.generate(message);
        return reply.send(response);
      } catch (error) {
        console.error("Error handling /chat request:", error);
        return reply.code(500);
      }
    },
  );

  return server;
};
