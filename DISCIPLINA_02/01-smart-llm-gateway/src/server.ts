import Fastify from "fastify";

export const createServer = () => {
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
        return reply.send({ response: `Received message: ${message}` });
      } catch (error) {
        console.error("Error handling /chat request:", error);
        return reply.status(500).send({ error: "Internal Server Error" });
      }
    },
  );

  return server;
};
