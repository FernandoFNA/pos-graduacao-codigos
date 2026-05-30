import { createServer } from "./server.ts";

const server = createServer();

await server.listen({ port: 3000, host: "0.0.0.0" });
// server.log.info("Server is running at http://localhost:3000");

server.inject({
  method: "POST",
  url: "/chat",
  body: { message: "Hello, this is a test message!" },
}).then((response) => {
  console.log("Status: " + response.statusCode + "\nResponse from /chat: ", response.json());
}).catch((error) => {
  console.error("Error during test request to /chat:", error);
});