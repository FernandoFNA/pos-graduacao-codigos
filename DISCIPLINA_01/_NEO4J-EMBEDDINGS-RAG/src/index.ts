import { HuggingFaceInference } from "@langchain/community/llms/hf";
import { CONFIG } from "./config.ts";
import { DocumentProcessor } from "./document-processor.ts";
import { HuggingFaceTransformersEmbeddings } from "@langchain/community/embeddings/huggingface_transformers";
import { Neo4jVectorStore } from "@langchain/community/vectorstores/neo4j_vector";
import { ChatOpenAI } from "@langchain/openai";
import { AI } from "./ai.ts";
import { mkdir, writeFile } from "fs/promises";

let _neo4jVectorStore = null; // Placeholder for Neo4j vector store instance

async function clearAll(vectorStore: Neo4jVectorStore, nodeLabel: string) {
  console.log(`Clearing all nodes with label '${nodeLabel}' from Neo4j...`);
  try {
    await vectorStore.query(`MATCH (n:${nodeLabel}) DETACH DELETE n`);
    console.log(
      `Finished clearing nodes with label '${nodeLabel}' from Neo4j.`,
    );
  } catch (error) {
    console.error(
      `Error clearing nodes with label '${nodeLabel}' from Neo4j:`,
      error,
    );
  }
}

try {
  console.log("Starting document processing...");

  const documentProcessorModule = new DocumentProcessor(
    CONFIG.pdf.path,
    CONFIG.textSplitter,
  );
  const document = await documentProcessorModule.loadAndSplit();

  const embeddings = new HuggingFaceTransformersEmbeddings({
    model: CONFIG.embedding.modelName,
    ...CONFIG.embedding.pretrainedOptions,
  });

  const nlpModel = new ChatOpenAI({
    temperature: CONFIG.openRouter.temperature,
    modelName: CONFIG.openRouter.nlpModel,
    openAIApiKey: CONFIG.openRouter.apiKey,
    maxRetries: CONFIG.openRouter.maxRetries,
    configuration: {
      baseURL: CONFIG.openRouter.url,
      defaultHeaders: CONFIG.openRouter.defaultHeaders,
    },
  });

  //   const response = await embeddings.embedQuery(
  //     "Carrapato quando pula se machuca?",
  //   );

  //   const response = await embeddings.embedDocuments([
  //     "Carrapato quando pula se machuca?",
  //   ]);

  //   console.log("Embedding response:", response);

  _neo4jVectorStore = await Neo4jVectorStore.fromExistingGraph(
    embeddings,
    CONFIG.neo4j,
  );

  await clearAll(_neo4jVectorStore, CONFIG.neo4j.nodeLabel);

  for (const [index, doc] of document.entries()) {
    console.log(`Processing chunk ${index + 1}/${document.length}...`);
    await _neo4jVectorStore.addDocuments([doc]);
  }
  console.log("Finished processing all chunks and adding to Neo4j.");

  //   PASSO 2: Consulta de similaridade
  console.log("Starting similarity search...");
  const questions = [
    // "Como treinar uma rede neural?",
    "O que é hot encoding?",
    "Quais são os tipos de aprendizado de máquina?",
  ];

  const ai = new AI({
    nlpModel,
    debugLog: console.log,
    vectorStore: _neo4jVectorStore,
    promptConfig: CONFIG.promptConfig,
    templateText: CONFIG.templateText,
    topK: CONFIG.similarity.topK,
  });

  for (const question of questions) {
    console.log(`Performing similarity search for question: "${question}"...`);

    const results = await ai.answerQuestion(question);

    if (results.error) {
      console.error(`Error answering question "${question}":`, results.error);
      continue; // Skip to the next question if there's an error
    }

    console.log(`Answer for question "${question}":`, results.anwer);

    const fileName = `${CONFIG.output.answersFolder}/${CONFIG.output.fileName}_${question.substring(0, 30).replace(/\s+/g, "_").replace(/[^a-zA-Z0-9_]/g, "")}.md`;

    await mkdir(CONFIG.output.answersFolder, { recursive: true });
    await writeFile(fileName, results.anwer!);
  }
} catch (error) {
  console.error("Error processing document:", error);
} finally {
  await _neo4jVectorStore?.close(); // Clean up Neo4j vector store instance if needed
  console.log("Finished all operations.");
}
