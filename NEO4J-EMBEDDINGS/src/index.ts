import { HuggingFaceInference } from "@langchain/community/llms/hf";
import { CONFIG } from "./config.ts";
import { DocumentProcessor } from "./document-processor.ts";
import { HuggingFaceTransformersEmbeddings } from "@langchain/community/embeddings/huggingface_transformers";
import { Neo4jVectorStore } from "@langchain/community/vectorstores/neo4j_vector";

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
  const questions = ["Como treinar uma rede neural?"];

  for (const question of questions) {
    console.log(`Performing similarity search for question: "${question}"...`);

    const results = await _neo4jVectorStore.similaritySearch(
      question,
      CONFIG.similarity.topK,
    );

    // console.log("Similarity search results:", results);
  }
} catch (error) {
  console.error("Error processing document:", error);
} finally {
  await _neo4jVectorStore?.close(); // Clean up Neo4j vector store instance if needed
  console.log("Finished all operations.");
}
