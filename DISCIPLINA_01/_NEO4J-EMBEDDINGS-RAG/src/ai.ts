import { Neo4jVectorStore } from "@langchain/community/vectorstores/neo4j_vector";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { RunnableSequence } from "@langchain/core/runnables";
import { ChatOpenAI } from "@langchain/openai";

type DebugLog = (...args: unknown[]) => void;

type params = {
  debugLog: DebugLog;
  vectorStore: Neo4jVectorStore;
  nlpModel: ChatOpenAI;
  promptConfig: any;
  templateText: string;
  topK: number;
};

interface ChainState {
  question: string;
  context?: string;
  topScore?: number;
  error?: string;
  anwer?: string;
}

export class AI {
  private params: params;

  constructor(params: params) {
    this.params = params;
  }

  async retriveVectorSearchResults(input: ChainState): Promise<ChainState> {
    console.log(
      "Retrieving vector search results for question:",
      input.question,
    );

    const vectorResults =
      await this.params.vectorStore.similaritySearchWithScore(
        input.question,
        this.params.topK,
      );

    if (!vectorResults || vectorResults.length === 0) {
      this.params.debugLog(
        "No vector search results found for question:",
        input.question,
      );
      return {
        ...input,
        error: "No relevant information found in the vector store.",
      };
    }

    // console.log("Vector search results:", { vectorResults });

    const topScore = vectorResults[0]![1];

    console.log("Top score from vector search:", topScore.toFixed(4));

    const context = vectorResults
      .filter(([_, score]) => score > 0.75)
      .map(([doc]) => doc.pageContent)
      .join("\n\n---\n\n");

    return {
      ...input,
      context,
      topScore,
    };
  }

  async generateNLPResponse(input: ChainState): Promise<ChainState> {
    if (input.error) return input;

    console.log("Generating NLP response using context and question...");

    const responsePrompt = ChatPromptTemplate.fromTemplate(
      this.params.templateText,
    );

    const responseChain = responsePrompt
      .pipe(this.params.nlpModel)
      .pipe(new StringOutputParser());

    const rawResponse = await responseChain.invoke({
      role: this.params.promptConfig.role,
      task: this.params.promptConfig.task,
      tone: this.params.promptConfig.constraints.tone,
      language: this.params.promptConfig.constraints.language,
      format: this.params.promptConfig.constraints.format,
      instructions: this.params.promptConfig.instructions
        .map((instruction: string, idx: number) => `${idx + 1}. ${instruction}`)
        .join("\n"),
      context: input.context,
      question: input.question,
    });

    // console.log("Raw NLP response:", rawResponse);

    return {
      ...input,
      anwer: rawResponse,
    };
  }

  async answerQuestion(question: string) {
    const chain = RunnableSequence.from([
      this.retriveVectorSearchResults.bind(this),
      this.generateNLPResponse.bind(this),
    ]);

    const result = await chain.invoke({ question });
    // console.log("Final result:", result);

    this.params.debugLog("Question:\n", question);
    (this,
      this.params.debugLog(
        "Answer:\n",
        result.anwer || result.error || "No answer generated.",
        "\n---\n",
      ));

      return result;
  }
}
