import { PDFLoader } from "../node_modules/@langchain/community/document_loaders/fs/pdf.js";
import { RecursiveCharacterTextSplitter } from "../node_modules/langchain/text_splitter.js";
import * as Config from "./config.ts";

export class DocumentProcessor {
  private pdfPath: string;
  private textSplitterConfig: Config.TextSplitterConfig;

  constructor(pdfPath: string, textSplitterConfig: Config.TextSplitterConfig) {
    this.pdfPath = pdfPath;
    this.textSplitterConfig = textSplitterConfig;
  }

  async loadAndSplit() {
    const loader = new PDFLoader(this.pdfPath);
    const rawDocuments = await loader.load();
    console.log(`Loaded ${rawDocuments.length} pages from PDF.`);
    const textSplitter = new RecursiveCharacterTextSplitter(
      this.textSplitterConfig,
    );
    const documents = await textSplitter.splitDocuments(rawDocuments);
    console.log(`Split into ${documents.length} chunks.`);

    return documents.map((doc) => ({
      ...doc,
      metadata: { source: doc.metadata.source },
    }));
  }
}
