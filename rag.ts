// Import necessary packages
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { QdrantVectorStore } from "@langchain/qdrant";
import { GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI } from "@langchain/google-genai";
import { SerpAPI } from "@langchain/community/tools/serpapi";
import { createRetrieverTool } from "langchain/tools/retriever";
import { createReactAgent } from "@langchain/langgraph/prebuilt"; // <-- USER REQUESTED IMPORT
import { HumanMessage, AIMessage } from "@langchain/core/messages";
import readline from 'readline';
import dotenv from 'dotenv';

// Helper to check for Qdrant collection
import { QdrantClient } from "@qdrant/js-client-rest";

// Load environment variables
dotenv.config(); 

// --- Configuration ---
const pdf_File = "./poradnik.pdf";
const collectionName = process.env.COLLECTION_NAME || "learning_langchain";

const qdrantClient = new QdrantClient({
    url: process.env.QDRANT_URL,
    apiKey: process.env.QDRANT_API_KEY
});

// --- Main Application ---
async function main() {
    console.log("Starting LangGraph Agent with RAG and SerpAPI...");

    // 1. Initialize Models (LangChain Wrappers)
    const embeddings = new GoogleGenerativeAIEmbeddings({
        model: "embedding-001",
    });
    // This LLM must be a tool-calling model
    const llm = new ChatGoogleGenerativeAI({
        model: "gemini-2.5-flash-latest",
        temperature: 0,
    });

    // 2. Setup Vector Store (Same logic as before)
    let vectorStore;
    const collections = await qdrantClient.getCollections();
    const exists = collections.collections.some(c => c.name === collectionName);

    const qdrantConfig = {
        collectionName: collectionName,
        url: process.env.QDRANT_URL,
        apiKey: process.env.QDRANT_API_KEY
    };

    if (!exists) {
        console.log(`Collection "${collectionName}" not found. Indexing document...`);
        const loader = new PDFLoader(pdf_File);
        const docs = await loader.load();
        const textSplitter = new RecursiveCharacterTextSplitter({
            chunkSize: 1000,
            chunkOverlap: 200,
        });
        const splitDocs = await textSplitter.splitDocuments(docs);
        vectorStore = await QdrantVectorStore.fromDocuments(
            splitDocs,
            embeddings,
            qdrantConfig
        );
        console.log("Indexing complete.");
    } else {
        console.log(`Collection "${collectionName}" already exists. Loading vector store.`);
        vectorStore = new QdrantVectorStore(embeddings, qdrantConfig);
    }

    // 3. Create Tools

    // Tool 1: SerpAPI Web Search
    const serpTool = new SerpAPI(process.env.SERPAPI_API_KEY, {
        hl: "en",
        gl: "us",
    });
    serpTool.name = "google_flights";
    serpTool.description = "A flight search tool. Use this tool to find information about flights: departurtes, routes, prices";

    // Tool 2: RAG Tool
    const retriever = vectorStore.asRetriever();
    const retrieverTool = createRetrieverTool(retriever, {
        name: "search_poradnik_document",
        description: "Searches the traveller's guide to retrieve information about travel best practices.",
    });

    const tools = [serpTool, retrieverTool];

    // 4. Create the LangGraph Agent
    // This one import replaces AgentExecutor, pull, and createToolCallingAgent.
    // This is the "brain" and the "executor" all in one.
    console.log("Compiling LangGraph ReAct agent...");
    const agent = createReactAgent({
        llm,
        tools,
        systemMessage: "You are a helpful travel assistant. Use your tools to search for flight information, luggage policies, and live flight prices."
    });
    console.log("Agent ready.");

    // 5. Start the Chat Interface
    startChatInterface(agent);
}

// Create readline interface
const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout
});


/**
 * Creates a command-line interface to chat with the compiled LangGraph agent.
 */
function startChatInterface(agent) {
    // We maintain a simple history list for the agent's memory
    let history = []; 

    async function ask() {
        rl.question('\nEnter your query (or type "exit" to quit): ', async (query) => {
            if (query.toLowerCase() === 'exit') {
                console.log('Goodbye!');
                rl.close();
                return;
            }

            try {
                console.log(`\n...Processing...\n`);

                // 1. Add the user's query to the message history
                history.push(new HumanMessage({ content: query }));

                // 2. Invoke the agent with the *entire* message history
                const result = await agent.invoke({ messages: history });

                // 3. Get the last message
                const messages = result?.messages;
                const finalAnswer = (Array.isArray(messages) && messages.length > 0) 
                    ? messages[messages.length - 1] 
                    : null; 

                if (finalAnswer instanceof AIMessage) {
                    // SUCCESS PATH
                    history.push(finalAnswer); // Add the good answer to history
                    
                    console.log("\n=== ANSWER ===");
                    console.log(finalAnswer.content);
                    console.log("==============");
                } else {
                    // FAILURE PATH
                    console.error("Agent did not return a valid AIMessage response.");
                    console.error("The last message in the agent state was:", finalAnswer || "No messages found.");
                    history.pop(); // Remove the user query we just added
                    console.log("Last query was not processed correctly and was removed from memory.");
                }

            } catch (error) {
                console.error("FATAL: Error invoking agent:", error);
                history = [];
                console.log("History has been cleared due to fatal error.");
            }

            // Ask the next question
            ask();
        });
    }
    
    // Start the loop
    ask();
}

// Run the application
main().catch(console.error);
