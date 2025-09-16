import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { QdrantVectorStore } from "@langchain/qdrant";
import { GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI } from "@langchain/google-genai";
import { createRetrieverTool } from "langchain/tools/retriever";
import { HumanMessage, AIMessage } from "@langchain/core/messages";
import readline from 'readline';
import dotenv from 'dotenv';
import { QdrantClient } from "@qdrant/js-client-rest";
import { z } from "zod";
import { DynamicStructuredTool } from "@langchain/core/tools";
import { getJson } from "serpapi";


import { AgentExecutor, createToolCallingAgent } from "langchain/agents";
import {
  ChatPromptTemplate,
  MessagesPlaceholder,
} from "@langchain/core/prompts";
// ------------------------------------------

// Load environment variables
dotenv.config();

// --- DEBUG SWITCH ---
// Set to 'true' for verbose logs, 'false' for clean output.
const DEBUG = process.env.DEBUG_MODE === 'true' || false;
// ------------------------------------

// --- Configuration --- 
const pdf_File = "./poradnik.pdf";
const collectionName = process.env.COLLECTION_NAME || "learning_langchain";
const qdrantClient = new QdrantClient({
    url: process.env.QDRANT_URL,
    apiKey: process.env.QDRANT_API_KEY
});

// --- Schema Definition --- (Unchanged, this is correct)
const FlightSearchSchema = z.object({
    departure_city_or_airport: z.string().describe("The departure city, airport name, or airport code. e.g., 'Wroclaw', 'WRO'"),
    arrival_city_or_airport: z.string().describe("The arrival city, airport name, or airport code. e.g., 'Warsaw', 'WAW'"),
    departure_date: z.string().describe("The departure date in YYYY-MM-DD format. Convert relative dates like 'this Friday' to this format."),
    return_date: z.string().optional().describe("The return date in YYYY-MM-DD format. Required only for round trips."),
});

// --- Flight Search Function --- 
async function searchFlights(args) { 
    if (DEBUG) {
        console.log(`[Flight Tool DEBUG]: Tool called with raw args: ${JSON.stringify(args, null, 2)}`);
    }

    const { 
        departure_city_or_airport, 
        arrival_city_or_airport, 
        departure_date, 
        return_date 
    } = args || {};

    if (!departure_city_or_airport || !arrival_city_or_airport || !departure_date) {
         if (DEBUG) {
            console.error("[Flight Tool ERROR]: Required arguments are missing. LLM failed to generate tool call correctly.");
         }
         return "Error: Tool failed. The user's query was missing a required argument (departure, arrival, or date). Ask the user to provide all three.";
    }

    if (DEBUG) {
        console.log(`[Flight Tool]: Processing search for: ${departure_city_or_airport} to ${arrival_city_or_airport} on ${departure_date}...`);
    }

    const params = {
        api_key: process.env.SERPAPI_API_KEY,
        engine: "google_flights",
        departure_id: departure_city_or_airport, 
        arrival_id: arrival_city_or_airport,     
        outbound_date: departure_date,           
        return_date: return_date,
        hl: "en",
        gl: "us",
    };

    try {
        const json = await getJson(params);
        
        let resultsToReturn = [];
        if (json.best_flights) {
             resultsToReturn = json.best_flights.slice(0, 5).map(flight => ({
                price: flight.price,
                airline: flight.flights[0].airline,
                departure: flight.flights[0].departure_airport.time,
                arrival: flight.flights[0].arrival_airport.time,
                duration_in_minutes: flight.total_duration,
                stops: flight.flights[0].travel_class ? flight.flights[0].length - 1 : flight.flights[0].length,
                type: flight.type,
            }));
        } else if (json.other_flights) {
             resultsToReturn = json.other_flights.slice(0, 5).map(flight => ({
                price: flight.price,
                airline: flight.flights[0].airline,
                departure: flight.flights[0].departure_airport.time,
                arrival: flight.flights[0].arrival_airport.time,
                duration_in_minutes: flight.total_duration,
                stops: flight.flights[0].travel_class ? flight.flights[0].length - 1 : flight.flights[0].length,
                type: flight.type,
            }));
        } else if (json.error) {
            console.warn(`[Flight Tool WARN]: SerpApi returned an error: ${json.error}`);
            return `API Error: ${json.error}`;
        } else {
            return "No flights found matching criteria.";
        }
        return JSON.stringify(resultsToReturn);

    } catch (e) {
        if (DEBUG) {
            console.error("[Flight Tool FATAL ERROR]: The 'getJson' call failed.");
            console.error("Error object:", e); 
        }

        if (e instanceof Error) {
           return `Error during flight search: ${e.message}`;
        } else {
           return `An unknown error occurred during flight search: ${JSON.stringify(e)}`;
        }
    }
}

// --- Main Application ---
async function main() {
    console.log("Starting Standard LangChain Agent...");

    // 1. Initialize Models
    const embeddings = new GoogleGenerativeAIEmbeddings({
        model: "embedding-001",
    });
    const llm = new ChatGoogleGenerativeAI({
        model: "gemini-2.0-flash", 
        temperature: 0,
    });

    // 2. Setup Vector Store
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
        const textSplitter = new RecursiveCharacterTextSplitter({ chunkSize: 1000, chunkOverlap: 200 });
        const splitDocs = await textSplitter.splitDocuments(docs);
        vectorStore = await QdrantVectorStore.fromDocuments(splitDocs, embeddings, qdrantConfig);
        console.log("Indexing complete.");
    } else {
        console.log(`Collection "${collectionName}" already exists. Loading vector store.`);
        vectorStore = new QdrantVectorStore(embeddings, qdrantConfig);
    }

    // 3. Create Tools 
    // Tool 1: Google Flights 
    const googleFlightsTool = new DynamicStructuredTool({
        name: "google_flights_search",
        description: "Searches for flight prices, routes, and schedules. Use this tool for any queries about finding specific flights, prices, departures, or arrivals.",
        func: searchFlights,
        schema: FlightSearchSchema
    });
    
    // Tool 2: RAG Tool 
    const retriever = vectorStore.asRetriever();
    const retrieverTool = createRetrieverTool(retriever, {
        name: "search_poradnik_document",
        description: "Searches the traveller's guide ('poradnik') to retrieve information about general travel advice, tips, and best practices. Use this for questions NOT related to specific flight bookings.",
    });

    const tools = [googleFlightsTool, retrieverTool];


    // SECTION 4: Agent Setup
    console.log("Compiling manual Tool-Calling Agent...");

    // 4a. Define the Agent Prompt
    // This is the prompt template the prebuilt agent was hiding from us.
    const prompt = ChatPromptTemplate.fromMessages([
        ["system", "You are a helpful travel assistant. Use 'google_flights_search' to find specific flights and prices. Use 'search_poradnik_document' to answer general questions about travel advice and best practices from the guide. You must respond to the user in the same language they used."],
        new MessagesPlaceholder("chat_history"), // For memory
        ["human", "{input}"], // For the user's new query
        new MessagesPlaceholder("agent_scratchpad"), // For tool inputs/outputs
    ]);

    // 4b. Create the "Brain"
    // We bind the tools to the LLM so it knows their schemas.
    const llmWithTools = llm.bindTools(tools);

    // 4c. Create the Agent
    const agent = await createToolCallingAgent({
        llm: llmWithTools,
        tools,
        prompt,
    });

    // 4d. Create the "Executor"
    // This is the runtime that will execute the agent's logic.
    const agentExecutor = new AgentExecutor({
        agent,
        tools,
        verbose: DEBUG, // Set to true for detailed agent logs!
    });
    
    console.log("Agent ready.");
    // --------------------------------------------------------------------


    // 5. Start the Chat Interface
    // We pass the new `agentExecutor` to the chat loop.
    startChatInterface(agentExecutor);
}


/**
 * Creates a command-line interface to chat with the AgentExecutor.
 */
function startChatInterface(agentExecutor) {
    // We maintain a simple history list for the agent's memory
    let history = []; 

    const rl = readline.createInterface({
        input: process.stdin,
        output: process.stdout
    });

    async function ask() {
        rl.question('\nEnter your query (or type "exit" to quit): ', async (query) => {
            if (query.toLowerCase() === 'exit') {
                console.log('Goodbye!');
                rl.close();
                return;
            }

            try {
                console.log(`\n...Processing...\n`);

                // 1. Invoke the AgentExecutor
                // Its signature is different: it takes 'input' and 'chat_history'
                const result = await agentExecutor.invoke({
                    input: query,
                    chat_history: history,
                });

                // 2. The final answer is in the 'output' key
                const finalAnswer = result.output; 

                if (typeof finalAnswer === "string") {
                    // SUCCESS PATH
                    console.log("\n=== ANSWER ===");
                    console.log(finalAnswer);
                    console.log("==============");
                    
                    // 3. Manually update the history
                    history.push(new HumanMessage(query));
                    history.push(new AIMessage(finalAnswer));
                } else {
                    // FAILURE PATH
                    console.error("Agent did not return a valid string response.");
                    console.error("The last agent state was:", result);
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
// --------------------------------------------------

// Run the application
main().catch(console.error);
