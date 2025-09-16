# **Travel Agent Bot**

This is a Node.js-based conversational agent that acts as a travel assistant. It can search for live flight information using the Google Flights API (via SerpApi) and answer general travel questions by searching a local PDF document (RAG).

The agent is built using LangChain.js, with a Google Gemini model as its "brain" and a Qdrant vector store for its memory.


## **🚀 How to Run**


### **1. Prerequisites**



*   [Node.js](https://nodejs.org/) (v18 or later)
*   An active[ Qdrant](https://qdrant.tech/) instance (either self-hosted or on Qdrant Cloud)
*   A **Google API Key** (for the Gemini model and embeddings)
*   A **SerpApi API Key** (for the flight search tool)


### **2. Installation**

Clone the repository and install the required dependencies:


```
npm install

```


This will install `@langchain/google-genai`, `@langchain/qdrant`, `langchain`, `serpapi`, `zod`, `@qdrant/js-client-rest`, `pdf-loader`, and other necessary packages.


### **3. Environment Setup**

Create a `.env` file in the root of the project and add your API keys and Qdrant information:


```
# Google API Key for Gemini and Embeddings
GOOGLE_API_KEY="YOUR_GOOGLE_API_KEY"

# SerpApi API Key for flight search
SERPAPI_API_KEY="YOUR_SERPAPI_API_KEY"

# Qdrant connection details
QDRANT_URL="YOUR_QDRANT_CLUSTER_URL"
QDRANT_API_KEY="YOUR_QDRANT_API_KEY"

# (Optional) Name for the Qdrant collection
COLLECTION_NAME="learning_langchain"

```



### **4. Add Your Document**

Place your travel guide PDF in the root of the project and name it `poradnik.pdf`. The first time you run the agent, it will automatically load, split, and index this file into your Qdrant vector store.


### **5. Run the Agent**

Once your `.env` file is set up, you can start the agent:


```
node app.js

```


The agent will start, connect to Qdrant (or index the doc if needed), and present you with a command-line prompt to start chatting.


#### **Debug Mode**

To see the agent's full thought process (which tools it's choosing, what data it's processing, etc.), you can run the app in debug mode:


```
DEBUG_MODE=true node app.js

```



## **📂 Project Structure**

The `app.js` file contains the entire application, which is organized as follows:



1. **Imports:** Loads all necessary modules from LangChain, Qdrant, Google, and SerpApi.
2. **Configuration:** Sets up the `DEBUG` switch and loads all environment variables.
3. **Tool**: <code>searchFlights</code> Function:</strong> This is a custom-built, structured tool. It uses a <strong>Zod schema</strong> (<code>FlightSearchSchema</code>) to define its required arguments (like <code>departure_city_or_airport</code>, <code>departure_date</code>, etc.). It then calls the <code>serpapi</code> <code>getJson</code> function with the correct <code>engine: "google_flights"</code> parameters and returns a pruned JSON string of the best flight options. It also contains robust error handling.
4. <strong><code>main()</code> Function:</strong> This is the core orchestrator of the application.
    *   <strong>Initializes Models:</strong> Sets up the <code>ChatGoogleGenerativeAI</code> (the LLM "brain," e.g., <code>gemini-2.0-flash</code>) and the <code>GoogleGenerativeAIEmbeddings</code> (for RAG).
    *   <strong>Initializes Vector Store:</strong> Connects to Qdrant. On the first run, it loads <code>poradnik.pdf</code>, splits it into chunks, and populates the vector store.
    *   <strong>Creates Tools:</strong>
        *   <code>googleFlightsTool</code>: An instance of <code>DynamicStructuredTool</code> that wraps our <code>searchFlights</code> function and its Zod schema.
        *   <code>retrieverTool</code>: A standard <code>createRetrieverTool</code> that connects the agent to the Qdrant vector store for RAG.
    *   <strong>Builds the Agent:</strong> This is the most critical part. It <strong>manually</strong> builds the agent (since the prebuilt <code>createReactAgent</code> was incompatible with the tools).
        *   It defines a <code>ChatPromptTemplate</code> (the agent's "system message" and memory).
        *   It binds the <code>tools</code> to the LLM using <code>llm.bindTools(tools)</code>.
        *   It creates the agent "brain" with <code>createToolCallingAgent</code>.
        *   It creates the agent "runner" with <code>AgentExecutor</code>, passing in the agent and tools.
5. <strong><code>startChatInterface()</code> Function:</strong> A simple <code>readline</code> chat loop that takes user input, manages a <code>history</code> array, and calls the <code>agentExecutor.invoke()</code> method to get a response.


## <strong>💡 Ideas for Improvement</strong>



*   **Use LangGraph (Manually):** Build a more robust agent using **LangGraph**. This would allow for more complex logic, like retrying a tool call if it fails or deciding if it has _enough_ information from the user _before_ calling the flight tool.
*   **Streaming Response:** The current `agentExecutor.invoke()` waits for the full, final answer. This can feel slow. You could replace it with `agentExecutor.stream()` to stream the agent's thoughts, tool calls, and final response back to the user token by token for a much more "live" feel.
*   **Error Handling in LLM:** When the `searchFlights` tool returns an error (like "Invalid API Key" or "No flights found"), the agent currently just reports that error. You could improve the system prompt to instruct the LLM to _interpret_ these errors and give a more helpful, natural-language response (e.g., "I'm sorry, I couldn't find any flights for that route. Would you like to try a different date?").
*   **More Tools:** Expand the agent's capabilities by adding more tools:
    *   A tool for **booking hotels** (e.g., via SerpApi's Google Hotels engine).
    *   A tool for **getting the weather** (e.g., via a free weather API).
    *   A tool to get the **current time** in different time zones.
