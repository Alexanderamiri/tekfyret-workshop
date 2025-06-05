package com.example.rag

import io.github.cdimascio.dotenv.dotenv
import io.ktor.client.*
import io.ktor.client.call.*
import io.ktor.client.engine.cio.*
import io.ktor.client.plugins.contentnegotiation.*
import io.ktor.client.request.*
// import io.ktor.client.statement.* // No longer needed for raw response handling
import io.ktor.http.*
import io.ktor.serialization.kotlinx.json.*
import kotlinx.coroutines.runBlocking
import kotlinx.serialization.ExperimentalSerializationApi
import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.jsonPrimitive


// --- Configuration ---
@OptIn(ExperimentalSerializationApi::class)
private val json = Json { ignoreUnknownKeys = true; prettyPrint = true; explicitNulls = false }

private val dotenv = dotenv {
    ignoreIfMissing = true 
}

private val openAiKey = dotenv["OPENAI_API_KEY"]
    ?: run {
        println("Error: OPENAI_API_KEY not found in .env file or as environment variable.")
        kotlin.system.exitProcess(1)
    }
private val anthropicKey = dotenv["ANTHROPIC_API_KEY"]
    ?: run {
        println("Error: ANTHROPIC_API_KEY not found in .env file or as environment variable.")
        kotlin.system.exitProcess(1)
    }

private const val EMBED_MODEL = "text-embedding-3-small"
private const val LLM_MODEL = "claude-sonnet-4-20250514" 
private const val QDRANT_URL = "http://localhost:6333"
private const val COLLECTION_NAME = "docs_kt"
private const val TOP_K_RESULTS = 5
private const val MAX_TOKENS_ANTHROPIC = 1024

// Ktor HTTP Client
private val httpClient = HttpClient(CIO) {
    install(ContentNegotiation) { json(json) }
    expectSuccess = true 
}

// --- Data classes for ChatCli (prefixed with Chat) ---
@Serializable
private data class ChatOpenAiEmbeddingRequest(val model: String, val input: List<String>)

@Serializable
private data class ChatEmbeddingData(val embedding: List<Double>)

@Serializable
private data class ChatOpenAiEmbeddingResponse(val data: List<ChatEmbeddingData>)

@Serializable
private data class ChatQdrantSearchRequest(
    val vector: List<Float>,
    val limit: Int,
    @SerialName("with_payload") val withPayload: Boolean
)

@Serializable
private data class ChatQdrantSearchHit(val id: JsonElement, val score: Float, val payload: Map<String, JsonElement>?)

@Serializable
private data class ChatQdrantSearchResponse(val result: List<ChatQdrantSearchHit>)

@Serializable
private data class ChatAnthropicMessage(val role: String, val content: String)

@Serializable
private data class ChatAnthropicRequest(
    val model: String,
    @SerialName("max_tokens") val maxTokens: Int,
    val system: String? = null, 
    val messages: List<ChatAnthropicMessage>
)

@Serializable
private data class ChatAnthropicContentBlock(val type: String, val text: String)

@Serializable
private data class ChatAnthropicResponse(
    val id: String,
    val type: String,
    val role: String,
    val content: List<ChatAnthropicContentBlock>,
)

private const val ANTHROPIC_SYSTEM_PROMPT = "You are an assistant that answers **only** from the provided <context>. " +
                                          "If the answer cannot be found, simply reply with I don't know. " +
                                          "Format your answer clearly and concisely."

// --- Core Chat Logic ---
suspend fun generateQueryEmbedding(query: String): List<Float> {
    try {
        val response: ChatOpenAiEmbeddingResponse = httpClient.post("https://api.openai.com/v1/embeddings") {
            header(HttpHeaders.Authorization, "Bearer $openAiKey")
            contentType(ContentType.Application.Json)
            setBody(ChatOpenAiEmbeddingRequest(EMBED_MODEL, listOf(query)))
        }.body()
        val vector = response.data.firstOrNull()?.embedding?.map(Double::toFloat) ?: emptyList()
        return vector
    } catch (e: Exception) { println("[Embedder] Error: ${e.message}"); return emptyList() }
}

suspend fun searchQdrant(queryVector: List<Float>): List<String> {
    if (queryVector.isEmpty()) { 
        return emptyList() 
    }
    try {
        val response: ChatQdrantSearchResponse = httpClient.post("$QDRANT_URL/collections/$COLLECTION_NAME/points/search") {
            contentType(ContentType.Application.Json)
            setBody(ChatQdrantSearchRequest(vector = queryVector, limit = TOP_K_RESULTS, withPayload = true))
        }.body()
        
        val contextTexts = response.result.mapNotNull { hit -> 
            hit.payload?.get("text")?.jsonPrimitive?.content
        }
        println("[Qdrant] Found ${contextTexts.size} relevant text chunks."); return contextTexts
    } catch (e: Exception) { 
        println("[Qdrant] Error during Qdrant search: ${e.message}")
        return emptyList() 
    }
}

suspend fun getAnthropicAnswer(userQuery: String, context: String): String {
    val requestBody = ChatAnthropicRequest(
        model = LLM_MODEL, maxTokens = MAX_TOKENS_ANTHROPIC, system = ANTHROPIC_SYSTEM_PROMPT,
        messages = listOf(ChatAnthropicMessage(role = "user", content = "<context>\n${context.trim()}\n</context>\n\nUser question: $userQuery"))
    )
    try {
        val response: ChatAnthropicResponse = httpClient.post("https://api.anthropic.com/v1/messages") {
            header("x-api-key", anthropicKey)
            header("anthropic-version", "2023-06-01")
            contentType(ContentType.Application.Json)
            setBody(requestBody)
        }.body()
        return response.content.firstOrNull { it.type == "text" }?.text?.trim() ?: "No answer text found in response."
    } catch (e: Exception) {
        println("[LLM] Error during Anthropic call: ${e.message}")
        return "Sorry, I encountered an error trying to get an answer from the AI model."
    }
}

// --- Main CLI Loop ---
fun main() = runBlocking {
    println("--- Kotlin RAG Chat CLI ---")
    println("Type your questions to query the indexed website content. Type 'exit' or 'quit' to stop.")

    // Check Qdrant connection first (optional but good practice)
    try {
        httpClient.get("$QDRANT_URL/collections/$COLLECTION_NAME")
        println("[System] Successfully connected to Qdrant and collection '$COLLECTION_NAME' seems to exist.")
    } catch (e: Exception) {
        println("[System] Error: Could not connect to Qdrant or collection '$COLLECTION_NAME' does not exist.")
        println("Please ensure Qdrant is running and you have run IndexWebsite.kt first.")
        println("Error details: ${e.message}")
        return@runBlocking
    }

    while (true) {
        print("\n> Question: ")
        val userInput = readLine()
        if (userInput.isNullOrBlank() || userInput.equals("exit", ignoreCase = true) || userInput.equals("quit", ignoreCase = true)) {
            break
        }

        // 1. Embed user query
        val queryVector = generateQueryEmbedding(userInput)
        if (queryVector.isEmpty()) {
            println("Could not process your query (embedding failed).")
            continue
        }

        // 2. Search Qdrant for context
        val contextChunks = searchQdrant(queryVector)
        if (contextChunks.isEmpty()) {
            println("I couldn't find any relevant information to answer that. Try rephrasing your question.")
            continue
        }
        val contextString = contextChunks.joinToString("\n---\n")

        // 3. Get answer from Anthropic
        val answer = getAnthropicAnswer(userInput, contextString)
        println("\n🤖 Answer:\n${answer}")
    }
    println("Exiting chat. Goodbye!")
    httpClient.close()
} 