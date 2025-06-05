package com.example.rag

import io.github.cdimascio.dotenv.dotenv
import io.ktor.client.*
import io.ktor.client.call.*
import io.ktor.client.engine.cio.*
import io.ktor.client.plugins.contentnegotiation.*
import io.ktor.client.request.*
import io.ktor.http.*
import io.ktor.serialization.kotlinx.json.*
import kotlinx.coroutines.runBlocking
import kotlinx.serialization.ExperimentalSerializationApi
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json
import org.jsoup.Jsoup
import java.net.URI
import java.util.*
import kotlin.math.min

// jtokkit imports for token-based chunking
import com.knuddels.jtokkit.Encodings
import com.knuddels.jtokkit.api.Encoding
import com.knuddels.jtokkit.api.EncodingType


// --- Configuration ---
@OptIn(ExperimentalSerializationApi::class)
private val json = Json { ignoreUnknownKeys = true; prettyPrint = true }

// Load environment variables from .env file using dotenv-kotlin
private val dotenv = dotenv {
    ignoreIfMissing = true
}

private val openAiKey = dotenv["OPENAI_API_KEY"]
    ?: run {
        println("Error: OPENAI_API_KEY not found in .env file or as environment variable.")
        kotlin.system.exitProcess(1)
    }


private const val EMBED_MODEL = "text-embedding-3-small"
private const val QDRANT_URL = "http://localhost:6333"
private const val COLLECTION_NAME = "docs_kt" 
private const val MAX_PAGES_TO_SCRAPE = 50
private const val CHUNK_SIZE_TOKENS = 800
private const val CHUNK_OVERLAP_TOKENS = 100
private const val QDRANT_RECREATE_TIMEOUT_SECONDS = 20L

// Ktor HTTP Client for network requests
private val httpClient = HttpClient(CIO) {
    install(ContentNegotiation) { json(json) }
    expectSuccess = true
}

// --- Data classes for OpenAI Embeddings ---
@Serializable
private data class IdxOpenAiEmbeddingRequest(val model: String, val input: List<String>)

@Serializable
private data class IdxEmbeddingData(val embedding: List<Double>)

@Serializable
private data class IdxOpenAiEmbeddingResponse(val data: List<IdxEmbeddingData>)

// --- Data classes for Qdrant ---
@Serializable
private data class IdxQdrantVectorParams(val size: Int, val distance: String)

@Serializable
private data class IdxQdrantCollectionCreateRequest(val vectors: IdxQdrantVectorParams)

@Serializable
private data class IdxQdrantPoint(val id: String, val vector: List<Float>, val payload: Map<String, String>)

@Serializable
private data class IdxQdrantUpsertPointsRequest(val points: List<IdxQdrantPoint>)

// --- Utility Functions ---
private fun normalizeUrl(url: String): String {
    val noFragmentOrQuery = url.split('#')[0].split('?')[0]
    return if (!noFragmentOrQuery.endsWith("/") && !noFragmentOrQuery.substringAfterLast('/').contains('.')) {
        "$noFragmentOrQuery/"
    } else {
        noFragmentOrQuery
    }
}

private fun isSameBaseDomain(url1: String, url2: String): Boolean {
    return try {
        URI(url1).host == URI(url2).host
    } catch (e: Exception) {
        false
    }
}

// --- Scraping Logic ---
private fun scrapeWebsite(baseUrl: String, maxPages: Int = MAX_PAGES_TO_SCRAPE): List<Pair<String, String>> {
    println("[Scraper] Starting scrape of $baseUrl, up to $maxPages pages.")
    val queue: ArrayDeque<String> = ArrayDeque(listOf(baseUrl)); val visitedUrls = mutableSetOf<String>()
    val scrapedPages = mutableListOf<Pair<String, String>>(); var pagesProcessedCount = 0
    while (queue.isNotEmpty() && scrapedPages.size < maxPages) {
        val currentUrl = normalizeUrl(queue.removeFirst())
        if (currentUrl in visitedUrls || !isSameBaseDomain(currentUrl, baseUrl)) continue
        visitedUrls.add(currentUrl); pagesProcessedCount++
        if (pagesProcessedCount == 1 || pagesProcessedCount % 20 == 0 || pagesProcessedCount == maxPages) {
            println("[Scraper] Processing page $pagesProcessedCount/$maxPages: $currentUrl")
        }
        try {
            val document = Jsoup.connect(currentUrl).timeout(10000).get()
            document.select("header, footer, nav, aside, script, style, noscript, svg").remove()
            val pageText = document.body()?.text()?.replace("\\s{3,}".toRegex(), " ")?.trim()
            if (!pageText.isNullOrEmpty()) scrapedPages.add(currentUrl to pageText)
            document.select("a[href]").forEach { link ->
                val absoluteUrl = normalizeUrl(link.absUrl("href"))
                if (absoluteUrl.isNotBlank() && isSameBaseDomain(absoluteUrl, baseUrl) && absoluteUrl !in visitedUrls) {
                    if (queue.size < maxPages * 2) queue.add(absoluteUrl)
                }
            }
        } catch (e: Exception) {  println("[Scraper] Warning: Failed to fetch/process $currentUrl: ${e.message?.take(100)}") }
    }
    println("[Scraper] Scraping complete. Found ${scrapedPages.size} pages with text."); return scrapedPages
}

// --- Text Chunking Logic (Refactored for jtokkit) ---
private fun chunkTextContent(text: String, chunkSizeInTokens: Int = CHUNK_SIZE_TOKENS, overlapInTokens: Int = CHUNK_OVERLAP_TOKENS): List<String> {
    if (text.isBlank()) return emptyList()
    val registry = Encodings.newDefaultEncodingRegistry()
    val encoding: Encoding = registry.getEncoding(EncodingType.CL100K_BASE)
    val tokenIds = encoding.encode(text)

    if (tokenIds.size <= chunkSizeInTokens) return listOf(text)

    val stepSize = (chunkSizeInTokens - overlapInTokens).coerceAtLeast(1)

    val textChunks = mutableListOf<String>()
    var i = 0
    while (i < tokenIds.size) {
        val end = min(i + chunkSizeInTokens, tokenIds.size)
        val chunkTokenIds = tokenIds.subList(i, end)
        if (chunkTokenIds.isNotEmpty()) {
            textChunks.add(encoding.decode(chunkTokenIds))
        }
        if (end == tokenIds.size) break
        i += stepSize
    }
    return textChunks
}

// --- Embedding Logic ---
suspend fun generateEmbeddings(texts: List<String>): List<List<Float>> {
    if (texts.isEmpty()) return emptyList()
    println("[Embedder] Generating embeddings for ${texts.size} text chunks...")
    val embeddings = mutableListOf<List<Float>>()
    try {
        val response: IdxOpenAiEmbeddingResponse = httpClient.post("https://api.openai.com/v1/embeddings") {
            header(HttpHeaders.Authorization, "Bearer $openAiKey")
            contentType(ContentType.Application.Json)
            setBody(IdxOpenAiEmbeddingRequest(EMBED_MODEL, texts))
        }.body()
        embeddings.addAll(response.data.map { it.embedding.map(Double::toFloat) })
        println("[Embedder] Successfully generated ${embeddings.size} embeddings.")
    } catch (e: Exception) {
        println("[Embedder] Error generating embeddings: ${e.message}")
    }
    return embeddings
}

// --- Qdrant Logic ---
suspend fun ensureQdrantCollection(vectorSize: Int) {
    println("[Qdrant] Ensuring collection '$COLLECTION_NAME' exists with vector size $vectorSize.")
    try {
        // Attempt to delete if it exists, to ensure a fresh state for workshop simplicity
        try {
            httpClient.delete("$QDRANT_URL/collections/$COLLECTION_NAME?timeout=$QDRANT_RECREATE_TIMEOUT_SECONDS")
            println("[Qdrant] Collection '$COLLECTION_NAME' deleted if it existed.")
        } catch (e: io.ktor.client.plugins.ClientRequestException) {
            if (e.response.status == HttpStatusCode.NotFound) {
                println("[Qdrant] Collection '$COLLECTION_NAME' did not exist. Will create anew.")
            } else {
                println("[Qdrant] Warning: Could not delete collection '$COLLECTION_NAME' (may not exist or other issue): ${e.message}")
            }
        }
        
        // Create the collection
        httpClient.put("$QDRANT_URL/collections/$COLLECTION_NAME?timeout=$QDRANT_RECREATE_TIMEOUT_SECONDS") {
            contentType(ContentType.Application.Json)
            setBody(IdxQdrantCollectionCreateRequest(IdxQdrantVectorParams(vectorSize, "Cosine")))
        }
        println("[Qdrant] Collection '$COLLECTION_NAME' created/recreated successfully.")
        kotlinx.coroutines.delay(2000) // Short delay for Qdrant to stabilize
    } catch (e: Exception) {
        println("[Qdrant] Error ensuring collection '$COLLECTION_NAME': ${e.message}")
        throw e // Re-throw to stop the process if collection setup fails
    }
}

private suspend fun uploadToQdrant(points: List<IdxQdrantPoint>) {
    if (points.isEmpty()) {
        println("[Qdrant] No points to upload.")
        return
    }
    println("[Qdrant] Uploading ${points.size} points to collection '$COLLECTION_NAME'...")
    try {
        httpClient.put("$QDRANT_URL/collections/$COLLECTION_NAME/points?wait=true") { // wait=true for synchronous indexing
            contentType(ContentType.Application.Json)
            setBody(IdxQdrantUpsertPointsRequest(points))
        }
        println("[Qdrant] Successfully uploaded ${points.size} points.")
    } catch (e: Exception) {
        println("[Qdrant] Error uploading points to Qdrant: ${e.message}")
    }
}

// --- Main Indexing Process ---
fun main(args: Array<String>) = runBlocking {
    println("--- Kotlin Website Indexing Script ---")
    if (args.isEmpty()) { println("Usage: ./gradlew runIndex --args=\"<base-url>\""); return@runBlocking }
    val baseUrlToIndex = args[0]; println("Target website: $baseUrlToIndex")
    val scrapedPages = scrapeWebsite(baseUrlToIndex)
    if (scrapedPages.isEmpty()) { println("No content scraped. Exiting."); return@runBlocking }
    val allTextChunks = mutableListOf<String>(); val chunkPayloads = mutableListOf<Map<String, String>>()
    
    scrapedPages.forEach { (pageUrl, pageText) ->
        val chunksFromPage = chunkTextContent(pageText)
        chunksFromPage.forEach {
            allTextChunks.add(it)
            chunkPayloads.add(mapOf("source" to pageUrl, "text" to it))
        }
    }
    
    if (allTextChunks.isEmpty()) { println("No text chunks. Exiting."); return@runBlocking }
    println("Total text chunks generated: ${allTextChunks.size}")
    val embeddings = generateEmbeddings(allTextChunks)
    if (embeddings.isEmpty() || embeddings.size != allTextChunks.size) { println("Embedding generation failed. Exiting."); return@runBlocking }
    ensureQdrantCollection(embeddings.first().size)
    val qdrantPoints = embeddings.mapIndexed { index, vector ->
        IdxQdrantPoint(id = UUID.randomUUID().toString(), vector = vector, payload = chunkPayloads[index])
    }
    uploadToQdrant(qdrantPoints)
    println("--- Indexing process complete for $baseUrlToIndex ---")
} 