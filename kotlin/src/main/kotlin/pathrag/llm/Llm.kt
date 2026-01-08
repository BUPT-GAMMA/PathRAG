package pathrag.llm

import dev.langchain4j.data.embedding.Embedding
import dev.langchain4j.data.segment.TextSegment
import dev.langchain4j.model.chat.ChatModel
import dev.langchain4j.model.embedding.EmbeddingModel
import dev.langchain4j.model.openai.OpenAiChatModel
import dev.langchain4j.model.openai.OpenAiEmbeddingModel
import dev.langchain4j.model.output.Response
import io.github.oshai.kotlinlogging.KotlinLogging
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.withContext
import pathrag.utils.EmbeddingFunc
import java.util.concurrent.ConcurrentHashMap
import kotlin.random.Random

private val logger = KotlinLogging.logger("PathRAG-LLM")
private const val DEFAULT_CHAT_MODEL = "gpt-4o-mini"
private const val DEFAULT_EMBED_MODEL = "text-embedding-3-small"

private val chatModels = ConcurrentHashMap<String, ChatModel>()
private val embeddingModels = ConcurrentHashMap<String, EmbeddingModel>()

suspend fun openAiComplete(
    model: String,
    prompt: String,
    systemPrompt: String? = null,
    historyMessages: List<Map<String, String>> = emptyList(),
    keywordExtraction: Boolean = false,
    stream: Boolean = false,
    maxTokens: Int? = null,
    hashingKv: Any? = null,
): String {
    val apiKey = System.getenv("OPENAI_API_KEY")
    if (apiKey.isNullOrBlank()) {
        logger.warn { "OPENAI_API_KEY not set. Falling back to stubbed response." }
        delay(50)
        return if (keywordExtraction) {
            """{"high_level_keywords": ["${prompt.take(10)}"], "low_level_keywords": ["${prompt.takeLast(10)}"]}"""
        } else {
            val history = if (historyMessages.isEmpty()) "" else " History size=${historyMessages.size}."
            "${systemPrompt.orEmpty()} $prompt$history".trim().take(maxTokens ?: 4000)
        }
    }

    val baseUrl = System.getenv("OPENAI_API_BASE")
    val modelName = model.ifBlank { DEFAULT_CHAT_MODEL }
    val chatModel: ChatModel =
        chatModels.computeIfAbsent("$modelName|$baseUrl") {
            val builder = OpenAiChatModel.builder().apiKey(apiKey).modelName(modelName)
            baseUrl?.takeIf { it.isNotBlank() }?.let { builder.baseUrl(it) }
            builder.build()
        }

    val fullPrompt =
        buildString {
            if (!systemPrompt.isNullOrBlank()) {
                appendLine(systemPrompt)
            }
            if (historyMessages.isNotEmpty()) {
                historyMessages.forEach { msg ->
                    appendLine("${msg["role"] ?: "user"}: ${msg["content"] ?: ""}")
                }
            }
            append(prompt)
        }

    val result: String =
        withContext(Dispatchers.IO) {
            chatModel.chat(fullPrompt)
        }
    return result
}

suspend fun openAiEmbedding(inputs: List<String>): List<DoubleArray> {
    val apiKey = System.getenv("OPENAI_API_KEY")
    if (apiKey.isNullOrBlank()) {
        logger.warn { "OPENAI_API_KEY not set. Falling back to stubbed embeddings." }
        return inputs.map { text ->
            val seed = text.hashCode()
            val random = Random(seed)
            DoubleArray(16) { random.nextDouble() }
        }
    }
    val baseUrl = System.getenv("OPENAI_API_BASE")
    val modelName = System.getenv("OPENAI_EMBEDDING_MODEL") ?: DEFAULT_EMBED_MODEL
    val embedModel: EmbeddingModel =
        embeddingModels.computeIfAbsent("$modelName|$baseUrl") {
            val builder = OpenAiEmbeddingModel.builder().apiKey(apiKey).modelName(modelName)
            baseUrl?.takeIf { it.isNotBlank() }?.let { builder.baseUrl(it) }
            builder.build()
        }

    return withContext(Dispatchers.IO) {
        val segments = inputs.map { TextSegment.from(it) }
        val response: Response<List<Embedding>> = embedModel.embedAll(segments)
        response.content().map { embedding ->
            val vector = embedding.vector()
            DoubleArray(vector.size) { idx -> vector[idx].toDouble() }
        }
    }
}

fun defaultEmbeddingFunc(): EmbeddingFunc =
    EmbeddingFunc(
        embeddingDim = 1536,
        maxTokenSize = 4096,
        func = ::openAiEmbedding,
    )
