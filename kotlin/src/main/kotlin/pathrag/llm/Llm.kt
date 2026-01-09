package pathrag.llm

import dev.langchain4j.data.embedding.Embedding
import dev.langchain4j.data.segment.TextSegment
import dev.langchain4j.model.chat.ChatModel
import dev.langchain4j.model.embedding.EmbeddingModel
import dev.langchain4j.model.ollama.OllamaChatModel
import dev.langchain4j.model.ollama.OllamaEmbeddingModel
import dev.langchain4j.model.openai.OpenAiChatModel
import dev.langchain4j.model.openai.OpenAiEmbeddingModel
import dev.langchain4j.model.output.Response
import io.github.oshai.kotlinlogging.KotlinLogging
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.withContext
import pathrag.prompt.Prompts
import pathrag.utils.EmbeddingFunc
import java.util.concurrent.ConcurrentHashMap
import kotlin.random.Random

private val logger = KotlinLogging.logger("PathRAG-LLM")
private const val DEFAULT_CHAT_MODEL = "gpt-4o-mini"
private const val DEFAULT_OLLAMA_MODEL = "llama3"
private const val DEFAULT_EMBED_MODEL = "text-embedding-3-small"
private const val DEFAULT_OLLAMA_EMBED_MODEL = "nomic-embed-text"
private const val DEFAULT_EMBED_DIM = 1536
private const val DEFAULT_EMBED_CTX = 8192
private const val DEFAULT_OLLAMA_EMBED_DIM = 768

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

    val logRequests = System.getenv("OPENAI_LOG_REQUESTS")?.toBoolean() ?: false
    val logResponses = System.getenv("OPENAI_LOG_RESPONSES")?.toBoolean() ?: false
    val baseUrl = System.getenv("OPENAI_API_BASE")
    val modelName = model.ifBlank { DEFAULT_CHAT_MODEL }
    val chatModel: ChatModel =
        chatModels.computeIfAbsent("$modelName|$baseUrl") {
            val builder =
                OpenAiChatModel
                    .builder()
                    .apiKey(apiKey)
                    .logRequests(logRequests)
                    .logResponses(logResponses)
                    .modelName(modelName)
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
            val maxAttempts = (System.getenv("OPENAI_RETRY_ATTEMPTS")?.toIntOrNull() ?: 3).coerceAtLeast(1)
            val backoffMs = (System.getenv("OPENAI_RETRY_BACKOFF_MS")?.toLongOrNull() ?: 500L).coerceAtLeast(0L)
            var lastError: Exception? = null
            repeat(maxAttempts) { attempt ->
                try {
                    return@withContext chatModel.chat(fullPrompt)
                } catch (e: Exception) {
                    lastError = e
                    logger.warn(e) { "OpenAI chat attempt ${attempt + 1} failed for model $modelName" }
                    if (attempt < maxAttempts - 1) {
                        delay(backoffMs)
                    }
                }
            }
            logger.error(lastError) { "OpenAI chat call failed after $maxAttempts attempts for model $modelName" }
            Prompts.FAIL_RESPONSE
        }
    return result
}

suspend fun ollamaComplete(
    model: String,
    prompt: String,
    systemPrompt: String? = null,
    historyMessages: List<Map<String, String>> = emptyList(),
    keywordExtraction: Boolean = false,
    stream: Boolean = false,
    maxTokens: Int? = null,
    hashingKv: Any? = null,
): String {
    val baseUrl = System.getenv("OLLAMA_BASE_URL") ?: System.getenv("OLLAMA_HOST") ?: "http://localhost:11434"
    val modelName = model.ifBlank { System.getenv("OLLAMA_MODEL") ?: DEFAULT_OLLAMA_MODEL }
    val chatModel: ChatModel =
        chatModels.computeIfAbsent("ollama|$modelName|$baseUrl") {
            OllamaChatModel
                .builder()
                .baseUrl(baseUrl)
                .modelName(modelName)
                .build()
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

    return withContext(Dispatchers.IO) {
        runCatching { chatModel.chat(fullPrompt) }
            .onFailure { logger.warn(it) { "Ollama chat call failed for model $modelName" } }
            .getOrElse { Prompts.FAIL_RESPONSE }
    }
}

suspend fun openAiEmbedding(inputs: List<String>): List<DoubleArray> {
    val apiKey = System.getenv("OPENAI_API_KEY")
    val sanitized = inputs.filter { it.isNotBlank() }
    if (sanitized.isEmpty()) return emptyList()
    if (apiKey.isNullOrBlank()) {
        logger.warn { "OPENAI_API_KEY not set. Falling back to stubbed embeddings." }
        return sanitized.map { text ->
            val seed = text.hashCode()
            val random = Random(seed)
            DoubleArray(1536) { random.nextDouble() }
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
        val maxAttempts = (System.getenv("OPENAI_RETRY_ATTEMPTS")?.toIntOrNull() ?: 3).coerceAtLeast(1)
        val backoffMs = (System.getenv("OPENAI_RETRY_BACKOFF_MS")?.toLongOrNull() ?: 500L).coerceAtLeast(0L)
        var lastError: Exception? = null
        repeat(maxAttempts) { attempt ->
            try {
                val segments = sanitized.map { TextSegment.from(it) }
                val response: Response<List<Embedding>> = embedModel.embedAll(segments)
                return@withContext response.content().map { embedding ->
                    val vector = embedding.vector()
                    DoubleArray(vector.size) { idx -> vector[idx].toDouble() }
                }
            } catch (e: Exception) {
                lastError = e
                logger.warn(e) { "OpenAI embedding attempt ${attempt + 1} failed for model $modelName" }
                if (attempt < maxAttempts - 1) {
                    delay(backoffMs)
                }
            }
        }
        logger.error(lastError) { "OpenAI embedding call failed after $maxAttempts attempts for model $modelName" }
        throw IllegalStateException("OpenAI embedding call failed after $maxAttempts attempts for model $modelName", lastError)
    }
}

suspend fun ollamaEmbedding(inputs: List<String>): List<DoubleArray> {
    val sanitized = inputs.filter { it.isNotBlank() }
    if (sanitized.isEmpty()) return emptyList()
    val baseUrl = System.getenv("OLLAMA_BASE_URL") ?: System.getenv("OLLAMA_HOST") ?: "http://localhost:11434"
    val modelName = System.getenv("OLLAMA_EMBED_MODEL") ?: DEFAULT_OLLAMA_EMBED_MODEL
    val embedModel: EmbeddingModel =
        embeddingModels.computeIfAbsent("ollama|$modelName|$baseUrl") {
            OllamaEmbeddingModel
                .builder()
                .baseUrl(baseUrl)
                .modelName(modelName)
                .build()
        }

    return withContext(Dispatchers.IO) {
        runCatching {
            val segments = sanitized.map { TextSegment.from(it) }
            val response: Response<List<Embedding>> = embedModel.embedAll(segments)
            response.content().map { embedding ->
                val vector = embedding.vector()
                DoubleArray(vector.size) { idx -> vector[idx].toDouble() }
            }
        }.onFailure { ex ->
            logger.warn(ex) { "Ollama embedding call failed for model $modelName" }
            throw IllegalStateException("Ollama embedding call failed for model $modelName", ex)
        }.getOrThrow()
    }
}

fun defaultEmbeddingFunc(): EmbeddingFunc =
    embeddingModelConfig().let { (provider, dim, ctx) ->
        val func =
            when (provider) {
                "ollama" -> ::ollamaEmbedding
                else -> ::openAiEmbedding
            }
        EmbeddingFunc(
            embeddingDim = dim,
            maxTokenSize = ctx,
            func = func,
        )
    }

private fun embeddingModelConfig(): Triple<String, Int, Int> {
    val provider = System.getenv("EMBED_PROVIDER")?.lowercase() ?: "openai"
    return when (provider) {
        "ollama" -> {
            val dim = System.getenv("OLLAMA_EMBED_DIM")?.toIntOrNull() ?: DEFAULT_OLLAMA_EMBED_DIM
            Triple("ollama", dim, DEFAULT_EMBED_CTX)
        }

        else -> {
            val modelName = System.getenv("OPENAI_EMBEDDING_MODEL") ?: DEFAULT_EMBED_MODEL
            val dim =
                when {
                    modelName.contains("3-large", ignoreCase = true) -> {
                        3072
                    }

                    modelName.contains("3-small", ignoreCase = true) -> {
                        1536
                    }

                    else -> {
                        logger.warn { "Unrecognized embedding model '$modelName'; using default dim $DEFAULT_EMBED_DIM" }
                        DEFAULT_EMBED_DIM
                    }
                }
            val ctx =
                when {
                    modelName.contains("3-large", ignoreCase = true) ||
                        modelName.contains("3-small", ignoreCase = true) -> 8192

                    else -> DEFAULT_EMBED_CTX
                }
            Triple("openai", dim, ctx)
        }
    }
}
