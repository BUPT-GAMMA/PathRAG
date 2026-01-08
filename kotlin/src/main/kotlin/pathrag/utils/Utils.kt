package pathrag.utils

import io.github.oshai.kotlinlogging.KotlinLogging
import kotlinx.coroutines.delay
import kotlinx.coroutines.sync.Semaphore
import kotlinx.coroutines.sync.withPermit
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import java.math.BigInteger
import java.security.MessageDigest
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.atomic.AtomicInteger
import kotlin.math.sqrt

private val internalLogger = KotlinLogging.logger("PathRAG")

fun log() = internalLogger

data class EmbeddingFunc(
    val embeddingDim: Int,
    val maxTokenSize: Int,
    val func: suspend (List<String>) -> List<DoubleArray>,
    val concurrentLimit: Int = 16,
) {
    private val semaphore = if (concurrentLimit > 0) Semaphore(concurrentLimit) else null

    suspend operator fun invoke(inputs: List<String>): List<DoubleArray> {
        val exec: suspend () -> List<DoubleArray> = { func(inputs) }
        val lock = semaphore
        return if (lock == null) {
            exec()
        } else {
            lock.withPermit { exec() }
        }
    }
}

fun convertResponseToJson(response: String): Map<String, Any?> {
    val regex = Regex("\\{.*\\}", RegexOption.DOT_MATCHES_ALL)
    val jsonString = regex.find(response)?.value ?: error("Unable to parse JSON from response: $response")
    val parsed = Json.parseToJsonElement(jsonString).jsonObject
    return parsed.mapValues { entry ->
        when (val value = entry.value) {
            is JsonObject -> value
            else -> value.jsonPrimitive.content
        }
    }
}

fun computeMdHashId(
    content: String,
    prefix: String = "",
): String {
    val md = MessageDigest.getInstance("MD5")
    val digest = md.digest(content.toByteArray())
    val bigInt = BigInteger(1, digest)
    val hashText = bigInt.toString(16).padStart(32, '0')
    return prefix + hashText
}

fun limitAsyncFuncCall(
    maxSize: Int,
    waitingTimeMillis: Long = 1,
): (suspend (() -> Unit) -> suspend () -> Unit) =
    { func ->
        val currentSize = AtomicInteger(0)
        suspend {
            while (currentSize.get() >= maxSize) {
                delay(waitingTimeMillis)
            }
            currentSize.incrementAndGet()
            try {
                func()
            } finally {
                currentSize.decrementAndGet()
            }
        }
    }

fun computeArgsHash(vararg args: Any?): String = computeMdHashId(args.toList().toString())

data class CacheData(
    val argsHash: String,
    val content: String,
    val prompt: String,
    val embedding: DoubleArray? = null,
    val mode: String = "default",
)

/**
 * Simple in-memory cache used to emulate the Python hashing KV behavior.
 */
class ResponseCache(
    val globalConfig: Map<String, Any?> = emptyMap(),
) {
    data class Entry(
        val content: String,
        val prompt: String,
        val embedding: DoubleArray?,
    )

    private val store = ConcurrentHashMap<String, MutableMap<String, Entry>>()

    suspend fun getById(mode: String): Map<String, Entry>? = store[mode]

    suspend fun upsert(
        mode: String,
        argsHash: String,
        content: String,
        prompt: String,
    ) {
        val embedCfg =
            globalConfig["embedding_cache_config"]?.takeIf { it is Map<*, *> }?.let {
                @Suppress("UNCHECKED_CAST")
                it as? Map<String, Any?>
            }
                ?: mapOf("enabled" to false, "similarity_threshold" to 0.95)
        val embedEnabled = embedCfg["enabled"] as? Boolean ?: false
        val embedding =
            if (embedEnabled) {
                val func = globalConfig["embedding_func"] as? EmbeddingFunc
                func?.invoke(listOf(prompt))?.firstOrNull()
            } else {
                null
            }
        store.computeIfAbsent(mode) { ConcurrentHashMap() }[argsHash] = Entry(content, prompt, embedding)
    }

    suspend fun handleCache(
        argsHash: String,
        prompt: String,
        mode: String,
    ): String? {
        val modeCache = store[mode]
        if (modeCache != null) {
            val direct = modeCache[argsHash]
            if (direct != null) return direct.content
        }

        val embedCfg =
            globalConfig["embedding_cache_config"]?.takeIf { it is Map<*, *> }?.let {
                @Suppress("UNCHECKED_CAST")
                it as? Map<String, Any?>
            }
                ?: mapOf("enabled" to false, "similarity_threshold" to 0.95)
        val embedEnabled = embedCfg["enabled"] as? Boolean ?: false
        if (!embedEnabled) return null

        val similarityThreshold = (embedCfg["similarity_threshold"] as? Number)?.toDouble() ?: 0.95
        val embeddingFunc = globalConfig["embedding_func"] as? EmbeddingFunc ?: return null
        val currentEmbedding = embeddingFunc(listOf(prompt)).firstOrNull() ?: return null

        var best: Entry? = null
        var bestSim = -1.0
        modeCache?.values?.forEach { entry ->
            val cachedEmb = entry.embedding ?: return@forEach
            val sim = cosineSimilarity(currentEmbedding, cachedEmb)
            if (sim > bestSim) {
                bestSim = sim
                best = entry
            }
        }
        return if (best != null && bestSim >= similarityThreshold) best.content else null
    }
}

private fun cosineSimilarity(
    a: DoubleArray,
    b: DoubleArray,
): Double {
    if (a.isEmpty() || b.isEmpty() || a.size != b.size) return -1.0
    var dot = 0.0
    var na = 0.0
    var nb = 0.0
    for (i in a.indices) {
        dot += a[i] * b[i]
        na += a[i] * a[i]
        nb += b[i] * b[i]
    }
    val denom = sqrt(na) * sqrt(nb)
    return if (denom == 0.0) -1.0 else dot / denom
}
