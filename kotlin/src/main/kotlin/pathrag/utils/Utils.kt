package pathrag.utils

import io.github.oshai.kotlinlogging.KotlinLogging
import kotlinx.coroutines.delay
import kotlinx.coroutines.sync.Semaphore
import kotlinx.coroutines.sync.withPermit
import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import java.io.File
import java.math.BigInteger
import java.security.MessageDigest
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.atomic.AtomicInteger
import kotlin.math.pow
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

@Serializable
private data class PersistEntry(
    val content: String,
    val prompt: String,
    val embedding: String? = null,
    val shape: List<Int>? = null,
    val min: Double? = null,
    val max: Double? = null,
)

class ResponseCache(
    val globalConfig: Map<String, Any?> = emptyMap(),
) {
    data class Entry(
        val content: String,
        val prompt: String,
        val embedding: DoubleArray?,
    )

    private val store = ConcurrentHashMap<String, MutableMap<String, Entry>>()
    private val cachePath: String? =
        (globalConfig["cache_path"] as? String)
            ?: (globalConfig["working_dir"] as? String)?.let { "$it/llm_cache.json" }

    init {
        loadFromDisk()
    }

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
                ?: mapOf("enabled" to false, "similarity_threshold" to 0.95, "use_llm_check" to false)
        val embedEnabled = embedCfg["enabled"] as? Boolean ?: false
        val embedding =
            if (embedEnabled) {
                val func = globalConfig["embedding_func"] as? EmbeddingFunc
                func?.invoke(listOf(prompt))?.firstOrNull()
            } else {
                null
            }
        store.computeIfAbsent(mode) { ConcurrentHashMap() }[argsHash] = Entry(content, prompt, embedding)
        persist()
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
                ?: mapOf("enabled" to false, "similarity_threshold" to 0.95, "use_llm_check" to false)
        val embedEnabled = embedCfg["enabled"] as? Boolean ?: false
        if (!embedEnabled) return null

        val similarityThreshold = (embedCfg["similarity_threshold"] as? Number)?.toDouble() ?: 0.95
        val embeddingFunc = globalConfig["embedding_func"] as? EmbeddingFunc ?: return null
        val currentEmbedding = embeddingFunc(listOf(prompt)).firstOrNull() ?: return null
        val useLlmCheck = embedCfg["use_llm_check"] as? Boolean ?: false
        val llmFunc =
            globalConfig["llm_model_func"] as? suspend (
                String,
                String?,
                List<Map<String, String>>,
                Boolean,
                Boolean,
                Int?,
                Any?,
            ) -> String

        var best: Entry? = null
        var bestSim = -1.0
        modeCache?.values?.forEach { entry ->
            val cachedEmb = entry.embedding ?: return@forEach
            val sim = cosineSimilarity(currentEmbedding, cachedEmb)
            if (!sim.isNaN() && sim > bestSim) {
                bestSim = sim
                best = entry
            }
        }
        if (best != null && bestSim >= similarityThreshold) {
            if (useLlmCheck && llmFunc != null) {
                val promptTemplate =
                    (globalConfig["similarity_check_prompt"] as? String)
                        ?: pathrag.prompt.Prompts.SIMILARITY_CHECK
                val promptCheck =
                    promptTemplate
                        .replace("{original_prompt}", prompt)
                        .replace("{cached_prompt}", best.prompt)
                val llmScore =
                    runCatching { llmFunc(promptCheck, null, emptyList(), false, false, 32, null).trim() }
                        .getOrNull()
                        ?.toDoubleOrNull()
                if (llmScore != null && llmScore < similarityThreshold) {
                    return null
                }
            }
            return best.content
        }
        return null
    }

    private fun persist() {
        val path = cachePath ?: return
        val payload =
            store.mapValues { (_, v) ->
                v.mapValues { (_, entry) ->
                    val q = entry.embedding?.let { quantizeEmbedding(it) }
                    PersistEntry(
                        content = entry.content,
                        prompt = entry.prompt,
                        embedding = q?.first?.let { bytesToHex(it) },
                        shape = q?.fourth,
                        min = q?.second,
                        max = q?.third,
                    )
                }
            }
        runCatching {
            val f = File(path)
            f.parentFile?.mkdirs()
            f.writeText(Json.encodeToString(payload))
        }.onFailure { internalLogger.warn(it) { "Failed to persist cache to $path" } }
    }

    private fun loadFromDisk() {
        val path = cachePath ?: return
        val f = File(path)
        if (!f.exists()) return
        runCatching {
            val text = f.readText()
            val decoded: Map<String, Map<String, PersistEntry>> = Json.decodeFromString(text)
            decoded.forEach { (mode, entries) ->
                entries.forEach { (hash, entry) ->
                    val embedding =
                        if (entry.embedding != null && entry.shape != null && entry.min != null && entry.max != null) {
                            val bytes = hexToBytes(entry.embedding)
                            dequantizeEmbedding(bytes, entry.min, entry.max, entry.shape)
                        } else {
                            null
                        }
                    store.computeIfAbsent(mode) { ConcurrentHashMap() }[hash] =
                        Entry(entry.content, entry.prompt, embedding)
                }
            }
        }.onFailure { internalLogger.warn(it) { "Failed to load cache from $path" } }
    }
}

private fun cosineSimilarity(
    a: DoubleArray,
    b: DoubleArray,
): Double {
    if (a.isEmpty() || b.isEmpty() || a.size != b.size) return Double.NaN
    var dot = 0.0
    var na = 0.0
    var nb = 0.0
    for (i in a.indices) {
        dot += a[i] * b[i]
        na += a[i] * a[i]
        nb += b[i] * b[i]
    }
    val denom = sqrt(na) * sqrt(nb)
    return if (denom == 0.0) Double.NaN else dot / denom
}

private fun quantizeEmbedding(
    embedding: DoubleArray,
    bits: Int = 8,
): Quadruple<ByteArray, Double, Double, List<Int>>? {
    if (embedding.isEmpty()) return null
    // Only 8-bit quantization is supported because we store values in a ByteArray and dequantize with 255-scale.
    require(bits == 8) { "Only 8-bit quantization is supported." }
    val min = embedding.min()
    val max = embedding.max()
    val maxVal = (1 shl bits) - 1
    if (max == min) {
        return Quadruple(ByteArray(embedding.size) { 0 }, min, max, listOf(embedding.size))
    }
    val scale = (max - min) / maxVal
    val bytes = ByteArray(embedding.size)
    for (i in embedding.indices) {
        val q = ((embedding[i] - min) / scale).toInt().coerceIn(0, maxVal)
        bytes[i] = q.toByte()
    }
    return Quadruple(bytes, min, max, listOf(embedding.size))
}

private fun dequantizeEmbedding(
    quantized: ByteArray,
    min: Double,
    max: Double,
    shape: List<Int>,
): DoubleArray {
    val size = shape.firstOrNull() ?: quantized.size
    val scale = (max - min) / 255.0
    val result = DoubleArray(size)
    for (i in 0 until size.coerceAtMost(quantized.size)) {
        val v = quantized[i].toInt() and 0xFF
        result[i] = v * scale + min
    }
    return result
}

private fun bytesToHex(bytes: ByteArray): String = bytes.joinToString("") { "%02x".format(it) }

private fun hexToBytes(hex: String): ByteArray {
    val clean = hex.trim()
    require(clean.length % 2 == 0) { "Hex string must have an even length." }
    val len = clean.length
    val data = ByteArray(len / 2)
    var i = 0
    while (i < len) {
        val high = Character.digit(clean[i], 16)
        val low = Character.digit(clean[i + 1], 16)
        require(high >= 0 && low >= 0) { "Invalid hex character at position $i" }
        data[i / 2] = ((high shl 4) + low).toByte()
        i += 2
    }
    return data
}

private data class Quadruple<A, B, C, D>(
    val first: A,
    val second: B,
    val third: C,
    val fourth: D,
)
