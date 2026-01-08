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
        var currentSize = 0
        suspend {
            while (currentSize >= maxSize) {
                delay(waitingTimeMillis)
            }
            currentSize += 1
            try {
                func()
            } finally {
                currentSize -= 1
            }
        }
    }

fun computeArgsHash(vararg args: Any?): String = computeMdHashId(args.toList().toString())

data class CacheData(
    val argsHash: String,
    val content: String,
    val prompt: String,
    val mode: String = "default",
)

/**
 * Simple in-memory cache used to emulate the Python hashing KV behavior.
 */
class ResponseCache {
    private val store = ConcurrentHashMap<String, MutableMap<String, String>>()

    suspend fun getById(mode: String): Map<String, String>? = store[mode]

    suspend fun upsert(
        mode: String,
        argsHash: String,
        content: String,
    ) {
        store.computeIfAbsent(mode) { ConcurrentHashMap() }[argsHash] = content
    }
}
