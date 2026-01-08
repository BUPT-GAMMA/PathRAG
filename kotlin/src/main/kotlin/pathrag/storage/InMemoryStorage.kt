package pathrag.storage

import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import pathrag.base.BaseGraphStorage
import pathrag.base.BaseKVStorage
import pathrag.base.BaseVectorStorage
import pathrag.utils.EmbeddingFunc
import pathrag.utils.computeMdHashId
import java.util.concurrent.ConcurrentHashMap

class JsonKVStorage<T : Any>(
    override val namespace: String,
    override val globalConfig: Map<String, Any?>,
    private val embeddingFunc: EmbeddingFunc?,
) : BaseKVStorage<T>(namespace, globalConfig) {
    private val mutex = Mutex()
    private val data = ConcurrentHashMap<String, T>()

    override suspend fun allKeys(): List<String> = data.keys().toList()

    override suspend fun getById(id: String): T? = data[id]

    override suspend fun getByIds(
        ids: List<String>,
        fields: Set<String>?,
    ): List<T?> = ids.map { data[it] }

    override suspend fun filterKeys(data: List<String>): Set<String> {
        val existing = allKeys().toSet()
        return data.filterNot { existing.contains(it) }.toSet()
    }

    override suspend fun upsert(data: Map<String, T>) {
        mutex.withLock {
            this.data.putAll(data)
        }
    }

    override suspend fun drop() {
        mutex.withLock { data.clear() }
    }
}

class NanoVectorDBStorage(
    override val namespace: String,
    override val globalConfig: Map<String, Any?>,
    private val embeddingFunc: EmbeddingFunc,
    private val metaFields: Set<String> = emptySet(),
) : BaseVectorStorage(namespace, globalConfig) {
    private val mutex = Mutex()
    private val entries = ConcurrentHashMap<String, StoredVector>()

    data class StoredVector(
        val embedding: DoubleArray,
        val content: String,
        val meta: Map<String, Any?> = emptyMap(),
    )

    override suspend fun query(
        query: String,
        topK: Int,
    ): List<Map<String, Any?>> {
        if (entries.isEmpty()) return emptyList()
        val queryEmbedding = embeddingFunc(listOf(query)).first()
        return entries.values
            .map { stored ->
                val similarity = cosineSimilarity(queryEmbedding, stored.embedding)
                mapOf("content" to stored.content, "score" to similarity) + stored.meta
            }.sortedByDescending { it["score"] as Double }
            .take(topK)
    }

    override suspend fun upsert(data: Map<String, Map<String, Any?>>) {
        val embeddings = embeddingFunc(data.values.map { it["content"].toString() })
        val items = data.entries.toList()
        mutex.withLock {
            embeddings.forEachIndexed { index, vector ->
                val key = items[index].key
                val value = items[index].value
                val meta = value.filterKeys { metaFields.contains(it) }
                val content = value["content"]?.toString().orEmpty()
                entries[key] = StoredVector(vector, content, meta)
            }
        }
    }

    override suspend fun deleteEntity(entityName: String) {
        mutex.withLock {
            entries.keys.filter { it.contains(entityName, ignoreCase = true) }.forEach { entries.remove(it) }
        }
    }

    override suspend fun deleteRelation(entityName: String) {
        deleteEntity(entityName)
    }

    private fun cosineSimilarity(
        a: DoubleArray,
        b: DoubleArray,
    ): Double {
        val dot = a.zip(b).sumOf { it.first * it.second }
        val normA = kotlin.math.sqrt(a.sumOf { it * it })
        val normB = kotlin.math.sqrt(b.sumOf { it * it })
        return if (normA == 0.0 || normB == 0.0) 0.0 else dot / (normA * normB)
    }
}

class NetworkXStorage(
    override val namespace: String,
    override val globalConfig: Map<String, Any?>,
    private val embeddingFunc: EmbeddingFunc?,
) : BaseGraphStorage(namespace, globalConfig) {
    private val mutex = Mutex()
    private val nodes = ConcurrentHashMap<String, MutableMap<String, Any?>>()
    private val edges = ConcurrentHashMap<Pair<String, String>, MutableMap<String, Any?>>()

    override suspend fun hasNode(nodeId: String): Boolean = nodes.containsKey(nodeId)

    override suspend fun hasEdge(
        sourceNodeId: String,
        targetNodeId: String,
    ): Boolean = edges.containsKey(sourceNodeId to targetNodeId)

    override suspend fun nodeDegree(nodeId: String): Int = edges.keys.count { it.first == nodeId || it.second == nodeId }

    override suspend fun edgeDegree(
        srcId: String,
        tgtId: String,
    ): Int = if (edges.containsKey(srcId to tgtId)) 1 else 0

    override suspend fun getNode(nodeId: String): Map<String, Any?>? = nodes[nodeId]

    override suspend fun getEdge(
        sourceNodeId: String,
        targetNodeId: String,
    ): Map<String, Any?>? = edges[sourceNodeId to targetNodeId]

    override suspend fun getNodeEdges(sourceNodeId: String): List<Pair<String, String>> =
        edges.keys.filter { it.first == sourceNodeId || it.second == sourceNodeId }

    override suspend fun upsertNode(
        nodeId: String,
        nodeData: Map<String, Any?>,
    ) {
        mutex.withLock {
            val existing = nodes[nodeId] ?: mutableMapOf()
            existing.putAll(nodeData)
            nodes[nodeId] = existing
        }
    }

    override suspend fun upsertEdge(
        sourceNodeId: String,
        targetNodeId: String,
        edgeData: Map<String, Any?>,
    ) {
        mutex.withLock {
            val existing = edges[sourceNodeId to targetNodeId] ?: mutableMapOf()
            existing.putAll(edgeData)
            edges[sourceNodeId to targetNodeId] = existing
        }
    }

    override suspend fun deleteNode(nodeId: String) {
        mutex.withLock {
            nodes.remove(nodeId)
            edges.keys.filter { it.first == nodeId || it.second == nodeId }.forEach { edges.remove(it) }
        }
    }

    override suspend fun nodes(): List<String> = nodes.keys().toList()

    override suspend fun edges(): List<Pair<String, String>> = edges.keys.toList()
}
