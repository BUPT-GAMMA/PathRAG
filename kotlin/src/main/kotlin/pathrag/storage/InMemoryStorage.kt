package pathrag.storage

import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import pathrag.base.BaseGraphStorage
import pathrag.base.BaseKVStorage
import pathrag.base.BaseVectorStorage
import pathrag.utils.EmbeddingFunc
import pathrag.utils.computeMdHashId
import java.util.concurrent.ConcurrentHashMap
import kotlin.math.abs

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
    private val metaFields: Set<String> = setOf("entity_name", "full_doc_id", "source_id"),
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
        if (query.isBlank()) return emptyList()
        val queryEmbeddings = embeddingFunc(listOf(query))
        if (queryEmbeddings.isEmpty()) return emptyList()
        val queryEmbedding = queryEmbeddings.first()
        return entries.values
            .map { stored ->
                val similarity = cosineSimilarity(queryEmbedding, stored.embedding)
                mapOf("content" to stored.content, "score" to similarity) + stored.meta
            }.sortedByDescending { it["score"] as Double }
            .take(topK)
    }

    override suspend fun upsert(data: Map<String, Map<String, Any?>>) {
        val items = data.entries.toList()
        val contents = items.map { it.value["content"]?.toString().orEmpty() }
        val validPairs = items.zip(contents).filter { it.second.isNotBlank() }
        if (validPairs.isEmpty()) return
        val embeddings = embeddingFunc(validPairs.map { it.second })
        mutex.withLock {
            embeddings.forEachIndexed { index, vector ->
                val (entry, content) = validPairs[index]
                val key = entry.key
                val value = entry.value
                val meta = value.filterKeys { metaFields.contains(it) }
                entries[key] = StoredVector(vector, content, meta)
            }
        }
    }

    override suspend fun deleteEntity(entityName: String) {
        val entityId = computeMdHashId(entityName, prefix = "ent-")
        mutex.withLock {
            entries.remove(entityId)
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
    private var cachedPagerank: Map<String, Double>? = null

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
            cachedPagerank = null
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
            cachedPagerank = null
            val existing = edges[sourceNodeId to targetNodeId] ?: mutableMapOf()
            existing.putAll(edgeData)
            edges[sourceNodeId to targetNodeId] = existing
        }
    }

    override suspend fun deleteNode(nodeId: String) {
        mutex.withLock {
            nodes.remove(nodeId)
            edges.keys.filter { it.first == nodeId || it.second == nodeId }.forEach { edges.remove(it) }
            cachedPagerank = null
        }
    }

    override suspend fun nodes(): List<String> = nodes.keys().toList()

    override suspend fun edges(): List<Pair<String, String>> = edges.keys.toList()

    override suspend fun getPagerank(nodeId: String): Double {
        val ranks = cachedPagerank ?: computePagerank().also { cachedPagerank = it }
        return ranks[nodeId] ?: 0.0
    }

    override suspend fun embedNodes(algorithm: String): Pair<DoubleArray, List<String>> {
        val labels = nodes.keys().toList()
        if (labels.isEmpty()) return DoubleArray(0) to emptyList()
        return when (algorithm.lowercase()) {
            "node2vec" -> runNode2Vec(labels)
            else -> runMetadataEmbedding(labels)
        }
    }

    private suspend fun runMetadataEmbedding(labels: List<String>): Pair<DoubleArray, List<String>> {
        val func = embeddingFunc
        return if (func != null) {
            val texts =
                labels.map { id ->
                    val n = nodes[id] ?: emptyMap()
                    val desc = n["description"]?.toString().orEmpty()
                    "$id ${n["entity_type"] ?: ""} $desc"
                }
            val vectors = func(texts)
            val flat = vectors.flatMap { it.asIterable() }.toDoubleArray()
            flat to labels
        } else {
            val ranks = computePagerank()
            val degs = labels.map { nodeDegree(it).toDouble() }
            val vecs =
                labels.mapIndexed { idx, id ->
                    doubleArrayOf(ranks[id] ?: 0.0, degs[idx])
                }
            val flat = vecs.flatMap { it.asIterable() }.toDoubleArray()
            flat to labels
        }
    }

    private suspend fun runNode2Vec(labels: List<String>): Pair<DoubleArray, List<String>> {
        val dim = globalConfig["node2vec_dim"] as? Int ?: 64
        if (labels.isEmpty()) return DoubleArray(0) to emptyList()
        val ranks = computePagerank()
        val degs = labels.map { nodeDegree(it).toDouble() }
        val vectors =
            labels.mapIndexed { idx, id ->
                DoubleArray(dim) { i ->
                    val r = ranks[id] ?: 0.0
                    val d = degs[idx]
                    if (i % 2 == 0) r else d
                }
            }
        val flat = vectors.flatMap { it.asIterable() }.toDoubleArray()
        return flat to labels
    }

    private fun computePagerank(
        damping: Double = 0.85,
        maxIter: Int = 100,
        tol: Double = 1e-6,
    ): Map<String, Double> {
        val nodesList = nodes.keys().toList()
        if (nodesList.isEmpty()) return emptyMap()
        val n = nodesList.size
        val adjacency =
            nodesList.associateWith { mutableListOf<String>() }.toMutableMap().also { adj ->
                edges.keys.forEach { (u, v) ->
                    adj[u]?.add(v)
                    adj[v]?.add(u)
                }
            }
        val rank = mutableMapOf<String, Double>()
        nodesList.forEach { rank[it] = 1.0 / n }

        repeat(maxIter) {
            var diff = 0.0
            val newRank = mutableMapOf<String, Double>()
            for (node in nodesList) {
                val neighbors = adjacency[node].orEmpty()
                val outDeg = neighbors.size
                val share = if (outDeg == 0) 0.0 else rank[node]!! / outDeg
                neighbors.forEach { dest -> newRank[dest] = (newRank[dest] ?: 0.0) + share }
            }
            for (node in nodesList) {
                val updated = (1 - damping) / n + damping * (newRank[node] ?: 0.0)
                diff += abs(updated - (rank[node] ?: 0.0))
                rank[node] = updated
            }
            if (diff < tol) return rank
        }
        return rank
    }
}
