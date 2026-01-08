package pathrag.base

import kotlinx.coroutines.runBlocking

data class QueryParam(
    val mode: String = "hybrid",
    val onlyNeedContext: Boolean = false,
    val onlyNeedPrompt: Boolean = false,
    val responseType: String = "Multiple Paragraphs",
    val stream: Boolean = false,
    val topK: Int = 40,
    val maxTokenForTextUnit: Int = 4000,
    val maxTokenForGlobalContext: Int = 3000,
    val maxTokenForLocalContext: Int = 5000,
)

open class StorageNameSpace(
    open val namespace: String,
    open val globalConfig: Map<String, Any?>,
) {
    open suspend fun indexDoneCallback() = Unit

    open suspend fun queryDoneCallback() = Unit
}

abstract class BaseVectorStorage(
    namespace: String,
    globalConfig: Map<String, Any?>,
) : StorageNameSpace(namespace, globalConfig) {
    abstract suspend fun query(
        query: String,
        topK: Int,
    ): List<Map<String, Any?>>

    abstract suspend fun upsert(data: Map<String, Map<String, Any?>>)

    open suspend fun deleteEntity(entityName: String) {}

    open suspend fun deleteRelation(entityName: String) {}
}

abstract class BaseKVStorage<T>(
    namespace: String,
    globalConfig: Map<String, Any?>,
) : StorageNameSpace(namespace, globalConfig) {
    abstract suspend fun allKeys(): List<String>

    abstract suspend fun getById(id: String): T?

    abstract suspend fun getByIds(
        ids: List<String>,
        fields: Set<String>? = null,
    ): List<T?>

    abstract suspend fun filterKeys(data: List<String>): Set<String>

    abstract suspend fun upsert(data: Map<String, T>)

    abstract suspend fun drop()
}

abstract class BaseGraphStorage(
    namespace: String,
    globalConfig: Map<String, Any?>,
) : StorageNameSpace(namespace, globalConfig) {
    open suspend fun hasNode(nodeId: String): Boolean = false

    open suspend fun hasEdge(
        sourceNodeId: String,
        targetNodeId: String,
    ): Boolean = false

    open suspend fun nodeDegree(nodeId: String): Int = 0

    open suspend fun edgeDegree(
        srcId: String,
        tgtId: String,
    ): Int = 0

    open suspend fun getPagerank(nodeId: String): Double = 0.0

    abstract suspend fun getNode(nodeId: String): Map<String, Any?>?

    abstract suspend fun getEdge(
        sourceNodeId: String,
        targetNodeId: String,
    ): Map<String, Any?>?

    abstract suspend fun getNodeEdges(sourceNodeId: String): List<Pair<String, String>>?

    open suspend fun getNodeInEdges(nodeId: String): List<Pair<String, String>>? = edges().filter { it.second == nodeId }

    open suspend fun getNodeOutEdges(nodeId: String): List<Pair<String, String>>? = edges().filter { it.first == nodeId }

    abstract suspend fun upsertNode(
        nodeId: String,
        nodeData: Map<String, Any?>,
    )

    abstract suspend fun upsertEdge(
        sourceNodeId: String,
        targetNodeId: String,
        edgeData: Map<String, Any?>,
    )

    abstract suspend fun deleteNode(nodeId: String)

    open suspend fun embedNodes(algorithm: String): Pair<DoubleArray, List<String>> =
        throw NotImplementedError("Node embedding is not implemented")

    open suspend fun nodes(): List<String> = emptyList()

    open suspend fun edges(): List<Pair<String, String>> = emptyList()
}

/**
 * Convenience helper to run suspend functions from blocking callers, mirroring the Python sync wrappers.
 */
fun <T> runBlockingMaybe(block: suspend () -> T): T = runBlocking { block() }
