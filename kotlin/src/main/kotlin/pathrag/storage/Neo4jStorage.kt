package pathrag.storage

import io.github.oshai.kotlinlogging.KotlinLogging
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.neo4j.driver.AuthTokens
import org.neo4j.driver.Driver
import org.neo4j.driver.GraphDatabase
import org.neo4j.driver.Record
import org.neo4j.driver.TransactionContext
import org.neo4j.driver.Values
import pathrag.base.BaseGraphStorage
import java.io.Closeable
import kotlin.math.abs

class Neo4jStorage(
    override val namespace: String,
    override val globalConfig: Map<String, Any?>,
) : BaseGraphStorage(namespace, globalConfig),
    Closeable {
    private val logger = KotlinLogging.logger("PathRAG-Neo4j")

    private val uri: String =
        (globalConfig["neo4j_uri"] as? String)
            ?: System.getenv("NEO4J_URI")
            ?: "bolt://localhost:7687"
    private val user: String =
        (globalConfig["neo4j_user"] as? String)
            ?: System.getenv("NEO4J_USER")
            ?: "neo4j"
    private val password: String =
        (globalConfig["neo4j_password"] as? String)
            ?: System.getenv("NEO4J_PASSWORD")
            ?: run {
                logger.warn { "Using default Neo4j password; set NEO4J_PASSWORD or neo4j_password in config for production." }
                "password"
            }
    private val driver: Driver = GraphDatabase.driver(uri, AuthTokens.basic(user, password))

    private val nodeLabel = namespace
    private val relType = "${namespace.uppercase()}_REL"

    override fun close() {
        driver.close()
    }

    private suspend fun <T> read(block: (TransactionContext) -> T): T =
        withContext(Dispatchers.IO) { driver.session().use { session -> session.executeRead { tx -> block(tx) } } }

    private suspend fun <T> write(block: (TransactionContext) -> T): T =
        withContext(Dispatchers.IO) { driver.session().use { session -> session.executeWrite { tx -> block(tx) } } }

    override suspend fun hasNode(nodeId: String): Boolean =
        read { tx ->
            tx
                .run(
                    "MATCH (n:$nodeLabel {id:\$id}) RETURN 1 LIMIT 1",
                    Values.parameters("id", nodeId),
                ).hasNext()
        }

    override suspend fun hasEdge(
        sourceNodeId: String,
        targetNodeId: String,
    ): Boolean =
        read { tx ->
            tx
                .run(
                    "MATCH (:$nodeLabel {id:\$src})-[r:$relType]->(:$nodeLabel {id:\$tgt}) RETURN 1 LIMIT 1",
                    Values.parameters("src", sourceNodeId, "tgt", targetNodeId),
                ).hasNext()
        }

    override suspend fun nodeDegree(nodeId: String): Int =
        read { tx ->
            tx
                .run(
                    "MATCH (n:$nodeLabel {id:\$id})-[r]-() RETURN count(r) AS deg",
                    Values.parameters("id", nodeId),
                ).list()
                .firstOrNull()
                ?.get("deg")
                ?.asInt() ?: 0
        }

    override suspend fun edgeDegree(
        srcId: String,
        tgtId: String,
    ): Int = if (hasEdge(srcId, tgtId)) 1 else 0

    override suspend fun getNode(nodeId: String): Map<String, Any?>? =
        read { tx ->
            tx
                .run(
                    "MATCH (n:$nodeLabel {id:\$id}) RETURN properties(n) AS props",
                    Values.parameters("id", nodeId),
                ).list()
                .firstOrNull()
                ?.get("props")
                ?.asMap { v -> v.asObject() }
        }

    override suspend fun getEdge(
        sourceNodeId: String,
        targetNodeId: String,
    ): Map<String, Any?>? =
        read { tx ->
            tx
                .run(
                    "MATCH (s:$nodeLabel {id:\$src})-[r:$relType]->(t:$nodeLabel {id:\$tgt}) RETURN properties(r) AS props",
                    Values.parameters("src", sourceNodeId, "tgt", targetNodeId),
                ).list()
                .firstOrNull()
                ?.get("props")
                ?.asMap { v -> v.asObject() }
        }

    override suspend fun getNodeEdges(sourceNodeId: String): List<Pair<String, String>> =
        read { tx ->
            tx
                .run(
                    "MATCH (s:$nodeLabel {id:\$id})-[r:$relType]->(t:$nodeLabel) RETURN s.id AS src, t.id AS tgt " +
                        "UNION " +
                        "MATCH (s:$nodeLabel)-[r:$relType]->(t:$nodeLabel {id:\$id}) RETURN s.id AS src, t.id AS tgt",
                    Values.parameters("id", sourceNodeId),
                ).list { rec -> rec.get("src").asString() to rec.get("tgt").asString() }
        }

    override suspend fun getNodeInEdges(nodeId: String): List<Pair<String, String>> =
        read { tx ->
            tx
                .run(
                    "MATCH (s:$nodeLabel)-[r:$relType]->(t:$nodeLabel {id:\$id}) RETURN s.id AS src, t.id AS tgt",
                    Values.parameters("id", nodeId),
                ).list { rec -> rec.get("src").asString() to rec.get("tgt").asString() }
        }

    override suspend fun getNodeOutEdges(nodeId: String): List<Pair<String, String>> =
        read { tx ->
            tx
                .run(
                    "MATCH (s:$nodeLabel {id:\$id})-[r:$relType]->(t:$nodeLabel) RETURN s.id AS src, t.id AS tgt",
                    Values.parameters("id", nodeId),
                ).list { rec -> rec.get("src").asString() to rec.get("tgt").asString() }
        }

    override suspend fun upsertNode(
        nodeId: String,
        nodeData: Map<String, Any?>,
    ) {
        write { tx ->
            tx.run(
                "MERGE (n:$nodeLabel {id:\$id}) SET n += \$props",
                Values.parameters("id", nodeId, "props", nodeData),
            )
        }
    }

    override suspend fun upsertEdge(
        sourceNodeId: String,
        targetNodeId: String,
        edgeData: Map<String, Any?>,
    ) {
        write { tx ->
            tx.run(
                "MERGE (s:$nodeLabel {id:\$src}) " +
                    "MERGE (t:$nodeLabel {id:\$tgt}) " +
                    "MERGE (s)-[r:$relType]->(t) " +
                    "SET r += \$props, r.src_id = \$src, r.tgt_id = \$tgt",
                Values.parameters("src", sourceNodeId, "tgt", targetNodeId, "props", edgeData),
            )
        }
    }

    override suspend fun deleteNode(nodeId: String) {
        write { tx ->
            tx.run(
                "MATCH (n:$nodeLabel {id:\$id}) DETACH DELETE n",
                Values.parameters("id", nodeId),
            )
        }
    }

    override suspend fun deleteEdge(
        sourceNodeId: String,
        targetNodeId: String,
    ) {
        write { tx ->
            tx.run(
                "MATCH (s:$nodeLabel {id:\$src})-[r:$relType]->(t:$nodeLabel {id:\$tgt}) DELETE r",
                Values.parameters("src", sourceNodeId, "tgt", targetNodeId),
            )
        }
    }

    override suspend fun nodes(): List<String> =
        read { tx -> tx.run("MATCH (n:$nodeLabel) RETURN n.id AS id").list { it.get("id").asString() } }

    override suspend fun edges(): List<Pair<String, String>> =
        read { tx ->
            tx
                .run("MATCH (s:$nodeLabel)-[r:$relType]->(t:$nodeLabel) RETURN s.id AS src, t.id AS tgt")
                .list { rec -> rec.get("src").asString() to rec.get("tgt").asString() }
        }

    override suspend fun getPagerank(nodeId: String): Double {
        val ranks = computePagerank()
        return ranks[nodeId] ?: 0.0
    }

    override suspend fun drop() {
        write { tx -> tx.run("MATCH (n:$nodeLabel) DETACH DELETE n") }
    }

    override suspend fun embedNodes(algorithm: String): Pair<DoubleArray, List<String>> {
        return if (algorithm.lowercase() == "node2vec") {
            runNode2VecGds(globalConfig["node2vec_dim"] as? Int ?: 64)
        } else {
            val labels = nodes()
            if (labels.isEmpty()) return DoubleArray(0) to emptyList()
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

    private suspend fun runNode2VecGds(dim: Int): Pair<DoubleArray, List<String>> =
        withContext(Dispatchers.IO) {
            val labels = mutableListOf<String>()
            val vectors = mutableListOf<DoubleArray>()
            val graphName = "pathrag_${namespace}_gds"
            try {
                driver.session().use { session ->
                    runCatching {
                        session
                            .run(
                                "CALL gds.graph.drop(\$g, false) YIELD graphName",
                                Values.parameters("g", graphName),
                            ).consume()
                    }
                    session
                        .run(
                            "CALL gds.graph.project(\$g, \$labels, {`$relType`:{orientation:'UNDIRECTED'}})",
                            Values.parameters("g", graphName, "labels", listOf(nodeLabel)),
                        ).consume()
                    val result =
                        session.run(
                            "CALL gds.beta.node2vec.stream(\$g, {embeddingDimension:\$dim}) " +
                                "YIELD nodeId, embedding " +
                                "RETURN gds.util.asNode(nodeId).id AS key, embedding",
                            Values.parameters("g", graphName, "dim", dim),
                        )
                    result.list { rec: Record ->
                        val key = rec.get("key").asString()
                        val emb = rec.get("embedding").asList { (it as Number).toDouble() }.toDoubleArray()
                        labels.add(key)
                        vectors.add(emb)
                        Unit
                    }
                }
            } catch (ex: Exception) {
                logger.warn(ex) { "Neo4j node2vec failed; falling back to pagerank/degree." }
                return@withContext computeFallbackEmbeddings()
            } finally {
                runCatching {
                    driver.session().use {
                        it.run("CALL gds.graph.drop(\$g, false) YIELD graphName", Values.parameters("g", graphName)).consume()
                    }
                }
            }

            if (labels.isEmpty() || vectors.isEmpty()) {
                return@withContext computeFallbackEmbeddings()
            }
            val flat = vectors.flatMap { it.asIterable() }.toDoubleArray()
            flat to labels
        }

    private suspend fun computeFallbackEmbeddings(): Pair<DoubleArray, List<String>> {
        val fallbackLabels = nodes()
        if (fallbackLabels.isEmpty()) return DoubleArray(0) to emptyList()
        val ranks = computePagerank()
        val degs = fallbackLabels.map { nodeDegree(it).toDouble() }
        val vecs =
            fallbackLabels.mapIndexed { idx, id ->
                doubleArrayOf(ranks[id] ?: 0.0, degs[idx])
            }
        val flat = vecs.flatMap { it.asIterable() }.toDoubleArray()
        return flat to fallbackLabels
    }

    private suspend fun fetchGraph(): Pair<List<String>, List<Pair<String, String>>> {
        val allNodes = nodes()
        val allEdges = edges()
        return allNodes to allEdges
    }

    private suspend fun computePagerank(
        damping: Double = 0.85,
        maxIter: Int = 100,
        tol: Double = 1e-6,
    ): Map<String, Double> {
        val gdsRanks = computePagerankGds()
        if (gdsRanks.isNotEmpty()) return gdsRanks
        return computePagerankLocal(damping, maxIter, tol)
    }

    private suspend fun computePagerankGds(): Map<String, Double> {
        val ranks = mutableMapOf<String, Double>()
        val graphName = "pathrag_${namespace}_gds_pagerank"
        withContext(Dispatchers.IO) {
            runCatching {
                driver.session().use { session ->
                    runCatching { session.run("CALL gds.graph.drop(\$g, false)", Values.parameters("g", graphName)).consume() }
                    session
                        .run(
                            "CALL gds.graph.project(\$g, \$labels, {`$relType`:{orientation:'UNDIRECTED'}})",
                            Values.parameters("g", graphName, "labels", listOf(nodeLabel)),
                        ).consume()
                    val result =
                        session.run(
                            "CALL gds.pageRank.stream(\$g) YIELD nodeId, score " +
                                "RETURN gds.util.asNode(nodeId).id AS id, score",
                            Values.parameters("g", graphName),
                        )
                    result.list { rec: Record ->
                        ranks[rec.get("id").asString()] = rec.get("score").asDouble()
                        Unit
                    }
                    runCatching { session.run("CALL gds.graph.drop(\$g, false)", Values.parameters("g", graphName)).consume() }
                }
            }.onFailure { ex -> logger.warn(ex) { "Neo4j GDS PageRank failed; falling back to in-memory computation." } }
        }
        return ranks
    }

    private suspend fun computePagerankLocal(
        damping: Double = 0.85,
        maxIter: Int = 100,
        tol: Double = 1e-6,
    ): Map<String, Double> {
        val (nodeList, edgeList) = fetchGraph()
        if (nodeList.isEmpty()) return emptyMap()
        val n = nodeList.size
        val adjacency =
            nodeList.associateWith { mutableListOf<String>() }.toMutableMap().also { adj ->
                edgeList.forEach { (u, v) ->
                    adj[u]?.add(v)
                    adj[v]?.add(u)
                }
            }
        val rank = mutableMapOf<String, Double>()
        nodeList.forEach { rank[it] = 1.0 / n }

        repeat(maxIter) {
            var diff = 0.0
            val newRank = mutableMapOf<String, Double>()
            for (node in nodeList) {
                val neighbors = adjacency[node].orEmpty()
                val outDeg = neighbors.size
                val share = if (outDeg == 0) 0.0 else rank[node]!! / outDeg
                neighbors.forEach { dest -> newRank[dest] = (newRank[dest] ?: 0.0) + share }
            }
            for (node in nodeList) {
                val updated = (1 - damping) / n + damping * (newRank[node] ?: 0.0)
                diff += abs(updated - (rank[node] ?: 0.0))
                rank[node] = updated
            }
            if (diff < tol) return rank
        }
        return rank
    }
}
