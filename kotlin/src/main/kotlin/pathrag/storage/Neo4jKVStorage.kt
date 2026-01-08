package pathrag.storage

import io.github.oshai.kotlinlogging.KotlinLogging
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.neo4j.driver.AuthTokens
import org.neo4j.driver.Driver
import org.neo4j.driver.GraphDatabase
import org.neo4j.driver.TransactionContext
import org.neo4j.driver.Values
import pathrag.base.BaseKVStorage

class Neo4jKVStorage<T : Any>(
    override val namespace: String,
    override val globalConfig: Map<String, Any?>,
) : BaseKVStorage<T>(namespace, globalConfig),
    AutoCloseable {
    private val logger = KotlinLogging.logger("PathRAG-Neo4jKVStorage")

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

    private val nodeLabel = "${namespace.uppercase()}_KV"

    override fun close() {
        driver.close()
    }

    private suspend fun <R> read(block: (TransactionContext) -> R): R =
        withContext(Dispatchers.IO) { driver.session().use { session -> session.executeRead { tx -> block(tx) } } }

    private suspend fun <R> write(block: (TransactionContext) -> R): R =
        withContext(Dispatchers.IO) { driver.session().use { session -> session.executeWrite { tx -> block(tx) } } }

    override suspend fun allKeys(): List<String> =
        read { tx ->
            tx
                .run("MATCH (n:$nodeLabel) RETURN n.id AS id")
                .list { it.get("id").asString() }
        }

    override suspend fun getById(id: String): T? =
        read { tx ->
            tx
                .run("MATCH (n:$nodeLabel {id:\$id}) RETURN properties(n) AS props", Values.parameters("id", id))
                .list()
                .firstOrNull()
                ?.get("props")
                ?.asMap { v -> v.asObject() }
                ?.filterKeys { it != "id" }
                ?.let {
                    @Suppress("UNCHECKED_CAST")
                    (it as T)
                }
        }

    override suspend fun getByIds(
        ids: List<String>,
        fields: Set<String>?,
    ): List<T?> {
        if (ids.isEmpty()) return emptyList()
        val results =
            read { tx ->
                tx
                    .run(
                        "MATCH (n:$nodeLabel) WHERE n.id IN \$ids RETURN n.id AS id, properties(n) AS props",
                        Values.parameters("ids", ids),
                    ).list { rec ->
                        rec.get("id").asString() to rec.get("props").asMap { v -> v.asObject() }
                    }.toMap()
            }
        return ids.map { id ->
            results[id]
                ?.filterKeys { it != "id" && (fields == null || fields.contains(it)) }
                ?.let {
                    @Suppress("UNCHECKED_CAST")
                    (it as T)
                }
        }
    }

    override suspend fun filterKeys(data: List<String>): Set<String> {
        val existing = allKeys().toSet()
        return data.filterNot { existing.contains(it) }.toSet()
    }

    override suspend fun upsert(data: Map<String, T>) {
        if (data.isEmpty()) return
        write { tx ->
            data.forEach { (id, value) ->
                val props =
                    when (value) {
                        is Map<*, *> -> value.filterKeys { it != "id" }
                        else -> mapOf("value" to value)
                    }
                tx.run(
                    "MERGE (n:$nodeLabel {id:\$id}) SET n += \$props, n.id = \$id",
                    Values.parameters("id", id, "props", props),
                )
            }
        }
    }

    override suspend fun drop() {
        write { tx -> tx.run("MATCH (n:$nodeLabel) DETACH DELETE n") }
    }
}
