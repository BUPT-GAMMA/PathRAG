package pathrag.operate

import io.github.oshai.kotlinlogging.KotlinLogging
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import pathrag.base.BaseGraphStorage
import pathrag.base.BaseKVStorage
import pathrag.base.BaseVectorStorage
import pathrag.base.QueryParam
import pathrag.prompt.Prompts
import pathrag.utils.ResponseCache
import pathrag.utils.Tokenizer
import pathrag.utils.computeArgsHash
import pathrag.utils.computeMdHashId
import kotlin.math.max
import kotlin.math.min

private val logger = KotlinLogging.logger("PathRAG-Operate")

fun chunkingByTokenSize(
    content: String,
    overlapTokenSize: Int = 128,
    maxTokenSize: Int = 1024,
    tiktokenModel: String = "gpt-4o-mini",
): List<Map<String, Any>> {
    val tokens = Tokenizer.encode(content, tiktokenModel)
    val chunks = mutableListOf<Map<String, Any>>()
    var index = 0
    var start = 0
    while (start < tokens.size) {
        val end = min(start + maxTokenSize, tokens.size)
        val slice = tokens.subList(start, end)
        val decoded = Tokenizer.decode(slice, tiktokenModel).trim()
        chunks.add(
            mapOf(
                "tokens" to slice.size,
                "content" to decoded,
                "chunk_order_index" to index,
            ),
        )
        index += 1
        start += maxTokenSize - overlapTokenSize
    }
    return chunks
}

/**
 * Lightweight placeholder that simply returns the existing graph storage.
 * Replace with a full entity/relationship extraction when an LLM backend is available.
 */
suspend fun extractEntities(
    chunks: Map<String, Map<String, Any>>,
    knowledgeGraphInst: BaseGraphStorage,
    entityVdb: BaseVectorStorage,
    relationshipsVdb: BaseVectorStorage,
    globalConfig: Map<String, Any?>,
): BaseGraphStorage {
    logger.info { "Stub entity extraction for ${chunks.size} chunks (no-op)." }
    return knowledgeGraphInst
}

suspend fun kgQuery(
    query: String,
    knowledgeGraphInst: BaseGraphStorage,
    entitiesVdb: BaseVectorStorage,
    relationshipsVdb: BaseVectorStorage,
    textChunksDb: BaseKVStorage<Map<String, Any>>,
    queryParam: QueryParam,
    globalConfig: Map<String, Any?>,
    llmModel: suspend (
        prompt: String,
        systemPrompt: String?,
        historyMessages: List<Map<String, String>>,
        keywordExtraction: Boolean,
        stream: Boolean,
        maxTokens: Int?,
        hashingKv: Any?,
    ) -> String,
    hashingKv: ResponseCache? = null,
): String =
    withContext(Dispatchers.Default) {
        val argsHash = computeArgsHash(queryParam.mode, query)
        val cached = hashingKv?.getById(queryParam.mode)?.get(argsHash)
        if (cached != null) return@withContext cached

        val systemContext =
            "PathRAG (Kotlin) | nodes=${knowledgeGraphInst.nodes().size} | mode=${queryParam.mode}"
        val response =
            when (queryParam.mode.lowercase()) {
                "local" -> {
                    runLocalMode(
                        query,
                        queryParam,
                        entitiesVdb,
                        knowledgeGraphInst,
                        textChunksDb,
                        llmModel,
                        systemContext,
                    )
                }

                "global" -> {
                    runGlobalMode(
                        query,
                        queryParam,
                        knowledgeGraphInst,
                        relationshipsVdb,
                        textChunksDb,
                        llmModel,
                        systemContext,
                    )
                }

                "hybrid" -> {
                    runHybridMode(
                        query,
                        queryParam,
                        knowledgeGraphInst,
                        entitiesVdb,
                        relationshipsVdb,
                        textChunksDb,
                        llmModel,
                        systemContext,
                        hashingKv,
                    )
                }

                else -> {
                    "Unknown mode ${queryParam.mode}"
                }
            }
        hashingKv?.upsert(queryParam.mode, argsHash, response)
        response
    }

private suspend fun runLocalMode(
    query: String,
    queryParam: QueryParam,
    entitiesVdb: BaseVectorStorage,
    knowledgeGraphInst: BaseGraphStorage,
    textChunksDb: BaseKVStorage<Map<String, Any>>,
    llmModel: suspend (
        prompt: String,
        systemPrompt: String?,
        historyMessages: List<Map<String, String>>,
        keywordExtraction: Boolean,
        stream: Boolean,
        maxTokens: Int?,
        hashingKv: Any?,
    ) -> String,
    systemContext: String,
): String {
    val keywords = query
    val (entitiesCsv, relationsCsv, textCsv) =
        getNodeData(
            keywords,
            knowledgeGraphInst,
            entitiesVdb,
            textChunksDb,
            queryParam,
        )

    val context =
        """
        -----local-information-----
        -----low-level entity information-----
        ```csv
        $entitiesCsv
        ```
        -----low-level relationship information-----
        ```csv
        $relationsCsv
        ```
        -----Sources-----
        ```csv
        $textCsv
        ```
        """.trimIndent()

    if (queryParam.onlyNeedContext) return context

    val sysPrompt =
        Prompts.RAG_RESPONSE.format(
            mapOf(
                "context_data" to context,
                "response_type" to queryParam.responseType,
            ),
        )

    return llmModel(
        query,
        "$systemContext\n$sysPrompt",
        emptyList(),
        false,
        queryParam.stream,
        queryParam.maxTokenForTextUnit,
        null,
    )
}

private fun String.format(values: Map<String, String>): String =
    values.entries.fold(this) { acc, (k, v) ->
        acc.replace("{$k}", v)
    }

private suspend fun getNodeData(
    keywords: String,
    knowledgeGraphInst: BaseGraphStorage,
    entitiesVdb: BaseVectorStorage,
    textChunksDb: BaseKVStorage<Map<String, Any>>,
    queryParam: QueryParam,
): Triple<String, String, String> {
    val results = entitiesVdb.query(keywords, topK = queryParam.topK)
    if (results.isEmpty()) {
        return Triple(
            emptyCsv(listOf("id", "entity", "type", "description", "rank")),
            emptyCsv(listOf("id", "context")),
            emptyCsv(listOf("id", "content")),
        )
    }

    val nodeDatas =
        results.mapNotNull { res ->
            val name = res["entity_name"]?.toString() ?: res["content"]?.toString()
            if (name != null) {
                val node = knowledgeGraphInst.getNode(name)
                val degree = knowledgeGraphInst.nodeDegree(name)
                val desc = node?.get("description") ?: res["content"]
                mapOf(
                    "entity_name" to name,
                    "entity_type" to (node?.get("entity_type") ?: "UNKNOWN"),
                    "description" to (desc ?: ""),
                    "rank" to degree,
                    "source_id" to (node?.get("source_id") ?: res["full_doc_id"] ?: ""),
                )
            } else {
                null
            }
        }

    val textUnits = findMostRelatedTextUnitFromEntities(nodeDatas, queryParam, textChunksDb)
    val relations = emptyList<String>() // local mode: no path exploration yet

    val entitiesCsv =
        toCsv(
            listOf("id", "entity", "type", "description", "rank"),
            nodeDatas.mapIndexed { idx, n ->
                listOf(
                    idx.toString(),
                    n["entity_name"].toString(),
                    n["entity_type"].toString(),
                    n["description"].toString(),
                    n["rank"].toString(),
                )
            },
        )
    val relationsCsv =
        toCsv(
            listOf("id", "context"),
            relations.mapIndexed { idx, r -> listOf(idx.toString(), r) },
        )
    val textCsv =
        toCsv(
            listOf("id", "content"),
            textUnits.mapIndexed { idx, t ->
                listOf(idx.toString(), t["content"].toString())
            },
        )

    return Triple(entitiesCsv, relationsCsv, textCsv)
}

private suspend fun findMostRelatedTextUnitFromEntities(
    nodeDatas: List<Map<String, Any>>,
    queryParam: QueryParam,
    textChunksDb: BaseKVStorage<Map<String, Any>>,
): List<Map<String, Any>> {
    val ids =
        nodeDatas.flatMap { node ->
            node["source_id"]
                ?.toString()
                ?.split(",")
                ?.map { it.trim() }
                ?.filter { it.isNotEmpty() }
                ?: emptyList()
        }
    val uniqueIds = ids.distinct()
    val chunks = textChunksDb.getByIds(uniqueIds)
    val valid = chunks.mapNotNull { it }.take(queryParam.topK)
    return truncateByToken(valid, queryParam.maxTokenForTextUnit)
}

private suspend fun runGlobalMode(
    query: String,
    queryParam: QueryParam,
    knowledgeGraphInst: BaseGraphStorage,
    relationshipsVdb: BaseVectorStorage,
    textChunksDb: BaseKVStorage<Map<String, Any>>,
    llmModel: suspend (
        prompt: String,
        systemPrompt: String?,
        historyMessages: List<Map<String, String>>,
        keywordExtraction: Boolean,
        stream: Boolean,
        maxTokens: Int?,
        hashingKv: Any?,
    ) -> String,
    systemContext: String,
): String {
    val (entitiesCsv, relationsCsv, textCsv) =
        getEdgeData(query, knowledgeGraphInst, relationshipsVdb, textChunksDb, queryParam)

    val context =
        """
        -----global-information-----
        -----high-level entity information-----
        ```csv
        $entitiesCsv
        ```
        -----high-level relationship information-----
        ```csv
        $relationsCsv
        ```
        -----Sources-----
        ```csv
        $textCsv
        ```
        """.trimIndent()

    if (queryParam.onlyNeedContext) return context

    val sysPrompt =
        Prompts.RAG_RESPONSE.format(
            mapOf(
                "context_data" to context,
                "response_type" to queryParam.responseType,
            ),
        )

    return llmModel(
        query,
        "$systemContext\n$sysPrompt",
        emptyList(),
        false,
        queryParam.stream,
        queryParam.maxTokenForGlobalContext,
        null,
    )
}

private suspend fun runHybridMode(
    query: String,
    queryParam: QueryParam,
    knowledgeGraphInst: BaseGraphStorage,
    entitiesVdb: BaseVectorStorage,
    relationshipsVdb: BaseVectorStorage,
    textChunksDb: BaseKVStorage<Map<String, Any>>,
    llmModel: suspend (
        prompt: String,
        systemPrompt: String?,
        historyMessages: List<Map<String, String>>,
        keywordExtraction: Boolean,
        stream: Boolean,
        maxTokens: Int?,
        hashingKv: Any?,
    ) -> String,
    systemContext: String,
    hashingKv: ResponseCache?,
): String {
    val (hlEntities, hlRelations, hlText) =
        getEdgeData(query, knowledgeGraphInst, relationshipsVdb, textChunksDb, queryParam)
    val (llEntities, llRelations, llText) =
        getNodeData(query, knowledgeGraphInst, entitiesVdb, textChunksDb, queryParam)

    val mergedContext =
        """
        -----global-information-----
        -----high-level entity information-----
        ```csv
        $hlEntities
        ```
        -----high-level relationship information-----
        ```csv
        $hlRelations
        ```
        -----Sources-----
        ```csv
        $hlText
        ```
        -----local-information-----
        -----low-level entity information-----
        ```csv
        $llEntities
        ```
        -----low-level relationship information-----
        ```csv
        $llRelations
        ```
        -----Sources-----
        ```csv
        $llText
        ```
        """.trimIndent()

    if (queryParam.onlyNeedContext) return mergedContext

    val sysPrompt =
        Prompts.RAG_RESPONSE.format(
            mapOf(
                "context_data" to mergedContext,
                "response_type" to queryParam.responseType,
            ),
        )

    return llmModel(
        query,
        "$systemContext\n$sysPrompt",
        emptyList(),
        false,
        queryParam.stream,
        queryParam.maxTokenForTextUnit,
        hashingKv,
    )
}

private fun truncateByToken(
    list: List<Map<String, Any>>,
    maxToken: Int,
): List<Map<String, Any>> {
    var count = 0
    val result = mutableListOf<Map<String, Any>>()
    for (item in list) {
        val content = item["content"]?.toString() ?: ""
        count += content.length
        if (count > maxToken) break
        result.add(item)
    }
    return result
}

private fun toCsv(
    headers: List<String>,
    rows: List<List<String>>,
): String {
    val allRows = listOf(headers) + rows
    return allRows.joinToString("\n") { it.joinToString(",") }
}

private fun emptyCsv(headers: List<String>): String = headers.joinToString(",")

private suspend fun getEdgeData(
    keywords: String,
    knowledgeGraphInst: BaseGraphStorage,
    relationshipsVdb: BaseVectorStorage,
    textChunksDb: BaseKVStorage<Map<String, Any>>,
    queryParam: QueryParam,
): Triple<String, String, String> {
    val results = relationshipsVdb.query(keywords, topK = queryParam.topK)
    if (results.isEmpty()) {
        return Triple(
            emptyCsv(listOf("id", "entity", "type", "description", "rank")),
            emptyCsv(listOf("id", "source", "target", "description", "keywords", "weight")),
            emptyCsv(listOf("id", "content")),
        )
    }

    val edges =
        results.mapNotNull { res ->
            val src = res["src_id"]?.toString()
            val tgt = res["tgt_id"]?.toString()
            if (src != null && tgt != null) {
                val edge = knowledgeGraphInst.getEdge(src, tgt)
                if (edge != null) {
                    edge + mapOf("src_id" to src, "tgt_id" to tgt)
                } else {
                    null
                }
            } else {
                null
            }
        }

    val entities =
        edges
            .flatMap { e ->
                listOfNotNull(
                    e["src_id"]?.toString(),
                    e["tgt_id"]?.toString(),
                )
            }.distinct()
    val nodeDatas =
        entities.mapNotNull { name ->
            val node = knowledgeGraphInst.getNode(name)
            val degree = knowledgeGraphInst.nodeDegree(name)
            node?.let {
                mapOf(
                    "entity_name" to name,
                    "entity_type" to (it["entity_type"] ?: "UNKNOWN"),
                    "description" to (it["description"] ?: ""),
                    "rank" to degree,
                    "source_id" to (it["source_id"] ?: ""),
                )
            }
        }

    val textUnits = findRelatedTextUnitFromRelationships(edges, textChunksDb, queryParam)

    val entitiesCsv =
        toCsv(
            listOf("id", "entity", "type", "description", "rank"),
            nodeDatas.mapIndexed { idx, n ->
                listOf(
                    idx.toString(),
                    n["entity_name"].toString(),
                    n["entity_type"].toString(),
                    n["description"].toString(),
                    n["rank"].toString(),
                )
            },
        )

    val relationsCsv =
        toCsv(
            listOf("id", "source", "target", "description", "keywords", "weight"),
            edges.mapIndexed { idx, e ->
                listOf(
                    idx.toString(),
                    e["src_id"].toString(),
                    e["tgt_id"].toString(),
                    e["description"]?.toString() ?: "",
                    e["keywords"]?.toString() ?: "",
                    e["weight"]?.toString() ?: "1.0",
                )
            },
        )

    val textCsv =
        toCsv(
            listOf("id", "content"),
            textUnits.mapIndexed { idx, t ->
                listOf(idx.toString(), t["content"].toString())
            },
        )

    return Triple(entitiesCsv, relationsCsv, textCsv)
}

private suspend fun findRelatedTextUnitFromRelationships(
    edges: List<Map<String, Any?>>,
    textChunksDb: BaseKVStorage<Map<String, Any>>,
    queryParam: QueryParam,
): List<Map<String, Any>> {
    val ids =
        edges.flatMap { e ->
            e["source_id"]
                ?.toString()
                ?.split(",")
                ?.map { it.trim() }
                ?.filter { it.isNotEmpty() }
                ?: emptyList()
        }
    val unique = ids.distinct()
    val chunks = textChunksDb.getByIds(unique)
    val valid = chunks.mapNotNull { it }.take(queryParam.topK)
    return truncateByToken(valid, queryParam.maxTokenForTextUnit)
}
