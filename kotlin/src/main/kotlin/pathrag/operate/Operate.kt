package pathrag.operate

import io.github.oshai.kotlinlogging.KotlinLogging
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import kotlinx.serialization.decodeFromString
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import pathrag.base.BaseGraphStorage
import pathrag.base.BaseKVStorage
import pathrag.base.BaseVectorStorage
import pathrag.base.QueryParam
import pathrag.prompt.Prompts
import pathrag.utils.ResponseCache
import pathrag.utils.Tokenizer
import pathrag.utils.computeArgsHash
import pathrag.utils.computeMdHashId
import kotlin.math.min

private val logger = KotlinLogging.logger("PathRAG-Operate")
private typealias LlmFunc =
    suspend (
        prompt: String,
        systemPrompt: String?,
        historyMessages: List<Map<String, String>>,
        keywordExtraction: Boolean,
        stream: Boolean,
        maxTokens: Int?,
        hashingKv: Any?,
    ) -> String

@Serializable
private data class LlmEntity(
    @SerialName("entity_name") val entityName: String = "",
    @SerialName("entity_type") val entityType: String = "UNKNOWN",
    val description: String = "",
    @SerialName("source_id") val sourceId: String? = null,
)

@Serializable
private data class LlmRelationship(
    @SerialName("src_id") val srcId: String = "",
    @SerialName("tgt_id") val tgtId: String = "",
    val description: String = "",
    val keywords: String = "",
    val weight: Double = 1.0,
    @SerialName("source_id") val sourceId: String? = null,
)

@Serializable
private data class ExtractionPayload(
    val entities: List<LlmEntity> = emptyList(),
    val relationships: List<LlmRelationship> = emptyList(),
)

@Serializable
private data class KeywordPayload(
    @SerialName("high_level_keywords") val highLevel: List<String> = emptyList(),
    @SerialName("low_level_keywords") val lowLevel: List<String> = emptyList(),
)

fun chunkingByTokenSize(
    content: String,
    overlapTokenSize: Int = 128,
    maxTokenSize: Int = 1024,
    tiktokenModel: String = "gpt-4o-mini",
): List<Map<String, Any>> {
    require(maxTokenSize > overlapTokenSize) {
        "maxTokenSize ($maxTokenSize) must be greater than overlapTokenSize ($overlapTokenSize)"
    }
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
 * Basic entity/relationship extraction that mirrors the Python flow shape:
 * - Uses LLM to parse entities/relationships from text chunks
 * - Merges entities by name, ensures graph + VDB consistency
 */
suspend fun extractEntities(
    chunks: Map<String, Map<String, Any>>,
    knowledgeGraphInst: BaseGraphStorage,
    entityVdb: BaseVectorStorage,
    relationshipsVdb: BaseVectorStorage,
    globalConfig: Map<String, Any?>,
): BaseGraphStorage {
    logger.info { "Extracting entities for ${chunks.size} chunks via LLM." }

    @Suppress("UNCHECKED_CAST")
    val llm = globalConfig["llm_model_func"] as? LlmFunc ?: return knowledgeGraphInst

    val allEntities = mutableListOf<Map<String, String>>()
    val allRelationships = mutableListOf<Map<String, Any>>()

    for ((chunkId, chunk) in chunks) {
        val content = chunk["content"]?.toString().orEmpty()
        if (content.isBlank()) continue
        val prompt = Prompts.ENTITY_REL_JSON.replace("{text}", content)
        val maxTokensForExtraction = (globalConfig["max_tokens_for_extraction"] as? Int) ?: 2048
        val response = llm(prompt, null, emptyList(), false, false, maxTokensForExtraction, null)
        val payload =
            runCatching { Json.decodeFromString<ExtractionPayload>(extractJsonPayload(response)) }
                .onFailure { logger.warn { "Failed to parse LLM extraction for chunk $chunkId: ${it.message}" } }
                .getOrElse { ExtractionPayload() }
        val entities = payload.entities.filter { it.entityName.isNotBlank() }
        val relationships = payload.relationships.filter { it.srcId.isNotBlank() && it.tgtId.isNotBlank() }

        entities.forEach { ent ->
            val name = normalizeId(ent.entityName)
            val entityType = ent.entityType.ifBlank { "UNKNOWN" }
            val description = ent.description
            val nodeData =
                mapOf(
                    "entity_type" to entityType,
                    "description" to description,
                    "source_id" to chunkId,
                    "entity_name" to name,
                )
            allEntities.add(nodeData)
            runCatching { knowledgeGraphInst.upsertNode(name, nodeData) }
                .onFailure { logger.error(it) { "Failed to upsert node $name" } }
        }

        relationships.forEach { rel ->
            val src = normalizeId(rel.srcId)
            val tgt = normalizeId(rel.tgtId)
            val description = rel.description
            val keywords = rel.keywords
            val edgeData =
                mapOf(
                    "weight" to rel.weight,
                    "description" to description,
                    "keywords" to keywords,
                    "source_id" to chunkId,
                )
            allRelationships.add(mapOf("src_id" to src, "tgt_id" to tgt) + edgeData)
            runCatching { knowledgeGraphInst.upsertEdge(src, tgt, edgeData) }
                .onFailure { logger.error(it) { "Failed to upsert edge $src -> $tgt" } }
        }
    }

    if (allEntities.isNotEmpty()) {
        val toStore =
            allEntities
                .mapNotNull { ent ->
                    val entityName = ent["entity_name"] ?: return@mapNotNull null
                    val id = computeMdHashId(entityName, prefix = "ent-")
                    id to
                        mapOf(
                            "content" to (ent["description"] ?: ""),
                            "entity_name" to entityName,
                        )
                }.toMap()
        runCatching { entityVdb.upsert(toStore) }
            .onFailure { logger.error(it) { "Failed to upsert ${toStore.size} entities into VDB" } }
    }

    if (allRelationships.isNotEmpty()) {
        val toStore =
            allRelationships
                .mapNotNull { edge ->
                    val src = edge["src_id"]?.toString() ?: return@mapNotNull null
                    val tgt = edge["tgt_id"]?.toString() ?: return@mapNotNull null
                    val description = edge["description"]?.toString() ?: ""
                    val keywords = edge["keywords"]?.toString() ?: ""
                    val id = computeMdHashId(src + tgt, prefix = "rel-")
                    id to
                        mapOf(
                            "src_id" to src,
                            "tgt_id" to tgt,
                            "content" to (description + keywords),
                            "keywords" to keywords,
                            "description" to description,
                        )
                }.toMap()
        runCatching { relationshipsVdb.upsert(toStore) }
            .onFailure { logger.error(it) { "Failed to upsert ${toStore.size} relationships into VDB" } }
    }

    return knowledgeGraphInst
}

private fun extractJsonPayload(response: String): String {
    val trimmed = response.trim()
    val fencedRegex = Regex("```(?:json)?\\s*([\\s\\S]*?)\\s*```", setOf(RegexOption.IGNORE_CASE, RegexOption.DOT_MATCHES_ALL))
    val fencedMatch = fencedRegex.find(trimmed)
    if (fencedMatch != null) {
        return fencedMatch.groupValues[1].trim()
    }
    val firstBrace = trimmed.indexOf('{')
    val lastBrace = trimmed.lastIndexOf('}')
    if (firstBrace != -1 && lastBrace > firstBrace) {
        return trimmed.substring(firstBrace, lastBrace + 1).trim()
    }
    return trimmed
}

private fun normalizeId(id: String): String = "\"${id.trim('"').uppercase()}\""

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
        val cached = hashingKv?.handleCache(argsHash, query, queryParam.mode)
        if (cached != null) return@withContext cached

        val (llKeywords, hlKeywords) = extractKeywords(llmModel, query, globalConfig)

        val systemContext = "PathRAG (Kotlin) | nodes=${knowledgeGraphInst.nodes().size} | mode=${queryParam.mode}"
        val response =
            when (queryParam.mode.lowercase()) {
                "local" -> {
                    runLocalMode(
                        llKeywords,
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
                        hlKeywords,
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
                        llKeywords,
                        hlKeywords,
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
        hashingKv?.upsert(queryParam.mode, argsHash, response, query)
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
    val (entitiesCsv, relationsCsv, textCsv) =
        getNodeData(
            query,
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

private fun String.format(values: Map<String, String>): String {
    val placeholder = Regex("\\{([A-Za-z0-9_]+)}")
    return placeholder.replace(this) { match ->
        val key = match.groupValues.getOrNull(1)
        values[key] ?: match.value
    }
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
    val relations = buildPathRelations(nodeDatas, knowledgeGraphInst, queryParam)

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

private suspend fun buildPathRelations(
    nodeDatas: List<Map<String, Any>>,
    knowledgeGraphInst: BaseGraphStorage,
    queryParam: QueryParam,
): List<String> {
    val targetNodes = nodeDatas.mapNotNull { it["entity_name"]?.toString() }.distinct()
    if (targetNodes.size < 2) return emptyList()

    val paths = findPathsBetweenTargets(knowledgeGraphInst, targetNodes, maxDepth = 3)
    if (paths.isEmpty()) return emptyList()

    val nodesCache = mutableMapOf<String, Map<String, Any?>>()
    for (name in targetNodes) {
        nodesCache[name] = knowledgeGraphInst.getNode(name) ?: emptyMap()
    }

    suspend fun describePath(path: List<String>): String {
        val pieces = mutableListOf<String>()
        for ((idx, nodeId) in path.withIndex()) {
            val node = nodesCache[nodeId] ?: emptyMap()
            val nodeType = node["entity_type"] ?: "UNKNOWN"
            val nodeDesc = node["description"] ?: ""
            pieces.add("Entity $nodeId ($nodeType): $nodeDesc")
            if (idx < path.lastIndex) {
                val next = path[idx + 1]
                val edgeData =
                    knowledgeGraphInst.getEdge(nodeId, next) ?: knowledgeGraphInst.getEdge(next, nodeId)
                val edgeKeywords = edgeData?.get("keywords") ?: ""
                val edgeDesc = edgeData?.get("description") ?: ""
                pieces.add("Edge [$nodeId -> $next]: keywords=($edgeKeywords), desc=($edgeDesc)")
            }
        }
        return pieces.joinToString(" | ")
    }

    val described = mutableListOf<String>()
    for (path in paths.sortedBy { it.size }) {
        described.add(describePath(path))
    }

    return truncateByToken(described.map { mapOf("content" to it) }, queryParam.maxTokenForLocalContext)
        .map { it["content"].toString() }
}

private suspend fun findPathsBetweenTargets(
    knowledgeGraphInst: BaseGraphStorage,
    targets: List<String>,
    maxDepth: Int = 3,
): List<List<String>> {
    val adjacency = mutableMapOf<String, MutableSet<String>>()
    knowledgeGraphInst.edges().forEach { (u, v) ->
        adjacency.computeIfAbsent(u) { mutableSetOf() }.add(v)
        adjacency.computeIfAbsent(v) { mutableSetOf() }.add(u)
    }
    if (adjacency.isEmpty()) return emptyList()

    fun neighbors(node: String): Set<String> = adjacency[node] ?: emptySet()

    val results = mutableSetOf<List<String>>()

    fun dfs(
        current: String,
        target: String,
        path: List<String>,
    ) {
        if (path.size > maxDepth + 1) return
        if (current == target) {
            results.add(path)
            return
        }
        for (n in neighbors(current)) {
            if (n !in path) {
                dfs(n, target, path + n)
            }
        }
    }

    for (i in targets.indices) {
        for (j in i + 1 until targets.size) {
            dfs(targets[i], targets[j], listOf(targets[i]))
        }
    }

    return results.toList()
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
    llKeywords: String,
    hlKeywords: String,
    userQuery: String,
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
        getEdgeData(hlKeywords, knowledgeGraphInst, relationshipsVdb, textChunksDb, queryParam)
    val (llEntities, llRelations, llText) =
        getNodeData(llKeywords, knowledgeGraphInst, entitiesVdb, textChunksDb, queryParam)

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
        userQuery,
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
    model: String = "gpt-4o-mini",
): List<Map<String, Any>> {
    var count = 0
    val result = mutableListOf<Map<String, Any>>()
    for (item in list) {
        val content = item["content"]?.toString() ?: ""
        val tokens = Tokenizer.encode(content, model).size
        count += tokens
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
    return allRows.joinToString("\n") { row -> row.joinToString(",") { escapeCsvField(it) } }
}

private fun emptyCsv(headers: List<String>): String = headers.joinToString(",")

private fun escapeCsvField(value: String): String {
    val needsQuotes = value.contains(',') || value.contains('\n') || value.contains('"')
    if (!needsQuotes) return value
    val escaped = value.replace("\"", "\"\"")
    return "\"$escaped\""
}

private suspend fun extractKeywords(
    llmModel: suspend (
        prompt: String,
        systemPrompt: String?,
        historyMessages: List<Map<String, String>>,
        keywordExtraction: Boolean,
        stream: Boolean,
        maxTokens: Int?,
        hashingKv: Any?,
    ) -> String,
    query: String,
    globalConfig: Map<String, Any?>,
): Pair<String, String> {
    val examples = (globalConfig["keywords_examples"] as? String).orEmpty()
    val language = (globalConfig["language"] as? String) ?: Prompts.DEFAULT_LANGUAGE
    val prompt =
        Prompts.KEYWORDS_EXTRACTION.format(
            mapOf(
                "query" to query,
                "examples" to examples,
                "language" to language,
            ),
        )
    val raw = llmModel(prompt, null, emptyList(), true, false, 512, null)
    val parsed =
        runCatching { Json.decodeFromString<KeywordPayload>(extractJsonPayload(raw)) }
            .getOrElse { KeywordPayload() }
    val hl = parsed.highLevel.joinToString(", ")
    val ll = parsed.lowLevel.joinToString(", ")
    return ll to hl
}

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
