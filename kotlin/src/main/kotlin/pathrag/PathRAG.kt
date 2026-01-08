package pathrag

import io.github.oshai.kotlinlogging.KotlinLogging
import kotlinx.coroutines.runBlocking
import pathrag.base.BaseGraphStorage
import pathrag.base.BaseKVStorage
import pathrag.base.BaseVectorStorage
import pathrag.base.QueryParam
import pathrag.base.runBlockingMaybe
import pathrag.llm.defaultEmbeddingFunc
import pathrag.llm.openAiComplete
import pathrag.operate.chunkingByTokenSize
import pathrag.operate.extractEntities
import pathrag.operate.kgQuery
import pathrag.storage.JsonKVStorage
import pathrag.storage.NanoVectorDBStorage
import pathrag.storage.NetworkXStorage
import pathrag.utils.ResponseCache
import pathrag.utils.computeMdHashId
import java.time.LocalDateTime
import java.time.format.DateTimeFormatter

class PathRAG(
    private val workingDir: String = "./PathRAG_cache_" + LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyy-MM-dd-HH-mm-ss")),
    private val kvStorage: String = "JsonKVStorage",
    private val vectorStorage: String = "NanoVectorDBStorage",
    private val graphStorage: String = "NetworkXStorage",
    private val chunkTokenSize: Int = 1200,
    private val chunkOverlapTokenSize: Int = 100,
) {
    private val logger = KotlinLogging.logger("PathRAG")
    private val llmModelName: String = System.getenv("OPENAI_MODEL") ?: "gpt-4o-mini"

    private val embeddingFunc = defaultEmbeddingFunc()
    private val llmModelFunc: suspend (String, String?, List<Map<String, String>>, Boolean, Boolean, Int?, Any?) -> String =
        { prompt, system, history, keyword, stream, maxTokens, hashingKv ->
            openAiComplete(
                llmModelName,
                prompt,
                systemPrompt = system,
                historyMessages = history,
                keywordExtraction = keyword,
                stream = stream,
                maxTokens = maxTokens,
                hashingKv = hashingKv,
            )
        }

    private val llmResponseCache = ResponseCache()

    private val storageFactory: Map<String, () -> BaseKVStorage<Map<String, Any>>> =
        mapOf(
            "JsonKVStorage" to { JsonKVStorage("kv", globalConfig(), embeddingFunc) },
        )
    private val vectorFactory: Map<String, () -> BaseVectorStorage> =
        mapOf(
            "NanoVectorDBStorage" to { NanoVectorDBStorage("vdb", globalConfig(), embeddingFunc) },
        )
    private val graphFactory: Map<String, () -> BaseGraphStorage> =
        mapOf(
            "NetworkXStorage" to { NetworkXStorage("graph", globalConfig(), embeddingFunc) },
        )

    private val fullDocs: BaseKVStorage<Map<String, Any>> =
        storageFactory[kvStorage]?.invoke()
            ?: error("Unknown kv storage: $kvStorage")
    private val textChunks: BaseKVStorage<Map<String, Any>> =
        storageFactory[kvStorage]?.invoke()
            ?: error("Unknown kv storage: $kvStorage")
    private var chunkEntityRelationGraph: BaseGraphStorage =
        graphFactory[graphStorage]?.invoke()
            ?: error("Unknown graph storage: $graphStorage")
    private val entitiesVdb: BaseVectorStorage =
        vectorFactory[vectorStorage]?.invoke()
            ?: error("Unknown vector storage: $vectorStorage")
    private val relationshipsVdb: BaseVectorStorage =
        vectorFactory[vectorStorage]?.invoke()
            ?: error("Unknown vector storage: $vectorStorage")
    private val chunksVdb: BaseVectorStorage =
        vectorFactory[vectorStorage]?.invoke()
            ?: error("Unknown vector storage: $vectorStorage")

    private fun globalConfig(): Map<String, Any?> =
        mapOf(
            "embedding_func" to embeddingFunc,
            "llm_model_func" to llmModelFunc,
            "chunk_token_size" to chunkTokenSize,
            "chunk_overlap_token_size" to chunkOverlapTokenSize,
        )

    fun insert(stringOrStrings: Any) = runBlockingMaybe { ainsert(stringOrStrings) }

    suspend fun ainsert(stringOrStrings: Any) {
        val inputs =
            when (stringOrStrings) {
                is String -> listOf(stringOrStrings)
                is Collection<*> -> stringOrStrings.filterIsInstance<String>()
                else -> emptyList()
            }
        if (inputs.isEmpty()) {
            logger.warn { "No documents provided for insertion." }
            return
        }
        val newDocs =
            inputs.associate { text ->
                val id = computeMdHashId(text.trim(), prefix = "doc-")
                id to mapOf("content" to text.trim())
            }
        val chunkMap = mutableMapOf<String, Map<String, Any>>()
        newDocs.forEach { (docKey, doc) ->
            val chunks =
                chunkingByTokenSize(
                    doc["content"]?.toString().orEmpty(),
                    overlapTokenSize = chunkOverlapTokenSize,
                    maxTokenSize = chunkTokenSize,
                )
            chunks.forEach { chunk ->
                val id = computeMdHashId(chunk["content"].toString(), prefix = "chunk-")
                chunkMap[id] = chunk + mapOf("full_doc_id" to docKey)
            }
        }
        chunksVdb.upsert(chunkMap)
        chunkEntityRelationGraph =
            extractEntities(
                chunkMap,
                chunkEntityRelationGraph,
                entitiesVdb,
                relationshipsVdb,
                globalConfig(),
            )
        fullDocs.upsert(newDocs)
        textChunks.upsert(chunkMap)
    }

    fun insertCustomKg(customKg: Map<String, Any?>) = runBlockingMaybe { ainsertCustomKg(customKg) }

    suspend fun ainsertCustomKg(customKg: Map<String, Any?>) {
        val chunks = (customKg["chunks"] as? List<Map<String, Any?>>).orEmpty()
        val entities = (customKg["entities"] as? List<Map<String, Any?>>).orEmpty()
        val relationships = (customKg["relationships"] as? List<Map<String, Any?>>).orEmpty()

        val chunkData =
            chunks.associate { chunk ->
                val id = computeMdHashId(chunk["content"].toString(), prefix = "chunk-")
                id to mapOf("content" to chunk["content"].toString(), "source_id" to chunk["source_id"].toString())
            }
        if (chunkData.isNotEmpty()) {
            chunksVdb.upsert(chunkData)
            textChunks.upsert(chunkData)
        }

        entities.forEach { entity ->
            val name = "\"${entity["entity_name"].toString().uppercase()}\""
            val nodeData =
                mapOf(
                    "entity_type" to (entity["entity_type"] ?: "UNKNOWN"),
                    "description" to (entity["description"] ?: "No description provided"),
                    "source_id" to (entity["source_id"] ?: "UNKNOWN"),
                )
            chunkEntityRelationGraph.upsertNode(name, nodeData)
        }

        relationships.forEach { rel ->
            val src = "\"${rel["src_id"].toString().uppercase()}\""
            val tgt = "\"${rel["tgt_id"].toString().uppercase()}\""
            val data =
                mapOf(
                    "weight" to (rel["weight"] ?: 1.0),
                    "description" to (rel["description"] ?: ""),
                    "keywords" to (rel["keywords"] ?: ""),
                    "source_id" to (rel["source_id"] ?: "UNKNOWN"),
                )
            chunkEntityRelationGraph.upsertEdge(src, tgt, data)
        }
    }

    fun query(
        query: String,
        param: QueryParam = QueryParam(),
    ): String =
        runBlocking {
            aquery(query, param)
        }

    suspend fun aquery(
        query: String,
        param: QueryParam = QueryParam(),
    ): String {
        val response =
            kgQuery(
                query,
                chunkEntityRelationGraph,
                entitiesVdb,
                relationshipsVdb,
                textChunks,
                param,
                globalConfig(),
                llmModelFunc,
                llmResponseCache,
            )
        return response
    }

    fun deleteByEntity(entityName: String) = runBlockingMaybe { adeleteByEntity(entityName) }

    suspend fun adeleteByEntity(entityName: String) {
        val key = "\"${entityName.uppercase()}\""
        entitiesVdb.deleteEntity(key)
        relationshipsVdb.deleteRelation(key)
        chunkEntityRelationGraph.deleteNode(key)
        logger.info { "Entity '$key' and relationships deleted." }
    }
}
