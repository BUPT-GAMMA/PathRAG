package pathrag

import kotlinx.coroutines.runBlocking

/**
 * Minimal sample showing how to use the Kotlin PathRAG port.
 *
 * It mirrors the flow of the provided snippet: set up the working data,
 * insert some text, and execute a query. Replace the sample text/query
 * with your own data as needed.
 */
fun main() =
    runBlocking {
        val rag =
            PathRAG(
                workingDir = "./sample_cache",
                chunkTokenSize = 800,
                chunkOverlapTokenSize = 120,
                language = "English",
                keywordExamples = "",
                similarityCheckPrompt = pathrag.prompt.Prompts.SIMILARITY_CHECK,
                embeddingCacheConfig =
                    mapOf(
                        "enabled" to true,
                        "similarity_threshold" to 0.9,
                        "use_llm_check" to false,
                    ),
                addonParams =
                    mapOf(
                        "entity_types" to listOf("organization", "person", "geo", "event", "category"),
                        "language" to "English",
                        "example_number" to 3,
                    ),
            )

        // Insert demo content (replace with real documents).
        rag.insert(
            listOf(
                """
                Charles Dickens was an English writer and social critic.
                He created some of the world's best-known fictional characters
                and is regarded as one of the greatest novelists of the Victorian era.
                """.trimIndent(),
            ),
        )

        val question = "What themes does Dickens explore?"

        // Local mode: entity-centric context only.
        val localAnswer = rag.query(question, param = pathrag.base.QueryParam(mode = "local"))
        println("Q (local): $question")
        println("A: $localAnswer\n")

        // Global mode: relationship-centric context.
        val globalAnswer = rag.query(question, param = pathrag.base.QueryParam(mode = "global"))
        println("Q (global): $question")
        println("A: $globalAnswer\n")

        // Hybrid mode: uses existing hybrid flow.
        val hybridAnswer = rag.query(question, param = pathrag.base.QueryParam(mode = "hybrid"))
        println("Q (hybrid): $question")
        println("A: $hybridAnswer\n")

        println("\nDone!")
    }
