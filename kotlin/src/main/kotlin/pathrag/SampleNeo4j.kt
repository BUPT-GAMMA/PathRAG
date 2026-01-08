package pathrag

import kotlinx.coroutines.runBlocking

/**
 * Sample showing how to run PathRAG with Neo4j-backed graph storage.
 *
 * Set Neo4j connection via env (NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD),
 * then run: ./gradlew execute -PmainClass=pathrag.SampleNeo4jKt
 */
fun main() =
    runBlocking {
        val rag =
            PathRAG(
                workingDir = "./sample_cache_neo4j",
                graphStorage = "Neo4jStorage",
                chunkTokenSize = 800,
                chunkOverlapTokenSize = 120,
                language = "English",
                // Optional: pin keywords to bypass LLM keyword extraction
                highLevelKeywords = listOf("themes", "Dickens"),
                lowLevelKeywords = listOf("poverty", "class struggle", "redemption"),
                embeddingCacheConfig =
                    mapOf(
                        "enabled" to true,
                        "similarity_threshold" to 0.9,
                        "use_llm_check" to false,
                    ),
                addonParams =
                    mapOf(
                        "entity_types" to listOf("organization", "person", "geo", "event", "category"),
                        "example_number" to 3,
                    ),
            )

        rag.insert(
            listOf(
                """
                Charles Dickens was an English writer and social critic.
                He created some of the world's best-known fictional characters
                and is regarded as one of the greatest novelists of the Victorian era.
                """.trimIndent(),
                """
                Oliver Twist is a novel by Dickens that critiques workhouses and child poverty.
                It follows an orphan navigating criminal underworlds and harsh social systems.
                """.trimIndent(),
                """
                A Christmas Carol tells the redemption story of Ebenezer Scrooge, shifting from greed to generosity.
                It explores themes of morality, compassion, and social responsibility.
                """.trimIndent(),
            ),
        )

        val question = "What themes does Dickens explore?"

        val localAnswer = rag.query(question, param = pathrag.base.QueryParam(mode = "local"))
        println("Q (local): $question")
        println("A: $localAnswer\n")

        val globalAnswer = rag.query(question, param = pathrag.base.QueryParam(mode = "global"))
        println("Q (global): $question")
        println("A: $globalAnswer\n")

        val hybridAnswer = rag.query(question, param = pathrag.base.QueryParam(mode = "hybrid"))
        println("Q (hybrid): $question")
        println("A: $hybridAnswer\n")

        println("\nDone with Neo4j-backed graph storage.")
    }
