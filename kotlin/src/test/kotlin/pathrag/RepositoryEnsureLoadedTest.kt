package pathrag

import io.mockk.every
import io.mockk.mockkStatic
import io.mockk.unmockkAll
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json
import java.nio.charset.Charset
import java.nio.file.Files
import java.util.concurrent.atomic.AtomicInteger
import kotlin.io.path.deleteIfExists
import kotlin.test.AfterTest
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

class RepositoryEnsureLoadedTest {
    private val json =
        Json {
            prettyPrint = true
            ignoreUnknownKeys = true
        }
    private val cleanupPaths = mutableListOf<Pair<java.io.File, java.io.File>>()

    @AfterTest
    fun tearDown() {
        unmockkAll()
        cleanupPaths.forEach { (file, dir) ->
            runCatching { file.toPath().deleteIfExists() }
            runCatching { dir.toPath().deleteIfExists() }
        }
        cleanupPaths.clear()
    }

    @Test
    fun userRepository_ensureLoaded_runsOnceWithConcurrentCalls() {
        runBlocking {
            val dir = Files.createTempDirectory("userRepoTest").toFile()
            val file = dir.resolve("users.json")
            cleanupPaths += file to dir
            file.writeText(json.encodeToString(listOf(User(id = 1, username = "alice", email = "a@example.com", hashedPassword = "pwd"))))

            val repo = UserRepository(file.toPath())
            val readCount = AtomicInteger(0)
            mockkStatic("kotlin.io.FilesKt__FileReadWriteKt")
            every { file.readText() } answers {
                readCount.incrementAndGet()
                file.inputStream().bufferedReader().use { it.readText() }
            }
            every { file.readText(any()) } answers {
                readCount.incrementAndGet()
                val charset = secondArg<Charset>()
                file.inputStream().bufferedReader(charset).use { it.readText() }
            }

            coroutineScope {
                repeat(2) { launch { repo.list() } }
            }

            assertEquals(1, readCount.get(), "User file should be read only once when ensureLoaded is invoked concurrently")
            val users = repo.list()
            assertEquals(1, users.size)
            assertEquals("alice", users.first().username)
        }
    }

    @Test
    fun chatRepository_ensureLoaded_runsOnceWithConcurrentCalls() {
        runBlocking {
            val dir = Files.createTempDirectory("chatRepoTest").toFile()
            val file = dir.resolve("chats.json")
            cleanupPaths += file to dir
            val chat =
                ChatThread(
                    id = 1,
                    uuid = "thread-1",
                    userId = 1,
                    title = "Sample",
                    chats =
                        listOf(
                            ChatMessage(
                                id = 1,
                                threadId = 1,
                                userId = 1,
                                message = "Hello",
                            ),
                        ),
                )
            file.writeText(json.encodeToString(listOf(chat)))

            val repo = ChatRepository(file.toPath())
            val readCount = AtomicInteger(0)
            mockkStatic("kotlin.io.FilesKt__FileReadWriteKt")
            every { file.readText() } answers {
                readCount.incrementAndGet()
                file.inputStream().bufferedReader().use { it.readText() }
            }
            every { file.readText(any()) } answers {
                readCount.incrementAndGet()
                val charset = secondArg<Charset>()
                file.inputStream().bufferedReader(charset).use { it.readText() }
            }

            coroutineScope {
                repeat(2) { launch { repo.allThreads() } }
            }

            assertEquals(1, readCount.get(), "Chat file should be read only once when ensureLoaded is invoked concurrently")
            val threads = repo.allThreads()
            assertEquals(1, threads.size)
            assertEquals("thread-1", threads.first().uuid)
            assertTrue(threads.first().chats.isNotEmpty())
        }
    }

    @Test
    fun documentRepository_ensureLoaded_runsOnceWithConcurrentCalls() {
        runBlocking {
            val dir = Files.createTempDirectory("docRepoTest").toFile()
            val file = dir.resolve("documents.json")
            cleanupPaths += file to dir
            val doc =
                DocumentInfo(
                    id = 1,
                    userId = 1,
                    filename = "file.txt",
                    contentType = "text/plain",
                    filePath = dir.resolve("1_file.txt").absolutePath,
                    fileSize = 10,
                )
            file.writeText(json.encodeToString(listOf(doc)))

            val repo = DocumentRepository(dir.absolutePath, file.toPath())
            val readCount = AtomicInteger(0)
            mockkStatic("kotlin.io.FilesKt__FileReadWriteKt")
            every { file.readText() } answers {
                readCount.incrementAndGet()
                file.inputStream().bufferedReader().use { it.readText() }
            }
            every { file.readText(any()) } answers {
                readCount.incrementAndGet()
                val charset = secondArg<Charset>()
                file.inputStream().bufferedReader(charset).use { it.readText() }
            }

            coroutineScope {
                repeat(2) { launch { repo.all() } }
            }

            assertEquals(1, readCount.get(), "Document file should be read only once when ensureLoaded is invoked concurrently")
            val docs = repo.all()
            assertEquals(1, docs.size)
            assertEquals("file.txt", docs.first().filename)
            assertNotNull(docs.first().filePath)
        }
    }
}
