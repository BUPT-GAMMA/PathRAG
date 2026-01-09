package pathrag

import io.github.oshai.kotlinlogging.KotlinLogging
import io.ktor.http.ContentType
import io.ktor.http.HttpHeaders
import io.ktor.http.HttpMethod
import io.ktor.http.HttpStatusCode
import io.ktor.http.charset
import io.ktor.http.content.PartData
import io.ktor.serialization.kotlinx.json.json
import io.ktor.server.application.Application
import io.ktor.server.application.ApplicationCall
import io.ktor.server.application.call
import io.ktor.server.application.install
import io.ktor.server.engine.embeddedServer
import io.ktor.server.http.content.staticResources
import io.ktor.server.netty.Netty
import io.ktor.server.plugins.contentnegotiation.ContentNegotiation
import io.ktor.server.plugins.cors.routing.CORS
import io.ktor.server.plugins.swagger.swaggerUI
import io.ktor.server.request.receive
import io.ktor.server.request.receiveMultipart
import io.ktor.server.response.respond
import io.ktor.server.response.respondRedirect
import io.ktor.server.routing.Route
import io.ktor.server.routing.delete
import io.ktor.server.routing.get
import io.ktor.server.routing.post
import io.ktor.server.routing.put
import io.ktor.server.routing.route
import io.ktor.server.routing.routing
import io.ktor.utils.io.core.readBytes
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import kotlinx.serialization.Serializable
import kotlinx.serialization.decodeFromString
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json
import pathrag.PathRAG
import pathrag.base.QueryParam
import java.io.File
import java.nio.charset.StandardCharsets
import java.nio.file.Files
import java.nio.file.Path
import java.nio.file.Paths
import java.security.MessageDigest
import java.time.Instant
import java.util.Base64
import java.util.UUID
import java.util.concurrent.atomic.AtomicBoolean
import javax.crypto.Mac
import javax.crypto.spec.SecretKeySpec

private val logger = KotlinLogging.logger("pathrag")

/**
 * Entry point that mirrors the behavior of the Python FastAPI server in [../main.py].
 * It loads environment variables, sets up default data, prepares directories, configures CORS,
 * and exposes placeholder routes for future feature parity.
 */
fun main() {
    val env = EnvironmentConfig.load(Paths.get("../.env"))
    val host = env["HOST"] ?: "0.0.0.0"
    val port = env["PORT"]?.toIntOrNull() ?: 8001

    logger.info { "Starting PathRAG API on $host:$port" }
    embeddedServer(Netty, port = port, host = host) {
        module(env)
    }.start(wait = true)
}

fun Application.module(env: EnvironmentConfig = EnvironmentConfig.empty()) {
    warnOnMissingRequiredVars(env)
    install(ContentNegotiation) {
        json()
    }
    install(CORS) {
        val origins = env.corsOrigins()
        if (origins == "*") {
            allowOrigins { true }
            logger.warn { "CORS is configured to allow all origins. Avoid this configuration in production." }
        } else {
            val allowed =
                origins
                    .split(",")
                    .map { it.trim() }
                    .filter { it.isNotBlank() }
            allowOrigins { origin -> allowed.any { origin.contains(it, ignoreCase = true) } }
        }
        allowCredentials = true
        allowMethod(HttpMethod.Get)
        allowMethod(HttpMethod.Post)
        allowMethod(HttpMethod.Put)
        allowMethod(HttpMethod.Delete)
        allowHeader(HttpHeaders.Authorization)
        allowHeader(HttpHeaders.ContentType)
    }

    val workingDir = env["WORKING_DIR"] ?: "./data"
    val uploadDir = env["UPLOAD_DIR"] ?: "./uploads"
    createDirectories(listOf(workingDir, uploadDir))

    val kvStorage = env["KV_STORAGE"] ?: "JsonKVStorage"
    val vectorStorage = env["VECTOR_STORAGE"] ?: "NanoVectorDBStorage"
    val graphStorage = env["GRAPH_STORAGE"] ?: "NetworkXStorage"
    val neo4jConfig =
        mapOf(
            "neo4j_uri" to env["NEO4J_URI"],
            "neo4j_user" to env["NEO4J_USER"],
            "neo4j_password" to env["NEO4J_PASSWORD"],
        ).filterValues { !it.isNullOrBlank() }

    val rag =
        PathRAG(
            workingDir = workingDir,
            kvStorage = kvStorage,
            vectorStorage = vectorStorage,
            graphStorage = graphStorage,
            extraConfig = neo4jConfig,
        )
    val userRepository = UserRepository(Paths.get(workingDir, "users.json"))
    val chatRepository = ChatRepository(Paths.get(workingDir, "chats.json"))
    val documentRepository = DocumentRepository(uploadDir, Paths.get(workingDir, "documents.json"))
    TokenService.configure(env)
    runBlocking { createDefaultUsers(userRepository) }
    runBlocking { preloadKnowledgeGraphIfEmpty(rag, documentRepository) }

    routing {
        get("/") { call.respondRedirect("/swagger") }
        swaggerUI(path = "swagger", swaggerFile = "openapi/documentation.yaml")
        staticResources("/ui", "static")
        get("/app") { call.respondRedirect("/ui/index.html") }
        authRoutes(userRepository)
        userRoutes(userRepository)
        chatRoutes(userRepository, chatRepository)
        documentRoutes(documentRepository, rag, userRepository)
        knowledgeGraphRoutes(rag)
        // query endpoint relocated under /documents/query
    }
}

private fun warnOnMissingRequiredVars(env: EnvironmentConfig) {
    val missing =
        buildList {
            if (!SecretKeyLoader.hasSecret(env)) add("SECRET_KEY_FILE (preferred) or SECRET_KEY")
        }
    if (missing.isNotEmpty()) {
        logger.warn { "Missing required environment variables: ${missing.joinToString(", ")}" }
        logger.warn { "Please set these variables in your .env file or environment. See sample.env for an example configuration." }
    }
}

private fun createDirectories(paths: List<String>) {
    paths.forEach { path ->
        runCatching {
            Files.createDirectories(Paths.get(path))
        }.onSuccess {
            logger.info { "Ensured directory exists: $path" }
        }.onFailure { ex ->
            logger.error(ex) { "Failed to create directory: $path" }
        }
    }
}

/**
 * In-memory repository used to mirror the Python default user creation.
 * Replace with a real database implementation when available.
 */
class UserRepository(
    private val filePath: Path? = null,
) {
    private val users = mutableListOf<User>()
    private var nextId = 1
    private val mutex = Mutex()
    private val json =
        Json {
            prettyPrint = true
            ignoreUnknownKeys = true
        }

    @Volatile
    private var initialized = false

    suspend fun count(): Int {
        ensureLoaded()
        return mutex.withLock { users.size }
    }

    suspend fun add(user: User): User {
        ensureLoaded()
        return mutex.withLock {
            val stored =
                if (user.id == null) user.copy(id = nextId++) else user.copy(id = user.id)
            users.add(stored)
            persist()
            stored
        }
    }

    suspend fun find(username: String): User? {
        ensureLoaded()
        return mutex.withLock { users.find { it.username == username } }
    }

    suspend fun updateTheme(
        username: String,
        theme: String,
    ): User? {
        ensureLoaded()
        return mutex.withLock {
            val idx = users.indexOfFirst { it.username == username }
            if (idx == -1) return@withLock null
            val updated = users[idx].copy(theme = theme, updatedAt = Instant.now().toString())
            users[idx] = updated
            persist()
            updated
        }
    }

    suspend fun list(): List<User> {
        ensureLoaded()
        return mutex.withLock { users.toList() }
    }

    suspend fun ensureLoaded() {
        if (initialized) return
        mutex.withLock {
            if (initialized) return
            val file = filePath?.toFile()
            val parsed =
                if (file != null && file.exists()) {
                    withContext(Dispatchers.IO) {
                        runCatching { json.decodeFromString<List<User>>(file.readText()) }
                            .onFailure { ex -> logger.warn(ex) { "Failed to load users from ${file.absolutePath}" } }
                            .getOrNull()
                    }
                } else {
                    emptyList()
                }
            if (!parsed.isNullOrEmpty()) {
                users.clear()
                users.addAll(parsed)
                nextId = (users.maxOfOrNull { it.id ?: 0 } ?: 0) + 1
            }
            initialized = true
        }
    }

    private suspend fun persist() {
        val file = filePath?.toFile() ?: return
        withContext(Dispatchers.IO) {
            runCatching {
                file.parentFile?.mkdirs()
                file.writeText(json.encodeToString(users))
            }.onFailure { ex ->
                logger.warn(ex) { "Failed to persist users to ${file.absolutePath}" }
            }
        }
    }
}

@Serializable
data class User(
    val id: Int? = null,
    val username: String,
    val email: String,
    val hashedPassword: String,
    val createdAt: String = Instant.now().toString(),
    val updatedAt: String = Instant.now().toString(),
    val theme: String = "blue",
)

private val defaultUsersCreated = AtomicBoolean(false)

private suspend fun createDefaultUsers(repository: UserRepository) {
    if (defaultUsersCreated.get()) return

    if (repository.count() == 0) {
        val defaults =
            listOf(
                DefaultUser("user1", "user1@example.com", "Pass@123"),
                DefaultUser("user2", "user2@example.com", "Pass@123"),
                DefaultUser("user3", "user3@example.com", "Pass@123"),
            )

        defaults.forEach { user ->
            val hashed = PasswordHasher.hash(user.password)
            repository.add(
                User(
                    id = null,
                    username = user.username,
                    email = user.email,
                    hashedPassword = hashed,
                ),
            )
        }
        defaultUsersCreated.set(true)
        logger.info { "Default users created successfully" }
    }
}

private data class DefaultUser(
    val username: String,
    val email: String,
    val password: String,
)

/**
 * Simple password hashing that mirrors the intent of get_password_hash in the Python code.
 * Swap this out for a stronger hashing strategy (e.g., bcrypt) when integrating authentication.
 */
object PasswordHasher {
    fun hash(input: String): String {
        val digest = MessageDigest.getInstance("SHA-256")
        val bytes = digest.digest(input.toByteArray())
        return bytes.joinToString(separator = "") { "%02x".format(it) }
    }
}

private object SecretKeyLoader {
    fun load(env: EnvironmentConfig): ByteArray {
        val secretFromFile =
            env["SECRET_KEY_FILE"]?.takeIf { it.isNotBlank() }?.let { pathString ->
                val path = runCatching { Paths.get(pathString) }.getOrNull()
                if (path == null) {
                    logger.error { "SECRET_KEY_FILE path is invalid: $pathString" }
                    null
                } else {
                    runCatching { Files.readString(path).trim() }
                        .onFailure { ex -> logger.error(ex) { "Failed to read SECRET_KEY_FILE at $pathString" } }
                        .getOrNull()
                        ?.takeIf { it.isNotBlank() }
                }
            }
        if (secretFromFile != null) {
            return secretFromFile.toByteArray(StandardCharsets.UTF_8)
        }

        val envSecret = env["SECRET_KEY"]?.takeIf { it.isNotBlank() }
        if (envSecret != null) {
            return envSecret.toByteArray(StandardCharsets.UTF_8)
        }

        logger.warn {
            "No SECRET_KEY_FILE configured and SECRET_KEY missing. Generated ephemeral secret; tokens will be invalidated on restart."
        }
        return UUID.randomUUID().toString().toByteArray(StandardCharsets.UTF_8)
    }

    fun hasSecret(env: EnvironmentConfig): Boolean {
        val fileSecretPresent =
            env["SECRET_KEY_FILE"]?.takeIf { it.isNotBlank() }?.let { pathString ->
                val path = runCatching { Paths.get(pathString) }.getOrNull() ?: return@let false
                if (!Files.exists(path)) return@let false
                val content = runCatching { Files.readString(path).trim() }.getOrNull()
                !content.isNullOrBlank()
            } ?: false

        return fileSecretPresent || !env["SECRET_KEY"].isNullOrBlank()
    }
}

object TokenService {
    private val encoder = Base64.getUrlEncoder().withoutPadding()
    private val decoder = Base64.getUrlDecoder()

    private const val DEFAULT_TOKEN_TTL_MINUTES = 30L

    @Volatile
    private var secret: ByteArray? = null

    @Volatile
    private var tokenTtlSeconds: Long = DEFAULT_TOKEN_TTL_MINUTES * 60

    fun configure(env: EnvironmentConfig) {
        if (secret == null) {
            secret = SecretKeyLoader.load(env)
        }
        tokenTtlSeconds =
            env["ACCESS_TOKEN_EXPIRE_MINUTES"]?.toLongOrNull()?.takeIf { it > 0 }?.times(60)
                ?: DEFAULT_TOKEN_TTL_MINUTES * 60
    }

    private fun secret(): ByteArray =
        secret ?: synchronized(this) {
            secret ?: SecretKeyLoader.load(EnvironmentConfig.empty()).also { secret = it }
        }

    fun issueToken(username: String): String {
        val issuedAt = Instant.now()
        val expiresAt = issuedAt.plusSeconds(tokenTtlSeconds)
        val nonce = UUID.randomUUID().toString()
        val payload = listOf(username, nonce, issuedAt.toString(), expiresAt.toString()).joinToString("|")
        val signature = hmacSha256(payload.toByteArray(StandardCharsets.UTF_8))
        return "${encoder.encodeToString(payload.toByteArray(StandardCharsets.UTF_8))}.${encoder.encodeToString(signature)}"
    }

    fun usernameFromToken(token: String?): String? {
        val parts = token?.split(".") ?: return null
        if (parts.size != 2) return null
        val payloadBytes =
            runCatching { decoder.decode(parts[0]) }.getOrElse { return null }
        val providedSignature =
            runCatching { decoder.decode(parts[1]) }.getOrElse { return null }
        val expected = hmacSha256(payloadBytes)
        if (!expected.contentEquals(providedSignature)) return null
        val payload = payloadBytes.toString(StandardCharsets.UTF_8)
        val segments = payload.split("|")
        if (segments.size < 4) return null
        val username = segments[0]
        val expiresAt = runCatching { Instant.parse(segments[3]) }.getOrNull() ?: return null
        if (Instant.now().isAfter(expiresAt)) return null
        return username
    }

    private fun hmacSha256(data: ByteArray): ByteArray {
        val mac = Mac.getInstance("HmacSHA256")
        mac.init(SecretKeySpec(secret(), "HmacSHA256"))
        return mac.doFinal(data)
    }
}

@Serializable
private data class LoginRequest(
    val username: String,
    val password: String,
)

@Serializable
private data class RegisterRequest(
    val username: String,
    val email: String,
    val password: String,
)

@Serializable
private data class ThemeRequest(
    val username: String,
    val theme: String,
)

@Serializable
data class ChatThread(
    val id: Int,
    val uuid: String,
    val userId: Int,
    val title: String,
    val createdAt: String = Instant.now().toString(),
    val updatedAt: String = Instant.now().toString(),
    val isDeleted: Boolean = false,
    val chats: List<ChatMessage> = emptyList(),
)

@Serializable
data class ChatMessage(
    val id: Int,
    val threadId: Int,
    val userId: Int,
    val role: String = "user",
    val message: String,
    val createdAt: String = Instant.now().toString(),
)

@Serializable
data class DocumentInfo(
    val id: Int,
    val userId: Int,
    val filename: String,
    val contentType: String,
    val filePath: String,
    val fileSize: Long,
    val uploadedAt: String = Instant.now().toString(),
    val status: String = "uploaded",
    val processedAt: String? = null,
    val errorMessage: String? = null,
)

class ChatRepository(
    private val filePath: Path? = null,
) {
    private val threads = mutableMapOf<String, ChatThread>()
    private var nextThreadId = 1
    private var nextMessageId = 1
    private val mutex = Mutex()
    private val json =
        Json {
            prettyPrint = true
            ignoreUnknownKeys = true
        }

    @Volatile
    private var initialized = false

    suspend fun allThreads(): List<ChatThread> {
        ensureLoaded()
        return mutex.withLock { threads.values.toList() }
    }

    suspend fun recentThreads(limit: Int = 5): List<ChatThread> {
        ensureLoaded()
        return mutex.withLock { threads.values.sortedByDescending { it.updatedAt }.take(limit) }
    }

    suspend fun thread(id: String): ChatThread? {
        ensureLoaded()
        return mutex.withLock { threads[id] }
    }

    suspend fun addThread(
        title: String,
        userId: Int,
    ): ChatThread {
        ensureLoaded()
        return mutex.withLock {
            val uuid =
                java.util.UUID
                    .randomUUID()
                    .toString()
            val thread =
                ChatThread(
                    id = nextThreadId++,
                    uuid = uuid,
                    userId = userId,
                    title = title,
                )
            threads[uuid] = thread
            persist()
            thread
        }
    }

    suspend fun updateThreadTitle(
        id: String,
        title: String,
    ): ChatThread? {
        ensureLoaded()
        return mutex.withLock {
            val current = threads[id] ?: return@withLock null
            val updated = current.copy(title = title, updatedAt = Instant.now().toString())
            threads[id] = updated
            persist()
            updated
        }
    }

    suspend fun markDeleted(id: String): ChatThread? {
        ensureLoaded()
        return mutex.withLock {
            val current = threads[id] ?: return@withLock null
            val updated = current.copy(isDeleted = true, updatedAt = Instant.now().toString())
            threads[id] = updated
            persist()
            updated
        }
    }

    suspend fun addChat(
        threadId: String,
        content: String,
        sender: String = "user",
        userId: Int,
    ): ChatMessage? {
        ensureLoaded()
        return mutex.withLock {
            val current = threads[threadId] ?: return@withLock null
            val message =
                ChatMessage(
                    id = nextMessageId++,
                    threadId = current.id,
                    userId = userId,
                    role = sender,
                    message = content,
                )
            val updated =
                current.copy(
                    chats = current.chats + message,
                    updatedAt = Instant.now().toString(),
                )
            threads[threadId] = updated
            persist()
            message
        }
    }

    private suspend fun ensureLoaded() {
        if (initialized) return
        mutex.withLock {
            if (initialized) return
            val file = filePath?.toFile()
            val parsed =
                if (file != null && file.exists()) {
                    withContext(Dispatchers.IO) {
                        runCatching { json.decodeFromString<List<ChatThread>>(file.readText()) }
                            .onFailure { ex -> logger.warn(ex) { "Failed to load chats from ${file.absolutePath}" } }
                            .getOrNull()
                    }
                } else {
                    emptyList()
                }
            if (!parsed.isNullOrEmpty()) {
                threads.clear()
                parsed.forEach { threads[it.uuid] = it }
                nextThreadId = (parsed.maxOfOrNull { it.id } ?: 0) + 1
                val maxMsgId =
                    parsed.flatMap { it.chats }.maxOfOrNull { it.id } ?: 0
                nextMessageId = maxMsgId + 1
            }
            initialized = true
        }
    }

    private suspend fun persist() {
        val file = filePath?.toFile() ?: return
        withContext(Dispatchers.IO) {
            runCatching {
                file.parentFile?.mkdirs()
                file.writeText(json.encodeToString(threads.values.toList()))
            }.onFailure { ex ->
                logger.warn(ex) { "Failed to persist chats to ${file.absolutePath}" }
            }
        }
    }
}

class DocumentRepository(
    private val uploadDir: String,
    private val filePath: Path? = null,
) {
    private val documents = mutableMapOf<Int, DocumentInfo>()
    private var nextId = 1
    private val mutex = Mutex()
    private val json =
        Json {
            prettyPrint = true
            ignoreUnknownKeys = true
        }

    @Volatile
    private var initialized = false

    suspend fun all(): List<DocumentInfo> {
        ensureLoaded()
        return mutex.withLock { documents.values.toList() }
    }

    suspend fun get(id: Int): DocumentInfo? {
        ensureLoaded()
        return mutex.withLock { documents[id] }
    }

    suspend fun add(
        name: String,
        content: String,
        contentType: String = "text/plain",
        userId: Int = 1,
    ): DocumentInfo {
        ensureLoaded()
        return mutex.withLock {
            val id = nextId++
            val filePath = File(uploadDir, "${id}_$name").absolutePath
            val info =
                DocumentInfo(
                    id = id,
                    userId = userId,
                    filename = name,
                    contentType = contentType,
                    filePath = filePath,
                    fileSize = content.toByteArray().size.toLong(),
                    status = "processing",
                )
            documents[id] = info
            saveToDisk(filePath, content)
            persist()
            info
        }
    }

    suspend fun addFile(
        name: String,
        data: ByteArray,
        contentType: String? = null,
        userId: Int = 1,
    ): DocumentInfo {
        ensureLoaded()
        return mutex.withLock {
            val id = nextId++
            val path = File(uploadDir, "${id}_$name").absolutePath
            val info =
                DocumentInfo(
                    id = id,
                    userId = userId,
                    filename = name,
                    contentType = contentType ?: "application/octet-stream",
                    filePath = path,
                    fileSize = data.size.toLong(),
                    status = "processing",
                )
            documents[id] = info
            saveBytes(path, data)
            persist()
            info
        }
    }

    suspend fun status(id: Int): String {
        ensureLoaded()
        return mutex.withLock { documents[id]?.status ?: "unknown" }
    }

    suspend fun markProcessed(id: Int) {
        ensureLoaded()
        mutex.withLock {
            val current = documents[id] ?: return@withLock
            documents[id] = current.copy(status = "processed", processedAt = Instant.now().toString(), errorMessage = null)
            persist()
        }
    }

    suspend fun markFailed(
        id: Int,
        message: String,
    ) {
        ensureLoaded()
        mutex.withLock {
            val current = documents[id] ?: return@withLock
            documents[id] = current.copy(status = "failed", errorMessage = message, processedAt = Instant.now().toString())
            persist()
        }
    }

    private suspend fun saveToDisk(
        path: String,
        content: String,
    ) {
        withContext(Dispatchers.IO) {
            runCatching {
                File(path).writeText(content)
            }.onFailure { ex ->
                logger.warn(ex) { "Failed to persist document at $path" }
            }
        }
    }

    private suspend fun saveBytes(
        path: String,
        data: ByteArray,
    ) {
        withContext(Dispatchers.IO) {
            runCatching {
                File(path).writeBytes(data)
            }.onFailure { ex ->
                logger.warn(ex) { "Failed to persist document at $path" }
            }
        }
    }

    private suspend fun ensureLoaded() {
        if (initialized) return
        mutex.withLock {
            if (initialized) return
            val file = filePath?.toFile()
            val parsed =
                if (file != null && file.exists()) {
                    withContext(Dispatchers.IO) {
                        runCatching { json.decodeFromString<List<DocumentInfo>>(file.readText()) }
                            .onFailure { ex -> logger.warn(ex) { "Failed to load documents from ${file.absolutePath}" } }
                            .getOrNull()
                    }
                } else {
                    emptyList()
                }
            if (!parsed.isNullOrEmpty()) {
                documents.clear()
                parsed.forEach { documents[it.id] = it }
                nextId = (documents.keys.maxOrNull() ?: 0) + 1
            }
            initialized = true
        }
    }

    private suspend fun persist() {
        val file = filePath?.toFile() ?: return
        withContext(Dispatchers.IO) {
            runCatching {
                file.parentFile?.mkdirs()
                file.writeText(json.encodeToString(documents.values.toList()))
            }.onFailure { ex ->
                logger.warn(ex) { "Failed to persist documents to ${file.absolutePath}" }
            }
        }
    }
}

/**
 * Placeholder route groups to keep parity with the FastAPI router structure.
 * Implement real logic when backing services are available.
 */
private fun Route.authRoutes(repository: UserRepository) {
    post("/token") {
        val req = call.receive<LoginRequest>()
        val user = repository.find(req.username)
        if (user == null || user.hashedPassword != PasswordHasher.hash(req.password)) {
            call.respond(HttpStatusCode.Unauthorized, mapOf("error" to "Invalid credentials"))
        } else {
            call.respond(mapOf("access_token" to TokenService.issueToken(user.username), "token_type" to "bearer"))
        }
    }
    post("/register") {
        val req = call.receive<RegisterRequest>()
        val existing = repository.find(req.username)
        if (existing != null) {
            call.respond(HttpStatusCode.Conflict, mapOf("error" to "User already exists"))
        } else {
            val user =
                User(
                    id = null,
                    username = req.username,
                    email = req.email,
                    hashedPassword = PasswordHasher.hash(req.password),
                )
            val stored = repository.add(user)
            call.respond(HttpStatusCode.Created, stored)
        }
    }
    get("/users/me") {
        val authHeader = call.request.headers[HttpHeaders.Authorization]
        val token =
            authHeader
                ?.takeIf { it.startsWith("Bearer ", ignoreCase = true) }
                ?.substring(7)
                ?.trim()
        val username = TokenService.usernameFromToken(token)
        if (username == null) {
            call.respond(HttpStatusCode.Unauthorized, mapOf("error" to "Invalid or missing token"))
            return@get
        }
        val current = repository.find(username)
        if (current == null) call.respond(HttpStatusCode.NotFound) else call.respond(current)
    }
}

private fun Route.userRoutes(repository: UserRepository) {
    route("/users") {
        get("/") {
            call.respond(mapOf("users" to repository.list()))
        }
        post("/theme") {
            val req = call.receive<ThemeRequest>()
            val updated = repository.updateTheme(req.username, req.theme)
            if (updated == null) {
                call.respond(HttpStatusCode.NotFound, mapOf("error" to "User not found"))
            } else {
                call.respond(updated)
            }
        }
    }
}

private suspend fun ApplicationCall.currentUser(userRepository: UserRepository): User? {
    val authHeader = request.headers[HttpHeaders.Authorization] ?: return null
    val token =
        authHeader
            .takeIf { it.startsWith("Bearer ", ignoreCase = true) }
            ?.substring(7)
            ?.trim()
    val username = TokenService.usernameFromToken(token) ?: return null
    return userRepository.find(username)
}

@Serializable
private data class CreateThreadRequest(
    val title: String,
)

@Serializable
private data class UpdateThreadRequest(
    val title: String,
)

@Serializable
private data class CreateChatRequest(
    val content: String,
    val sender: String? = "user",
)

private fun Route.chatRoutes(
    userRepository: UserRepository,
    chatRepository: ChatRepository,
) {
    route("/chats") {
        get("/") {
            val currentUser = call.currentUser(userRepository)
            if (currentUser == null || currentUser.id == null) {
                call.respond(HttpStatusCode.Unauthorized, mapOf("error" to "Invalid or missing token"))
                return@get
            }
            val chats =
                chatRepository
                    .allThreads()
                    .filter { it.userId == currentUser.id }
                    .flatMap { it.chats }
            call.respond(mapOf("chats" to chats))
        }
        get("/recent") {
            val currentUser = call.currentUser(userRepository)
            if (currentUser == null || currentUser.id == null) {
                call.respond(HttpStatusCode.Unauthorized, mapOf("error" to "Invalid or missing token"))
                return@get
            }
            val threads =
                chatRepository
                    .recentThreads()
                    .filter { it.userId == currentUser.id }
            call.respond(mapOf("threads" to threads))
        }
        route("/threads") {
            get {
                val currentUser = call.currentUser(userRepository)
                if (currentUser == null || currentUser.id == null) {
                    call.respond(HttpStatusCode.Unauthorized, mapOf("error" to "Invalid or missing token"))
                    return@get
                }
                val threads =
                    chatRepository
                        .allThreads()
                        .filter { it.userId == currentUser.id }
                call.respond(mapOf("threads" to threads))
            }
            post {
                val currentUser = call.currentUser(userRepository)
                if (currentUser == null || currentUser.id == null) {
                    call.respond(HttpStatusCode.Unauthorized, mapOf("error" to "Invalid or missing token"))
                    return@post
                }
                val req = call.receive<CreateThreadRequest>()
                val thread = chatRepository.addThread(req.title, currentUser.id)
                call.respond(HttpStatusCode.Created, thread)
            }
            get("/{thread_uuid}") {
                val id = call.parameters["thread_uuid"]
                val thread = id?.let { chatRepository.thread(it) }
                if (thread == null) call.respond(HttpStatusCode.NotFound) else call.respond(thread)
            }
            put("/{thread_uuid}") {
                val id = call.parameters["thread_uuid"]
                val req = call.receive<UpdateThreadRequest>()
                val updated = id?.let { chatRepository.updateThreadTitle(it, req.title) }
                if (updated == null) call.respond(HttpStatusCode.NotFound) else call.respond(updated)
            }
            delete("/{thread_uuid}") {
                val id = call.parameters["thread_uuid"]
                val deleted = id?.let { chatRepository.markDeleted(it) }
                if (deleted == null) call.respond(HttpStatusCode.NotFound) else call.respond(deleted)
            }
        }
        post("/chat/{thread_uuid}") {
            val id = call.parameters["thread_uuid"]
            val req = call.receive<CreateChatRequest>()
            val currentUser = call.currentUser(userRepository)
            if (currentUser == null || currentUser.id == null) {
                call.respond(HttpStatusCode.Unauthorized, mapOf("error" to "Invalid or missing token"))
                return@post
            }
            val message = id?.let { chatRepository.addChat(it, req.content, req.sender ?: "user", currentUser.id) }
            if (message == null) {
                call.respond(HttpStatusCode.NotFound, mapOf("error" to "Thread not found"))
            } else {
                call.respond(HttpStatusCode.Created, message)
            }
        }
    }
}

@Serializable
private data class UploadDocumentRequest(
    val name: String,
    val content: String,
    val contentType: String? = "text/plain",
)

@Serializable
private data class QueryRequest(
    val query: String,
    val mode: String? = null,
)

@Serializable
private data class KnowledgeGraphQuery(
    val query: String? = null,
    val q: String? = null,
)

@Serializable
private data class DocumentStatusResponse(
    val documentId: Int,
    val status: String,
)

@Serializable
private data class GraphNodeDto(
    val id: String,
    val properties: Map<String, String> = emptyMap(),
)

@Serializable
private data class GraphEdgeDto(
    val source: String,
    val target: String,
    val properties: Map<String, String> = emptyMap(),
)

@Serializable
private data class GraphResponse(
    val nodes: List<GraphNodeDto>,
    val edges: List<GraphEdgeDto>,
)

private fun isSupportedTextContent(contentType: ContentType?): Boolean {
    if (contentType == null) return false
    if (contentType.match(ContentType.Text.Any)) return true
    if (contentType.contentType.equals("application", ignoreCase = true)) {
        val subtype = contentType.contentSubtype.lowercase()
        if (subtype in setOf("json", "xml", "x-yaml", "yaml", "javascript", "csv")) return true
    }
    return false
}

private fun Route.documentRoutes(
    repository: DocumentRepository,
    rag: PathRAG,
    userRepository: UserRepository,
) {
    route("/documents") {
        get("/") {
            val currentUser = call.currentUser(userRepository)
            if (currentUser == null || currentUser.id == null) {
                call.respond(HttpStatusCode.Unauthorized, mapOf("error" to "Invalid or missing token"))
                return@get
            }
            val docs = repository.all().filter { it.userId == currentUser.id }
            call.respond(mapOf("documents" to docs))
        }
        post("/upload") {
            val req = call.receive<UploadDocumentRequest>()
            val currentUser = call.currentUser(userRepository)
            if (currentUser == null || currentUser.id == null) {
                call.respond(HttpStatusCode.Unauthorized, mapOf("error" to "Invalid or missing token"))
                return@post
            }
            val doc = repository.add(req.name, req.content, req.contentType ?: "text/plain", currentUser.id)
            launch {
                runCatching { rag.ainsert(req.content) }
                    .onSuccess { repository.markProcessed(doc.id) }
                    .onFailure { ex ->
                        logger.warn(ex) { "Failed to ingest uploaded document ${doc.id}" }
                        repository.markFailed(doc.id, ex.message ?: "Ingestion failed")
                    }
            }
            call.respond(HttpStatusCode.Created, doc)
        }
        post("/upload-file") {
            val currentUser = call.currentUser(userRepository)
            if (currentUser == null || currentUser.id == null) {
                call.respond(HttpStatusCode.Unauthorized, mapOf("error" to "Invalid or missing token"))
                return@post
            }
            val multipart = call.receiveMultipart()
            var saved: DocumentInfo? = null
            while (true) {
                val part = multipart.readPart() ?: break
                try {
                    when (part) {
                        is PartData.FileItem -> {
                            if (saved == null) {
                                val contentType = part.contentType
                                if (!isSupportedTextContent(contentType)) {
                                    call.respond(
                                        HttpStatusCode.UnsupportedMediaType,
                                        mapOf(
                                            "error" to
                                                "Unsupported content type '${contentType ?: "unknown"}'. Only text uploads are accepted.",
                                        ),
                                    )
                                    return@post
                                }
                                val bytes = withContext(Dispatchers.IO) { part.provider().readBytes() }
                                val filename = part.originalFileName ?: "upload_${System.currentTimeMillis()}"
                                saved = repository.addFile(filename, bytes, contentType?.toString(), currentUser.id)
                                launch {
                                    runCatching {
                                        val charset = contentType?.charset() ?: StandardCharsets.UTF_8
                                        val text = String(bytes, charset)
                                        rag.ainsert(text)
                                    }.onSuccess { repository.markProcessed(saved.id) }
                                        .onFailure { ex ->
                                            logger.warn(ex) { "Failed to ingest uploaded file $filename into RAG" }
                                            repository.markFailed(saved.id, ex.message ?: "Ingestion failed")
                                        }
                                }
                            }
                        }

                        else -> {}
                    }
                } finally {
                    part.dispose()
                }
            }
            val doc = saved
            if (doc == null) {
                call.respond(HttpStatusCode.BadRequest, mapOf("error" to "No file found in request"))
            } else {
                call.respond(HttpStatusCode.Created, doc)
            }
        }
        post("/query") {
            val req = call.receive<QueryRequest>()
            val mode = req.mode ?: QueryParam().mode
            val result = rag.query(req.query, QueryParam(mode = mode))
            call.respond(mapOf("answer" to result))
        }
        get("/{document_id}") {
            val id = call.parameters["document_id"]?.toIntOrNull()
            val doc = id?.let { repository.get(it) }
            if (doc == null) call.respond(HttpStatusCode.NotFound) else call.respond(doc)
        }
        get("/{document_id}/status") {
            val id = call.parameters["document_id"]?.toIntOrNull() ?: return@get call.respond(HttpStatusCode.BadRequest)
            call.respond(DocumentStatusResponse(id, repository.status(id)))
        }
        post("/reload") {
            call.respond(mapOf("message" to "Reload request accepted. PathRAG will recognize new documents."))
        }
    }
}

private fun Route.knowledgeGraphRoutes(rag: PathRAG) {
    suspend fun respondGraph(call: ApplicationCall) {
        val g = rag.graph()
        val nodeIds = g.nodes()
        val nodes =
            nodeIds.map { id ->
                GraphNodeDto(
                    id = id,
                    properties = toStringMap(g.getNode(id)),
                )
            }
        val edges =
            g
                .edges()
                .map { (u, v) ->
                    GraphEdgeDto(
                        source = u,
                        target = v,
                        properties = toStringMap(g.getEdge(u, v) ?: g.getEdge(v, u)),
                    )
                }
        call.respond(GraphResponse(nodes, edges))
    }
    route("/knowledge-graph") {
        get { respondGraph(call) }
        get("/") { respondGraph(call) } // tolerate trailing slash
        post("/query") {
            val payload = runCatching { call.receive<KnowledgeGraphQuery>() }.getOrNull()
            val question = payload?.query ?: payload?.q
            if (question.isNullOrBlank()) {
                call.respond(HttpStatusCode.BadRequest, mapOf("error" to "Missing 'query' in payload"))
            } else {
                val result = rag.query(question, QueryParam(mode = "hybrid"))
                call.respond(mapOf("answer" to result))
            }
        }
    }
}

private fun toStringMap(data: Map<String, Any?>?): Map<String, String> = data?.mapValues { (_, v) -> v?.toString() ?: "" } ?: emptyMap()

private suspend fun preloadKnowledgeGraphIfEmpty(
    rag: PathRAG,
    documentRepository: DocumentRepository,
) {
    val g = rag.graph()
    val hasNodes =
        runCatching { g.nodes().isNotEmpty() }
            .onFailure { ex -> logger.warn(ex) { "Failed to inspect knowledge graph; skipping preload." } }
            .getOrDefault(false)
    if (hasNodes) return

    val docs =
        runCatching { documentRepository.all() }
            .onFailure { ex -> logger.warn(ex) { "Failed to load documents for graph preload." } }
            .getOrDefault(emptyList())
    if (docs.isEmpty()) return

    logger.info { "Knowledge graph empty; preloading from ${docs.size} documents." }
    docs.forEach { doc ->
        runCatching {
            val content =
                withContext(Dispatchers.IO) {
                    File(doc.filePath).takeIf { it.exists() }?.readText()
                }
            if (!content.isNullOrBlank()) {
                rag.ainsert(content)
            } else {
                logger.warn { "Skipping preload for document ${doc.id}; file missing or empty at ${doc.filePath}" }
            }
        }.onFailure { ex -> logger.warn(ex) { "Failed to ingest document ${doc.id} for graph preload." } }
    }
}

/**
 * Minimal .env loader that emulates python-dotenv behavior for local development.
 */
class EnvironmentConfig private constructor(
    private val values: Map<String, String>,
) {
    operator fun get(key: String): String? = System.getenv(key) ?: values[key]

    fun corsOrigins(): String = this["CORS_ORIGINS"] ?: "*"

    companion object {
        fun empty() = EnvironmentConfig(emptyMap())

        fun load(path: Path): EnvironmentConfig {
            val file = path.toFile()
            if (!file.exists()) {
                logger.warn { "No .env file found at ${file.absolutePath}. Using system environment variables." }
                return EnvironmentConfig(emptyMap())
            }

            val pairs =
                file
                    .readLines()
                    .mapNotNull { line ->
                        val trimmed = line.trim()
                        if (trimmed.isBlank() || trimmed.startsWith("#")) return@mapNotNull null
                        val delimiterIndex = trimmed.indexOf("=")
                        if (delimiterIndex <= 0) return@mapNotNull null
                        val key = trimmed.substring(0, delimiterIndex).trim()
                        val value = trimmed.substring(delimiterIndex + 1).trim().removeSurrounding("\"")
                        key to value
                    }.toMap()

            logger.info { "Loaded environment variables from ${file.absolutePath}" }
            return EnvironmentConfig(pairs)
        }
    }
}
