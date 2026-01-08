package pathrag

import io.github.oshai.kotlinlogging.KotlinLogging
import io.ktor.http.HttpHeaders
import io.ktor.http.HttpMethod
import io.ktor.serialization.kotlinx.json.json
import io.ktor.server.application.Application
import io.ktor.server.application.call
import io.ktor.server.application.install
import io.ktor.server.engine.embeddedServer
import io.ktor.server.netty.Netty
import io.ktor.server.plugins.contentnegotiation.ContentNegotiation
import io.ktor.server.plugins.cors.routing.CORS
import io.ktor.server.plugins.cors.routing.allowHost
import io.ktor.server.plugins.cors.routing.anyHost
import io.ktor.server.response.respond
import io.ktor.server.routing.Route
import io.ktor.server.routing.get
import io.ktor.server.routing.route
import io.ktor.server.routing.routing
import kotlinx.serialization.Serializable
import java.io.File
import java.nio.file.Files
import java.nio.file.Path
import java.nio.file.Paths
import java.security.MessageDigest
import java.time.Instant
import java.util.concurrent.atomic.AtomicBoolean

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
            anyHost()
            logger.warn { "CORS is configured to allow all origins. Avoid this configuration in production." }
        } else {
            origins
                .split(",")
                .map { it.trim() }
                .filter { it.isNotBlank() }
                .forEach { allowHost(it, listOf("http", "https")) }
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

    val userRepository = UserRepository()
    createDefaultUsers(userRepository)

    routing {
        get("/") {
            call.respond(
                mapOf(
                    "message" to "Welcome to PathRAG API",
                    "docs" to "/docs",
                    "version" to "1.0.0",
                ),
            )
        }
        authRoutes()
        userRoutes(userRepository)
        chatRoutes()
        documentRoutes()
        knowledgeGraphRoutes()
    }
}

private fun warnOnMissingRequiredVars(env: EnvironmentConfig) {
    val required = listOf("SECRET_KEY")
    val missing = required.filter { env[it].isNullOrBlank() }
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
class UserRepository {
    private val users = mutableListOf<User>()

    fun count(): Int = users.size

    fun add(user: User) {
        users.add(user)
    }

    fun list(): List<User> = users.toList()
}

@Serializable
data class User(
    val username: String,
    val email: String,
    val hashedPassword: String,
    val createdAt: String = Instant.now().toString(),
)

private val defaultUsersCreated = AtomicBoolean(false)

private fun createDefaultUsers(repository: UserRepository) {
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

/**
 * Placeholder route groups to keep parity with the FastAPI router structure.
 * Implement real logic when backing services are available.
 */
private fun Route.authRoutes() {
    route("/auth") {
        get {
            call.respond(mapOf("status" to "ok", "message" to "Auth routes not implemented"))
        }
    }
}

private fun Route.userRoutes(repository: UserRepository) {
    route("/users") {
        get {
            call.respond(mapOf("users" to repository.list()))
        }
    }
}

private fun Route.chatRoutes() {
    route("/chats") {
        get {
            call.respond(mapOf("status" to "ok", "message" to "Chat routes not implemented"))
        }
    }
}

private fun Route.documentRoutes() {
    route("/documents") {
        get {
            call.respond(mapOf("status" to "ok", "message" to "Document routes not implemented"))
        }
    }
}

private fun Route.knowledgeGraphRoutes() {
    route("/knowledge-graph") {
        get {
            call.respond(mapOf("status" to "ok", "message" to "Knowledge graph routes not implemented"))
        }
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
