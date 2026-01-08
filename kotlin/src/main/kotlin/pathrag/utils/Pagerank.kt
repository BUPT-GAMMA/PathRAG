package pathrag.utils

import kotlin.math.abs

/**
 * Lightweight PageRank for small in-memory graphs; treats edges as undirected for scoring.
 */
fun computePagerankLocal(
    nodes: List<String>,
    edges: List<Pair<String, String>>,
    damping: Double = 0.85,
    maxIter: Int = 100,
    tol: Double = 1e-6,
): Map<String, Double> {
    if (nodes.isEmpty()) return emptyMap()
    val n = nodes.size
    val adjacency =
        nodes.associateWith { mutableListOf<String>() }.toMutableMap().also { adj ->
            edges.forEach { (u, v) ->
                adj[u]?.add(v)
                adj[v]?.add(u)
            }
        }
    val rank = mutableMapOf<String, Double>()
    nodes.forEach { rank[it] = 1.0 / n }

    repeat(maxIter) {
        var diff = 0.0
        val newRank = mutableMapOf<String, Double>()
        for (node in nodes) {
            val neighbors = adjacency[node].orEmpty()
            val outDeg = neighbors.size
            val share = if (outDeg == 0) 0.0 else rank[node]!! / outDeg
            neighbors.forEach { dest -> newRank[dest] = (newRank[dest] ?: 0.0) + share }
        }
        for (node in nodes) {
            val updated = (1 - damping) / n + damping * (newRank[node] ?: 0.0)
            diff += abs(updated - (rank[node] ?: 0.0))
            rank[node] = updated
        }
        if (diff < tol) return rank
    }
    return rank
}
