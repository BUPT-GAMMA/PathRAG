package pathrag.prompt

object Prompts {
    const val GRAPH_FIELD_SEP = "<SEP>"
    const val DEFAULT_LANGUAGE = "English"
    val DEFAULT_ENTITY_TYPES = listOf("organization", "person", "geo", "event", "category")
    val PROCESS_TICKERS = listOf("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏")

    const val FAIL_RESPONSE = "Sorry, I'm not able to provide an answer to that question."

    const val RAG_RESPONSE = """---Role---
You are a helpful assistant responding to questions about data in the tables provided.
---Goal---
Generate a response of the target length and format that responds to the user's question.
If you don't know the answer, just say so. Do not make anything up.
---Target response length and format---
{response_type}
---Data tables---
{context_data}
Add sections and commentary to the response as appropriate for the length and format. Style the response in markdown.
"""

    const val KEYWORDS_EXTRACTION = """Given the query, list both high-level and low-level keywords in JSON.
Query: {query}
Examples:
{examples}
Output keys: "high_level_keywords" and "low_level_keywords".
The Output should be human text, not unicode characters. Keep the same language as Query.
"""
}
