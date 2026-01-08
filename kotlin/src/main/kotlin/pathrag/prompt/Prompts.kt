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

    const val KEYWORDS_EXTRACTION = """---Role---
You extract concise keywords from user queries.
---Goal---
Return high-level and low-level keywords as JSON arrays.
---Language---
Use {language}.
---Input Query---
{query}
---Examples---
{examples}
---Output Format---
{ "high_level_keywords": [...], "low_level_keywords": [...] }
Only return JSON.
"""

    const val ENTITY_REL_JSON = """
Extract entities and relationships from the given text and return strict JSON.
Rules:
- Entities: produce array under "entities" with objects { "entity_name": string, "entity_type": string, "description": string }
- Relationships: produce array under "relationships" with objects { "src_id": string, "tgt_id": string, "description": string, "keywords": string, "weight": number }
- Use uppercase for entity_name/src_id/tgt_id and wrap names in quotes (e.g., "LONDON")
- Keep descriptions concise (<= 200 chars)
- Respond with ONLY JSON, no extra text.

Text:
{text}
"""
}
