import json
import logging
from neo4j import GraphDatabase
from app.config import settings
from app.prompts.topic_prompt import build_topic_extraction_prompt, build_batch_topic_extraction_prompt
from app.services.groq_key_pool import chat_completion

logger = logging.getLogger(__name__)

# How many chunks to bundle into a single topic-extraction LLM call during
# ingestion. Larger batches mean fewer Groq calls, but a bigger blast radius
# if the model's JSON response is malformed (the whole batch falls back to
# empty topics for every chunk in it). 15 is a reasonable middle ground for
# short (~300-token) FAQ chunks.
TOPIC_BATCH_SIZE = 15


class _SuppressBenignWriteRetry(logging.Filter):
    """
    The neo4j driver retries a failed write against a stale/cold connection
    and succeeds on the very next attempt (confirmed: KAG queries return
    correct data despite this line appearing — both failed and successful
    attempts resolve to the same IP, pointing to a cold-connection retry
    rather than a genuinely stale routing address). It logs the failed
    first attempt at ERROR level even though it self-recovers, so a plain
    setLevel(ERROR) can't filter it out. This filter drops only that exact
    message — every other Neo4j error (auth failure, timeout, real
    connection loss, etc.) still logs normally.
    """
    def filter(self, record):
        return "Failed to write data to connection" not in record.getMessage()


logging.getLogger("neo4j").addFilter(_SuppressBenignWriteRetry())

driver = None
try:
    driver = GraphDatabase.driver(
        settings.NEO4J_URI,
        auth=(settings.NEO4J_USERNAME, settings.NEO4J_PASSWORD),
        max_connection_lifetime=60,
        max_connection_pool_size=10,
        connection_acquisition_timeout=30,
        keep_alive=True,
    )
    driver.verify_connectivity()
    with driver.session() as session:
        session.run("RETURN 1")
    logger.info("Neo4j driver initialized successfully")
except Exception as e:
    logger.warning(f"Could not initialize Neo4j driver: {e}. Knowledge graph features will be disabled.")
    driver = None


def _create_topic_relationship_tx(tx, topic: str, related_topic: str, source_url: str) -> None:
    tx.run(
        """
        MERGE (a:Topic {name: $topic})
        MERGE (b:Topic {name: $related_topic})
        MERGE (a)-[r:RELATED_TO]->(b)
        SET r.source_url = $source_url
        """,
        topic=topic,
        related_topic=related_topic,
        source_url=source_url,
    )


def create_topic_relationship(topic: str, related_topic: str, source_url: str = "") -> None:
    if not driver:
        logger.warning("Neo4j not available, skipping topic relationship creation")
        return
    try:
        with driver.session() as session:
            session.execute_write(_create_topic_relationship_tx, topic, related_topic, source_url)
    except Exception as e:
        logger.warning(f"Could not create topic relationship: {e}")


def _get_related_topics_tx(tx, topic: str, limit: int) -> list[str]:
    result = tx.run(
        """
        MATCH (a:Topic {name: $topic})-[:RELATED_TO]-(b:Topic)
        RETURN b.name AS name LIMIT $limit
        """,
        topic=topic,
        limit=limit,
    )
    return [record["name"] for record in result]


def get_related_topics(topic: str, limit: int = 5) -> list[str]:
    if not driver:
        logger.warning("Neo4j not available, returning empty topics")
        return []
    try:
        with driver.session() as session:
            return session.execute_read(_get_related_topics_tx, topic, limit)
    except Exception as e:
        logger.warning(f"Could not get related topics: {e}")
        return []


def extract_topics(text: str) -> list[str]:
    """
    Uses the LLM to pull 2-4 topic keywords out of a chunk of text.
    Returns a clean list of lowercase topic strings.
    """
    try:
        messages = build_topic_extraction_prompt(text[:2000])
        raw = chat_completion(messages, model=settings.LLM_MODEL, temperature=0.0)
        topics = [t.strip().lower() for t in raw.split(",") if t.strip()]
        return topics[:4]
    except Exception as e:
        logger.warning(f"Could not extract topics: {e}")
        return []


def link_topics_from_text(text: str, source_url: str) -> list[str]:
    """
    Extracts topics from a chunk and links every pair of them together
    in the graph (since they co-occurred in the same chunk).
    Returns the topic list so the caller can also tag Pinecone metadata.
    """
    topics = extract_topics(text)
    _link_topic_pairs(topics, source_url)
    return topics


def _link_topic_pairs(topics: list[str], source_url: str) -> None:
    if not driver or not topics:
        return
    try:
        for i in range(len(topics)):
            for j in range(i + 1, len(topics)):
                create_topic_relationship(topics[i], topics[j], source_url)
    except Exception as e:
        logger.warning(f"Could not link topics: {e}")


def extract_topics_batch(texts: list[str]) -> list[list[str]]:
    """
    Same as extract_topics, but for many chunks in ONE LLM call instead of
    one call per chunk. This is the main lever for cutting Groq API usage
    during large ingests (thousands of chunks -> one call per TOPIC_BATCH_SIZE
    chunks instead of one call per chunk).

    Falls back to per-chunk extract_topics for the whole batch if the
    model's JSON response can't be parsed or doesn't match the expected
    length, so a single malformed response never silently drops topics for
    an entire batch.
    """
    if not texts:
        return []

    try:
        messages = build_batch_topic_extraction_prompt([t[:2000] for t in texts])
        content = chat_completion(messages, model=settings.LLM_MODEL, temperature=0.0)
        raw = content.strip()
        # Models sometimes wrap JSON in ```json fences despite instructions.
        if raw.startswith("```"):
            raw = raw.strip("`")
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        parsed = json.loads(raw)
        if not isinstance(parsed, list) or len(parsed) != len(texts):
            raise ValueError(f"Expected {len(texts)} topic lists, got {len(parsed) if isinstance(parsed, list) else type(parsed)}")

        result: list[list[str]] = []
        for entry in parsed:
            if isinstance(entry, list):
                result.append([str(t).strip().lower() for t in entry if str(t).strip()][:4])
            else:
                result.append([])
        return result
    except Exception as e:
        logger.warning(f"Batch topic extraction failed ({e}), falling back to per-chunk extraction")
        return [extract_topics(t) for t in texts]


def link_topics_from_texts_batch(texts: list[str], source_urls: list[str]) -> list[list[str]]:
    """
    Batched version of link_topics_from_text for use during bulk ingestion.
    texts and source_urls must be the same length (one source_url per chunk).
    Processes in groups of TOPIC_BATCH_SIZE so a single LLM call covers many
    chunks, then links topic pairs per chunk in the graph as before.
    """
    if not texts:
        return []

    all_topics: list[list[str]] = []
    for start in range(0, len(texts), TOPIC_BATCH_SIZE):
        batch_texts = texts[start:start + TOPIC_BATCH_SIZE]
        batch_urls = source_urls[start:start + TOPIC_BATCH_SIZE]
        batch_topics = extract_topics_batch(batch_texts)
        for topics, source_url in zip(batch_topics, batch_urls):
            _link_topic_pairs(topics, source_url)
        all_topics.extend(batch_topics)
    return all_topics