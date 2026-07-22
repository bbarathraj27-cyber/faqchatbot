import logging
from neo4j import GraphDatabase
from openai import OpenAI
from app.config import settings
from app.prompts.topic_prompt import build_topic_extraction_prompt

logger = logging.getLogger(__name__)


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
        # Shorter than before (was 300s): Aura can rotate/rebalance the
        # underlying server address, especially after a free-tier instance
        # wakes from being paused. Recycling connections more often means
        # the pool holds fewer stale sockets pointed at a since-changed
        # address, so fewer failed-then-retried attempts overall.
        max_connection_lifetime=60,
        max_connection_pool_size=10,
        connection_acquisition_timeout=30,
        keep_alive=True,
    )
    # Force an eager routing-table fetch against the *current* address right
    # now, at startup, rather than lazily on the first real request — which
    # is what caused the very first query after a resume to hit a stale
    # cached address.
    driver.verify_connectivity()
    # Warm-up call: absorbs the one-time cold-connection retry here at
    # import/startup time instead of during the first real user request.
    with driver.session() as session:
        session.run("RETURN 1")
    logger.info("Neo4j driver initialized successfully")
except Exception as e:
    logger.warning(f"Could not initialize Neo4j driver: {e}. Knowledge graph features will be disabled.")
    driver = None

_llm_client = OpenAI(api_key=settings.GROQ_API_KEY, base_url="https://api.groq.com/openai/v1")


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
        response = _llm_client.chat.completions.create(
            model=settings.LLM_MODEL,
            messages=messages,
            temperature=0.0,
        )
        raw = response.choices[0].message.content
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
    if driver:
        try:
            for i in range(len(topics)):
                for j in range(i + 1, len(topics)):
                    create_topic_relationship(topics[i], topics[j], source_url)
        except Exception as e:
            logger.warning(f"Could not link topics: {e}")
    return topics