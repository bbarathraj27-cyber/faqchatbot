from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # --- App ---
    APP_NAME: str = "FAQ Chatbot"
    ENV: str = "development"

    # --- PostgreSQL ---
    DATABASE_URL: str = ""

    # --- Gemini (embeddings) ---
    GEMINI_API_KEY: str = ""
    GEMINI_API_KEYS: str = ""  # optional: comma-separated list of multiple keys for rotation
    EMBEDDING_MODEL: str = "gemini-embedding-001"

    # --- Gemini (LLM fallback, used only when Groq fails) ---
    GEMINI_LLM_API_KEYS: str = ""
    GEMINI_LLM_MODEL: str = "gemini-2.5-flash"
    LLM_FALLBACK_TO_GEMINI: bool = True

    # --- LLM ---
    GROQ_API_KEY: str = ""
    GROQ_API_KEYS: str = ""  # optional: comma-separated list of multiple keys for rotation
    LLM_MODEL: str = "openai/gpt-oss-20b"

    # --- Pinecone ---
    PINECONE_API_KEY: str = ""
    PINECONE_ENVIRONMENT: str = "us-east-1"
    PINECONE_INDEX_NAME: str = "faq-chatbot"
    EMBEDDING_DIMENSION: int = 1536

    # --- Redis (cache) ---
    REDIS_HOST: str = ""
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    REDIS_PASSWORD: str = ""
    REDIS_URL: str = ""
    CACHE_TTL_SECONDS: int = 60 * 60 * 24

    # --- Neo4j (knowledge graph) ---
    NEO4J_URI: str = ""
    NEO4J_USERNAME: str = ""
    NEO4J_PASSWORD: str = ""

    # --- Auth ---
    JWT_SECRET_KEY: str = "change-this-secret-in-production"
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRE_MINUTES: int = 60 * 24

    # --- Chunking ---
    CHUNK_SIZE_TOKENS: int = 300
    CHUNK_OVERLAP_TOKENS: int = 50

    # --- Tokenization budgets ---
    TOPIC_EXTRACTION_CHUNK_TOKENS: int = 500
    TOPIC_BATCH_TOKEN_BUDGET: int = 5000
    TOPIC_BATCH_CHUNK_TOKENS: int = 500

    # --- CORS ---
    CORS_ALLOWED_ORIGINS: str = "*"

    # --- Scraper: JS rendering ---
    SCRAPER_JS_RENDER_ENABLED: bool = True
    # If a static fetch returns less than this many characters of visible
    # text, we assume the page is a JS-rendered SPA and retry with a
    # headless browser instead of trusting the (likely empty) static HTML.
    SCRAPER_JS_FALLBACK_MIN_TEXT_LENGTH: int = 200
    SCRAPER_JS_RENDER_TIMEOUT_MS: int = 20000

    # --- Scraper: anti-bot handling ---
    SCRAPER_MAX_RETRIES: int = 3
    SCRAPER_RETRY_BACKOFF_BASE_SECONDS: float = 1.5
    SCRAPER_MIN_DELAY_SECONDS: float = 0.5  # politeness delay between requests in a crawl

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


settings = Settings()