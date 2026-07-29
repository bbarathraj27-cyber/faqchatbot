from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # --- App ---
    APP_NAME: str = "FAQ Chatbot"
    ENV: str = "development"

    # --- PostgreSQL ---
    DATABASE_URL: str = ""

    # --- Gemini ---
    GEMINI_API_KEY: str = ""
    GEMINI_API_KEYS: str = ""  # optional: comma-separated list of multiple keys for rotation
    EMBEDDING_MODEL: str = "gemini-embedding-001"

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
    # NOTE: MAX_CONTEXT_TOKENS lives in app/prompts/faq_prompt.py, not here -
    # this used to be duplicated in both places (and out of sync: 3000 here
    # vs whatever faq_prompt.py had), which was just confusing. Removed.
    TOPIC_EXTRACTION_CHUNK_TOKENS: int = 500
    TOPIC_BATCH_TOKEN_BUDGET: int = 5000
    TOPIC_BATCH_CHUNK_TOKENS: int = 500

    # --- CORS ---
    # Comma-separated list of allowed origins, e.g.
    # "https://myapp.com,https://admin.myapp.com". Falls back to "*" (allow
    # any origin) only when unset, so local dev keeps working untouched.
    CORS_ALLOWED_ORIGINS: str = "*"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


settings = Settings()