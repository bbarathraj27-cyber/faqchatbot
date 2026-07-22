from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # --- App ---
    APP_NAME: str = "FAQ Chatbot"
    ENV: str = "development"

    # --- PostgreSQL ---
    DATABASE_URL: str = ""

    # --- Gemini ---
    GEMINI_API_KEY: str = ""
    EMBEDDING_MODEL: str = "gemini-embedding-001"

    # --- LLM ---
    GROQ_API_KEY: str = ""
    LLM_MODEL: str = "openai/gpt-oss-20b"

    # --- Pinecone ---
    PINECONE_API_KEY: str = ""
    PINECONE_ENVIRONMENT: str = "us-east-1"
    PINECONE_INDEX_NAME: str = "faq-chatbot"
    EMBEDDING_DIMENSION: int = 1536

    # --- Redis (cache) ---
    REDIS_HOST: str = "relative-sturgeon-74102.upstash.io"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    REDIS_PASSWORD: str = ""
    REDIS_URL: str = ""
    CACHE_TTL_SECONDS: int = 60 * 60 * 24

    # --- Neo4j (knowledge graph) ---
    NEO4J_URI: str = "neo4j+s://416204ba.databases.neo4j.io"
    NEO4J_USERNAME: str = ""
    NEO4J_PASSWORD: str = ""

    # --- Auth ---
    JWT_SECRET_KEY: str = "change-this-secret-in-production"
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRE_MINUTES: int = 60 * 24

    # --- Chunking ---
    CHUNK_SIZE_TOKENS: int = 300
    CHUNK_OVERLAP_TOKENS: int = 50

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


settings = Settings()
