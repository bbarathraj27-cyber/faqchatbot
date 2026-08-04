import json
from sqlalchemy import Column, Integer, String, DateTime, Boolean, Text, func
from app.database import Base


class ScrapeJob(Base):
    __tablename__ = "scrape_jobs"

    id = Column(String, primary_key=True, index=True)
    urls_json = Column(Text, nullable=False)
    question_selector = Column(String, nullable=True)
    answer_selector = Column(String, nullable=True)
    status = Column(String, default="running", index=True)
    total_pages = Column(Integer, default=0)
    current_index = Column(Integer, default=0)
    chunks_created = Column(Integer, default=0)
    total_words = Column(Integer, default=0)
    current_url = Column(String, nullable=True)
    pause_requested = Column(Boolean, default=False)
    cancel_requested = Column(Boolean, default=False)
    last_error = Column(Text, nullable=True)
    consecutive_errors = Column(Integer, default=0)
    last_resume_check = Column(String, nullable=True)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

    @property
    def urls(self) -> list[str]:
        return json.loads(self.urls_json)