import json

from sqlalchemy import Column, Integer, String, DateTime, Boolean, Text, func

from app.database import Base


class ScrapeJob(Base):
    """
    Durable record of a bulk scrape/import job (the "Importing in
    background..." progress bar in the admin tab).

    Unlike the old in-memory job_tracker dict, this survives server
    restarts, dropped tabs, and crashes — everything a resume needs
    (the full URL list, selectors, and exactly how far it got) lives in
    the DB, not in process memory. That's what makes pause/resume and
    crash-recovery possible: a job can be picked back up by any later
    request, even from a brand new process.
    """

    __tablename__ = "scrape_jobs"

    id = Column(String, primary_key=True, index=True)  # uuid4 string

    # JSON-encoded list[str] of every URL this job covers, in scrape order.
    # Stored once at creation and never mutated — current_index is the only
    # thing that moves, so "resume" always means "continue this exact list
    # from this exact position," never a re-derived/re-fetched URL set.
    urls_json = Column(Text, nullable=False)

    question_selector = Column(String, nullable=True)
    answer_selector = Column(String, nullable=True)

    # running -> paused (user-requested or auto, after repeated failures)
    #         -> running again (resume) -> completed
    status = Column(String, default="running", index=True)

    total_pages = Column(Integer, default=0)
    # How many URLs have been *successfully* processed. This is the resume
    # pointer: urls_json[:current_index] is done, urls_json[current_index]
    # is where the next run picks up. A URL that failed and triggered an
    # auto-pause is deliberately NOT counted here, so resuming retries it
    # instead of silently skipping it.
    current_index = Column(Integer, default=0)

    chunks_created = Column(Integer, default=0)
    total_words = Column(Integer, default=0)
    current_url = Column(String, nullable=True)

    # Set by the pause endpoint; the running loop polls this and stops
    # cleanly (after finishing whatever page it's mid-way through) rather
    # than being killed mid-request, so a page is never left half-saved.
    pause_requested = Column(Boolean, default=False)

    last_error = Column(Text, nullable=True)
    # How many URLs in a row have failed. Reset to 0 on any success. Used
    # to auto-pause instead of burning through the rest of the list once
    # something structural (an exhausted API key pool, a dead network) is
    # clearly wrong rather than a one-off bad page.
    consecutive_errors = Column(Integer, default=0)

    # Result of the "does the last completed page still look right"
    # self-check that runs at the start of a resume, before continuing
    # forward. Purely informational — a failed check logs a warning but
    # never blocks the resume.
    last_resume_check = Column(String, nullable=True)

    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

    @property
    def urls(self) -> list[str]:
        return json.loads(self.urls_json)