import re

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.source_model import Source
from app.services import pinecone_service

router = APIRouter(prefix="/sources", tags=["sources"])


class BulkDeleteRequest(BaseModel):
    ids: list[int] = Field(...)

_CATEGORY_LABELS = {
    "guide": ("Guides & Concepts", "Core TensorFlow concepts, APIs, and how-tos."),
    "tutorials": ("Tutorials", "Step-by-step walkthroughs and applied examples."),
    "api_docs": ("API Reference", "Detailed API signatures and usage."),
    "install": ("Installation & Setup", "Getting TensorFlow installed and configured."),
    "lite": ("TensorFlow Lite", "Mobile and edge deployment."),
    "tfx": ("TFX & Serving", "Production pipelines and model serving."),
    "js": ("TensorFlow.js", "Running TensorFlow in the browser and Node.js."),
    "hub": ("TensorFlow Hub", "Reusable pretrained models."),
}


def _slug_to_label(url: str) -> str:
    path = url.split("tensorflow.org", 1)[-1] if "tensorflow.org" in url else url
    path = path.strip("/")
    skip = {"guide", "tutorials", "api_docs", "python", "en", "lite", "js", "hub", "tfx", "install"}
    segments = [s for s in path.split("/") if s and s not in skip]
    slug = segments[-1] if segments else path
    slug = re.sub(r"[-_]+", " ", slug)
    slug = re.sub(r"\.\w+$", "", slug)
    return slug.strip().title()


@router.get("/")
def list_sources(db: Session = Depends(get_db)):
    # Ordered to match the site's sitemap order (same alphabetical sort
    # analyze_site()/the Range tab use for page 1, 2, 3...) rather than
    # scrape recency, so S.No lines up with "position on the site" and the
    # homepage naturally lands at #1.
    return db.query(Source).order_by(Source.url.asc()).all()


@router.get("/stats")
def get_stats(db: Session = Depends(get_db)):
    total_sources = db.query(Source).count()
    total_chunks = sum(s.chunks_created or 0 for s in db.query(Source).all())
    index_stats = pinecone_service.get_index_stats()
    return {
        "total_sources": total_sources,
        "total_chunks": total_chunks,
        "total_vectors": index_stats["total_vectors"],
        "estimated_storage_mb": index_stats["estimated_storage_mb"],
    }


@router.get("/topics")
def get_topics(db: Session = Depends(get_db)):
    """
    Derives homepage suggestions/categories directly from what's actually
    been successfully scraped and chunked - so the frontend never shows a
    question or topic with no grounding in the DB.
    """
    sources = (
        db.query(Source)
        .filter(Source.status == "success", Source.chunks_created > 0)
        .order_by(Source.chunks_created.desc())
        .all()
    )

    seen_labels = set()
    suggestions = []
    for s in sources:
        label = _slug_to_label(s.url)
        key = label.lower()
        if not label or key in seen_labels:
            continue
        seen_labels.add(key)
        suggestions.append({"label": label, "url": s.url, "chunks": s.chunks_created})
        if len(suggestions) >= 6:
            break

    category_totals: dict[str, dict] = {}
    for s in sources:
        path = s.url.split("tensorflow.org", 1)[-1].strip("/") if "tensorflow.org" in s.url else "general"
        top = path.split("/")[0] if path else "general"
        bucket = category_totals.setdefault(top, {"chunks": 0, "pages": 0})
        bucket["chunks"] += s.chunks_created or 0
        bucket["pages"] += 1

    ranked_categories = sorted(category_totals.items(), key=lambda kv: kv[1]["chunks"], reverse=True)[:3]
    categories = []
    for i, (key, info) in enumerate(ranked_categories):
        label, desc = _CATEGORY_LABELS.get(key, (key.replace("_", " ").title(), "Explore this topic."))
        categories.append({
            "n": f"0{i + 1} — {key.upper()[:6]}",
            "title": label,
            "desc": desc,
        })

    return {"suggestions": suggestions, "categories": categories}


@router.delete("/")
def delete_all_sources(db: Session = Depends(get_db)):
    """Permanently wipes every source from Postgres and every vector from Pinecone."""
    db.query(Source).delete()
    db.commit()
    pinecone_service.delete_all_chunks()
    return {"status": "all sources deleted"}


@router.post("/delete-bulk")
def delete_sources_bulk(request: BulkDeleteRequest, db: Session = Depends(get_db)):
    """Deletes multiple sources (and their Pinecone vectors) by id in one
    call, for the checkbox-select-then-delete flow in the admin table."""
    if not request.ids:
        raise HTTPException(status_code=400, detail="No source ids provided")

    matched = db.query(Source).filter(Source.id.in_(request.ids)).all()
    if not matched:
        raise HTTPException(status_code=404, detail="No matching sources found")

    deleted_ids = []
    for source in matched:
        try:
            pinecone_service.delete_chunks_by_source(source.url)
        except Exception:
            # Don't let a Pinecone hiccup on one source block deleting the
            # rest of the selection - the DB row still gets removed below,
            # matching the tolerant style of the rest of this pipeline.
            pass
        db.delete(source)
        deleted_ids.append(source.id)

    db.commit()
    return {"status": "deleted", "deleted_ids": deleted_ids, "count": len(deleted_ids)}


@router.get("/{source_id}")
def get_source(source_id: int, db: Session = Depends(get_db)):
    source = db.query(Source).filter(Source.id == source_id).first()
    if not source:
        raise HTTPException(status_code=404, detail="Source not found")
    return source


@router.delete("/{source_id}")
def delete_source(source_id: int, db: Session = Depends(get_db)):
    """Permanently removes a source record and its vectors from Pinecone."""
    source = db.query(Source).filter(Source.id == source_id).first()
    if not source:
        raise HTTPException(status_code=404, detail="Source not found")
    pinecone_service.delete_chunks_by_source(source.url)
    db.delete(source)
    db.commit()
    return {"status": "deleted", "id": source_id}