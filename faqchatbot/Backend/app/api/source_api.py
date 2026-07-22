from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.source_model import Source
from app.services import pinecone_service

router = APIRouter(prefix="/sources", tags=["sources"])


@router.get("/")
def list_sources(db: Session = Depends(get_db)):
    return db.query(Source).order_by(Source.created_at.desc()).all()


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


@router.delete("/")
def delete_all_sources(db: Session = Depends(get_db)):
    """Permanently wipes every source from Postgres and every vector from Pinecone."""
    db.query(Source).delete()
    db.commit()
    pinecone_service.delete_all_chunks()
    return {"status": "all sources deleted"}


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