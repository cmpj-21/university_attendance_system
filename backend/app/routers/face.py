from fastapi import APIRouter, File, UploadFile, HTTPException
from app.services.face_recognition_service import get_embedding_from_bytes, compare_faces
from app.database.face_db import save_face, get_all_faces, delete_face_by_id

router = APIRouter(
    prefix="/face",
    tags=["facial-recognition"]
)

@router.post("/register")
async def register(name: str, file: UploadFile = File(...)):
    """Register a new person. Send name + face image."""
    data = await file.read()
    embedding = get_embedding_from_bytes(data)
    save_face(name, embedding)
    return {"message": f"Registered '{name}' successfully."}

from typing import List

@router.post("/recognize")
async def recognize(files: List[UploadFile] = File(...), threshold: float = 0.5):
    """
    Send a sequence of face images (min 3), 
    checks for liveness (blinking/motion), 
    then gets back who it is.
    """
    if not files or len(files) < 3:
        raise HTTPException(status_code=400, detail="Please provide at least 3 frames for liveness verification.")

    # Read all files into bytes
    frames_bytes = [await f.read() for f in files]
    
    # 1. Perform Liveness Check
    from app.services.face_recognition_service import check_liveness_multi_frame
    if not check_liveness_multi_frame(frames_bytes):
        raise HTTPException(status_code=403, detail="Liveness verification failed.")

    # 2. Get embedding from the LAST frame (assuming it's the clearest/latest)
    query_embedding = get_embedding_from_bytes(frames_bytes[-1])
    
    rows = get_all_faces()
    if not rows:
        raise HTTPException(status_code=404, detail="No faces registered yet.")
    
    return compare_faces(query_embedding, rows, threshold)

@router.get("/faces")
def list_faces():
    """List all registered names."""
    rows = get_all_faces()
    return [{"id": r[0], "name": r[1]} for r in rows]

@router.delete("/faces/{face_id}")
def delete_face(face_id: int):
    """Remove a registered face by ID."""
    delete_face_by_id(face_id)
    return {"message": f"Deleted face ID {face_id}."}
