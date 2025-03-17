from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import List
from app.services.footprint_service import process_footprint

router = APIRouter()

@router.post("/upload")
async def upload_footprint(file: UploadFile = File(...)):
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")
    
    try:
        result = await process_footprint(file)
        return {"species": result['species'], "confidence": result['confidence']}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))