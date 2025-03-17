from typing import List, Dict
from fastapi import HTTPException
from app.db.repositories import FootprintRepository
from ml_model.inference.predict import predict_footprint
from ml_model.inference.preprocess import preprocess_image

class FootprintService:
    def __init__(self, footprint_repository: FootprintRepository):
        self.footprint_repository = footprint_repository

    def identify_footprint(self, image: bytes) -> Dict[str, float]:
        try:
            # Preprocess the image for the model
            processed_image = preprocess_image(image)
            # Get predictions from the model
            predictions = predict_footprint(processed_image)
            # Return the predictions
            return predictions
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    def save_footprint_data(self, footprint_data: Dict) -> None:
        try:
            self.footprint_repository.save(footprint_data)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    def get_footprint_history(self) -> List[Dict]:
        try:
            return self.footprint_repository.get_all()
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))