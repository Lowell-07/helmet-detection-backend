import os
import cv2
import numpy as np
import requests
import firebase_admin
from firebase_admin import credentials, firestore, storage
from firebase_functions import https_fn, options
from ultralytics import YOLO
from paddleocr import PaddleOCR
from datetime import datetime
import tempfile

# Initialize Firebase
cred = credentials.ApplicationDefault()
firebase_admin.initialize_app(cred)
db = firestore.client()

# Load models once (cold start optimization)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
yolo_model = YOLO(os.path.join(BASE_DIR, 'model_files/best.pt'))
ocr = PaddleOCR(
    rec_model_dir=os.path.join(BASE_DIR, 'model_files/indian_plate'),
    use_angle_cls=True,
    lang='en',
    show_log=False,
    use_gpu=False  # Cloud Functions CPU only
)

@https_fn.on_call(
    memory=options.MemoryOption.GB_2,
    timeout_sec=300,
    cors=options.CorsOptions(cors_origins=["*"], cors_methods=["get", "post"])
)
def processViolationImage(req: https_fn.CallableRequest):
    try:
        image_url = req.data.get("imageUrl")
        user_id = req.auth.uid if req.auth else "anonymous"
        
        if not image_url:
            return {"error": "No image URL provided"}
        
        # Download image to temp
        temp_dir = tempfile.mkdtemp()
        temp_path = os.path.join(temp_dir, 'image.jpg')
        
        response = requests.get(image_url, timeout=30)
        with open(temp_path, 'wb') as f:
            f.write(response.content)
        
        # Load image
        img = cv2.imread(temp_path)
        if img is None:
            return {"error": "Failed to load image"}
        
        # Run YOLO detection
        results = yolo_model(img)
        
        has_helmet = False
        without_helmet_found = False
        plate_crop = None
        plate_confidence = 0
        
        for box in results[0].boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            class_name = yolo_model.names[cls]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            if class_name == "with_helmet":
                has_helmet = True
            elif class_name == "without_helmet":
                without_helmet_found = True
            elif class_name == "number_plate" and conf > 0.5:
                plate_crop = img[y1:y2, x1:x2]
                plate_confidence = conf
        
        # Logic: Only process if no helmet + plate detected
        if without_helmet_found and plate_crop is not None:
            # Run OCR
            ocr_result = ocr.ocr(plate_crop, det=False, rec=True)
            plate_text = ""
            
            if ocr_result and ocr_result[0]:
                plate_text = ocr_result[0][0][0]
                confidence_score = ocr_result[0][0][1]
                
                # Save to Firestore
                violation_data = {
                    "userId": user_id,
                    "timestamp": datetime.now(),
                    "imageUrl": image_url,
                    "numberPlateText": plate_text,
                    "hasHelmet": False,
                    "confidence": float(confidence_score),
                    "violationType": "No Helmet"
                }
                
                db.collection("violations").add(violation_data)
                
                return {
                    "violation": True,
                    "plateText": plate_text,
                    "confidence": float(confidence_score),
                    "message": "Violation detected and recorded"
                }
        
        return {
            "violation": False,
            "message": "No violation detected (wearing helmet or no plate found)"
        }
        
    except Exception as e:
        return {"error": str(e), "violation": False}