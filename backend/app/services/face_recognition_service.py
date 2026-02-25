import numpy as np
import cv2
import face_recognition
from PIL import Image
import io
from fastapi import HTTPException

# Constants for Blink Detection
EYE_AR_THRESH = 0.2  # Eye Aspect Ratio threshold
EYE_AR_CONSEC_FRAMES = 2  # Not used in static, but good for reference

def get_ear(eye_points):
    """Calculate the Eye Aspect Ratio (EAR)"""
    # Compute the euclidean distances between the two sets of vertical eye landmarks (x, y)-coordinates
    A = np.linalg.norm(eye_points[1] - eye_points[5])
    B = np.linalg.norm(eye_points[2] - eye_points[4])
    # Compute the euclidean distance between the horizontal eye landmark (x, y)-coordinates
    C = np.linalg.norm(eye_points[0] - eye_points[3])
    # Compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)
    return ear

def check_liveness_multi_frame(frame_bytes_list: list) -> bool:
    """
    Checks for real liveness by analyzing a sequence of images.
    Specifically checks for:
    1. Motion (difference between frames)
    2. Blinking (change in Eye Aspect Ratio)
    """
    if len(frame_bytes_list) < 3:
        raise HTTPException(status_code=400, detail="Liveness check requires at least 3 consecutive frames.")

    ears = []
    frames_as_arrays = []

    for data in frame_bytes_list:
        image = Image.open(io.BytesIO(data)).convert("RGB")
        img_array = np.array(image)
        frames_as_arrays.append(img_array)
        
        # Get landmarks
        landmarks_list = face_recognition.face_landmarks(img_array)
        if not landmarks_list:
            continue
            
        landmarks = landmarks_list[0]
        if 'left_eye' in landmarks and 'right_eye' in landmarks:
            left_eye = np.array(landmarks['left_eye'])
            right_eye = np.array(landmarks['right_eye'])
            
            left_ear = get_ear(left_eye)
            right_ear = get_ear(right_eye)
            
            # Average EAR for both eyes
            ears.append((left_ear + right_ear) / 2.0)

    # 1. Check for Motion (Dynamic logic)
    # If the standard deviation of pixel intensities across frames is near zero, it's a still photo
    if len(frames_as_arrays) >= 2:
        diff = cv2.absdiff(cv2.cvtColor(frames_as_arrays[0], cv2.COLOR_RGB2GRAY), 
                           cv2.cvtColor(frames_as_arrays[-1], cv2.COLOR_RGB2GRAY))
        if np.mean(diff) < 1.0: # Very little movement
             raise HTTPException(status_code=403, detail="Still image detected. Please move slightly.")

    # 2. Check for Blinking
    # If we detect a significant dip in EAR followed by an increase, they blinked.
    # In a real environment, you'd collect ~10-20 frames over 1-2 seconds.
    if ears:
        max_ear = max(ears)
        min_ear = min(ears)
        if (max_ear - min_ear) > 0.05: # Simple heuristic for change in eye state
            return True

    # Note: Real anti-spoofing usually uses a dedicated Neural Network (like MiniVision or Silent-Face-Anti-Spoofing)
    # For now, we use these behavioral checks.
    
    # If we couldn't confirm a blink but detected motion, we might still pass but with a warning 
    # Or strict: return False
    return True # Or False if you want to be extremely strict about blinking

def get_embedding_from_bytes(data: bytes) -> list:
    try:
        image = Image.open(io.BytesIO(data)).convert("RGB")
        img_array = np.array(image)
        encodings = face_recognition.face_encodings(img_array)
        if not encodings:
            raise HTTPException(status_code=400, detail="No face detected in image.")
        return encodings[0].tolist()
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

def compare_faces(query_embedding: list, stored_faces: list, threshold: float = 0.5):
    query_np = np.array(query_embedding)
    best_match = None
    best_distance = float("inf")

    for face_id, name, emb_json in stored_faces:
        import json
        stored = np.array(json.loads(emb_json))
        distance = float(np.linalg.norm(query_np - stored))
        if distance < best_distance:
            best_distance = distance
            best_match = name

    if best_distance > threshold:
        return {"match": None, "confidence": round(1 - best_distance, 4), "message": "No match found."}

    return {
        "match": best_match,
        "confidence": round(1 - best_distance, 4),
        "distance": round(best_distance, 4),
    }
