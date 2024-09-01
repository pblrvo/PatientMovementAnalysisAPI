# main.py
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
import numpy as np
import tensorflow as tf
from src.json_file_keypoint_extraction import KeypointProcessor
import json

model = tf.keras.models.load_model("model/video_classifier_model.keras")
app = FastAPI()

class KeypointsInput(BaseModel):
    keypoints: list

processor = KeypointProcessor(max_seq_length=370)
class_mapping = {
    0: "NORMAL",
    1: "MILD",
    2: "MODERATE",
    3: "SEVERE"
}

@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> str:
    try:
        contents = await file.read()
        try:
            json_data = json.loads(contents)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid JSON file")

        print("Received JSON data:", json_data)

        if isinstance(json_data, list):
            keypoints_list = []
            for frame in json_data:
                if isinstance(frame, dict) and 'keypoints' in frame:
                    keypoints_list.extend(frame['keypoints'])
                else:
                    raise HTTPException(status_code=400, detail="Invalid frame format in JSON data")
        else:
            raise HTTPException(status_code=400, detail="JSON data should be a list of frames")

        if not keypoints_list:
            raise HTTPException(status_code=400, detail="No keypoints found in JSON data")

        # Skip confidence values
        keypoints_list = [value for i, value in enumerate(keypoints_list) if (i + 1) % 3 != 0]

        # Process keypoints
        extended_keypoints = processor.extend_keypoints(keypoints_list)
        standardized_keypoints = processor.standardize_keypoints(extended_keypoints)
        tensor_data = processor.build_tensors(standardized_keypoints)

        prediction = model.predict(tensor_data)
        predicted_class = int(np.argmax(prediction, axis=1)[0])
        return "prediction: " + class_mapping[predicted_class]
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

