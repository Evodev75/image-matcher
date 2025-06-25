
from fastapi import FastAPI, File, UploadFile
from PIL import Image
import numpy as np
import uvicorn
import os
import re
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import Model
from scipy.spatial.distance import cosine

app = FastAPI()

# Charger le modèle MobileNetV2 sans la couche finale (embedding 1280D)
base_model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg')
model = Model(inputs=base_model.input, outputs=base_model.output)

# Fonction d'extraction d'embedding
def get_embedding(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB").resize((224, 224))
    img_array = img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    embedding = model.predict(img_array, verbose=0)
    return embedding[0]

# Charger les images de référence
REFERENCE_IMAGES = {}
REFERENCE_DIR = "reference_images"

for filename in os.listdir(REFERENCE_DIR):
    if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
        continue
    match = re.search(r"0000s_\d+_(.+?)\.(jpg|png|jpeg)$", filename)
    if match:
        img_id = match.group(1)
        img_path = os.path.join(REFERENCE_DIR, filename)
        img = Image.open(img_path)
        embedding = get_embedding(img)
        REFERENCE_IMAGES[img_id] = embedding

@app.post("/match")
async def match_image(file: UploadFile = File(...)):
    uploaded_img = Image.open(file.file)
    uploaded_embedding = get_embedding(uploaded_img)

    best_match_id = None
    best_score = float("inf")

    for ref_id, ref_embedding in REFERENCE_IMAGES.items():
        score = cosine(ref_embedding, uploaded_embedding)
        if score < best_score:
            best_score = score
            best_match_id = ref_id

    similarity = 1 - best_score  # 1 = parfait, 0 = orthogonal

    return {
        "match_id": best_match_id,
        "similarity": similarity
    }
