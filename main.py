
from fastapi import FastAPI, File, UploadFile
from skimage.metrics import structural_similarity as ssim
from PIL import Image
import numpy as np
import uvicorn
import os
import re

app = FastAPI()

# Charger les images de référence une fois au démarrage
REFERENCE_IMAGES = {}
REFERENCE_DIR = "reference_images"

for filename in os.listdir(REFERENCE_DIR):
    if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
        continue
    match = re.search(r"0000s_\d+_(.+?)\.(jpg|png|jpeg)$", filename)
    if match:
        img_id = match.group(1)
        img_path = os.path.join(REFERENCE_DIR, filename)
        img = Image.open(img_path).convert('L')
        REFERENCE_IMAGES[img_id] = np.array(img)

@app.post("/match")
async def match_image(file: UploadFile = File(...)):
    uploaded_img = Image.open(file.file).convert('L')
    uploaded_arr = np.array(uploaded_img)

    best_match_id = None
    best_score = -1

    for ref_id, ref_arr in REFERENCE_IMAGES.items():
        try:
            score = ssim(ref_arr, uploaded_arr)
            if score > best_score:
                best_score = score
                best_match_id = ref_id
        except Exception:
            continue

    return {"match_id": best_match_id}
