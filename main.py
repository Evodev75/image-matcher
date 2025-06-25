
from fastapi import FastAPI, File, UploadFile
from PIL import Image
import imagehash
import os
import re

app = FastAPI()

# Fonction pour calculer le hash d'une image
def get_image_hash(image: Image.Image) -> imagehash.ImageHash:
    image = image.convert("L").resize((256, 256))
    return imagehash.phash(image)

# Charger les images de référence avec leurs hashes
REFERENCE_HASHES = {}
REFERENCE_DIR = "reference_images"

for filename in os.listdir(REFERENCE_DIR):
    if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
        continue
    match = re.search(r"0000s_\d+_(.+?)\.(jpg|png|jpeg)$", filename)
    if match:
        img_id = match.group(1)
        img_path = os.path.join(REFERENCE_DIR, filename)
        img = Image.open(img_path)
        hash_val = get_image_hash(img)
        REFERENCE_HASHES[img_id] = hash_val

@app.post("/match")
async def match_image(file: UploadFile = File(...)):
    uploaded_img = Image.open(file.file)
    uploaded_hash = get_image_hash(uploaded_img)

    best_match_id = None
    best_distance = float("inf")

    for ref_id, ref_hash in REFERENCE_HASHES.items():
        distance = uploaded_hash - ref_hash  # Hamming distance
        if distance < best_distance:
            best_distance = distance
            best_match_id = ref_id

    return {
        "match_id": best_match_id,
        "distance": int(best_distance)  # ← conversion ici
    }
