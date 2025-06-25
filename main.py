from fastapi import FastAPI, UploadFile, File
from torchvision import models, transforms
import torch
from PIL import Image
import os
import torch.nn.functional as F
import re

app = FastAPI()

# Chemin vers les images de référence
REFERENCE_DIR = "reference_images"

# Prétraitement des images pour MobileNetV2
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Charger MobileNetV2 sans la dernière couche (embedding)
model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
model.classifier = torch.nn.Identity()
model.eval()

# Préparer les embeddings des images de référence
reference_embeddings = {}
def extract_id(filename):
    match = re.search(r"0000s_\d+_(.+?)\.(jpg|jpeg|png)", filename)
    return match.group(1) if match else filename

for filename in os.listdir(REFERENCE_DIR):
    if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
        continue
    img_path = os.path.join(REFERENCE_DIR, filename)
    img = Image.open(img_path).convert("RGB")
    tensor = preprocess(img).unsqueeze(0)
    with torch.no_grad():
        embedding = model(tensor).squeeze()
    img_id = extract_id(filename)
    reference_embeddings[img_id] = embedding

@app.post("/match")
async def match(file: UploadFile = File(...)):
    image = Image.open(file.file).convert("RGB")
    tensor = preprocess(image).unsqueeze(0)
    with torch.no_grad():
        embedding = model(tensor).squeeze()

    best_match = None
    best_score = -1

    for ref_id, ref_emb in reference_embeddings.items():
        sim = F.cosine_similarity(embedding, ref_emb, dim=0).item()
        if sim > best_score:
            best_score = sim
            best_match = ref_id

    return {"match_id": best_match, "score": best_score}

# Ajout pour port binding correct sur Render
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)