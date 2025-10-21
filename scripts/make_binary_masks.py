import os
import numpy as np
from PIL import Image

MASK_DIR = "data/data/masks/train" #IMPUT YOUR FILE PATH HERE [JUST FOR REFERENCE] 
OUT_DIR = "data/data/masks/train_bin" #IMPUT YOUR FILE PATH HERE [JUST FOR REFERENCE]
os.makedirs(OUT_DIR, exist_ok=True)

THRESHOLD = 50

for fname in os.listdir(MASK_DIR):
    if not fname.lower().endswith((".png", ".jpg", ".jpeg", ".tif")):
        continue

    img = Image.open(os.path.join(MASK_DIR, fname)).convert("L")
    arr = np.array(img)
    binary = (arr > THRESHOLD).astype(np.uint8) * 255
    Image.fromarray(binary).save(os.path.join(OUT_DIR, fname))

print(f"✅ All masks converted to binary and saved in {OUT_DIR}")
