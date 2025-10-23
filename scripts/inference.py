import torch
import numpy as np
import cv2
import os
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp

# --- Import contour editor ---
from contour_editor_gui import ContourEditorGUI

# --- 1️⃣ Paths ---
MODEL_PATH = r"C:\Users\Celes\PycharmProjects\unet_small_gpu\unet_finetuned.pth"  # your trained weights
TEST_IMAGES_DIR = "data/images/test"
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# --- 2️⃣ Device Selection ---
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("✅ Using NVIDIA GPU (CUDA)")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("🍎 Using Apple Metal (MPS GPU)")
else:
    device = torch.device("cpu")
    print("🧠 Using CPU")

# --- 3️⃣ Load Model (ResNet34 UNet) ---
model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights=None,
    in_channels=3,
    classes=1
).to(device)

# --- 4️⃣ Load trained weights ---
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()
print(f"✅ Model loaded successfully from {MODEL_PATH}")

# --- 5️⃣ Image Transform ---
transform = transforms.Compose([transforms.ToTensor()])

# --- 6️⃣ Inference Loop ---
test_images = [f for f in os.listdir(TEST_IMAGES_DIR) if f.lower().endswith((".png", ".jpg", ".jpeg"))]

for img_name in test_images:
    img_path = os.path.join(TEST_IMAGES_DIR, img_name)
    image = Image.open(img_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        output_sigmoid = torch.sigmoid(output).squeeze().cpu().numpy()

    print(f"{img_name} → output min: {output_sigmoid.min():.4f}, max: {output_sigmoid.max():.4f}")

    # --- Threshold + Contours ---
    mask_clean = (output_sigmoid > 0.5).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_cv = np.array(image)
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
    cv2.drawContours(img_cv, contours, -1, (0, 0, 255), 2)

    # --- 🪟 Open Windows-style interactive contour editor ---
    print("🖋 Launching interactive contour editor window...")
    editor = ContourEditorGUI(img_cv.copy(), contours)
    edited_contours = editor.start()  # blocks until user saves or cancels

    # Use edited contours if available
    if edited_contours and len(edited_contours) > 0:
        contours = edited_contours
        print(f"✅ User edited {len(contours)} contours successfully.")
    else:
        print("⚠️ Contours unchanged or edit canceled.")

    # --- Re-draw final contours ---
    final_img = np.array(image)
    final_img = cv2.cvtColor(final_img, cv2.COLOR_RGB2BGR)
    cv2.drawContours(final_img, contours, -1, (0, 0, 255), 2)

    # --- Save Results ---
    result_path = os.path.join(RESULTS_DIR, img_name)
    cv2.imwrite(result_path, final_img)
    print(f"💾 Saved final result to {result_path}")

    # --- Preview ---
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.title("Original")
    plt.imshow(image)
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title("Predicted Mask")
    plt.imshow(mask_clean, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title("Final Overlay (After Edit)")
    plt.imshow(cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB))
    plt.axis("off")

    plt.show()
