# 🧠 BoundaryMaker for Mac  
**Hair Follicle and T-Cell Segmentation using UNet (ResNet34 backbone)**  

BoundaryMaker is a cross-platform project for biomedical image segmentation.  
It uses a fine-tuned ResNet34-based UNet model to detect and draw contour boundaries around hair follicles and attached T-cells from microscope images.  
This version is optimized for macOS (with Apple Silicon support through Metal Performance Shaders, MPS) and also runs seamlessly on Windows with CUDA or CPU.

---

## 🚀 Features
- Fine-tuned UNet model using **ResNet34** encoder (`segmentation_models_pytorch`)
- **Automatic device selection** — CUDA, MPS, or CPU
- **Binary mask segmentation** and **boundary contour visualization**
- Compatible with both macOS (M-series chips) and Windows (NVIDIA GPUs)
- Clear output overlays and low-noise boundary filtering
- Easily extendable for lab or microscopy applications

---

## 🧩 Project Structure
BoundaryMaker-for-Mac/
│
├── scripts/
│ ├── train_finetune.py # Fine-tuning script (ResNet34 UNet)
│ ├── inference.py # Inference + boundary overlay
│
├── models/
│ └── unet_finetuned.pth # Place your trained model here
│
├── data/
│ └── data/
│ ├── images/
│ │ ├── train/
│ │ └── test/
│ └── masks/
│ └── train_bin/
│
├── results/ # Output images with boundaries
├── requirements.txt
├── README.md
└── .gitignore

yaml
Copy code

---

## ⚙️ Installation

### macOS or Linux
```bash
git clone https://github.com/johnarrackalceles-ui/BoundaryMaker-for-Mac.git
cd BoundaryMaker-for-Mac
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
Windows
bash
Copy code
git clone https://github.com/johnarrackalceles-ui/BoundaryMaker-for-Mac.git
cd BoundaryMaker-for-Mac
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
💾 Model Setup
Since GitHub restricts large files, the trained model (unet_finetuned.pth) is not included.
You can:

Copy your trained model from the Windows project into:
BoundaryMaker-for-Mac/models/unet_finetuned.pth

Or download it from your shared drive (Google Drive, Dropbox, etc.)

🧠 Training
To train or fine-tune the model (optional):

bash
Copy code
python3 scripts/train_finetune.py
You can configure:

Dataset directories (TRAIN_IMAGES_DIR, TRAIN_MASKS_DIR)

Epochs and batch size

Whether to load pretrained weights

If a pretrained model isn’t found, it will start training from scratch.

🔍 Inference
Run segmentation and generate boundary overlays:

bash
Copy code
python3 scripts/inference.py
Results will appear inside results/ as red-outlined images with segmented boundaries.
During execution, the script will automatically choose the fastest available device:

⚡ CUDA GPU (Windows / NVIDIA)

🍎 MPS GPU (macOS / Apple Silicon)

🧠 CPU fallback

🧰 Requirements
Main dependencies:

Python 3.10+

torch (with CUDA or MPS support)

torchvision

segmentation-models-pytorch

opencv-python

matplotlib

pillow

numpy

Install all at once:

bash
Copy code
pip install -r requirements.txt
📦 Notes
Training on macOS with MPS is supported but slower than NVIDIA GPUs.

Inference (prediction + contour drawing) is fast on both platforms.

The dataset should contain binary masks (white = object, black = background).

🧪 Example Output
Red boundaries highlight segmented hair follicles.

Low-threshold masks visualize early learning or weaker predictions.

Noise filtering can be customized to isolate follicles while ignoring skin layers.

🤝 Credits
Developed and fine-tuned with collaboration between:

Windows GPU training setup (CUDA)

macOS inference and portability optimization (MPS)

📜 License
MIT License © 2025
Created by johnarrackalceles-ui

download weights here https://drive.google.com/file/d/1ehdq3NOiItEMSd8pxZH0imdzm8H8CDwg/view?usp=sharing

