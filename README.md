# 🧠 BoundaryMaker for Mac  
**Hair Follicle and T-Cell Segmentation using UNet (ResNet34 backbone)**  

BoundaryMaker is a cross-platform project for biomedical image segmentation.  
It uses a fine-tuned ResNet34-based UNet model to detect and draw contour boundaries around hair follicles and attached T-cells from microscope images.  
This version is optimized for macOS (with Apple Silicon support through Metal Performance Shaders, MPS) and also runs seamlessly on Windows with CUDA or CPU.

## 🚀 Features
- Fine-tuned UNet model using **ResNet34** encoder (`segmentation_models_pytorch`)
- **Automatic device selection** — CUDA, MPS, or CPU
- **Binary mask segmentation** and **boundary contour visualization**
- Compatible with both macOS (M-series chips) and Windows (NVIDIA GPUs)
- Clear output overlays and low-noise boundary filtering
- Easily extendable for lab or microscopy applications

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


## Training
Run: 
python3 scripts/train_finetune.py

You can configure:

- Dataset directories (TRAIN_IMAGES_DIR, TRAIN_MASKS_DIR)
- Epochs and batch size
- Whether to load pretrained weights
- If a pretrained model isn’t found, it will start training from scratch.

## Inference
Run:
python3 scripts/inference.py

Results will appear inside results/ as red-outlined images with segmented boundaries.
During execution, the script will automatically choose the fastest available device:

⚡ CUDA GPU (Windows / NVIDIA)
🍎 MPS GPU (macOS / Apple Silicon)
🧠 CPU fallback

## Requirementes (Dependencies)
Main dependencies:
Python 3.10+
torch (with CUDA or MPS support)
torchvision
segmentation-models-pytorch
opencv-python
matplotlib
pillow
numpy

To Install all at once run: 
pip install -r requirements.txt

## Notes
Training on macOS with MPS is supported but slower than NVIDIA GPUs.
Inference (prediction + contour drawing) is fast on both platforms.
The dataset should contain binary masks (white = object, black = background).

## Example Output
<img width="1493" height="566" alt="image" src="https://github.com/user-attachments/assets/a97961d5-26be-4449-a8d4-407c77aa8505" />

## Credits
Developed and Fine-tuned with the collaboration of 
- Windows GPU training setup (CUDA)
- macOS inference and portability optimization (MPS (Metal GPU)).

## License
MIT License 2025


## ⚙️ Installation
For Windows
```bash
git clone https://github.com/johnarrackalceles-ui/BoundaryMaker-for-Mac.git
cd BoundaryMaker-for-Mac
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
For MacOS or Linux 

### macOS or Linux
```bash
git clone https://github.com/johnarrackalceles-ui/BoundaryMaker-for-Mac.git
cd BoundaryMaker-for-Mac
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt



