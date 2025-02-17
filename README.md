# 🚗 Parking Space Detection using Mask R-CNN & CUDA

This project implements **real-time parking space detection** using **Mask R-CNN**, leveraging **NVIDIA CUDA** for parallel processing. The model is trained on the **COCO dataset**, and it specifically detects **cars** while ignoring other objects like trees, people, or buildings.

---

## 🚀 Features
- **Real-time Detection**: Identifies occupied and free parking spaces.
- **Parallel Processing**: Uses **NVIDIA CUDA** to accelerate inference.
- **Mask R-CNN with COCO Dataset**: Detects **only cars** (COCO label `3`).
- **Optimized for Performance**: Frame resizing, score filtering, and CUDA optimizations ensure high FPS.
- **Flexible Grid System**: Detects available spots based on a **30x30 pixel grid**.

---

## 📌 Installation
### 1️⃣ **Clone the Repository**
```bash
git clone https://github.com/yourusername/parking-space-detection.git
cd parking-space-detection
```

### 2️⃣ **Install Dependencies**
Ensure you have Python 3.8+ installed.
```bash
pip install torch torchvision opencv-python numpy
```

### 3️⃣ **Check CUDA Availability**
Ensure CUDA is installed and recognized by PyTorch:
```python
python -c "import torch; print(torch.cuda.is_available())"
```
If the output is `True`, CUDA is available. If not, install the correct CUDA version from [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit).

---

## ▶️ Run the Detection
Simply execute the script to start parking detection:
```bash
python park_detection.py
```
To stop the program, press `q`.

---

## 📌 How It Works
1. **Loads a video file (`parking_space.mp4`).**
2. **Resizes frames to 960x540** for optimized performance.
3. **Uses Mask R-CNN trained on COCO dataset** to detect cars (label `3`).
4. **Parallel Video Processing:** Uses multi-threading for better FPS.
5. **CUDA Acceleration:** Runs the model on GPU for fast inference.
6. **Draws grids:** Green (free), Red (occupied).

---

## ⚡ Performance Optimizations
- ✅ **CUDA acceleration** for model inference.
- ✅ **Filters detections** (`score > 0.7`) for better accuracy.
- ✅ **Parallel processing** with Python threads to avoid frame drops.
- ✅ **Resizes images** to reduce computational overhead.

---


## 🤝 Contributing
Pull requests are welcome! If you have any suggestions or issues, feel free to open an [issue](https://github.com/kdrmz1695/parking-space-detection/issues).

