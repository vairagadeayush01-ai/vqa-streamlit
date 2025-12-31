# 🖼️ Visual Question Answering (VQA) System  
**Python · PyTorch · ResNet50 · BERT**

A deep learning–based **Visual Question Answering (VQA)** system that answers natural-language questions about images by jointly reasoning over **visual content and text**.

The model is implemented in **PyTorch**, using:
- **ResNet50** for image feature extraction
- **BERT-base** for question encoding
- **Spatial attention** to focus on relevant image regions
- **Gated fusion** for multimodal reasoning

This repository contains the **complete pipeline**: model definition, inference utilities, and a **Streamlit web application** for interactive testing.

---

## ✨ Features

- End-to-end VQA pipeline
- ResNet50 (ImageNet-pretrained) image encoder
- BERT-base-uncased text encoder
- Spatial attention over CNN feature maps
- Gated fusion of image and text features
- Top-3000 answer classification (VQA v2)
- Resume-safe training with checkpoints
- Streamlit app for real-time inference

---

## 📊 Model Performance

**Dataset:** VQA v2 (COCO 2014 – Validation)

| Metric | Score |
|------|------|
| Hard Accuracy | **~44.9%** |

**Hard Accuracy Definition:**  
Exact match between the predicted answer and the most frequent ground-truth answer  
(Top-3000 answer classification setting)

---

## 🧠 Model Architecture (Conceptual Overview)

1. **Image Processing**
   - Input image resized to 224×224
   - Feature extraction using ResNet50
   - Spatial feature map generation

2. **Question Processing**
   - Input question tokenized
   - Encoded using BERT-base
   - Fixed-length text embedding obtained

3. **Attention & Fusion**
   - Spatial attention highlights relevant image regions
   - Gated fusion combines visual and textual features

4. **Answer Prediction**
   - Fully connected classifier
   - Predicts the most likely answer from top-3000 candidates

---

## 🔑 Core Components

| Component | Description |
|--------|------------|
| Image Encoder | ResNet50 (pretrained on ImageNet) |
| Text Encoder | BERT-base-uncased |
| Attention | Spatial attention over image feature maps |
| Fusion | Gated fusion with dropout |
| Classifier | Fully connected layer over top-3000 answers |

---

## 📁 Repository Structure
VQA/
- ├── checkpoints/     # Saved model weights
- ├── .devcontainer/   # Dev container configuration
- ├── model.py         # VQA model architecture
- ├── inference.py     # Inference utilities
- ├── app.py           # Streamlit web application
- ├── requirements.txt # Python dependencies
- └── README.md

---

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- PyTorch
- CUDA-enabled GPU (recommended for training)
- 16GB+ RAM

---
### Installation

bash
- git clone https://github.com/vairagadeayush01-ai/vqa-project.git
- cd vqa-project
- pip install -r requirements.txt
---
### Inference (Quick Test)

- Run inference on a single image and question:
- python inference.py \
   -- image path/to/image.jpg \
   -- question "What is the person doing?"
---
### 🖥️ Streamlit Demo

- Launch the interactive web application:
- streamlit run app.py
- The app allows you to:
- 💠Upload an image
- 💠Enter a natural-language question
- 💠Get the model-predicted answer instantly
---
### 🏋️ Training (Summary)

- 💠Trained on VQA v2 (COCO 2014) dataset
- 💠Training performed on Kaggle (GPU P100)
- 💠Initial epochs with frozen ResNet50 and BERT
- 💠Fine-tuning after unfreezing encoders
- 💠Model checkpoints saved after every epoch
- 💠Training safely resumed after session interruptions

### ⚙️ Training Configuration

- 💠Optimizer: Adam
- 💠Learning Rate: 1e-4 → 1e-5
- 💠Batch Size: 32
- 💠Image Size: 224 × 224
- 💠Max Question Length: 21 tokens
- 💠Answer Vocabulary Size: Top-3000
---
🧪 Key Learnings
- Built an end-to-end multimodal deep learning system
- Integrated CNN and Transformer-based encoders
- Implemented spatial attention and gated fusion
- Handled large-scale datasets and long GPU training
- Designed a deployable inference pipeline
- Built an interactive Streamlit demo
---
🔮 Future Work

- Replace ResNet50 with Vision Transformer (ViT)
- Increase answer vocabulary size (3k → 5k)
- Add learning-rate scheduling and label smoothing
- Experiment with advanced fusion methods (BAN, MCB)
- Explore multimodal transformers (ViLBERT, LXMERT)
---
📚 References

- VQA v2 Dataset – Agrawal et al.
- Deep Residual Learning for Image Recognition – He et al.
- BERT: Pre-training of Deep Bidirectional Transformers – Devlin et al.
- Making the V in VQA Matter – Goyal et al.
---
🤝 Contributing

- Contributions are welcome!
- Fork the repository
- Create a feature branch
- Commit your changes
- Open a Pull Request
