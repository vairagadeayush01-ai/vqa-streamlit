🖼️ Visual Question Answering (VQA) Model

Python · PyTorch · ResNet50 · BERT

A deep learning–based Visual Question Answering (VQA) system that combines visual and textual understanding to answer natural-language questions about images.
The model is built using PyTorch, leveraging ResNet50 for image feature extraction, BERT for question encoding, and a gated fusion mechanism with spatial attention for multimodal reasoning.

📊 Model Performance
Dataset	Hard Accuracy
VQA v2 Validation	~44.9%

Hard Accuracy: Exact match with the most frequent ground-truth answer
(Top-3000 answer classification setting)

🏗️ Architecture Overview
┌─────────────────┐     ┌─────────────────┐
│   Input Image   │     │  Input Question │
└────────┬────────┘     └────────┬────────┘
         │                       │
         ▼                       ▼
┌─────────────────┐     ┌─────────────────┐
│   ResNet50      │     │  BERT Encoder   │
│ (Image Encoder) │     │ (Text Encoder)  │
└────────┬────────┘     └────────┬────────┘
         │                       │
         └───────────┬───────────┘
                     │
              ┌──────▼──────┐
              │  Spatial    │
              │  Attention  │
              └──────┬──────┘
                     │
              ┌──────▼──────┐
              │   Gated     │
              │   Fusion    │
              └──────┬──────┘
                     │
              ┌──────▼──────┐
              │ Classifier  │
              │  (FC Layer) │
              └──────┬──────┘
                     │
                     ▼
              ┌─────────────┐
              │   Answer    │
              └─────────────┘


Key Components
Component	Description
Image Encoder	ResNet50, pretrained on ImageNet
Text Encoder	BERT-base-uncased
Attention	Spatial attention over image features
Fusion	Gated fusion combining image & text features
Classifier	Fully connected layer over top-3000 answers

VQA-ResNet50-BERT/
├── data/
│   ├── vqa_dataset.py        # Dataset loader
│   └── answer_vocab.json     # Answer → index mapping
├── models/
│   ├── vqa_model.py          # Main VQA model
│   ├── encoders.py           # Image & text encoders
│   ├── attention.py          # Spatial attention module
│   └── fusion.py             # Gated fusion module
├── training/
│   ├── train.py              # Training script
│   └── evaluate.py           # Validation evaluation
├── inference/
│   └── predict.py            # Inference on custom images
├── checkpoints/
│   ├── vqa_checkpoint.pth    # Training checkpoints
│   └── vqa_final_model.pth   # Final trained model
├── notebooks/
│   ├── training.ipynb
│   └── inference.ipynb
├── requirements.txt
└── README.md

VQA-ResNet50-BERT/
├── data/
│   ├── vqa_dataset.py        # Dataset loader
│   └── answer_vocab.json     # Answer → index mapping
├── models/
│   ├── vqa_model.py          # Main VQA model
│   ├── encoders.py           # Image & text encoders
│   ├── attention.py          # Spatial attention module
│   └── fusion.py             # Gated fusion module
├── training/
│   ├── train.py              # Training script
│   └── evaluate.py           # Validation evaluation
├── inference/
│   └── predict.py            # Inference on custom images
├── checkpoints/
│   ├── vqa_checkpoint.pth    # Training checkpoints
│   └── vqa_final_model.pth   # Final trained model
├── notebooks/
│   ├── training.ipynb
│   └── inference.ipynb
├── requirements.txt
└── README.md

🚀 Getting Started
Prerequisites
->Python 3.9+
->PyTorch
->CUDA-enabled GPU (recommended)
->16GB+ RAM

Installation

  git clone https://github.com/your-username/VQA-ResNet50-BERT.git
  cd VQA-ResNet50-BERT
  pip install -r requirements.txt
  
  📦 Dataset
  This project uses VQA v2.0 (COCO 2014).
  
  Required data:
  ->COCO 2014 train & validation images
  ->VQA train & validation questions
  ->VQA train & validation annotations
  Expected structure:
    dataset/
    ├── images/
    │   ├── train2014/
    │   └── val2014/
    ├── questions/
    │   ├── train_questions.json
    │   └── val_questions.json
    └── annotations/
        ├── train_annotations.json
        └── val_annotations.json

Training
  ->Build Answer Vocabulary
  ->Train Model
  ->Training Strategy
    💠Freeze ResNet50 + BERT for initial epochs
    💠Unfreeze for fine-tuning
    💠Checkpoint saving after each epoch
    💠Resume-safe training
      
