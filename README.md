# Visual Question Answering (VQA)

A Visual Question Answering system built using:
- ResNet50 for image feature extraction
- BERT for question encoding
- Gated multimodal fusion
- Streamlit for deployment

## Features
- Upload image + ask natural language question
- Predicts best answer from top-3000 answers
- CPU-only inference (deployment friendly)

## Dataset
VQA v2

## Run Locally
pip install -r requirements.txt  
streamlit run app.py

## Deployment
Deployed on Streamlit Cloud with GitHub CI/CD
