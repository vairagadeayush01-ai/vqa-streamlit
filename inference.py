import torch
import requests
import streamlit as st
from PIL import Image
from torchvision import transforms
from transformers import BertTokenizer

from model import VQAModel

DEVICE = torch.device("cpu")


@st.cache_resource
def load_everything():
    checkpoint = torch.load(
        "checkpoints/vqa_final_model.pth",
        map_location="cpu"
    )

    idx2ans = checkpoint["idx2ans"]
    max_question_len = checkpoint["max_question_len"]

    model = VQAModel(num_answers=len(idx2ans))
    model.load_state_dict(checkpoint["model_state"])
    model.to(DEVICE)
    model.eval()

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    return model, tokenizer, idx2ans, transform, max_question_len


def load_image(image_input):
    if isinstance(image_input, str):  # URL-based
        image = Image.open(requests.get(image_input, stream=True).raw)
    else:  # Streamlit upload (PIL Image)
        image = image_input
    return image.convert("RGB")


def predict(image_input, question):
    model, tokenizer, idx2ans, transform, max_len = load_everything()

    image = transform(load_image(image_input)).unsqueeze(0).to(DEVICE)

    encoding = tokenizer(
        question,
        padding="max_length",
        truncation=True,
        max_length=max_len,
        return_tensors="pt"
    )

    input_ids = encoding["input_ids"].to(DEVICE)
    attention_mask = encoding["attention_mask"].to(DEVICE)

    with torch.no_grad():
        logits = model(image, input_ids, attention_mask)
        pred_idx = torch.argmax(logits, dim=1).item()

    return idx2ans[pred_idx]

