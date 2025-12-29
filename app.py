
import streamlit as st
from PIL import Image
from inference import predict

st.set_page_config(page_title="VQA App", layout="centered")

st.title("Visual Question Answering")
st.write("Upload an image and ask a question")

image_file = st.file_uploader("Upload Image", type=["jpg", "png"])
question = st.text_input("Ask a question")

if image_file is not None:
    image = Image.open(image_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

if st.button("Get Answer"):
    if image_file is None or question.strip() == "":
        st.warning("Please upload an image and enter a question")
    else:
        with st.spinner("Thinking..."):
            answer = predict(image, question)
        st.success(f"**Answer:** {answer}")
