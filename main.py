import streamlit as st
from PIL import Image
import torch
from torchvision import models, transforms
import requests

# Load a pre-trained image recognition model
model = models.resnet50(pretrained=True)
model.eval()

# Define the image transformation pipeline
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load ImageNet class labels
LABELS_URL = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
LABELS = requests.get(LABELS_URL).json()

def classify_image(image):
    image = preprocess(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(image)
    # Get top 5 predictions
    probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
    top5_prob, top5_catid = torch.topk(probabilities, 5)
    top5_labels = [LABELS[catid] for catid in top5_catid]
    top5_probabilities = [prob.item() for prob in top5_prob]
    return top5_labels, top5_probabilities

# Streamlit application
st.title("Image Component Identifier")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    labels, probabilities = classify_image(image)
    st.write("Top 5 Predictions:")
    for label, probability in zip(labels, probabilities):
        st.write(f"{label}: {probability:.4f}")
