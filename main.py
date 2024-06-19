import streamlit as st
from PIL import Image
import torch
from torchvision import models, transforms
import requests

# Loading PyTorch's Pretrained resnet50
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
model.eval()

# transforming the image to pass it to the model
preprocess = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# Loading the classes labels  to get the results
LABELS_URL = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
LABELS = requests.get(LABELS_URL).json()


def classify_image(image):
    image = preprocess(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(image)
    probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
    top5_prob, top5_labels = torch.topk(probabilities, 5)
    top5_labels = [LABELS[label] for label in top5_labels]
    top5_probabilities = [prob.item() for prob in top5_prob]
    return top5_labels, top5_probabilities


st.title("Image Component Identifier")
uploaded_file = st.file_uploader(
    "Choose an image...", type=["png", "jpg", "webp", "jpeg"]
)
submit = st.button("Analyse Image")
if submit:
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        st.write("")
        wait=st.empty()
        wait.write("Classifying...")
        labels, probabilities = classify_image(image)
        wait.empty()
        st.write("This photo has:")
        for label, probability in zip(labels, probabilities):
            st.write(f"{label}: {probability*100:.2f}%")
    else :
        st.write("Please upload a Photo!")

