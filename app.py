import os
import streamlit as st
import torch
import torch.nn as nn
from torchvision.models import resnet18
from torchvision import transforms
from PIL import Image, ImageOps

class EnhancedResNet18(nn.Module):
    def __init__(self):
        super(EnhancedResNet18, self).__init__()
        self.resnet = resnet18(pretrained=True) 
        # Update the classifier for two classes (Not Tumor and Tumor)
        self.resnet.fc = nn.Sequential(
            nn.Linear(self.resnet.fc.in_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(256, 2)
        )

    def forward(self, x):
        return self.resnet(x)

MODEL_PATH = "enhanced_model.pth"
device = torch.device("cpu")
model = EnhancedResNet18()

if not os.path.exists(MODEL_PATH):
    st.error(f"Model file not found: {MODEL_PATH}. Please ensure the file is present.")
    st.stop()
    
checkpoint = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(checkpoint, strict=False)
model.to(device)
model.eval()

preprocess = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)), 
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), 
])

class_labels = {0: "Not Tumor", 1: "Tumor"}

st.title("Medical Brain Tumor Classifier")
st.write("Upload a random image")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    grayscale_image = ImageOps.grayscale(image)
    st.image(grayscale_image, caption="Uploaded Image (Converted to Grayscale)", use_column_width=True)
    
    input_tensor = preprocess(grayscale_image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
    
    st.write("Prediction Results (Probability Scores):")
    for idx, prob in enumerate(probabilities):
        st.write(f"Class {idx} ({class_labels[idx]}): {prob:.4f}")
    
    predicted_class = torch.argmax(probabilities).item()
    st.write(f"### **Predicted Label: {class_labels[predicted_class]}**")
