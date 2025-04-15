import streamlit as st
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import pickle
import os

# ==== Config ====
MODEL_PATH = "resnet_model.pth"
CLASSES_PATH = "class_names.pkl"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_model_and_classes():
    with open(CLASSES_PATH, "rb") as f:
        class_names = pickle.load(f)
    
    model = models.resnet18(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, len(class_names))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    return model.to(DEVICE), class_names

model, class_names = load_model_and_classes()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

st.title(" Sports Image Classifier")
st.write("Upload une image pour prédire le sport détecté.")

uploaded_file = st.file_uploader("📤 Choisissez une image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Image uploadée", use_container_width=True)

    input_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.nn.functional.softmax(output[0], dim=0)
        top5_probs, top5_indices = torch.topk(probs, 5)

    st.markdown("### 🔍 Prédictions :")
    for i in range(5):
        st.write(f"**{i+1}. {class_names[top5_indices[i]]}** — {top5_probs[i].item()*100:.2f}%")

    st.success("✅ Prédiction effectuée.")
