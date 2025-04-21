import streamlit as st
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import pickle
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch.nn.functional as F

# === Config ===
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

# === Transformations ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# === Grad-CAM Hooks ===
activations = []
gradients = []
target_layer = model.layer4[1].conv2
target_layer.register_forward_hook(lambda m, i, o: activations.append(o))
target_layer.register_full_backward_hook(lambda m, gi, go: gradients.append(go[0]))

# === Interface Streamlit ===
st.title("🏅 Sports Image Classifier + Grad-CAM")
st.write("Upload une image pour prédire le sport détecté et voir les zones activées.")

uploaded_file = st.file_uploader("📤 Choisissez une image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Image uploadée", use_container_width=True)

    input_tensor = transform(image).unsqueeze(0).to(DEVICE)

    model.eval()
    activations.clear()
    gradients.clear()

    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.nn.functional.softmax(output[0], dim=0)
        top5_probs, top5_indices = torch.topk(probs, 5)

    st.markdown("### 🔍 Prédictions :")
    for i in range(5):
        st.write(f"**{i+1}. {class_names[top5_indices[i]]}** — {top5_probs[i].item()*100:.2f}%")

    st.success("✅ Prédiction effectuée.")

    # === Grad-CAM ===
    # Refaire une passe avec gradients
    output = model(input_tensor)
    pred_class = output.argmax().item()
    model.zero_grad()
    output[0, pred_class].backward()

    grad = gradients[0]
    act = activations[0]
    weights = grad.mean(dim=(2, 3), keepdim=True)
    cam = (weights * act).sum(dim=1).squeeze()
    cam = F.relu(cam)
    cam -= cam.min()
    cam /= cam.max()
    cam = cam.cpu().detach().numpy()
    cam = cv2.resize(cam, (224, 224))

    # Superposition avec image d'origine
    img_np = transform(image).permute(1, 2, 0).numpy() * 0.5 + 0.5
    img_np = np.clip(img_np, 0, 1)
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0
    fusion = np.clip(0.6 * img_np + 0.4 * heatmap, 0, 1)

    # === Affichage dans Streamlit ===
    st.markdown("### 🔥 Carte d’activation Grad-CAM")
    col1, col2, col3 = st.columns(3)
    col1.image(img_np, caption="Image originale", use_container_width=True)
    # Convertir cam en image RGB simulée
    cam_jet = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    cam_jet = cv2.cvtColor(cam_jet, cv2.COLOR_BGR2RGB) / 255.0

    # Puis afficher dans Streamlit
    col2.image(cam_jet, caption="Carte Grad-CAM (JET)", use_container_width=True)

    col3.image(fusion, caption=f"Grad-CAM sur : {class_names[pred_class]}", use_container_width=True) 