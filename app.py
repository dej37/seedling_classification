import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import streamlit as st

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 12)

model.load_state_dict(torch.load("model_weights.pth", map_location = "cpu",weights_only = True))
model.eval()


uploaded_file = st.file_uploader(
    "upload image here !",
    type = ["jpg" , "jpeg" , "png"]
)

transform = transforms.Compose([
        transforms.Resize((128,128)),
        transforms.ToTensor(),
])


class_names = {
    0: "Black-grass",
    1: "Charlock",
    2: "Cleavers",
    3: "Common Chickweed",
    4: "Common wheat",
    5: "Fat Hen",
    6: "Loose Silky-bent",
    7: "Maize",
    8: "Scentless Mayweed",
    9: "Shepherds Purse",
    10: "Small-flowered Cranesbill",
    11: "Sugar beet"
}

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded seedling", use_container_width=True)

    image_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.softmax(output, dim=1)
        confidence, pred = torch.max(probabilities, 1)

    predicted_class = pred.item()
    confidence_pct = confidence.item() * 100

    st.success(f"Predicted species: {class_names[predicted_class]}")
    st.write(f"Confidence: {confidence_pct:.2f}%")
