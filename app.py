import os
import tempfile
import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import gdown

# ---------------- Page setup ----------------
st.set_page_config(
    page_title="AC & Fan Classifier",
    page_icon="❄️",
    layout="centered",
)

st.title("❄️ AC & Fan Image Classifier")
st.write(
    "Upload a photo of an air conditioner (AC) or fan. "
    "This model was trained on images collected at our college."
)

# ---------------- Model configuration ----------------
# 1. Upload ac_fan_cnn.pth to Google Drive.
# 2. Set its sharing permission to: Anyone with the link.
# 3. Copy the file ID from the share URL and paste it below.
# Example URL: https://drive.google.com/file/d/FILE_ID/view?usp=sharing
DRIVE_FILE_ID = "1TZPyf1r5XXrxaRdSCQUnaeWhCU7CLw3h"
MODEL_PATH = os.path.join(tempfile.gettempdir(), "ac_fan_cnn.pth")


class SimpleCNN(nn.Module):
    """CNN architecture matching the supplied Google Colab notebook."""
    def __init__(self, num_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 62 * 62, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


@st.cache_resource
def load_model_and_classes(drive_file_id):
    if not drive_file_id or drive_file_id == "PASTE_YOUR_GOOGLE_DRIVE_FILE_ID_HERE":
        raise ValueError(
            "Google Drive file ID is not set. Open app.py and replace "
            "DRIVE_FILE_ID with the ID of your shared ac_fan_cnn.pth file."
        )

    if not os.path.exists(MODEL_PATH):
        url = f"https://drive.google.com/uc?id={drive_file_id}"
        downloaded = gdown.download(url, MODEL_PATH, quiet=False)
        if not downloaded or not os.path.exists(MODEL_PATH):
            raise RuntimeError(
                "Could not download the model. Check the Google Drive file ID "
                "and make sure sharing is set to 'Anyone with the link'."
            )

    checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    classes = checkpoint["classes"]
    model = SimpleCNN(num_classes=len(classes))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, classes


image_transform = transforms.Compose([
    transforms.Resize((250, 250)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png", "bmp", "webp"],
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded image", use_container_width=True)

        if st.button("Predict", type="primary", use_container_width=True):
            with st.spinner("Loading model and predicting..."):
                model, classes = load_model_and_classes(DRIVE_FILE_ID)
                input_tensor = image_transform(image).unsqueeze(0)

                with torch.no_grad():
                    logits = model(input_tensor)
                    probabilities = torch.softmax(logits, dim=1)[0]
                    predicted_index = int(torch.argmax(probabilities).item())
                    confidence = float(probabilities[predicted_index].item())

            st.success(f"Prediction: **{classes[predicted_index]}**")
            st.metric("Model confidence", f"{confidence * 100:.1f}%")
            st.caption(
                "Confidence is the model's estimated probability, not a guarantee "
                "that the prediction is correct."
            )
    except Exception as exc:
        st.error(f"Could not make a prediction: {exc}")
else:
    st.info("Choose an image above to get started.")

st.divider()
st.caption("College-collected AC and Fan dataset • CNN image classification")
