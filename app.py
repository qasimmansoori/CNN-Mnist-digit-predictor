import streamlit as st
from streamlit_drawable_canvas import st_canvas
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import pandas as pd
import hashlib

st.set_page_config(layout="wide")

class_names = [str(i) for i in range(10)]


model = nn.Sequential(
    nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(64),
    nn.ReLU(),
    nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
    nn.BatchNorm2d(128),
    nn.ReLU(),
    nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
    nn.BatchNorm2d(256),
    nn.ReLU(),
    nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(512),
    nn.ReLU(),
    nn.AdaptiveAvgPool2d((1, 1)),
    nn.Flatten(),
    nn.Dropout(0.5),
    nn.Linear(512, 10)
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.load_state_dict(torch.load("best_model (1).pt", map_location=device))
model.to(device)
model.eval()

st.markdown("<h2 style='text-align:center; font-size:34px;'>🖌️ Handwritten Digit Recognition</h2>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; margin-top:-10px;'>Draw a digit (0-9) and the app will predict in real time.</p>", unsafe_allow_html=True)

if "last_hash" not in st.session_state:
    st.session_state["last_hash"] = None
if "last_prediction" not in st.session_state:
    st.session_state["last_prediction"] = None
if "last_probs" not in st.session_state:
    st.session_state["last_probs"] = None

left_col, right_col = st.columns([1, 1])

with left_col:
    st.subheader("Draw here")
    canvas_result = st_canvas(
        fill_color="white",
        stroke_width=10,
        stroke_color="black",
        background_color="white",
        width=280,
        height=280,
        drawing_mode="freedraw",
        key="canvas",
    )

with right_col:
    pred_placeholder = st.empty()
    chart_placeholder = st.empty()
    if st.session_state["last_prediction"] is None:
        pred_placeholder.info("Prediction and probabilities will update automatically as you draw.")

if canvas_result and canvas_result.image_data is not None:
    img_arr = (255 - canvas_result.image_data[:, :, 0]).astype(np.uint8)
    img_hash = hashlib.md5(img_arr.tobytes()).hexdigest()

    if img_hash != st.session_state["last_hash"]:
        st.session_state["last_hash"] = img_hash

        img = Image.fromarray(img_arr)
        img = img.resize((28, 28)).convert("L")

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        input_tensor = transform(img).unsqueeze(0).to(device)

        with st.spinner("Predicting..."):
            with torch.no_grad():
                output = model(input_tensor)
                probs = torch.softmax(output, dim=1).cpu().numpy().flatten()
                pred_class = int(np.argmax(probs))

        st.session_state["last_prediction"] = pred_class
        st.session_state["last_probs"] = probs

    if st.session_state["last_prediction"] is not None:
        pred_placeholder.subheader(f"🧠 Predicted Digit: **{class_names[st.session_state['last_prediction']]}**")

        prob_df = pd.DataFrame({
            "Digit": class_names,
            "Probability (%)": (st.session_state["last_probs"] * 100)
        }).set_index("Digit").sort_values("Probability (%)", ascending=False)

        chart_placeholder.bar_chart(prob_df["Probability (%)"])
else:
    with left_col:
        st.info("Start drawing on the canvas to get a prediction.")

st.sidebar.header("About")
st.sidebar.write("""
This app uses a custom CNN trained on the MNIST dataset.
It now predicts in real time as you draw and uses a smaller header so controls stay visible.
""")
