import streamlit as st
from streamlit_drawable_canvas import st_canvas
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import pandas as pd

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


st.title("🖌️ Handwritten Digit Recognition")
st.write("Draw a digit (0-9) below and click **Predict** to see what the model thinks!")

# Create a canvas for drawing
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


if st.button("Predict"):
    if canvas_result.image_data is not None:
        
        img = Image.fromarray((255 - canvas_result.image_data[:, :, 0]).astype(np.uint8))
        img = img.resize((28, 28)).convert("L")

        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        input_tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(input_tensor)
            probs = torch.softmax(output, dim=1).cpu().numpy().flatten()
            pred_class = np.argmax(probs)

        st.subheader(f"🧠 Predicted Digit: **{class_names[pred_class]}**")

        
        prob_df = pd.DataFrame({
            "Digit": class_names,
            "Probability (%)": (probs * 100)
        }).set_index("Digit").sort_values("Probability (%)", ascending=False)
        st.write("Prediction probabilities")
        st.bar_chart(prob_df["Probability (%)"]) 
    else:
        st.warning("Please draw something first!")


st.sidebar.header("About")
st.sidebar.write("""
This app uses a custom CNN trained on the MNIST dataset.
You can draw digits 0-9, and the model will predict what you wrote!
""")
