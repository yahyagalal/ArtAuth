# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 15:29:58 2024

@author: anaya
"""
import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
import torch.nn as nn
import torchvision.models as models
import numpy as np


def set_custom_css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Pacifico&display=swap');  /* Custom font */

        .stApp {
            background-color: #ffe4e1;  /* Misty Rose background */
        }
        h1, h2, h3, h4, h5, h6, p, div, label, span {
            color: #800020 !important;  /* Burgundy text */
            font-family: 'Pacifico', cursive !important;  /* Custom font for text */
        }
        .stButton>button {
            background-color: #ff69b4 !important;  /* Hot Pink for buttons */
            color: white !important;  /* White text for buttons */
            border-radius: 12px !important;  /* Rounded corners */
            font-family: 'Pacifico', cursive !important;  /* Custom font for buttons */
        }
        .stSelectbox>div, .stTextInput>div {
            background-color: #ffc0cb !important;  /* Pink for select boxes and text inputs */
            color: #800020 !important;  /* Burgundy text for select boxes and text inputs */
            border-radius: 8px !important;  /* Rounded corners */
            font-family: 'Pacifico', cursive !important;  /* Custom font */
        }
        .stFileUploader>div {
            background-color: #ffb6c1 !important;  /* Light Pink for file uploader */
            color: #800020 !important;  /* Burgundy text for file uploader */
            border-radius: 8px !important;  /* Rounded corners */
            font-family: 'Pacifico', cursive !important;  /* Custom font */
        }
        header .css-1gpf7j8 {
            background-color: #800020 !important;  /* Burgundy for header */
            color: white !important;  /* White text for header */
            font-family: 'Pacifico', cursive !important;  /* Custom font */
        }
        .css-1d391kg {
            background-color: #800020 !important;  /* Burgundy for sidebar */
            color: white !important;  /* White text for sidebar */
            font-family: 'Pacifico', cursive !important;  /* Custom font */
        }
        </style>
        """,
        unsafe_allow_html=True
    )

# Example usage
set_custom_css()


st.title("ArtAuth: Check whether a piece of art is AI made or Human Made")
st.write("Project by Yahia Galal")

# Load the fine-tuned AlexNet model
def load_finetuned_alexnet():
    model = models.alexnet(pretrained=False)
    model.classifier = nn.Sequential(
        nn.Linear(9216, 1024),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(1024, 2),
        nn.LogSoftmax(dim=1)
    )
    model.load_state_dict(torch.load('ArtAuthAlexNetModel.pt', map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_finetuned_alexnet()

#st.write(model)
#st.write("Test")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    st.image(image, caption='Uploaded Image.', use_column_width=True)

    # Apply transformations
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension

    # Inverse normalize for visualization
    inv_normalize = transforms.Normalize(
        mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
        std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
    )

    im = inv_normalize(image_tensor.squeeze()).permute(1, 2, 0).numpy()
    im = np.clip(im, 0, 1)

   #st.write("Classifying...")
    
    with torch.no_grad():
        prediction = model(image_tensor).argmax(dim=1)

    prediction = prediction.item()
    if prediction==0:
        prediction="AI Made"
    else:
        prediction="Human Made"
    
    st.markdown(f"<p style='font-size:48px; color:#800020; font-family:Pacifico;'>Prediction: {prediction}</p>", unsafe_allow_html=True)

    # Optionally, display the image after inverse normalization (to verify)
 #  st.image(im, caption='Processed Image', use_column_width=True)
else:
    st.write("Please upload an image file.")
