import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
from io import BytesIO

# Load YOLO model
@st.cache_resource
def load_model():
    model = torch.load('best.pt', map_location='cpu')  # Ensure 'best.pt' is in the same folder
    model.eval()  # Set the model to evaluation mode
    return model

model = load_model()

# Streamlit UI
st.title("Gallstone Detection App")
st.write("Upload an image, and the app will detect gallstones.")

# Upload an image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Convert PIL image to OpenCV format
    img_array = np.array(image)
    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    # Perform detection
    st.write("Detecting gallstones...")
    with torch.no_grad():  # Disable gradient computation for inference
        results = model(img_array)  # Run YOLO detection

    # Convert results to an image
    detected_img = np.squeeze(results.render())  # Render results on the image
    detected_img = cv2.cvtColor(detected_img, cv2.COLOR_BGR2RGB)  # Convert to RGB for display

    # Display detected image
    st.image(detected_img, caption="Detection Results", use_column_width=True)

    # Allow user to download the resulting image
    st.write("Download the detection result:")
    detected_pil = Image.fromarray(detected_img)
    buf = BytesIO()
    detected_pil.save(buf, format="JPEG")
    byte_im = buf.getvalue()
    st.download_button(
        label="Download Image",
        data=byte_im,
        file_name="detection_result.jpg",
        mime="image/jpeg",
    )
