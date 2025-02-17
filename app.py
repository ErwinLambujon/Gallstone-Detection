import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
from ultralytics import YOLO

@st.cache_resource
def load_model():
    try:
        return YOLO("best.pt")
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        return None

model = load_model()

def render_without_confidence(image, boxes, class_names):
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        class_id = int(box.cls[0])
        label = class_names[class_id]

        color = (0, 255, 0)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            image,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
            cv2.LINE_AA,
        )
    return image

st.title("🩺 **AGDC: Automated Gallstone Detection and Classification** 🩺")
st.markdown("""
    **AGAD** is an **AI-powered tool** that leverages **YOLOv11**, a state-of-the-art deep learning model, to detect and classify gallstones in **gallbladder ultrasound images**.

    Upon uploading an image make sure that it is a gallbladder ultrasound image, the app classifies it into two categories:
    - **"Gallstones Detected"**, with the average detection confidence displayed
    - **"No Gallstones Detected"**

    The results are visually represented with bounding boxes around detected gallstones, and users can download the processed image for further analysis.
""", unsafe_allow_html=True)

with st.expander("Privacy Policy"):
    st.write("""
    **Effective Date:** [Insert Date]
    
    At AGAD, we are committed to protecting your privacy and the confidentiality of your personal information. This Privacy Policy explains how we collect, use, and protect the data you provide when using our automated gallstone detection tool.
    
    ### 1. Information We Collect
    When you upload an image to our platform, we collect the following information:
    - **Uploaded Image**: The ultrasound image you provide for analysis.
    - **Detection Results**: The processed results, including bounding boxes or labels, generated from the image analysis.

    ### 2. How We Use Your Information
    We use the information collected for the following purposes:
    - **Image Processing and Detection**: To process and analyze the ultrasound images for the detection of gallstones using our AI-powered model.
    - **Providing Results**: To display the detection results, including any bounding boxes or annotations, for you to review.
    - **Improving Our Services**: To enhance the accuracy and effectiveness of the detection model.

    ### 3. Data Storage and Retention
    - We **do not store your uploaded images** or detection results permanently. They are processed temporarily during the session and are deleted once the session ends, unless otherwise required by law.
    - Detection results may be stored temporarily for the purpose of providing downloadable images or to improve the detection model, but we will not retain personal or medical data beyond what is necessary for operational purposes.

    ### 4. Security of Your Information
    We take the security of your personal data seriously. We implement appropriate technical and organizational measures to protect your data from unauthorized access, alteration, disclosure, or destruction during transmission and processing.

    ### 5. Data Sharing
    We **do not share your images or personal data** with any third parties, except as required by law or for the operation of the service (e.g., hosting providers, cloud services). We may share aggregate, anonymized data with partners for research or development purposes, but this will not contain any personal or identifiable information.

    ### 6. User Rights
    Depending on your jurisdiction, you may have the right to:
    - **Access**: Request access to the images and results you have uploaded.
    - **Delete**: Request that we delete any data we have processed related to you.

    If you wish to exercise these rights, please contact us at [contact email or phone number].

    ### 7. Third-Party Links
    Our service may contain links to third-party websites or services. We are not responsible for the privacy practices of these third-party sites. Please review the privacy policies of any third-party sites you visit.

    ### 8. Changes to This Privacy Policy
    We may update this Privacy Policy from time to time to reflect changes in our practices or legal requirements. Any updates will be posted on this page with an updated effective date.

    ### 9. Contact Us
    If you have any questions or concerns about this Privacy Policy or how your data is handled, please contact us at:

    - Email: erwin.lambujon@cit.edu 
    - Phone: +63 999 419 5922
    - Address: 084 B. Aranas Extension Cebu City
    """)

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img_array = np.array(image)
    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    # Perform detection
    st.write("Detecting gallstones...")
    with torch.no_grad():
        results = model(img_array)

    if results:
        result = results[0]

        detections = result.boxes
        class_names = result.names

        if len(detections) > 0:
            confidences = detections.conf.cpu().numpy()
            avg_confidence = np.mean(confidences) * 100
            caption = f"Gallstones Detected - Average Confidence: {avg_confidence:.2f}%"

            detected_img = render_without_confidence(img_array.copy(), detections, class_names)

            detected_img = cv2.cvtColor(detected_img, cv2.COLOR_BGR2RGB)

            st.image(detected_img, caption=caption, use_column_width=True)

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
        else:
            detected_img = img_array.copy()
            caption = "No Gallstones Detected"

            detected_img = cv2.cvtColor(detected_img, cv2.COLOR_BGR2RGB)

            st.image(detected_img, caption=caption, use_column_width=True)

    else:
        st.error("No results returned from the model!")

st.markdown("""
### 📝 Provide Your Feedback
We value your insights! Please help us improve **AGDC** by filling out our usability testing form. Click the link below to share your experience:

[**👉 Fill out the form**](https://docs.google.com/forms/d/e/1FAIpQLSfjFAEA6hvM66nl-5KkNL3HyTO2gKarKATFsuuY7GI7paj2WQ/viewform?usp=header)
""", unsafe_allow_html=True)
