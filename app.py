import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json

# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="Plant Disease Detection AI",
    page_icon="🌿",
    layout="centered"
)

# ==========================================
# LOAD MODEL
# ==========================================

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(
        "plant_disease_efficientnet.keras"
    )
    return model

model = load_model()

# ==========================================
# LOAD CLASS NAMES
# ==========================================

with open("class_names.json", "r") as f:
    class_names = json.load(f)

# ==========================================
# DISEASE INFORMATION
# ==========================================

disease_info = {

    "Apple_scab": {
        "description": "Apple Scab is one of the most common diseases in apple trees. It is caused by a fungus that attacks leaves, fruits, and young shoots.",
        "symptoms": "Olive-green or dark brown spots appear on leaves. Fruits may develop rough, cracked areas and become misshapen.",
        "causes": "The disease spreads in cool and wet weather through fungal spores.",
        "treatment": "Remove infected leaves and fruits. Spray recommended fungicides during the growing season.",
        "prevention": "Keep the orchard clean, prune trees regularly, and avoid excess moisture around plants."
    },

    "Apple_Black_rot": {
        "description": "Black Rot is a fungal disease that damages apple leaves, branches, and fruits.",
        "symptoms": "Brown circular spots on leaves, dark rotten patches on fruits, and cankers on branches.",
        "causes": "The fungus survives in dead branches and infected fruits.",
        "treatment": "Prune infected branches and apply fungicides.",
        "prevention": "Remove fallen fruits and maintain orchard sanitation."
    },

    "Apple_healthy": {
        "description": "The apple leaf appears healthy and free from disease.",
        "symptoms": "Bright green leaves with no visible spots or damage.",
        "causes": "Proper plant growth and disease-free conditions.",
        "treatment": "No treatment required.",
        "prevention": "Continue good farming practices and regular monitoring."
    },

    "Corn_Common_rust": {
        "description": "Common Rust is a fungal disease that frequently affects maize crops.",
        "symptoms": "Small reddish-brown or orange pustules appear on both sides of leaves.",
        "causes": "Fungal spores spread through wind and humid weather conditions.",
        "treatment": "Use fungicides when infection becomes severe.",
        "prevention": "Plant resistant varieties and practice crop rotation."
    },

    "Corn_Leaf_Blight": {
        "description": "Leaf Blight is a fungal disease that reduces photosynthesis and crop yield.",
        "symptoms": "Long gray-green or brown lesions on leaves that gradually enlarge.",
        "causes": "Warm temperatures and high humidity favor disease development.",
        "treatment": "Apply recommended fungicides and remove infected plant debris.",
        "prevention": "Use resistant hybrids and avoid planting corn repeatedly in the same field."
    },

    "Corn_healthy": {
        "description": "The corn leaf is healthy and shows normal growth.",
        "symptoms": "Green leaves without lesions, spots, or discoloration.",
        "causes": "Healthy growing conditions.",
        "treatment": "No treatment needed.",
        "prevention": "Maintain proper irrigation and fertilization."
    },

    "Grape_Black_rot": {
        "description": "Black Rot is a serious fungal disease that affects grape leaves and fruits.",
        "symptoms": "Brown spots with black dots on leaves and shriveled black grapes.",
        "causes": "Fungal spores spread through rain and infected plant material.",
        "treatment": "Apply fungicides and remove infected grapes.",
        "prevention": "Prune vines regularly and keep the vineyard clean."
    },

    "Grape_Leaf_blight": {
        "description": "Leaf Blight causes damage to grape leaves and reduces plant productivity.",
        "symptoms": "Brown patches and drying of leaf tissues.",
        "causes": "Fungal infection encouraged by moist weather.",
        "treatment": "Use suitable fungicides and remove affected leaves.",
        "prevention": "Provide good air circulation and avoid overcrowding."
    },

    "Grape_healthy": {
        "description": "Healthy grape leaf with no signs of disease.",
        "symptoms": "Uniform green color and healthy leaf structure.",
        "causes": "Good plant health and proper management.",
        "treatment": "No treatment required.",
        "prevention": "Continue routine monitoring."
    },

    "Strawberry_Leaf_scorch": {
        "description": "Leaf Scorch is a fungal disease that affects strawberry plants.",
        "symptoms": "Purple spots develop on leaves, and leaf edges appear burnt or scorched.",
        "causes": "Fungal spores spread through water splashes and humid conditions.",
        "treatment": "Remove infected leaves and apply fungicides if necessary.",
        "prevention": "Avoid overhead watering and maintain field cleanliness."
    },

    "Strawberry_healthy": {
        "description": "Healthy strawberry leaf without disease symptoms.",
        "symptoms": "Fresh green leaves with no discoloration.",
        "causes": "Healthy plant growth.",
        "treatment": "No treatment needed.",
        "prevention": "Continue good cultivation practices."
    },

    "Tomato_Early_blight": {
        "description": "Early Blight is a common fungal disease that affects tomato leaves, stems, and fruits.",
        "symptoms": "Brown spots with concentric rings, often called a target-board pattern.",
        "causes": "The fungus survives in soil and crop debris.",
        "treatment": "Remove infected leaves and apply fungicides.",
        "prevention": "Practice crop rotation and avoid wetting the leaves."
    },

    "Tomato_Late_blight": {
        "description": "Late Blight is a highly destructive disease that can rapidly destroy tomato crops.",
        "symptoms": "Dark water-soaked lesions on leaves and stems, followed by rapid wilting.",
        "causes": "Cool and humid weather encourages the pathogen.",
        "treatment": "Apply copper-based fungicides immediately and remove infected plants.",
        "prevention": "Use resistant varieties and avoid excess moisture."
    },

    "Tomato_healthy": {
        "description": "The tomato leaf is healthy and disease-free.",
        "symptoms": "Uniform green leaves without spots or damage.",
        "causes": "Healthy plant growth conditions.",
        "treatment": "No treatment required.",
        "prevention": "Continue proper irrigation, fertilization, and monitoring."
    }
}

# ==========================================
# HEADER
# ==========================================

st.title("🌿 Plant Disease Detection System")

st.markdown("""
### AI-Based Plant Disease Detection Using EfficientNetB0

Upload a plant leaf image and the model will:
- Identify the disease
- Show confidence score
- Display disease information
- Suggest treatment and prevention methods
""")

# ==========================================
# FILE UPLOAD
# ==========================================

uploaded_file = st.file_uploader(
    "Upload Leaf Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Leaf Image",
        use_container_width=True
    )

    img = image.resize((224, 224))

    img_array = np.array(img)

    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    prediction = model.predict(
        img_array,
        verbose=0
    )

    pred_idx = np.argmax(prediction)

    disease = class_names[pred_idx]

    confidence = np.max(prediction) * 100

    st.success(f"Prediction: {disease}")

    st.info(f"Confidence Score: {confidence:.2f}%")

    st.progress(float(confidence / 100))

    if "healthy" in disease.lower():
        st.success("✅ Plant is Healthy")
    else:
        st.warning("⚠️ Disease Detected")

    st.subheader("Top 3 Predictions")

    top3 = np.argsort(
        prediction[0]
    )[-3:][::-1]

    for idx in top3:
        st.write(
            f"**{class_names[idx]}** : {prediction[0][idx] * 100:.2f}%"
        )

    if disease in disease_info:

        info = disease_info[disease]

        st.subheader("📖 Disease Information")

        st.markdown("### Description")
        st.write(info["description"])

        st.markdown("### Symptoms")
        st.write(info["symptoms"])

        st.markdown("### Causes")
        st.write(info["causes"])

        st.markdown("### Treatment")
        st.write(info["treatment"])

        st.markdown("### Prevention")
        st.write(info["prevention"])

st.markdown("---")
st.caption(
    "Plant Disease Detection using EfficientNetB0 | Final Year Major Project"
)