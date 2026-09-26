import json
import numpy as np
from PIL import Image
import gradio as gr
import tensorflow as tf
import spaces

# Load the trained model
model = tf.keras.models.load_model("carboniq_model.h5")

# Load labels mapping
with open("labels.json", "r") as f:
    labels_data = json.load(f)

# Handles both dictionary and list label structures
LABELS = labels_data["classes"] if isinstance(labels_data, dict) and "classes" in labels_data else labels_data

@spaces.GPU
def predict(image):
    if image is None:
        return {}
    
    # Resize and match the 1.0/255 training normalization
    img = Image.fromarray(image).convert("RGB").resize((224, 224))
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Run inference
    predictions = model.predict(img_array)[0]
    
    # Return dictionary of class -> confidence for Gradio
    return {LABELS[i]: float(predictions[i]) for i in range(len(LABELS))}

# Build the Gradio UI
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="numpy", label="Upload Food Image"),
    outputs=gr.Label(num_top_classes=3, label="Top Predictions"),
    title="CarbonIQ — Indian Food Classifier",
    description="Custom MobileNetV2 fine-tuned on 20 Indian food classes."
)

if __name__ == "__main__":
    demo.launch(ssr_mode=False)
