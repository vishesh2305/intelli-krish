from flask import Flask, request, jsonify, render_template, Response
import tensorflow as tf
import numpy as np
from PIL import Image
import ollama

app = Flask(__name__)

# Load trained model
model = tf.keras.models.load_model("crop_disease_model_final.keras")

# Class labels
class_names = [
    "Wheat_Aphid", "Wheat_BlackRust", "Wheat_Blast", "Wheat_BrownRust",
    "Wheat_CommonRootRot", "Wheat_FusariumHeadBlight", "Wheat_Healthy",
    "Wheat_LeafBlight", "Wheat_Mildew", "Wheat_Mite", "Wheat_Septoria",
    "Wheat_Smut", "Wheat_Stemfly", "Wheat_Tanspot", "Wheat_YellowRust",
    "Rice_BrownSpot", "Rice_Healthy", "Rice_Hispa", "Rice_LeafBlast",
    "Potato___Early_Blight", "Potato___Healthy", "Potato___Late_Blight",
    "Corn___Common_Rust", "Corn___Gray_Leaf_Spot", "Corn___Healthy",
    "Corn___Northern_Leaf_Blight"
]

@app.route('/')
def index():
    return render_template("index.html")

# Generator to stream Ollama response
def stream_ollama_response(prompt):
    response = ollama.chat(
        model='llama3.1:latest',
        messages=[{"role": "user", "content": prompt}],
        stream=True
    )
    for chunk in response:
        yield chunk['message']['content']

@app.route('/stream', methods=['POST'])
def stream():
    file = request.files['file']
    user_prompt = request.form.get("prompt", "").strip()

    # Preprocess image
    image = Image.open(file).convert("RGB").resize((224, 224))
    image_array = np.array(image) / 255.0
    image_tensor = np.expand_dims(image_array, axis=0)

    predictions = model.predict(image_tensor)[0]
    top_index = int(np.argmax(predictions))
    confidence = float(predictions[top_index])
    full_class = class_names[top_index]

    if "___" in full_class:
        crop, disease = full_class.split("___")
    elif "_" in full_class:
        crop, disease = full_class.split("_", 1)
    else:
        crop = "Unknown"
        disease = full_class

    # 🔥 Print to terminal
    print(f"📸 Prediction: {full_class} | Crop: {crop} | Disease: {disease} | Confidence: {confidence:.2%}")


    prompt = (
        f"You are an expert plant doctor. The identified crop is '{crop}' and the disease is '{disease}'.\n\n"
        "1. Briefly explain what this disease is and how it affects the crop.\n"
        "2. Provide both organic and chemical treatment options.\n"
        "3. Share a recovery plan to reduce the impact on crop yield.\n"
        "4. Suggest a long-term farming strategy to avoid this and other similar diseases.\n"
    )

    if user_prompt:
        prompt += f"\n\nAdditional user question: {user_prompt}"

    def generate():
        yield f"event: meta\ndata: {crop}|{disease}|{confidence:.2%}\n\n"
        for chunk in stream_ollama_response(prompt):
            yield f"data: {chunk}\n\n"

    return Response(generate(), mimetype='text/event-stream')

if __name__ == '__main__':
    app.run(debug=True)
