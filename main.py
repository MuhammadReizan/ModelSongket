from flask import Flask, request, jsonify
from flask_cors import CORS  # Untuk mengaktifkan CORS
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from io import BytesIO
import numpy as np
import os

# Inisialisasi Flask app
app = Flask(__name__)

# Aktifkan CORS agar API bisa diakses dari berbagai domain
CORS(app)

# Memuat model ResNet50
MODEL_PATH = "songket_brand_clf_resnet50.h5"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file '{MODEL_PATH}' not found.")

model = load_model(MODEL_PATH)

# Label yang digunakan sesuai dengan model yang dilatih
LABELS = ['Bintang Berante', 'Nago Besaung', 'Nampan Perak', 'Pulir']

# Fungsi preprocessing gambar
def prepare_image(image, target_size=(224, 224)):
    # Ubah ukuran gambar menjadi 224x224 (sesuai dengan ResNet50)
    image = image.resize(target_size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    return image

# Endpoint untuk prediksi
@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No image file provided."}), 400

    # Ambil file gambar dari request
    image_file = request.files["file"]
    try:
        # Konversi file menjadi BytesIO untuk kompatibilitas dengan load_img
        image = load_img(BytesIO(image_file.read()))
    except Exception as e:
        return jsonify({"error": f"Invalid image format. {str(e)}"}), 400

    # Preprocess gambar
    processed_image = prepare_image(image)

    # Prediksi menggunakan model
    predictions = model.predict(processed_image)[0]
    predicted_class = int(np.argmax(predictions))  # Pastikan tipe Python int

    # Konversi prediksi menjadi persentase
    predictions_percent = (predictions / np.sum(predictions)) * 100

    # Kembalikan hasil prediksi
    response = {
        "predictions": {LABELS[i]: f"{predictions_percent[i]:.2f}%" for i in range(len(LABELS))},
        "predicted_class": predicted_class,
        "predicted_label": LABELS[predicted_class]
    }
    return jsonify(response)

# Menjalankan Flask app
if __name__ == "__main__":
    app.run(debug=True)
