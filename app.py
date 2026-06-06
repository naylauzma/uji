import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import tempfile
import zipfile
import os

st.set_page_config(
    page_title="Klasifikasi Gambar",
    page_icon="🖼️",
    layout="centered"
)

st.title("Klasifikasi Gambar Menggunakan Model ZIP")

# Upload ZIP model
zip_file = st.file_uploader(
    "Upload File ZIP Model",
    type=["zip"]
)

# Upload gambar
image_file = st.file_uploader(
    "Upload Gambar Uji",
    type=["jpg", "jpeg", "png"]
)

def extract_model(zip_path, extract_dir):
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_dir)

    model_path = None

    for root, dirs, files in os.walk(extract_dir):
        for file in files:
            if file.endswith(".keras") or file.endswith(".h5"):
                model_path = os.path.join(root, file)
                return model_path

    return None

if zip_file is not None and image_file is not None:

    try:
        # Simpan ZIP sementara
        with tempfile.TemporaryDirectory() as temp_dir:

            zip_path = os.path.join(temp_dir, "model.zip")

            with open(zip_path, "wb") as f:
                f.write(zip_file.getbuffer())

            # Ekstrak model
            model_path = extract_model(zip_path, temp_dir)

            if model_path is None:
                st.error(
                    "Tidak ditemukan file model .keras atau .h5 di dalam ZIP"
                )
                st.stop()

            # Load model
            model = tf.keras.models.load_model(model_path)

            # Buka gambar
            image = Image.open(image_file).convert("RGB")

            st.image(
                image,
                caption="Gambar Uji",
                use_container_width=True
            )

            # Ambil ukuran input model
            input_shape = model.input_shape

            img_height = input_shape[1]
            img_width = input_shape[2]

            # Preprocessing
            img = image.resize((img_width, img_height))

            img_array = np.array(img) / 255.0

            img_array = np.expand_dims(
                img_array,
                axis=0
            )

            # Prediksi
            prediction = model.predict(img_array)

            st.subheader("Hasil Prediksi")

            if prediction.shape[1] == 1:

                score = float(prediction[0][0])

                if score > 0.5:
                    st.success("Kelas 1")
                else:
                    st.success("Kelas 0")

                st.write(f"Skor Prediksi: {score:.4f}")

            else:

                predicted_class = np.argmax(prediction)

                st.success(
                    f"Kelas Prediksi: {predicted_class}"
                )

                st.write(
                    "Probabilitas:"
                )

                st.write(prediction[0])

    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")