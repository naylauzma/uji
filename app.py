import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import tempfile

st.title("Klasifikasi Gambar dengan CNN")

# Upload model
model_file = st.file_uploader(
    "Upload Model (.keras)",
    type=["keras"]
)

# Upload gambar
image_file = st.file_uploader(
    "Upload Gambar",
    type=["jpg", "jpeg", "png"]
)

# Nama kelas
class_names = [
    "Kelinci",
    "Lumba-Lumba"
]

if model_file is not None:

    try:

        # Simpan model sementara
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".keras"
        ) as tmp:

            tmp.write(model_file.read())
            model_path = tmp.name

        # Load model
        model = tf.keras.models.load_model(
            model_path,
            compile=False,
            safe_mode=False
        )
        except Exception as e:
            st.error(f"Gagal memuat model: {e}")
        

        st.success("Model berhasil dimuat")

        if image_file is not None:

            # Tampilkan gambar
            img = Image.open(image_file).convert("RGB")

            st.image(
                img,
                caption="Gambar Uji",
                use_container_width=True
            )

            # Ambil ukuran input model
            _, h, w, _ = model.input_shape

            # Resize gambar
            img = img.resize((w, h))

            # Preprocessing
            img_array = np.array(img)

            img_array = (
                img_array.astype("float32")
                / 255.0
            )

            img_array = np.expand_dims(
                img_array,
                axis=0
            )

            # Prediksi
            prediction = model.predict(
                img_array,
                verbose=0
            )

            predicted_index = np.argmax(
                prediction
            )

            predicted_class = class_names[
                predicted_index
            ]

            confidence = (
                np.max(prediction) * 100
            )

            st.subheader(
                "Hasil Klasifikasi"
            )

            st.success(
                f"{predicted_class}"
            )

            st.write(
                f"Keyakinan: {confidence:.2f}%"
            )

            st.subheader(
                "Detail Probabilitas"
            )

            for i, label in enumerate(
                class_names
            ):

                prob = (
                    prediction[0][i] * 100
                )

                st.write(
                    f"{label}: {prob:.2f}%"
                )

                st.progress(
                    int(prob)
                )

    except Exception as e:

        st.error(
            f"Terjadi kesalahan: {e}"
        )

else:

    st.info(
        "Silakan upload model .keras terlebih dahulu"
    )
