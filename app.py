import streamlit as st
import tensorflow as tf
import tempfile

st.title("Pembaca File Model .keras")

model_file = st.file_uploader(
    "Upload Model .keras",
    type=["keras"]
)

if model_file is not None:

    try:
        # Simpan sementara
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".keras"
        ) as tmp:

            tmp.write(model_file.read())
            model_path = tmp.name

        # Load model
        model = tf.keras.models.load_model(
            model_path,
            compile=False
        )

        st.success("Model berhasil dibaca")

        st.subheader("Informasi Model")

        st.write("Nama Model:")
        st.write(model.name)

        st.write("Input Shape:")
        st.write(model.input_shape)

        st.write("Output Shape:")
        st.write(model.output_shape)

        st.write("Jumlah Layer:")
        st.write(len(model.layers))

        st.subheader("Daftar Layer")

        for i, layer in enumerate(model.layers):

            st.write(
                f"{i+1}. {layer.name} ({layer.__class__.__name__})"
            )

    except Exception as e:

        st.error(f"Gagal membaca model: {e}")
