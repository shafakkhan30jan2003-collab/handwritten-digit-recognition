import streamlit as st
from PIL import Image
import os
import io
from db_utils import init_db, save_image, load_images, clear_all_data
from model_utils import train_from_db, predict_image

# ---------- Streamlit Page Setup ----------
st.set_page_config(page_title="Digit Trainer (DB-Based)", page_icon="🔢")
st.title(" Handwritten Digit Recognition — SQL + Streamlit + PyTorch")

# ---------- Setup ----------
init_db()
os.makedirs("models", exist_ok=True)
os.makedirs("database", exist_ok=True)

tab1, tab2, tab3 = st.tabs(["📤 Upload", "📈 Train", "🔍 Predict"])

# ---------- UPLOAD TAB ----------
with tab1:
    st.header("Upload Handwritten Digit")
    uploaded = st.file_uploader("Upload a digit image", type=["jpg", "png", "jpeg"])
    label = st.number_input("Enter the digit label (0–9):", min_value=0, max_value=9, step=1)

    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        st.image(img, caption="Uploaded Image", width=150)
        if st.button("Save to Database"):
            # Convert image to bytes for storage
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            image_bytes = buf.getvalue()
            save_image(label, image_bytes)
            st.success(f"✅ Saved digit {label} to database!")

    if st.button("Clear Database"):
        clear_all_data()
        st.warning("🗑️ All data cleared from database.")

    if "show_samples" not in st.session_state:
        st.session_state.show_samples = False

    # --- VIEW DATABASE SAMPLES ---
    if st.button("View Database Samples"):
        st.session_state.show_samples = True  # ✅ remember to show

    if st.session_state.show_samples:
        rows = load_images()
        total = len(rows)
        st.write(f"📦 Database contains **{total} images.**")

        if total > 0:
            per_page = 20
            pages = (total // per_page) + (1 if total % per_page != 0 else 0)
            page = st.number_input("Page:", 1, pages, 1, key="page_num")

            start = (page - 1) * per_page
            end = start + per_page

            st.write(f"Showing images **{start+1}–{min(end, total)} of {total}**")

            cols = st.columns(5)
            for i, (lbl, blob) in enumerate(rows[start:end]):
                img = Image.open(io.BytesIO(blob))
                with cols[i % 5]:
                    st.image(img, caption=f"Label: {lbl}", width=100)

            if st.button("❌ Close Viewer"):
                st.session_state.show_samples = False  # ✅ hide viewer again
        else:
            st.info("Database is empty. Upload some samples first.")

# ---------- TRAIN TAB ----------
with tab2:
    st.header("Train Model from Database")
    epochs = st.slider("Select number of epochs", 3, 25, 10)

    if st.button("Start Training"):
        with st.spinner("Training model..."):
            try:
                train_from_db(epochs=epochs)
                st.success("✅ Model trained successfully and saved to models/digit_cnn.pth")
            except Exception as e:
                st.error(f"Error: {e}")

# ---------- PREDICT TAB ----------
with tab3:
    st.header("Test Model Prediction")
    test_img = st.file_uploader("Upload an image to test", type=["jpg", "png", "jpeg"], key="test")

    if test_img:
        image = Image.open(test_img).convert("RGB")
        st.image(image, width=150)
        if st.button("Predict Digit"):
            with st.spinner("Analyzing..."):
                result = predict_image(test_img)
            st.success(f"Predicted Digit: {result}")









