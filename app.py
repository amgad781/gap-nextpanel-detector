import streamlit as st
from ultralytics import YOLO
from PIL import Image

st.set_page_config(page_title="Gap vs Next-Panel Classifier", page_icon="🧩")

@st.cache_resource
def load_model():
    return YOLO("best.pt")  # or "gap_model.pt" if you renamed it

model = load_model()

st.title("🧩 Gap vs Next-Panel Classifier")
st.write("Upload a camera view to check whether it's a large air gap (do not cross) or a small crossable gap to the next panel.")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded image", use_container_width=True)

    with st.spinner("Analyzing..."):
        results = model(image, verbose=False)
        probs = results[0].probs
        class_names = results[0].names

    top_class = class_names[probs.top1]
    confidence = probs.top1conf.item()

    if top_class == "next-panel" and confidence >= 0.85:
        st.success(f"✅ Safe to cross — **{top_class}** ({confidence:.1%} confidence)")
    else:
        st.error(f"⛔ Do not cross — predicted **{top_class}** ({confidence:.1%} confidence)")

    st.write("---")
    st.write("All class probabilities:")
    for i in range(len(class_names)):
        st.write(f"{class_names[i]}: {float(probs.data[i]):.1%}")
        st.progress(float(probs.data[i]))
