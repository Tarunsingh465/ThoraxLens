from pathlib import Path
import hashlib
import json
from io import BytesIO

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image

from src.inference import predict, gradcam
from src.labels import LABELS

st.set_page_config(page_title="Chest X-ray Research Demo", page_icon="🩻", layout="wide")
st.title("Chest X-ray · Multi-label Research Demo")
st.warning(
    "Educational/research prototype only. Not a medical device; scores, threshold flags, "
    "and heatmaps are not diagnoses. Do not use for clinical decisions."
)
st.markdown(
    "This research demo uses DenseNet-121 for 14 NIH ChestX-ray14 labels. "
    "The scores below are uncalibrated model outputs; the optional per-label thresholds "
    "were selected to maximize F1 on a validation split, not as clinical cutoffs."
)

checkpoint = st.text_input(
    "Checkpoint path", "D:\chest_xray_project_v3\models\densenet121_xray_label_fixed_best.pth"
)
image_size = st.selectbox("Model input size", [224, 256, 320], index=0)
threshold_path = Path("reports/validation_thresholds.json")
thresholds = None
threshold_checkpoint = None
if threshold_path.is_file():
    try:
        threshold_payload = json.loads(threshold_path.read_text(encoding="utf-8"))
        threshold_checkpoint = Path(str(threshold_payload.get("checkpoint", ""))).name
        thresholds = threshold_payload.get("thresholds", {})
        if Path(checkpoint).name != threshold_checkpoint:
            st.warning(
                f"Thresholds were calibrated for '{threshold_checkpoint}', but the selected "
                f"checkpoint is '{Path(checkpoint).name}'. Threshold flags are disabled "
                "until they match."
            )
            thresholds = None
    except (OSError, json.JSONDecodeError, TypeError):
        st.warning("Threshold file could not be read; showing raw model scores only.")
else:
    st.info("Optional validation thresholds not found yet. Raw scores will be shown only.")

upload = st.file_uploader("Upload a chest X-ray (PNG/JPG)", type=["png", "jpg", "jpeg"])

if upload:
    payload = upload.getvalue()
    upload_key = hashlib.sha256(payload).hexdigest()
    image = Image.open(BytesIO(payload)).convert("L")
    st.image(image, caption="Uploaded image", width=520)

    if st.button("Run prediction", type="primary"):
        if not Path(checkpoint).is_file():
            st.error("Checkpoint not found. Select a valid trained checkpoint path.")
        else:
            try:
                with st.spinner("Running model..."):
                    ranked = predict(image, checkpoint, image_size)
                st.session_state["prediction_key"] = upload_key
                st.session_state["ranked"] = ranked
                st.session_state.pop("cam_key", None)
            except Exception as exc:
                st.error(f"Unable to process image: {exc}")

    ranked = (
        st.session_state.get("ranked")
        if st.session_state.get("prediction_key") == upload_key
        else None
    )
    if ranked:
        col1, col2 = st.columns([1, 1])
        with col1:
            st.subheader("Model scores")
            st.caption("Raw, uncalibrated scores; these are not clinical probabilities.")
            score_map = dict(ranked)
            rows = []
            for label, score in ranked:
                row = {"Finding": label, "Score": round(float(score), 4)}
                if thresholds is not None and label in thresholds:
                    threshold = float(thresholds[label])
                    row["Validation threshold"] = round(threshold, 4)
                    row["Model flag"] = "Above threshold" if score >= threshold else "Below threshold"
                rows.append(row)
            st.dataframe(rows, use_container_width=True, hide_index=True)
            if thresholds is not None:
                flags = [label for label in LABELS if label in thresholds and score_map.get(label, -1) >= float(thresholds[label])]
                st.metric("Findings above validation thresholds", len(flags))
                st.caption(
                    "Thresholds were chosen by maximizing F1 on validation data. "
                    "They can produce false positives and false negatives and are not medical cutoffs."
                )
                if flags:
                    st.write("**Above-threshold model flags:** " + ", ".join(flags))
                else:
                    st.write("No finding scores exceed their validation thresholds for this image.")
        with col2:
            st.subheader("Grad-CAM-style visualization")
            target = st.selectbox("Finding to visualize", LABELS, index=LABELS.index(ranked[0][0]))
            if st.button("Create heatmap"):
                if not Path(checkpoint).is_file():
                    st.error("Checkpoint not found.")
                else:
                    try:
                        with st.spinner("Generating exploratory heatmap..."):
                            scores, cam = gradcam(image, checkpoint, target, image_size)
                        arr = np.array(image.resize((image_size, image_size)))
                        fig, ax = plt.subplots(figsize=(6, 6))
                        ax.imshow(arr, cmap="gray")
                        ax.imshow(cam, cmap="jet", alpha=0.38)
                        ax.axis("off")
                        ax.set_title(f"Exploratory saliency for {target} (score {scores[target]:.3f})")
                        st.pyplot(fig)
                        plt.close(fig)
                    except Exception as exc:
                        st.error(f"Unable to generate heatmap: {exc}")
