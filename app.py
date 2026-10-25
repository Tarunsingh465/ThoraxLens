from __future__ import annotations
import hashlib, json
from io import BytesIO
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch
from PIL import Image
from src.inference import predict, gradcam
from src.labels import LABELS

ROOT = Path(__file__).resolve().parent
REPORTS = ROOT / "reports"
FILES = {
    "ranking": REPORTS / "patient_disjoint_test_metrics_fixed.csv",
    "threshold": REPORTS / "patient_disjoint_test_threshold_metrics.csv",
    "validation": REPORTS / "validation_threshold_metrics.csv",
    "history": REPORTS / "training_history.csv",
    "threshold_json": REPORTS / "validation_thresholds.json",
}
DEFAULT_CHECKPOINT = "models/densenet121_xray_label_fixed_best.pth"
st.set_page_config(page_title="ThoraxLens | Research Lab", page_icon="🩻", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>
[data-testid="stAppViewContainer"]{background:#f4f7fb}[data-testid="stHeader"]{background:rgba(244,247,251,.94)}
[data-testid="stSidebar"]{background:#eef3f8;border-right:1px solid #dde6ef}.block-container{max-width:1440px;padding-top:1.35rem;padding-bottom:3rem}
h1,h2,h3{color:#12243a;letter-spacing:-.025em}div[data-testid="stMetric"]{background:white;border:1px solid #e2eaf2;padding:14px 16px;border-radius:15px;box-shadow:0 3px 13px rgba(25,50,80,.04)}
div[data-testid="stMetricLabel"] p{color:#687b91;font-weight:600}.stButton>button{border-radius:10px;font-weight:650;min-height:2.6rem}
.tl-hero{background:linear-gradient(115deg,#12243a 0%,#1e4562 60%,#167e82 100%);padding:28px 32px;border-radius:22px;color:white;margin:2px 0 20px;box-shadow:0 12px 28px rgba(19,54,81,.14)}
.tl-hero h1{color:white;font-size:2.15rem;margin:0 0 .35rem}.tl-hero p{color:#e0edf5;margin:0;line-height:1.55;max-width:960px}.tl-eyebrow{text-transform:uppercase;letter-spacing:.15em;font-size:.72rem;font-weight:750;color:#9fe7dc;margin-bottom:.55rem}
.tl-pill{display:inline-block;border:1px solid #ffffff44;background:#ffffff1a;color:#f1f7fb;padding:5px 10px;border-radius:999px;font-size:.78rem;margin:14px 5px 0 0}
.tl-card{background:white;border:1px solid #e2eaf2;border-radius:16px;padding:16px 18px;margin-bottom:12px}.tl-card b{color:#14283d}.tl-note{color:#60738a;font-size:.9rem;margin:0 0 .7rem}
</style>
""", unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def load_csv(path_str: str) -> pd.DataFrame:
    path = Path(path_str)
    if not path.is_file(): return pd.DataFrame()
    try: return pd.read_csv(path)
    except (OSError, pd.errors.ParserError, UnicodeDecodeError): return pd.DataFrame()

def info(title: str, body: str) -> None:
    if hasattr(st, "popover"):
        with st.popover(f"ⓘ {title}"): st.markdown(body)
    else:
        with st.expander(f"ⓘ {title}"): st.markdown(body)

def project_path(value: str) -> Path:
    p = Path(value).expanduser()
    return p if p.is_absolute() else ROOT / p

def basename(value: str) -> str:
    return str(value).replace("\\", "/").rstrip("/").split("/")[-1]

def showfig(fig) -> None:
    fig.tight_layout(pad=1.5); st.pyplot(fig, use_container_width=True); plt.close(fig)

def missing(path: Path) -> None:
    st.info(f"Saved report not found locally: {path.relative_to(ROOT)}. Add the tracked report file to enable this chart.")

def heading(title: str, note: str = "") -> None:
    st.markdown(f"### {title}")
    if note: st.markdown(f'<p class="tl-note">{note}</p>', unsafe_allow_html=True)

ranking = load_csv(str(FILES["ranking"]))
threshold = load_csv(str(FILES["threshold"]))
validation = load_csv(str(FILES["validation"]))
history = load_csv(str(FILES["history"]))
try: threshold_payload = json.loads(FILES["threshold_json"].read_text(encoding="utf-8"))
except (OSError, json.JSONDecodeError, TypeError): threshold_payload = {}
threshold_map = threshold_payload.get("thresholds", {}) if isinstance(threshold_payload, dict) else {}
threshold_ckpt = basename(threshold_payload.get("checkpoint", "")) if isinstance(threshold_payload, dict) else ""

with st.sidebar:
    st.markdown('<div style="font-size:1.45rem;font-weight:850;color:#14283d;letter-spacing:-.04em;">🩻 ThoraxLens</div><div style="font-size:.81rem;color:#6b7c90;margin:3px 0 16px;">Chest X-ray research lab</div>', unsafe_allow_html=True)
    heading("Explore")
    page = st.radio("Dashboard section", ["Overview", "Model performance", "Predict & explain", "Data & methodology"], label_visibility="collapsed")
    st.divider(); heading("Local model settings")
    checkpoint_text = st.text_input("Checkpoint path", DEFAULT_CHECKPOINT, help="Model weights are intentionally excluded from the public repository. Set a path to a compatible local checkpoint.")
    image_size = st.selectbox("Model input size", [224, 256, 320], index=0, help="Use the size expected by your selected checkpoint and preprocessing configuration.")
    checkpoint_path = project_path(checkpoint_text)
    if checkpoint_path.is_file(): st.success("Checkpoint found locally")
    else: st.warning("Checkpoint not found locally")
    st.caption(f"Compute device: {'CUDA GPU' if torch.cuda.is_available() else 'CPU'}")
    info("Checkpoint help", "The repository does not distribute model weights. A compatible checkpoint must exist locally. Keep its architecture, image size, and preprocessing consistent with the inference code.")

st.markdown('''<div class="tl-hero"><div class="tl-eyebrow">Research prototype · Multi-label imaging</div><h1>Understand the model, not just the score.</h1><p>Explore saved performance reports, review dataset integrity, and inspect image-level scores with optional exploratory saliency. Metric and feature info controls explain how to interpret each view.</p><span class="tl-pill">DenseNet-121</span><span class="tl-pill">14 findings</span><span class="tl-pill">PyTorch</span><span class="tl-pill">Validation-selected thresholds</span></div>''', unsafe_allow_html=True)
st.warning("Research and education only — not a medical device. ThoraxLens is not clinically validated. Scores, flags, and saliency maps must not be used for diagnosis or treatment decisions.")

auc = float(ranking["AUROC"].mean()) if not ranking.empty and "AUROC" in ranking else np.nan
ap = float(ranking["average_precision"].mean()) if not ranking.empty and "average_precision" in ranking else np.nan
prec = float(threshold["test_precision_at_threshold"].mean()) if not threshold.empty and "test_precision_at_threshold" in threshold else np.nan
rec = float(threshold["test_recall_at_threshold"].mean()) if not threshold.empty and "test_recall_at_threshold" in threshold else np.nan
f1 = float(threshold["test_f1_at_threshold"].mean()) if not threshold.empty and "test_f1_at_threshold" in threshold else np.nan

if page == "Overview":
    heading("Project overview", "A guided summary of the current experiment, key performance numbers, and evaluation safeguards.")
    info("About ThoraxLens", "ThoraxLens predicts scores for 14 findings from a frontal chest X-ray. More than one label can be relevant to one image, so the task is multi-label classification. Saved evaluation results below are separate from predictions on newly uploaded images.")
    cols = st.columns(4)
    cards = [("Macro AUROC", auc, "Mean of the 14 per-finding AUROC values; describes ranking discrimination over thresholds."),("Macro Average Precision", ap, "Mean per-finding AP; summarizes precision-recall behavior and is sensitive to label prevalence."),("Macro precision", prec, "Mean precision at thresholds selected on validation data; low precision indicates many false positives against dataset labels."),("Macro recall", rec, "Mean recall at validation thresholds; recall increases when more labelled positives are detected, often at the cost of more false positives.")]
    for col, (name, value, help_text) in zip(cols, cards):
        with col: st.metric(name, f"{value:.3f}" if np.isfinite(value) else "N/A", help=help_text)
    c1, c2 = st.columns([1.1, 1])
    with c1:
        st.markdown('<div class="tl-card"><b>Dashboard features</b><br><br>• Per-finding AUROC and Average Precision graphs<br>• Precision, recall and F1 at validation-selected thresholds<br>• Training loss and validation-metric curves<br>• Image-level score ranking and optional saliency overlay<br>• Dataset split accounting and interpretation guidance</div>', unsafe_allow_html=True)
        info("How to navigate", "Use Model performance for saved report graphs. Predict & explain runs a local checkpoint on your image. Data & methodology describes the dataset split, threshold process, and limitations.")
    with c2:
        st.markdown('<div class="tl-card"><b>Documented experiment</b><br><br>Dataset: NIH ChestX-ray14<br>Images audited: 112,120<br>Official train/validation pool: 86,524<br>Internal training: 73,023<br>Internal validation: 13,501<br>Held-out test: 25,596<br>Training: 2 fine-tuning epochs from an earlier project checkpoint</div>', unsafe_allow_html=True)
        info("Dataset counts", "The official train/validation pool was divided into training and validation partitions. The official test list remained held out. The tracked split audit reports zero filename-list overlap and zero patient overlap between the development group and the held-out test group.")
    heading("Findings with highest and lowest AUROC")
    if not ranking.empty and {"label", "AUROC"}.issubset(ranking.columns):
        x, y = st.columns(2)
        with x: st.caption("Highest values in the test report"); st.dataframe(ranking.nlargest(3, "AUROC")[["label", "AUROC"]].style.format({"AUROC":"{:.3f}"}), hide_index=True, use_container_width=True)
        with y: st.caption("Lowest values in the test report"); st.dataframe(ranking.nsmallest(3, "AUROC")[["label", "AUROC"]].style.format({"AUROC":"{:.3f}"}), hide_index=True, use_container_width=True)

elif page == "Model performance":
    heading("Model performance", "Charts are rendered from the project's tracked CSV reports, not sample or fabricated scores.")
    info("Metric guide", "AUROC measures ranking discrimination across thresholds. Average Precision summarizes the precision-recall curve. Precision, recall and F1 describe predictions after applying fixed decision thresholds. Macro scores average the per-label metrics equally.")
    tabs = st.tabs(["AUROC & AP", "Threshold metrics", "Training history", "Label prevalence"])
    with tabs[0]:
        if ranking.empty or not {"label", "AUROC", "average_precision"}.issubset(ranking.columns): missing(FILES["ranking"])
        else:
            a, b = st.columns(2)
            with a:
                heading("AUROC by finding")
                info("Read AUROC", "Each bar shows one finding's AUROC on held-out test images. Values nearer 1 indicate stronger ranking discrimination for that finding; AUROC is not the same as accuracy and does not specify a decision threshold.")
                d = ranking.dropna(subset=["AUROC"]).sort_values("AUROC"); fig, ax = plt.subplots(figsize=(7.2, 6.2)); bars = ax.barh(d["label"], d["AUROC"], color="#278f96", height=.68); ax.set_xlim(0,1); ax.set_xlabel("Area under ROC curve (AUROC)"); ax.grid(axis="x", alpha=.2); ax.set_axisbelow(True)
                for bar, value in zip(bars, d["AUROC"]): ax.text(min(value+.015,.96), bar.get_y()+bar.get_height()/2, f"{value:.3f}", va="center", fontsize=8)
                showfig(fig)
            with b:
                heading("Average Precision by finding")
                info("Read Average Precision", "AP summarizes precision-recall behavior and is especially useful for rare findings. It depends partly on prevalence, so interpret values together with each finding's number of positive cases.")
                d = ranking.dropna(subset=["average_precision"]).sort_values("average_precision"); fig, ax = plt.subplots(figsize=(7.2, 6.2)); bars = ax.barh(d["label"], d["average_precision"], color="#6175dc", height=.68); ax.set_xlim(0, max(.1, float(d["average_precision"].max())*1.18)); ax.set_xlabel("Average Precision (AP)"); ax.grid(axis="x", alpha=.2); ax.set_axisbelow(True)
                for bar, value in zip(bars, d["average_precision"]): ax.text(bar.get_width()+.004, bar.get_y()+bar.get_height()/2, f"{value:.3f}", va="center", fontsize=8)
                showfig(fig)
            st.info("The repository stores per-finding AUROC values but not raw test-set labels and prediction scores needed to reconstruct true ROC curves. This dashboard plots the actual saved AUROC results by finding instead of inventing a ROC curve.")
            with st.expander("Exact per-finding ranking table"):
                cols = [c for c in ["label", "positive_count", "prevalence", "AUROC", "average_precision"] if c in ranking]; st.dataframe(ranking[cols].style.format({c:"{:.4f}" for c in ["prevalence","AUROC","average_precision"] if c in cols}), hide_index=True, use_container_width=True)
                st.download_button("Download ranking metrics CSV", ranking.to_csv(index=False).encode(), "thoraxlens_test_ranking_metrics.csv", "text/csv")
    with tabs[1]:
        if threshold.empty or not {"label","test_precision_at_threshold","test_recall_at_threshold","test_f1_at_threshold"}.issubset(threshold.columns): missing(FILES["threshold"])
        else:
            heading("Precision, recall and F1 by finding")
            info("Threshold interpretation", "Each finding's threshold was selected by maximizing F1 on validation data, then applied unchanged to the held-out test set. These are research operating points, not medical cutoffs.")
            d = threshold.sort_values("test_f1_at_threshold"); y = np.arange(len(d)); h = .24; fig, ax = plt.subplots(figsize=(11.5, 7.3)); ax.barh(y-h,d["test_precision_at_threshold"],h,label="Precision",color="#278f96"); ax.barh(y,d["test_recall_at_threshold"],h,label="Recall",color="#6175dc"); ax.barh(y+h,d["test_f1_at_threshold"],h,label="F1",color="#e39b48"); ax.set_yticks(y,d["label"]); ax.set_xlim(0,1); ax.set_xlabel("Metric value"); ax.legend(ncol=3,frameon=False); ax.grid(axis="x",alpha=.2); ax.set_axisbelow(True); showfig(fig)
            st.metric("Macro F1 at validation thresholds", f"{f1:.3f}" if np.isfinite(f1) else "N/A", help="Mean per-finding F1 after applying validation-selected thresholds to held-out test predictions.")
            with st.expander("Thresholds and error counts"):
                cols=[c for c in ["label","threshold_from_validation","test_precision_at_threshold","test_recall_at_threshold","test_f1_at_threshold","TP","FP","FN","TN"] if c in threshold]; fm={c:"{:.4f}" for c in ["threshold_from_validation","test_precision_at_threshold","test_recall_at_threshold","test_f1_at_threshold"] if c in cols}; st.dataframe(threshold[cols].style.format(fm),hide_index=True,use_container_width=True); st.download_button("Download threshold metrics CSV",threshold.to_csv(index=False).encode(),"thoraxlens_test_threshold_metrics.csv","text/csv")
    with tabs[2]:
        if history.empty or "epoch" not in history: missing(FILES["history"])
        else:
            heading("Training and validation curves")
            info("Training notes", "The saved history covers two fine-tuning epochs initialized from an earlier project checkpoint, not training from scratch. Two epochs are only an early diagnostic snapshot and do not establish convergence.")
            a,b=st.columns(2)
            with a:
                fig,ax=plt.subplots(figsize=(6.5,4.4))
                if "train_loss" in history: ax.plot(history["epoch"],history["train_loss"],marker="o",linewidth=2.3,label="Training loss",color="#278f96")
                if "val_loss" in history: ax.plot(history["epoch"],history["val_loss"],marker="o",linewidth=2.3,label="Validation loss",color="#6175dc")
                ax.set_xlabel("Epoch");ax.set_ylabel("Loss");ax.set_xticks(history["epoch"].astype(int));ax.legend(frameon=False);ax.grid(alpha=.2);showfig(fig)
            with b:
                fig,ax=plt.subplots(figsize=(6.5,4.4))
                if "val_macro_auroc" in history: ax.plot(history["epoch"],history["val_macro_auroc"],marker="o",linewidth=2.3,label="Validation macro AUROC",color="#278f96")
                if "val_macro_ap" in history: ax.plot(history["epoch"],history["val_macro_ap"],marker="o",linewidth=2.3,label="Validation macro AP",color="#e39b48")
                ax.set_xlabel("Epoch");ax.set_ylabel("Metric value");ax.set_ylim(0,1);ax.set_xticks(history["epoch"].astype(int));ax.legend(frameon=False);ax.grid(alpha=.2);showfig(fig)
            st.dataframe(history,hide_index=True,use_container_width=True)
    with tabs[3]:
        if ranking.empty or "positive_count" not in ranking: missing(FILES["ranking"])
        else:
            heading("Positive test cases by finding")
            info("Why prevalence matters", "Findings have different frequencies. Rare labels have fewer positive examples, so their metrics can be more uncertain. AP should be read alongside the number and fraction of positives.")
            d=ranking.sort_values("positive_count");fig,ax=plt.subplots(figsize=(10,6.2));bars=ax.barh(d["label"],d["positive_count"],color="#6886a4");ax.set_xlabel("Positive test cases");ax.grid(axis="x",alpha=.2);ax.set_axisbelow(True)
            for bar,value in zip(bars,d["positive_count"]): ax.text(bar.get_width()+max(d["positive_count"])*.01,bar.get_y()+bar.get_height()/2,f"{int(value):,}",va="center",fontsize=8)
            showfig(fig)
            if "prevalence" in ranking: st.dataframe(ranking[["label","positive_count","prevalence"]].sort_values("prevalence",ascending=False).style.format({"prevalence":"{:.2%}"}),hide_index=True,use_container_width=True)

elif page == "Predict & explain":
    heading("Predict & explain", "Upload a local X-ray to inspect 14 scores and generate an optional exploratory saliency view.")
    info("Before you begin", "Use images you are permitted to process. The selected local checkpoint runs inference. Raw output scores are not guaranteed calibrated probabilities. Threshold flags are only enabled if the threshold file names the same checkpoint.")
    threshold_ok = bool(threshold_map) and basename(checkpoint_text) == threshold_ckpt
    if threshold_ok: st.success("Validation thresholds match the selected checkpoint name.")
    elif threshold_map: st.warning(f"Threshold flags are disabled: thresholds name {threshold_ckpt or 'an unknown checkpoint'}, selected checkpoint is {basename(checkpoint_text)}.")
    else: st.info("No usable threshold file found; raw scores will be shown only.")
    upload=st.file_uploader("Choose a frontal chest X-ray",type=["png","jpg","jpeg"],help="Supported formats are PNG and JPEG. This prototype is not validated for clinical use.")
    if upload is not None:
        try: image=Image.open(BytesIO(upload.getvalue())).convert("L")
        except Exception as exc: st.error(f"Could not read image: {exc}"); image=None
        if image is not None:
            key=hashlib.sha256(upload.getvalue()+str(checkpoint_path).encode()+str(image_size).encode()).hexdigest();a,b=st.columns([.9,1.1],gap="large")
            with a:
                heading("Image preview");st.image(image,caption=f"{image.width} × {image.height} pixels",use_container_width=True);st.caption("The inference pipeline converts the image to grayscale and applies the configured resize and normalization transform.")
                if st.button("Analyze image",type="primary",use_container_width=True):
                    if not checkpoint_path.is_file(): st.error(f"Checkpoint not found: {checkpoint_path}")
                    else:
                        try:
                            with st.spinner("Running DenseNet-121 inference…"): ranked=predict(image,str(checkpoint_path),int(image_size))
                            st.session_state["tl_ranked"]=ranked;st.session_state["tl_key"]=key;st.session_state.pop("tl_cam",None)
                        except Exception as exc: st.error(f"Inference failed: {exc}")
            ranked=st.session_state.get("tl_ranked") if st.session_state.get("tl_key")==key else None
            with b:
                heading("Finding scores")
                if ranked:
                    data=pd.DataFrame(ranked,columns=["Finding","Raw score"]);usable=threshold_ok
                    if usable:
                        data["Validation threshold"]=data["Finding"].map(threshold_map).astype(float);data["Model flag"]=np.where(data["Raw score"]>=data["Validation threshold"],"Above threshold","Below threshold")
                    top=data.head(10).sort_values("Raw score");fig,ax=plt.subplots(figsize=(7.3,5.1));colors=["#df8a3b" if (usable and score>=float(threshold_map.get(label,2))) else "#278f96" for label,score in zip(top["Finding"],top["Raw score"])];bars=ax.barh(top["Finding"],top["Raw score"],color=colors);ax.set_xlim(0,1);ax.set_xlabel("Raw model score (not a calibrated probability)");ax.grid(axis="x",alpha=.2);ax.set_axisbelow(True)
                    for bar,value in zip(bars,top["Raw score"]): ax.text(min(value+.012,.96),bar.get_y()+bar.get_height()/2,f"{value:.3f}",va="center",fontsize=8)
                    showfig(fig)
                    if usable:
                        flags=data[data["Model flag"]=="Above threshold"]["Finding"].tolist();st.metric("Scores above validation thresholds",len(flags),help="This is a count of model outputs above experimental thresholds, not a diagnosis.")
                        if flags: st.markdown("**Above-threshold model flags:** "+", ".join(flags))
                        else: st.caption("No scores exceed the saved validation thresholds for this image.")
                    with st.expander("All 14 finding scores"):
                        fm={"Raw score":"{:.4f}"}
                        if "Validation threshold" in data: fm["Validation threshold"]="{:.4f}"
                        st.dataframe(data.style.format(fm),hide_index=True,use_container_width=True);st.download_button("Download scores CSV",data.to_csv(index=False).encode(),"thoraxlens_image_scores.csv","text/csv")
                else: st.markdown('<div class="tl-card"><b>Your results will appear here</b><br><br>Upload an image and select Analyze image. The model returns 14 separate scores.</div>',unsafe_allow_html=True)
            if ranked:
                st.divider();heading("Exploratory saliency")
                info("What is this heatmap?", "The Grad-CAM-style overlay highlights regions associated with a selected model output. It is not verified lesion segmentation and does not prove that a clinically meaningful feature was used. Do not use it to guide care.")
                target=st.selectbox("Finding to visualize",LABELS,index=LABELS.index(ranked[0][0]),help="Choose the output used to generate the exploratory Grad-CAM-style overlay.")
                if st.button("Generate saliency overlay"):
                    if not checkpoint_path.is_file(): st.error("Checkpoint not found.")
                    else:
                        try:
                            with st.spinner("Generating overlay…"): scores,cam=gradcam(image,str(checkpoint_path),target,int(image_size))
                            arr=np.asarray(image.resize((int(image_size),int(image_size))));fig,ax=plt.subplots(figsize=(6.5,6.5));ax.imshow(arr,cmap="gray");ax.imshow(cam,cmap="jet",alpha=.4,vmin=0,vmax=1);ax.set_title(f"Exploratory saliency · {target} · score {scores[target]:.3f}");ax.axis("off");showfig(fig)
                        except Exception as exc: st.error(f"Could not generate saliency: {exc}")

else:
    heading("Data & methodology", "Dataset accounting, patient separation, threshold selection, and responsible interpretation.")
    info("Why methodology matters", "Reliable evaluation depends on which examples enter each partition, whether patients overlap, and whether test data influences tuning. This page summarizes the protocol documented with this project.")
    cols=st.columns(4)
    for col,(label,value,help_text) in zip(cols,[("Images audited","112,120","Extracted images and matching metadata rows."),("Training","73,023","Internal training partition."),("Validation","13,501","Used for validation-based threshold selection."),("Held-out test","25,596","Official test images evaluated after model development.")]):
        with col: st.metric(label,value,help=help_text)
    heading("Dataset partition sizes")
    split=pd.DataFrame({"Partition":["Internal train","Internal validation","Held-out test"],"Images":[73023,13501,25596]});fig,ax=plt.subplots(figsize=(9.5,3.6));bars=ax.bar(split["Partition"],split["Images"],color=["#278f96","#6175dc","#e39b48"],width=.62);ax.set_ylabel("Images");ax.set_ylim(0,93000);ax.grid(axis="y",alpha=.2);ax.set_axisbelow(True)
    for bar,value in zip(bars,split["Images"]): ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+1600,f"{value:,}",ha="center",fontweight="bold")
    showfig(fig)
    x,y=st.columns(2)
    with x: st.markdown('<div class="tl-card"><b>Split integrity</b><br><br>Official train/validation: 86,524<br>Official test: 25,596<br>Filename-list overlap: <b>0</b><br>Patient overlap reported by audit: <b>0</b><br>Image/metadata filename mismatches: <b>0</b></div>',unsafe_allow_html=True)
    with y: st.markdown('<div class="tl-card"><b>Threshold protocol</b><br><br>Each finding gets a threshold by maximizing F1 on the validation split. Those values are then applied unchanged to the held-out test set. These are experimental operating points, not medical cutoffs.</div>',unsafe_allow_html=True)
    with st.expander("ⓘ AUROC and Average Precision"):
        st.markdown("AUROC summarizes ranking across thresholds. Average Precision summarizes precision-recall behavior and is particularly useful for rare labels. Neither metric proves clinical utility or safety.")
    with st.expander("ⓘ Precision, recall and F1"):
        st.markdown("Precision = TP / (TP + FP). Recall = TP / (TP + FN). F1 is the harmonic mean of precision and recall. Choosing a threshold based on F1 expresses one trade-off and does not determine the right clinical trade-off.")
    with st.expander("ⓘ Limitations"):
        st.markdown("NIH report-derived labels can be noisy or incomplete. Prevalence varies widely across findings. The reported run completed two fine-tuning epochs from an earlier checkpoint, not training from scratch. Saliency maps are exploratory, no confidence intervals or external clinical validation are documented, and the model is not for diagnosis, triage, or treatment.")
    heading("Tracked report files")
    for label,path,frame in [("Per-finding test ranking metrics",FILES["ranking"],ranking),("Thresholded test metrics",FILES["threshold"],threshold),("Validation threshold diagnostics",FILES["validation"],validation),("Training history",FILES["history"],history)]: st.markdown(f"- **{label}:** {path.relative_to(ROOT)} — {'Available' if not frame.empty else 'Missing locally'}")
    st.caption("Dataset images and trained checkpoints are intentionally excluded from the public repository. See data/DATASET_DOWNLOAD.md for dataset setup.")

st.divider()
st.markdown('<div style="text-align:center;color:#7b899a;font-size:.82rem;line-height:1.6;">ThoraxLens · DenseNet-121 · Multi-label chest X-ray research<br>Results describe the tracked research experiment only. Not for clinical use.</div>',unsafe_allow_html=True)
