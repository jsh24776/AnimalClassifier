import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import json, os, time

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Animal Classifier",
    page_icon="🐾",
    layout="centered",
)

# ── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Hide streamlit default elements */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2rem; padding-bottom: 2rem; max-width: 780px; }

/* Hero title */
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 3rem;
    font-weight: 800;
    background: linear-gradient(135deg, #f0f0f0 0%, #a0a0a0 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1.1;
    margin-bottom: 0.2rem;
}
.hero-sub {
    font-size: 0.95rem;
    color: #666;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    margin-bottom: 2.5rem;
}

/* Upload zone */
.upload-hint {
    font-size: 0.85rem;
    color: #555;
    text-align: center;
    margin-top: -0.5rem;
    margin-bottom: 1.5rem;
}

/* Result card */
.result-card {
    background: #111;
    border: 1px solid #222;
    border-radius: 16px;
    padding: 1.5rem 2rem;
    margin-top: 1.5rem;
}
.winner-label {
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 800;
    color: #fff;
    margin: 0;
}
.winner-conf {
    font-size: 0.85rem;
    color: #888;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 1.2rem;
}

/* Bar rows */
.bar-row { margin-bottom: 0.9rem; }
.bar-label {
    display: flex;
    justify-content: space-between;
    font-size: 0.85rem;
    color: #aaa;
    margin-bottom: 4px;
}
.bar-label span:first-child { color: #fff; }
.bar-track {
    height: 6px;
    background: #222;
    border-radius: 99px;
    overflow: hidden;
}
.bar-fill {
    height: 100%;
    border-radius: 99px;
    background: linear-gradient(90deg, #e8e8e8, #888);
}

/* Divider */
.divider {
    border: none;
    border-top: 1px solid #222;
    margin: 1.2rem 0;
}

/* Info badge */
.badge {
    display: inline-block;
    background: #1a1a1a;
    border: 1px solid #2a2a2a;
    border-radius: 99px;
    padding: 4px 14px;
    font-size: 0.75rem;
    color: #666;
    margin-bottom: 1.5rem;
}
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────
NUM_CLASSES = 10
DEVICE      = torch.device("cpu")   # Streamlit Cloud has no GPU
IMG_SIZE    = 224

CLASS_NAMES = {
    0: "Dog",      1: "Horse",     2: "Elephant",
    3: "Butterfly",4: "Chicken",   5: "Cat",
    6: "Cow",      7: "Sheep",     8: "Spider",
    9: "Squirrel",
}
EMOJIS = {
    "Dog":"🐶","Horse":"🐴","Elephant":"🐘","Butterfly":"🦋",
    "Chicken":"🐔","Cat":"🐱","Cow":"🐄","Sheep":"🐑",
    "Spider":"🕷️","Squirrel":"🐿️",
}
FUN_FACTS = {
    "Dog":       "Dogs have a sense of smell 40× stronger than humans.",
    "Horse":     "Horses can sleep both standing up and lying down.",
    "Elephant":  "Elephants are the only animals that can't jump.",
    "Butterfly": "Butterflies taste with their feet.",
    "Chicken":   "Chickens have better color vision than humans.",
    "Cat":        "Cats spend 70% of their lives sleeping.",
    "Cow":        "Cows have best friends and get stressed when separated.",
    "Sheep":     "Sheep can recognize up to 50 other sheep faces.",
    "Spider":    "Spiders recycle their webs by eating them.",
    "Squirrel":  "Squirrels forget where they bury 50% of their nuts.",
}

infer_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# ── Model loader (cached) ─────────────────────────────────────
# Replace the load_model() function in app.py with this:
@st.cache_resource(show_spinner=False)
def load_model():
    from huggingface_hub import hf_hub_download

    weights_path = hf_hub_download(
        repo_id="YOUR_HF_USERNAME/animal-classifier",  # ← change this
        filename="best_model.pth",
        cache_dir="/tmp"
    )

    model = models.efficientnet_b2(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_features, 512),
        nn.SiLU(),
        nn.Dropout(p=0.3),
        nn.Linear(512, NUM_CLASSES),
    )
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    model.eval()
    return model 

# ── Load class mapping if exists, else use default ────────────
def get_class_names():
    mapping_path = os.path.join(os.path.dirname(__file__), "class_mapping.json")
    if os.path.exists(mapping_path):
        with open(mapping_path) as f:
            raw = json.load(f)
        return {int(k): v for k, v in raw.items()}
    return CLASS_NAMES

# ── Predict ───────────────────────────────────────────────────
def predict(img: Image.Image, model, class_names: dict, top_k=3):
    tensor = infer_tf(img.convert("RGB")).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = F.softmax(model(tensor), dim=1)[0]
    top_probs, top_idxs = probs.topk(top_k)
    return [
        {
            "label":      class_names.get(i.item(), f"class_{i}"),
            "confidence": float(p) * 100,
            "emoji":      EMOJIS.get(class_names.get(i.item(), ""), "🐾"),
        }
        for p, i in zip(top_probs.cpu(), top_idxs.cpu())
    ]

# ── UI ────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">Animal<br>Classifier</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">EfficientNet-B2 · Animals-10 Dataset · 10 Classes</div>', unsafe_allow_html=True)
st.markdown('<div class="badge">🐶 🐴 🐘 🦋 🐔 🐱 🐄 🐑 🕷️ 🐿️ &nbsp;·&nbsp; Supports dog, horse, elephant, butterfly, chicken, cat, cow, sheep, spider, squirrel</div>', unsafe_allow_html=True)

# Load model
with st.spinner("Loading model..."):
    model      = load_model()
    class_names = get_class_names()

# Upload
uploaded = st.file_uploader(
    "Upload an animal photo",
    type=["jpg", "jpeg", "png", "webp"],
    label_visibility="collapsed",
)
st.markdown('<div class="upload-hint">JPG · PNG · WEBP &nbsp;·&nbsp; Max 200 MB</div>', unsafe_allow_html=True)

if uploaded:
    img = Image.open(uploaded)

    col1, col2 = st.columns([1, 1], gap="large")
    with col1:
        st.image(img, use_container_width=True)

    with col2:
        with st.spinner("Classifying..."):
            time.sleep(0.3)   # tiny pause for UX feel
            results = predict(img, model, class_names)

        winner = results[0]

        # Winner
        st.markdown(f"""
        <div class="result-card">
            <div style="font-size:2.5rem; margin-bottom:0.3rem;">{winner['emoji']}</div>
            <div class="winner-label">{winner['label']}</div>
            <div class="winner-conf">{winner['confidence']:.1f}% confidence</div>
            <hr class="divider">
        """, unsafe_allow_html=True)

        # Bars
        for r in results:
            w = r['confidence']
            st.markdown(f"""
            <div class="bar-row">
                <div class="bar-label">
                    <span>{r['emoji']} {r['label']}</span>
                    <span>{w:.1f}%</span>
                </div>
                <div class="bar-track">
                    <div class="bar-fill" style="width:{w}%"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Fun fact
        fact = FUN_FACTS.get(winner['label'], "")
        if fact:
            st.markdown(f"""
            <hr class="divider">
            <div style="font-size:0.8rem; color:#555; line-height:1.6;">
                <span style="color:#444; font-weight:500;">Did you know?</span><br>{fact}
            </div>
            """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    # Try another
    st.markdown("<br>", unsafe_allow_html=True)
