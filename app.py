import streamlit as st
import fitz
from paddleocr import PaddleOCR
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline

# -------------------------------
# LOAD MODELS
# -------------------------------
@st.cache_resource
def load_models():
    ocr = PaddleOCR(use_angle_cls=True, lang='en')

    embedder = SentenceTransformer('all-MiniLM-L6-v2')

    qa_pipeline = pipeline(
        "text2text-generation",
        model="google/flan-t5-base",
        max_length=256
    )

    return ocr, embedder, qa_pipeline

ocr, embedder, qa_model = load_models()

# -------------------------------
# IMAGE PREPROCESSING
# -------------------------------
def preprocess_variants(image):
    variants = []

    variants.append(np.array(image.convert("RGB")))

    gray = image.convert("L")
    gray = ImageEnhance.Contrast(gray).enhance(2)
    variants.append(np.stack((np.array(gray),)*3, axis=-1))

    sharp = image.filter(ImageFilter.SHARPEN)
    variants.append(np.array(sharp))

    return variants

# -------------------------------
# OCR MULTI-PASS
# -------------------------------
def extract_text_from_image(image):
    variants = preprocess_variants(image)
    all_lines = []

    for var in variants:
        result = ocr.ocr(var)
        if result:
            for line in result[0]:
                text = line[1][0]
                if text.strip():
                    all_lines.append(text.strip())

    # Remove duplicates (keep order)
    seen = set()
    final = []
    for t in all_lines:
        if t not in seen:
            seen.add(t)
            final.append(t)

    return final

# -------------------------------
# PDF EXTRACTION
# -------------------------------
def extract_text_from_pdf(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    data = []

    for i, page in enumerate(doc):
        page_num = i + 1

        text = page.get_text().strip()

        if text:
            lines = text.split("\n")
        else:
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            lines = extract_text_from_image(img)

        chunk = []
        for line in lines:
            if line.strip():
                chunk.append(line.strip())

            if len(chunk) >= 4:
                data.append({
                    "page": page_num,
                    "text": " ".join(chunk)
                })
                chunk = []

        if chunk:
            data.append({
                "page": page_num,
                "text": " ".join(chunk)
            })

    return data

# -------------------------------
# EMBEDDINGS
# -------------------------------
@st.cache_data
def create_embeddings(data):
    texts = [d["text"] for d in data]
    return embedder.encode(texts)

# -------------------------------
# SEARCH
# -------------------------------
def search(query, data, embeddings, top_k=5):
    q_emb = embedder.encode([query])
    scores = cosine_similarity(q_emb, embeddings)[0]

    idxs = np.argsort(scores)[-top_k:][::-1]

    results = []
    for i in idxs:
        results.append({
            "text": data[i]["text"],
            "page": data[i]["page"],
            "score": scores[i]
        })

    return results

# -------------------------------
# ANSWER GENERATION WITH VALIDATION
# -------------------------------
def generate_answer(query, results):
    max_score = max([r["score"] for r in results])

    # Reject out-of-context questions
    if max_score < 0.35:
        return "I don’t have enough information in the document to answer that."

    context = " ".join([r["text"] for r in results])

    prompt = f"""
    Answer the question based ONLY on the context below.

    Context:
    {context}

    Question:
    {query}
    """

    response = qa_model(prompt)[0]["generated_text"]
    return response

# -------------------------------
# UI
# -------------------------------
st.set_page_config(page_title="PaperBrain AI", layout="wide")

st.title("🧠 PaperBrain AI")
st.write("Accurate AI PDF + Handwriting Reader")

# Session state
if "data" not in st.session_state:
    st.session_state.data = None
    st.session_state.embeddings = None

# Upload
file = st.file_uploader("📄 Upload PDF/Image", type=["pdf", "png", "jpg", "jpeg"])

if file:
    if st.button("📥 Process Document"):
        with st.spinner("Processing..."):
            if file.type == "application/pdf":
                data = extract_text_from_pdf(file)
            else:
                img = Image.open(file)
                lines = extract_text_from_image(img)
                data = [{"page": 1, "text": l} for l in lines]

            embeddings = create_embeddings(data)

            st.session_state.data = data
            st.session_state.embeddings = embeddings

        st.success("✅ Document Ready!")

# Show extracted data
if st.session_state.data:
    if st.checkbox("📜 Show Extracted Data"):
        for d in st.session_state.data:
            st.write(f"Page {d['page']} → {d['text']}")

# Ask section
if st.session_state.data:
    st.subheader("💬 Ask PaperBrain")

    q = st.text_input("Ask your question")

    col1, col2 = st.columns(2)

    with col1:
        ask = st.button("🧠 Ask PaperBrain")

    with col2:
        clear = st.button("🧹 Clear Chat")

    if clear:
        st.session_state.data = None
        st.session_state.embeddings = None
        st.rerun()   # ✅ FIXED HERE

    if ask and q:
        results = search(q, st.session_state.data, st.session_state.embeddings)

        ans = generate_answer(q, results)

        st.subheader("📌 Answer")
        st.write(ans)

        st.subheader("📍 Sources")
        for r in results:
            st.write(f"Page {r['page']} → {r['text']}")