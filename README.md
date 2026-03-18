# 🧠 PaperBrain AI

PaperBrain AI is an intelligent document assistant that can read PDFs and images (including handwritten text), extract information, and answer questions using AI.

---

## 🚀 Features

- 📄 Extract text from PDFs (digital + scanned)
- ✍️ Read handwritten text using OCR
- 🔍 Semantic search using embeddings
- 🤖 AI-powered question answering
- 📊 Shows source context from document
- ⚡ Fast and interactive UI with Streamlit

---

## 🛠 Tech Stack

- Python
- Streamlit
- PyMuPDF (fitz)
- PaddleOCR
- Sentence Transformers
- Scikit-learn (cosine similarity)
- HuggingFace Transformers (FLAN-T5)

---

## ⚙️ How It Works

1. Upload a PDF or image  
2. Text is extracted using:
   - Direct PDF parsing OR
   - OCR (for scanned/handwritten content)
3. Text is split into chunks  
4. Embeddings are created using Sentence Transformers  
5. User asks a question  
6. System finds most relevant chunks  
7. AI model generates answer based on context  

---

## 💻 Installation

```bash
git clone https://github.com/adityakakade897/simple_ai_note.git
cd simple_ai_note
pip install -r requirements.txt
