import os
import html
import hashlib
import tempfile
import textwrap

import streamlit as st
import numpy as np
import faiss
import torch

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="ResearchMate AI",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# SAFE HTML RENDER FUNCTION
# ============================================================

def render_html(content):
    """
    Render custom HTML reliably.
    st.html prevents Streamlit from showing nested HTML tags as plain text.
    A fallback is kept for older Streamlit versions.
    """
    content = textwrap.dedent(content).strip()

    if hasattr(st, "html"):
        st.html(content)
    else:
        st.markdown(content, unsafe_allow_html=True)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at 5% 10%,
            rgba(255, 0, 128, 0.18),
            transparent 25%),

        radial-gradient(circle at 95% 5%,
            rgba(0, 229, 255, 0.16),
            transparent 25%),

        radial-gradient(circle at 50% 100%,
            rgba(124, 58, 237, 0.20),
            transparent 30%),

        linear-gradient(
            135deg,
            #020617,
            #0f172a,
            #111827
        );

    color: #f8fafc;
}

.block-container {
    max-width: 1350px;
    padding-top: 1.5rem;
    padding-bottom: 3rem;
}


/* =========================================================
   RGB BORDER
   ========================================================= */

.rgb-border {
    padding: 2px;
    border-radius: 25px;

    background: linear-gradient(
        90deg,
        #ff0080,
        #7928ca,
        #00d4ff,
        #00ff87,
        #ff0080
    );

    background-size: 400% 400%;

    animation: rgbAnimation 8s ease infinite;

    margin-bottom: 28px;
}

@keyframes rgbAnimation {

    0% {
        background-position: 0% 50%;
    }

    50% {
        background-position: 100% 50%;
    }

    100% {
        background-position: 0% 50%;
    }

}


/* =========================================================
   HERO
   ========================================================= */

.hero {
    background: rgba(2, 6, 23, 0.92);
    border-radius: 20px;
    padding: 25px 20px;
    text-align: center;
    backdrop-filter: blur(20px);
}

.hero-title-row {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 12px;
}

.hero-icon {
    font-size: 42px;
    line-height: 1;
}

.hero-title {
    font-size: 46px;
    font-weight: 800;
    background: linear-gradient(
        90deg,
        #ff0080,
        #7c3aed,
        #00d4ff,
        #00ff87
    );
    background-size: 300% auto;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: titleRGB 6s linear infinite;
}

@keyframes titleRGB {
    0% {
        background-position: 0% center;
    }
    100% {
        background-position: 300% center;
    }
}

.hero-subtitle {
    color: #cbd5e1;
    font-size: 16px;
    margin-top: 8px;
}

.hero-tag {
    display: inline-block;
    margin-top: 12px;
    padding: 7px 16px;
    border-radius: 30px;
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.14);
    color: #e2e8f0;
    font-size: 13px;
}


/* =========================================================
   SECTION TITLE
   ========================================================= */

.section-title {
    font-size: 28px;

    font-weight: 800;

    margin-top: 25px;

    margin-bottom: 15px;

    background: linear-gradient(
        90deg,
        #60a5fa,
        #c084fc,
        #f472b6
    );

    -webkit-background-clip: text;

    -webkit-text-fill-color: transparent;
}


/* =========================================================
   GLASS CARD
   ========================================================= */

.glass-card {

    background:
        linear-gradient(
            135deg,
            rgba(255,255,255,0.08),
            rgba(255,255,255,0.025)
        );

    border:
        1px solid rgba(255,255,255,0.12);

    border-radius: 20px;

    padding: 22px;

    margin-bottom: 18px;

    backdrop-filter: blur(15px);

    box-shadow:
        0 15px 40px rgba(0,0,0,0.25);
}


/* =========================================================
   QUESTION CARD
   ========================================================= */

.question-card {

    background:
        linear-gradient(
            135deg,
            rgba(255,0,128,0.12),
            rgba(124,58,237,0.13),
            rgba(0,212,255,0.10)
        );

    border:
        1px solid rgba(255,255,255,0.15);

    border-radius: 20px;

    padding: 24px;

    margin: 18px 0;

    box-shadow:
        0 12px 40px rgba(0,0,0,0.25);
}

.question-label {

    color: #67e8f9;

    font-size: 13px;

    font-weight: 800;

    letter-spacing: 1.5px;

    margin-bottom: 10px;
}

.question-text {

    color: #ffffff;

    font-size: 20px;

    font-weight: 600;

    line-height: 1.6;
}


/* =========================================================
   ANSWER CARD
   ========================================================= */

.answer-card {

    background:
        linear-gradient(
            135deg,
            rgba(0,255,135,0.08),
            rgba(0,212,255,0.08),
            rgba(124,58,237,0.10)
        );

    border:
        1px solid rgba(0,229,255,0.22);

    border-radius: 20px;

    padding: 26px;

    margin-bottom: 20px;

    box-shadow:
        0 12px 45px rgba(0,0,0,0.25);
}

.answer-label {

    color: #00e5ff;

    font-size: 13px;

    font-weight: 800;

    letter-spacing: 1.5px;

    margin-bottom: 12px;
}

.answer-text {

    color: #f8fafc;

    font-size: 17px;

    line-height: 1.8;
}


/* =========================================================
   SOURCE CARD
   ========================================================= */

.source-card {

    background:
        rgba(15,23,42,0.72);

    border-left:
        4px solid #00d4ff;

    border-radius: 14px;

    padding: 17px;

    margin-bottom: 12px;
}

.source-title {

    color: #ffffff;

    font-size: 15px;

    font-weight: 700;
}

.source-info {

    color: #94a3b8;

    font-size: 13px;

    line-height: 1.8;

    margin-top: 8px;
}


/* =========================================================
   STAT CARD
   ========================================================= */

.stat-card {

    background:
        linear-gradient(
            135deg,
            rgba(255,255,255,0.08),
            rgba(255,255,255,0.025)
        );

    border:
        1px solid rgba(255,255,255,0.12);

    border-radius: 18px;

    padding: 20px;

    text-align: center;

    transition: 0.3s;
}

.stat-card:hover {

    transform: translateY(-4px);

    border-color:
        rgba(0,212,255,0.40);

    box-shadow:
        0 10px 35px rgba(0,212,255,0.12);
}

.stat-number {

    font-size: 27px;

    font-weight: 800;

    background:
        linear-gradient(
            90deg,
            #60a5fa,
            #c084fc,
            #f472b6
        );

    -webkit-background-clip: text;

    -webkit-text-fill-color: transparent;
}

.stat-label {

    color: #94a3b8;

    font-size: 13px;

    margin-top: 5px;
}


/* =========================================================
   SIDEBAR
   ========================================================= */

section[data-testid="stSidebar"] {

    background:
        linear-gradient(
            180deg,
            #020617,
            #0f172a,
            #111827
        );

    border-right:
        1px solid rgba(255,255,255,0.08);
}

.sidebar-title {

    font-size: 26px;

    font-weight: 800;

    background:
        linear-gradient(
            90deg,
            #ff0080,
            #7c3aed,
            #00d4ff
        );

    -webkit-background-clip: text;

    -webkit-text-fill-color: transparent;
}

.sidebar-box {
    background:
        linear-gradient(
            135deg,
            rgba(30, 41, 59, 0.92),
            rgba(15, 23, 42, 0.96)
        );
    border: 1px solid rgba(148, 163, 184, 0.22);
    border-radius: 16px;
    padding: 16px;
    margin-bottom: 15px;
    color: #e2e8f0 !important;
    box-shadow: 0 10px 30px rgba(0,0,0,0.18);
}

.sidebar-box,
.sidebar-box * {
    color: #e2e8f0 !important;
}

section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
    color: #e2e8f0 !important;
}

section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h1,
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h2,
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h3 {
    color: #f8fafc !important;
}

section[data-testid="stSidebar"] .stAlert {
    border-radius: 14px;
}



/* =========================================================
   FILE UPLOADER
   ========================================================= */

[data-testid="stFileUploader"] {

    background:
        linear-gradient(
            135deg,
            rgba(255,0,128,0.06),
            rgba(0,212,255,0.06)
        );

    border:
        1px dashed rgba(96,165,250,0.45);

    border-radius: 18px;

    padding: 10px;
}


/* =========================================================
   TEXT AREA
   ========================================================= */

textarea {

    background:
        rgba(2,6,23,0.80) !important;

    color:
        #ffffff !important;

    border:
        1px solid rgba(255,255,255,0.15) !important;

    border-radius:
        15px !important;
}


/* =========================================================
   BUTTON
   ========================================================= */

.stButton > button {

    border-radius: 14px !important;

    min-height: 50px;

    font-weight: 700;

    background:
        linear-gradient(
            90deg,
            #7c3aed,
            #2563eb
        );

    color: white;

    border:
        1px solid rgba(255,255,255,0.12);

    transition: 0.3s;
}

.stButton > button:hover {

    transform: translateY(-2px);

    box-shadow:
        0 10px 30px rgba(124,58,237,0.35);
}


/* =========================================================
   FOOTER
   ========================================================= */

.footer {

    text-align: center;

    color: #64748b;

    font-size: 13px;

    padding: 30px 0;
}


#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# HERO
# ============================================================

render_html("""
<div class="rgb-border">
    <div class="hero">
        <div class="hero-title-row">
            <div class="hero-icon">📚</div>
            <div class="hero-title">ResearchMate AI</div>
        </div>

        <div class="hero-subtitle">
            Intelligent Research Paper Assistant powered by Retrieval-Augmented Generation
        </div>

        <div class="hero-tag">
            ✨ PDF → Embeddings → FAISS → Semantic Search → AI Answer
        </div>
    </div>
</div>
""")


# ============================================================
# LOAD MODELS
# ============================================================

@st.cache_resource
def load_models():

    embedding_model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    model_name = "google/flan-t5-base"

    tokenizer = AutoTokenizer.from_pretrained(
        model_name
    )

    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name
    )

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model = model.to(device)

    return (
        embedding_model,
        tokenizer,
        model,
        device
    )


with st.spinner("🧠 Loading AI models..."):

    embedding_model, tokenizer, llm_model, device = load_models()


# ============================================================
# SESSION STATE
# ============================================================

if "documents" not in st.session_state:
    st.session_state.documents = []

if "chunks" not in st.session_state:
    st.session_state.chunks = []

if "index" not in st.session_state:
    st.session_state.index = None

if "processed_files" not in st.session_state:
    st.session_state.processed_files = []

if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_question" not in st.session_state:
    st.session_state.last_question = ""

if "last_answer" not in st.session_state:
    st.session_state.last_answer = ""

if "last_sources" not in st.session_state:
    st.session_state.last_sources = []


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    render_html("""
    <div class="sidebar-title">
    ⚡ ResearchMate
    </div>
    """)

    st.success("🟢 RAG Engine Ready")

    st.markdown("---")

    st.markdown("### 🧠 AI Architecture")

    render_html("""
    <div class="sidebar-box">

    🔹 Sentence Transformers<br>
    🔹 FAISS Vector Database<br>
    🔹 Semantic Retrieval<br>
    🔹 FLAN-T5 LLM<br>
    🔹 Retrieval-Augmented Generation

    </div>
    """)

    st.markdown("### 💻 System")

    vector_count = (
        st.session_state.index.ntotal
        if st.session_state.index
        else 0
    )

    render_html(f"""
    <div class="sidebar-box">

    ⚡ Device:
    <b>{device.upper()}</b>
    <br><br>

    📄 Documents:
    <b>{len(st.session_state.documents)}</b>
    <br><br>

    🧩 Chunks:
    <b>{len(st.session_state.chunks)}</b>
    <br><br>

    🔎 Vectors:
    <b>{vector_count}</b>

    </div>
    """)

    st.markdown("---")

    if st.button(
        "🗑️ Clear Everything",
        use_container_width=True
    ):

        st.session_state.documents = []
        st.session_state.chunks = []
        st.session_state.index = None
        st.session_state.processed_files = []
        st.session_state.messages = []
        st.session_state.last_question = ""
        st.session_state.last_answer = ""
        st.session_state.last_sources = []

        st.rerun()


# ============================================================
# UPLOAD
# ============================================================

render_html("""
<div class="section-title">
📄 Upload Research Papers
</div>
""")

uploaded_files = st.file_uploader(
    "Upload one or more PDF research papers",
    type=["pdf"],
    accept_multiple_files=True
)


# ============================================================
# PROCESS PDF
# ============================================================

if uploaded_files:

    new_files = []

    for uploaded_file in uploaded_files:

        file_hash = hashlib.md5(
            uploaded_file.getvalue()
        ).hexdigest()

        if file_hash not in st.session_state.processed_files:

            new_files.append(
                (uploaded_file, file_hash)
            )

    if new_files:

        progress = st.progress(0)

        status = st.empty()

        all_chunks = []

        for file_number, (
            uploaded_file,
            file_hash
        ) in enumerate(new_files):

            status.info(
                f"📖 Reading {uploaded_file.name}..."
            )

            temp_path = None

            try:

                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".pdf"
                ) as temp_file:

                    temp_file.write(
                        uploaded_file.getvalue()
                    )

                    temp_path = temp_file.name

                reader = PdfReader(temp_path)

                for page_number, page in enumerate(
                    reader.pages,
                    start=1
                ):

                    text = page.extract_text()

                    if not text:
                        continue

                    text = text.strip()

                    chunk_size = 700
                    overlap = 120

                    start = 0

                    while start < len(text):

                        end = start + chunk_size

                        chunk_text = text[start:end]

                        if chunk_text.strip():

                            all_chunks.append({

                                "text":
                                    chunk_text.strip(),

                                "source":
                                    uploaded_file.name,

                                "page":
                                    page_number
                            })

                        start += (
                            chunk_size - overlap
                        )

                st.session_state.processed_files.append(
                    file_hash
                )

                st.session_state.documents.append(
                    uploaded_file.name
                )

                progress.progress(
                    (file_number + 1)
                    / len(new_files)
                )

            except Exception as e:

                st.error(
                    f"❌ Error reading PDF: {e}"
                )

            finally:

                if temp_path:

                    try:
                        os.remove(temp_path)
                    except:
                        pass

        # ====================================================
        # ADD CHUNKS
        # ====================================================

        st.session_state.chunks.extend(
            all_chunks
        )

        # ====================================================
        # EMBEDDINGS
        # ====================================================

        if all_chunks:

            status.info(
                "🧠 Creating embeddings..."
            )

            texts = [
                item["text"]
                for item in all_chunks
            ]

            embeddings = embedding_model.encode(
                texts,
                convert_to_numpy=True,
                show_progress_bar=False,
                normalize_embeddings=True
            )

            embeddings = np.asarray(
                embeddings
            ).astype("float32")

            dimension = embeddings.shape[1]

            new_index = faiss.IndexFlatIP(
                dimension
            )

            new_index.add(
                embeddings
            )

            if st.session_state.index is None:

                st.session_state.index = new_index

            else:

                st.session_state.index.add(
                    embeddings
                )

            status.success(
                f"✅ {len(all_chunks)} chunks indexed!"
            )


# ============================================================
# KNOWLEDGE BASE
# ============================================================

if st.session_state.documents:

    render_html("""
    <div class="section-title">
    📊 Knowledge Base
    </div>
    """)

    col1, col2, col3, col4 = st.columns(4)

    vector_count = (
        st.session_state.index.ntotal
        if st.session_state.index
        else 0
    )

    with col1:

        render_html(f"""
        <div class="stat-card">

        <div class="stat-number">
        📄 {len(st.session_state.documents)}
        </div>

        <div class="stat-label">
        Documents
        </div>

        </div>
        """)

    with col2:

        render_html(f"""
        <div class="stat-card">

        <div class="stat-number">
        🧩 {len(st.session_state.chunks)}
        </div>

        <div class="stat-label">
        Text Chunks
        </div>

        </div>
        """)

    with col3:

        render_html(f"""
        <div class="stat-card">

        <div class="stat-number">
        🔎 {vector_count}
        </div>

        <div class="stat-label">
        Vectors
        </div>

        </div>
        """)

    with col4:

        render_html(f"""
        <div class="stat-card">

        <div class="stat-number">
        ⚡ {device.upper()}
        </div>

        <div class="stat-label">
        Compute
        </div>

        </div>
        """)


# ============================================================
# DOCUMENTS
# ============================================================

if st.session_state.documents:

    with st.expander(
        "📚 View Uploaded Research Papers"
    ):

        for document in st.session_state.documents:

            st.write(
                f"📄 {document}"
            )


# ============================================================
# CHAT HISTORY
# ============================================================

if st.session_state.messages:

    render_html("""
    <div class="section-title">
    💬 Conversation History
    </div>
    """)

    for message in st.session_state.messages:

        if message["role"] == "user":

            render_html(f"""
            <div class="question-card">

            <div class="question-label">
            👤 YOUR QUESTION
            </div>

            <div class="question-text">
            {html.escape(message["content"])}
            </div>

            </div>
            """)

        else:

            render_html(f"""
            <div class="answer-card">

            <div class="answer-label">
            🤖 RESEARCHMATE AI
            </div>

            <div class="answer-text">
            {html.escape(message["content"])}
            </div>

            </div>
            """)


# ============================================================
# QUESTION
# ============================================================

render_html("""
<div class="section-title">
🔍 Ask Your Research Paper
</div>
""")

question = st.text_area(
    "Enter your question",
    placeholder=(
        "Example: What is the main objective of U-Net?"
    ),
    height=130,
    key="question_box"
)


st.caption(
    "💡 Ask any question related to your uploaded research paper."
)


# ============================================================
# EXAMPLE QUESTIONS
# ============================================================

col1, col2, col3 = st.columns(3)

with col1:
    st.info("🎯 What is the main objective?")

with col2:
    st.info("🧠 What methodology was used?")

with col3:
    st.info("📊 What are the key results?")


# ============================================================
# GENERATE ANSWER
# ============================================================

generate = st.button(
    "🚀 Generate AI Answer",
    use_container_width=True,
    type="primary"
)


if generate:

    if not st.session_state.documents:

        st.warning(
            "⚠️ Please upload a PDF first."
        )

    elif not question.strip():

        st.warning(
            "⚠️ Please enter a question."
        )

    elif st.session_state.index is None:

        st.error(
            "❌ Knowledge base is not ready."
        )

    else:

        clean_question = question.strip()

        st.session_state.last_question = (
            clean_question
        )

        st.session_state.messages.append({

            "role": "user",

            "content": clean_question

        })

        # ====================================================
        # EMBED QUESTION
        # ====================================================

        with st.spinner(
            "🔎 Searching relevant information..."
        ):

            question_embedding = (
                embedding_model.encode(
                    [clean_question],
                    convert_to_numpy=True,
                    normalize_embeddings=True
                )
                .astype("float32")
            )

        # ====================================================
        # SEARCH
        # ====================================================

        top_k = min(
            4,
            st.session_state.index.ntotal
        )

        scores, indices = (
            st.session_state.index.search(
                question_embedding,
                top_k
            )
        )

        retrieved_chunks = []

        for score, idx in zip(
            scores[0],
            indices[0]
        ):

            if idx < 0:
                continue

            retrieved_chunks.append({

                **st.session_state.chunks[idx],

                "score":
                    float(score)

            })

        # ====================================================
        # CONTEXT
        # ====================================================

        context_parts = []

        for item in retrieved_chunks:

            context_parts.append(
                f"""
SOURCE: {item['source']}
PAGE: {item['page']}

{item['text']}
"""
            )

        context = "\n\n".join(
            context_parts
        )

        # ====================================================
        # PROMPT
        # ====================================================

        prompt = f"""
You are ResearchMate AI, a research paper assistant.

Answer the user's question using ONLY the information
provided in the CONTEXT.

Do not use outside knowledge.

If the answer cannot be found in the context,
respond exactly:

The answer is not available in the uploaded document.

Give a clear and concise answer.

CONTEXT:

{context}

QUESTION:

{clean_question}

ANSWER:
"""

        # ====================================================
        # GENERATE
        # ====================================================

        with st.spinner(
            "🤖 Generating AI answer..."
        ):

            try:

                inputs = tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=2048
                )

                inputs = {
                    key: value.to(device)
                    for key, value in inputs.items()
                }

                output = llm_model.generate(

                    **inputs,

                    max_new_tokens=220,

                    do_sample=False,

                    num_beams=2

                )

                answer = tokenizer.decode(
                    output[0],
                    skip_special_tokens=True
                )

            except Exception as e:

                answer = (
                    f"Error generating answer: {e}"
                )

        # ====================================================
        # SAVE
        # ====================================================

        st.session_state.last_answer = answer

        st.session_state.last_sources = (
            retrieved_chunks
        )

        st.session_state.messages.append({

            "role": "assistant",

            "content": answer

        })

        st.rerun()


# ============================================================
# CURRENT QUESTION
# ============================================================

if st.session_state.last_question:

    render_html("""
    <div class="section-title">
    🎯 Current Question
    </div>
    """)

    render_html(f"""
    <div class="question-card">

    <div class="question-label">
    ❓ QUESTION ASKED
    </div>

    <div class="question-text">
    {html.escape(
        st.session_state.last_question
    )}
    </div>

    </div>
    """)


# ============================================================
# CURRENT ANSWER
# ============================================================

if st.session_state.last_answer:

    render_html("""
    <div class="section-title">
    🤖 AI Generated Answer
    </div>
    """)

    render_html(f"""
    <div class="answer-card">

    <div class="answer-label">
    ✨ RESEARCHMATE AI RESPONSE
    </div>

    <div class="answer-text">
    {html.escape(
        st.session_state.last_answer
    )}
    </div>

    </div>
    """)


# ============================================================
# SOURCES
# ============================================================

if st.session_state.last_sources:

    render_html("""
    <div class="section-title">
    📚 Retrieved Sources
    </div>
    """)

    for i, source in enumerate(
        st.session_state.last_sources,
        start=1
    ):

        render_html(f"""
        <div class="source-card">

        <div class="source-title">
        🔹 Source {i}
        </div>

        <div class="source-info">

        📄 Document:
        <b>{html.escape(source["source"])}</b>

        <br>

        📑 Page:
        <b>{source["page"]}</b>

        <br>

        🎯 Similarity Score:
        <b>{source["score"]:.3f}</b>

        </div>

        </div>
        """)


# ============================================================
# RETRIEVED CONTEXT
# ============================================================

if st.session_state.last_sources:

    with st.expander(
        "🔎 View Retrieved Context"
    ):

        for i, source in enumerate(
            st.session_state.last_sources,
            start=1
        ):

            st.markdown(
                f"### 🧩 Retrieved Chunk {i}"
            )

            st.write(
                f"📄 Document: {source['source']}"
            )

            st.write(
                f"📑 Page: {source['page']}"
            )

            st.write(
                f"🎯 Similarity: {source['score']:.3f}"
            )

            st.write(
                source["text"]
            )

            st.markdown("---")


# ============================================================
# FOOTER
# ============================================================

render_html("""
<div class="footer">

<b>ResearchMate AI</b>

<br><br>

Intelligent RAG-Based Research Paper Assistant

<br><br>

PDF → Text Extraction → Chunking → Embeddings
→ FAISS → Retrieval → FLAN-T5

<br><br>

🚀 Built with Python • Streamlit • FAISS • Transformers

</div>
""")