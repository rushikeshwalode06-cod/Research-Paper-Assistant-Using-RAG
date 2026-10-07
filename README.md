# 📚 Research Paper Assistant Using RAG
ResearchMate AI is a RAG-based research paper assistant that lets users upload PDFs and ask questions in natural language. It extracts and chunks document text, generates embeddings using Sentence Transformers, performs semantic search with FAISS, and generates context-aware answers using FLAN-T5 through an interactive Streamlit interface.

# 📚 ResearchMate AI

## 🧠 Intelligent RAG-Based Research Paper Assistant

ResearchMate AI is an intelligent **Retrieval-Augmented Generation (RAG)** based research paper assistant that allows users to upload research papers in PDF format and ask questions related to their documents.

The application extracts text from uploaded research papers, divides the text into smaller chunks, converts those chunks into vector embeddings, stores them using **FAISS**, retrieves the most relevant information using semantic similarity search, and generates a concise answer using the **FLAN-T5** language model.

The system is designed to provide answers based only on the information available in the uploaded research papers.

---

## 🚀 Project Overview

Reading and understanding lengthy research papers can be time-consuming. ResearchMate AI simplifies this process by allowing users to upload research papers and interact with them using natural-language questions.

The application follows a complete RAG pipeline:

```text
Research Paper PDF
        ↓
PDF Text Extraction
        ↓
Text Chunking
        ↓
Sentence Transformer Embeddings
        ↓
FAISS Vector Database
        ↓
Semantic Similarity Search
        ↓
Relevant Context Retrieval
        ↓
FLAN-T5
        ↓
AI Generated Answer
```

---

## ✨ Key Features

* 📄 Upload one or multiple research papers
* 🔍 Ask questions about uploaded research papers
* 🧠 Semantic search using vector embeddings
* ⚡ Fast similarity search using FAISS
* 🤖 AI-generated answers using FLAN-T5
* 📚 Shows retrieved document sources
* 📑 Displays source page numbers
* 🎯 Displays similarity scores
* 🔎 Allows users to view retrieved context
* 💬 Maintains conversation history
* 📊 Displays knowledge-base statistics
* 🗑️ Clear all uploaded documents and session data
* 🎨 Modern dark-themed Streamlit interface
* 🌈 RGB animated UI elements
* 📱 Wide and responsive application layout

---

## 🧠 Technologies Used

### Programming Language

* 🐍 Python

### Framework

* Streamlit

### Natural Language Processing

* Sentence Transformers
* Hugging Face Transformers
* FLAN-T5

### Vector Database

* FAISS

### Deep Learning

* PyTorch

### PDF Processing

* pypdf

### Numerical Computing

* NumPy

### Frontend / UI

* Streamlit
* HTML
* CSS

---

## 🤖 AI Models Used

### 1. Sentence Transformer

The project uses:

```text
sentence-transformers/all-MiniLM-L6-v2
```

This model converts research-paper text and user questions into numerical vector representations called embeddings.

These embeddings are used to find semantically similar pieces of information.

---

### 2. FLAN-T5

The project uses:

```text
google/flan-t5-base
```

FLAN-T5 is responsible for generating the final answer from the context retrieved from the uploaded research papers.

The model is loaded using Hugging Face Transformers.

---
![ml](https://github.com/rushikeshwalode06-cod/Research-Paper-Assistant-Using-RAG/blob/main/Research%20ppr%20image.png?raw=true)

## 🔎 What is RAG?

RAG stands for:

**Retrieval-Augmented Generation**

Instead of directly asking a language model to answer a question from its general knowledge, RAG first retrieves relevant information from a specific knowledge source and then provides that information to the language model.

In this project, the knowledge source is the uploaded research paper.

### RAG Process

```text
User Question
      ↓
Question Embedding
      ↓
FAISS Similarity Search
      ↓
Top Relevant Chunks
      ↓
Context Creation
      ↓
FLAN-T5
      ↓
Final Answer
```

---

# 🏗️ System Architecture

```text
                 ┌──────────────────────┐
                 │   Research Paper PDF │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │   PDF Text Extraction│
                 │       using pypdf    │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │     Text Chunking    │
                 │  Chunk Size = 700    │
                 │  Overlap = 120       │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │ Sentence Transformer │
                 │ all-MiniLM-L6-v2     │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │        FAISS         │
                 │   Vector Database    │
                 └──────────┬───────────┘
                            │
                            │
                  User Question
                            │
                            ▼
                 ┌──────────────────────┐
                 │ Question Embedding  │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │ Semantic Similarity │
                 │      Search         │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │ Top Relevant Chunks │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │     FLAN-T5 Model   │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │    AI Generated     │
                 │       Answer        │
                 └──────────────────────┘
```

---

## ⚙️ How the Project Works

### 1. PDF Upload

Users can upload one or multiple research papers in PDF format.

The application accepts multiple PDF files at the same time.

---

### 2. PDF Text Extraction

The project uses `PdfReader` from `pypdf` to read the uploaded research papers.

Each page is processed individually and its text is extracted.

The application also stores the document name and page number along with each chunk.

---

### 3. Text Chunking

Large documents are divided into smaller pieces before creating embeddings.

The project uses:

```text
Chunk Size = 700
Overlap = 120
```

This allows the retrieval system to work with smaller and more meaningful sections of the research paper.

---

### 4. Embedding Generation

The extracted chunks are converted into embeddings using:

```text
all-MiniLM-L6-v2
```

The embeddings are normalized before being stored.

The generated embeddings are converted into NumPy arrays and prepared for vector indexing.

---

### 5. FAISS Vector Database

The project uses FAISS for efficient vector similarity search.

The generated embeddings are added to the FAISS index.

---

### 6. User Question

The user can enter a natural-language question related to the uploaded research paper.

Example questions:

```text
What is the main objective of the research paper?
```

```text
What methodology was used?
```

```text
What are the key results?
```

---

### 7. Question Embedding

The user's question is converted into an embedding using the same Sentence Transformer model.

This allows the system to compare the question with the document chunks.

---

### 8. Semantic Search

The system searches the FAISS vector database to retrieve the most relevant chunks.

The project retrieves up to **4 relevant chunks** for answering a question.

---

### 9. Context Creation

The retrieved chunks are combined into a context containing:

* Document name
* Page number
* Retrieved text

This context is then passed to the language model.

---

### 10. Prompt Engineering

The project uses a controlled prompt instructing the model to answer using the retrieved context.

The system also provides a fallback response when the required information cannot be found in the retrieved context.

---

### 11. AI Answer Generation

The FLAN-T5 model generates the final response from the retrieved context.

The generated output is decoded and displayed as the AI answer.

---

## 📚 Source Tracking

ResearchMate AI provides source information for the retrieved content.

For each retrieved source, the application displays:

* Document name
* Page number
* Similarity score

This helps users identify where the retrieved information came from.

---

## 🔎 Retrieved Context

Users can expand the **View Retrieved Context** section to inspect the text chunks retrieved from the research paper.

For each retrieved chunk, the application displays:

* 📄 Document name
* 📑 Page number
* 🎯 Similarity score
* 📝 Retrieved text

---

## 💬 Conversation History

ResearchMate AI maintains questions and generated answers during the current Streamlit session.

The conversation history allows users to see previous questions and their corresponding AI responses.

---

## 📊 Knowledge Base

After uploading research papers, the application maintains information about:

* 📄 Uploaded documents
* 🧩 Text chunks
* 🔢 Vector count
* 📚 Current knowledge base

The application displays these statistics through the dashboard.

---

## 🎯 Project Objectives

The main objectives of ResearchMate AI are:

* 🎯 Simplify research-paper analysis.
* ❓ Allow users to ask questions directly from research papers.
* 🔍 Retrieve semantically relevant information.
* ⏱️ Reduce the time required to manually search through papers.
* 🤖 Generate concise answers from retrieved document context.
* 📑 Provide source and page information for retrieved content.
* 🧠 Demonstrate a practical implementation of Retrieval-Augmented Generation.
* 🚀 Provide an interactive AI-powered research assistant.

---

## 🌟 Advantages

* 😊 Easy to use
* 📚 Supports multiple research papers
* 🔍 Semantic rather than simple keyword-based retrieval
* ⚡ Uses vector search
* 📑 Source-aware responses
* 👀 Shows retrieved context
* 💬 Maintains conversation history
* 🎨 Interactive Streamlit interface
* 🎓 Useful for research and academic document analysis
---

## ⚠️ Limitations

The current implementation has some limitations:

* 📄 The application works with PDF documents containing extractable text.
* 🖼️ Scanned/image-only PDFs may not provide usable text without OCR.
* 💾 The knowledge base is maintained in the current Streamlit session.
* 🗂️ FAISS index data is not permanently stored on disk.
* 🎯 Answer quality depends on extracted text and retrieved chunks.
* 📚 Very large documents may require additional optimization.
* 🔍 The current retrieval configuration uses up to four relevant chunks.
* 🖥️ FLAN-T5 generation can require significant computational resources.

---

## 🔮 Future Improvements

Possible future improvements include:

* 🔹 OCR support for scanned research papers
* 🔹 Persistent vector database
* 🔹 ChromaDB or Pinecone integration
* 🔹 Improved chunking strategies
* 🔹 Hybrid keyword + semantic search
* 🔹 Reranking models
* 🔹 Citation-aware answer generation
* 🔹 Multi-language research-paper support
* 🔹 Research-paper summarization
* 🔹 Automatic abstract generation
* 🔹 Voice-based questions
* 🔹 Chat history persistence
* 🔹 User authentication
* 🔹 Cloud deployment
* 🔹 Larger and more powerful LLM integration
* 🔹 Document comparison
* 🔹 Multiple-paper cross-question answering

---

## 🧠 Skills Demonstrated

🐍 Python
🎨 Streamlit
📝 Natural Language Processing
🔄 Retrieval-Augmented Generation
🧠 Large Language Models
🔤 Sentence Transformers
📊 Text Embeddings
🗄️ Vector Databases
⚡ FAISS
🔍 Semantic Search
🤗 Hugging Face Transformers
🤖 FLAN-T5
🔥 PyTorch
📄 PDF Processing
✍️ Prompt Engineering
📚 Context Retrieval
✨ Generative AI
🚀 AI Application Development

---

## 🔑 Keywords

```text
🔹 RAG
🔹 Retrieval-Augmented Generation
🔹 Generative AI
🔹 NLP
🔹 LLM
🔹 FAISS
🔹 Vector Database
🔹 Semantic Search
🔹 Sentence Transformers
🔹 Embeddings
🔹 FLAN-T5
🔹 Hugging Face
🔹 Transformers
🔹 PyTorch
🔹 Streamlit
🔹 Python
🔹 PDF Processing
🔹 Prompt Engineering
🔹 Artificial Intelligence
🔹 Research Assistant
```

---

![ml](https://github.com/rushikeshwalode06-cod/Research-Paper-Assistant-Using-RAG/blob/main/R1.png?raw=true)

## 🏁 Conclusion

ResearchMate AI demonstrates a practical implementation of a **Retrieval-Augmented Generation system for research-paper question answering**.

The application combines **PDF processing, text chunking, Sentence Transformer embeddings, FAISS vector search, semantic retrieval, prompt engineering, and FLAN-T5 generation** into a single interactive application.

Instead of manually searching through lengthy research papers, users can upload their documents and ask questions in natural language. The system retrieves relevant information from the uploaded papers and generates an answer based on the retrieved context.

Overall, ResearchMate AI demonstrates how modern **NLP, Generative AI, Vector Search, and RAG techniques** can be combined to build a useful real-world research assistant.

---
![ml](https://github.com/rushikeshwalode06-cod/Research-Paper-Assistant-Using-RAG/blob/main/Re%20ppr.jpeg?raw=true)


