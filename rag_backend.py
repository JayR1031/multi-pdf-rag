import tempfile
import os
from ctransformers import AutoModelForCausalLM
from huggingface_hub import hf_hub_download
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


# ----------------------------------------
# 1. Embeddings
# ----------------------------------------
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


# ----------------------------------------
# 2. Build Chroma vector database from PDFs
# ----------------------------------------
def build_vectordb_from_uploaded_files(uploaded_files):
    docs = []
    for uploaded in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded.getvalue())
            path = tmp.name

        loader = PyPDFLoader(path)
        docs.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(docs)
    embeddings = get_embeddings()

    vectordb = Chroma.from_documents(chunks, embeddings)
    return vectordb


# ----------------------------------------
# 3. Load GGUF Model (local or Spaces)
# ----------------------------------------

def get_llm(model_path=None, model_file=None):
    """
    Loads a small GGUF model via ctransformers.
    Uses a small Llama model as OLMo GGUF models are not readily available.
    Auto-selects the correct model depending on environment.
    Downloads the model from HuggingFace if not already cached.
    """
    model_type = "llama"  # Default model type

    if model_path is None:
        # Detect HuggingFace Spaces GPU
        if "HF_SPACE_ID" in os.environ:
            print("Running on HF Spaces → loading larger model")
            repo_id = "TheBloke/Llama-2-7B-Chat-GGUF"
            model_file = "llama-2-7b-chat.Q4_K_M.gguf"
            model_type = "llama"
        else:
            print("Running locally → loading small model (TinyLlama)")
            repo_id = "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF"
            model_file = "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
            model_type = "llama"

        # Download model file from HuggingFace if not already cached
        print(f"Downloading model from {repo_id}...")
        try:
            local_model_path = hf_hub_download(
                repo_id=repo_id,
                filename=model_file,
                cache_dir=None  # Uses default cache directory
            )
            model_path = os.path.dirname(local_model_path)
        except Exception as e:
            print(f"Error downloading model: {e}")
            print("Falling back to alternative model...")
            # Fallback to a different small model
            repo_id = "microsoft/Phi-3-mini-4k-instruct-gguf"
            model_file = "Phi-3-mini-4k-instruct-q4.gguf"
            model_type = "phi3"
            local_model_path = hf_hub_download(
                repo_id=repo_id,
                filename=model_file,
                cache_dir=None
            )
            model_path = os.path.dirname(local_model_path)

    llm = AutoModelForCausalLM.from_pretrained(
        model_path,
        model_file=model_file,
        model_type=model_type,
        gpu_layers=50,        # allows partial GPU acceleration (Metal on M3)
        temperature=0.2,
        context_length=4096,
    )

    return llm


# ----------------------------------------
# 4. Streaming Text Generator
# ----------------------------------------
def stream_llm_answer(model, prompt, max_new_tokens=256):
    """
    Stream tokens from the model.
    Yields each token as it's generated.
    """
    try:
        for token in model(prompt, stream=True, max_new_tokens=max_new_tokens, temperature=0.2):
            if token:  # Only yield non-empty tokens
                yield token
    except Exception as e:
        # If streaming fails, try non-streaming as fallback
        print(f"Streaming error: {e}, falling back to non-streaming")
        result = model(prompt, stream=False,
                       max_new_tokens=max_new_tokens, temperature=0.2)
        if result:
            # Yield the result character by character to simulate streaming
            for char in result:
                yield char


# ----------------------------------------
# 5. RAG Prompt Construction
# ----------------------------------------
def answer_question(vectordb, llm, question, k=3):
    retriever = vectordb.as_retriever(search_kwargs={"k": k})
    # LangChain 1.0+ uses invoke() instead of get_relevant_documents()
    docs = retriever.invoke(question)

    context = "\n\n".join(d.page_content for d in docs)

    # Llama 2 chat format with safety guidelines and strict context adherence
    prompt = f"""<s>[INST] <<SYS>>
You are a document Q&A assistant. You ONLY answer questions using information from the provided context below. You have NO other knowledge.

STRICT RULES:
1. ONLY use information from the "Context from uploaded documents" section below.
2. If the question cannot be answered from the context, you MUST respond EXACTLY with: "Thank you for your questions but this question is outside of context and suggest you ask about the files you uploaded."
3. Do NOT use any knowledge from your training data. Do NOT make assumptions. Do NOT provide general information.
4. If the context is empty or the question is not about the documents, use the exact response from rule 2.
5. You must NEVER generate, discuss, or reference content related to:
   - Violence, harm, or threats
   - Sexual content or explicit material
   - Murder, death, or graphic descriptions
   - Racism, prejudice, discrimination, or hate speech
   - Any offensive, inappropriate, or harmful topics
6. If asked about inappropriate topics, respond: "I can only answer questions about the content in the uploaded documents, and I cannot discuss that topic."
7. Keep responses professional, respectful, and focused solely on the document content.

Remember: You have NO personal information, NO general knowledge, and NO ability to answer questions outside the provided context.
<</SYS>>

Context from uploaded documents:
{context}

Question: {question} [/INST]"""

    return prompt, docs
