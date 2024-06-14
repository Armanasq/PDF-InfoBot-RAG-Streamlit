import os
import base64
from llama_index.core import (
    load_index_from_storage,
    VectorStoreIndex,
    SimpleDirectoryReader,
    ChatPromptTemplate,
    Settings,
    StorageContext
)
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from dotenv import load_dotenv
from llama_index.llms.huggingface import HuggingFaceInferenceAPI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from langdetect import detect
import torch
import streamlit as st
from llama_index.core import get_response_synthesizer

# Load environment variables
load_dotenv()

# Define the directory for persistent storage and data
PERSIST_DIR = "./db"
DATA_DIR = "data"

# Ensure data directory exists
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(PERSIST_DIR, exist_ok=True)

# Load GPT-2 model and tokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "mistralai/Mistral-7B-Instruct-v0.3"
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Configure the embedding model to use a local HuggingFace model
embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    device=0 
)

# Configure the language model to use a local HuggingFace model
generator = pipeline('text-generation', model=model, tokenizer=tokenizer)

# Updating Settings to use the correct classes
Settings.embed_model = embed_model
Settings.llm = HuggingFaceInferenceAPI(model_name=model_name, tokenizer_name=model_name)

def generate_text(prompt, max_new_tokens, do_sample, top_p, top_k):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(
        inputs["input_ids"],
        max_length=max_new_tokens,
        do_sample=do_sample,
        top_p=top_p,
        top_k=top_k,
        pad_token_id=tokenizer.eos_token_id
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def display_pdf(file_path):
    with open(file_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'
    return pdf_display

def data_ingestion(chunk_size, chunk_overlap):
    documents = SimpleDirectoryReader(DATA_DIR).load_data()
    storage_context = StorageContext.from_defaults()
    index = VectorStoreIndex.from_documents(documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    index.storage_context.persist(persist_dir=PERSIST_DIR)
def handle_query(query, lang, max_new_tokens, do_sample, top_p, top_k):
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index = load_index_from_storage(storage_context)
    chat_text_qa_msgs = [
        (
            "user",
            """You are a Q&A assistant named FarzanBot. Your main goal is to provide answers as accurately as possible, based on the instructions and context you have been given. If a question does not match the provided context or is outside the scope of the document, kindly advise the user to ask questions within the context of the document.
            Context:
            {context_str}
            Question:
            {query_str}
            """
        )
    ]
    text_qa_template = ChatPromptTemplate.from_messages(chat_text_qa_msgs)
    
    response_synthesizer = get_response_synthesizer(response_mode="compact")  # or use the appropriate mode you need
    query_engine = index.as_query_engine(response_synthesizer=response_synthesizer)
    
    response = query_engine.query(query)
    context_str = response.response  # assuming 'response.response' contains the synthesized response
    
    full_prompt = text_qa_template.format(context_str=context_str, query_str=query)
    answer = generate_text(full_prompt, max_new_tokens, do_sample, top_p, top_k)
    
    return answer, lang

def process_file(uploaded_file, chunk_size, chunk_overlap):
    if uploaded_file:
        filepath = os.path.join(DATA_DIR, "saved_pdf.pdf")
        with open(filepath, "wb") as f:
            f.write(uploaded_file.getbuffer())
        data_ingestion(chunk_size, chunk_overlap)
        return display_pdf(filepath)
    return "No file uploaded."

# Streamlit UI
st.sidebar.title("RAG Hyperparameters")
uploaded_file = st.sidebar.file_uploader("Upload your PDF file", type="pdf")
chunk_size = st.sidebar.slider("Chunk Size", 100, 2000, 512)
chunk_overlap = st.sidebar.slider("Chunk Overlap", 0, 500, 50)
max_new_tokens = st.sidebar.slider("Max New Tokens", 50, 2000, 1024)
do_sample = st.sidebar.checkbox("Do Sample", True)
top_p = st.sidebar.slider("Top P", 0.0, 1.0, 0.9)
top_k = st.sidebar.slider("Top K", 1, 100, 50)

st.title("(PDF) Information and Inference 🗞️")
st.markdown("## Retrieval-Augmented Generation")
st.markdown("Start chat ...🚀")

if uploaded_file:
    pdf_display = process_file(uploaded_file, chunk_size, chunk_overlap)
    st.markdown(pdf_display, unsafe_allow_html=True)

query = st.text_input("Ask me anything about the content of the PDF:")
chat_history = st.empty()

if query:
    lang = detect(query)
    response, lang = handle_query(query, lang, max_new_tokens, do_sample, top_p, top_k)
    chat_history.text_area("Chat History", f"User: {query}\nAssistant: {response}", height=300)
