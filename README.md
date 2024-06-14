# PDF-InfoBot

PDF-InfoBot is a Streamlit application leveraging Retrieval-Augmented Generation (RAG) to analyze and interact with PDF documents. It uses state-of-the-art models from Hugging Face tenable intelligent document querying and contextual responses.

## Features

- **PDF Upload and Display**: Upload and view PDF documents within the app.
- **Chunk-Based Data Ingestion**: Efficiently processes documents in adjustable chunk sizes.
- **RAG for Q&A**: Combines document retrieval with text generation for precise answers.
- **Hyperparameter Customization**: Fine-tune chunk size, chunk overlap, and text generation settings.
- **Multilingual Query Support**: Detects query language for appropriate response generation.

## Installation

1. **Clone the repository**:
    ```sh
    git clone https://github.com/Armanasq/PDF-InfoBot-RAG-Streamlit.git
    cd PDF-InfoBot
    ```

2. **Create a virtual environment**:
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install dependencies**:
    ```sh
    pip install -r requirements.txt
    ```

4. **Set up environment variables**:
    - Create a `.env` file.
    - Add necessary environment variables, such as API keys.

## Usage

1. **Run the application**:
    ```sh
    streamlit run app.py
    ```

2. **Upload a PDF**:
    - Use the sidebar to upload a PDF.
    - Adjust chunk size and overlap for document processing.

3. **Ask Questions**:
    - Enter queries in the text input field.
    - Receive detailed, context-aware responses.

## Code Overview

### Environment Setup

- Load environment variables using `python-dotenv`.

    ```python
    from dotenv import load_dotenv
    load_dotenv()
    ```

### Model and Tokenizer

- Load the Mistral-7B model and tokenizer from Hugging Face.

    ```python
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "mistralai/Mistral-7B-Instruct-v0.3"
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    ```

### Embedding Model

- Configure local Hugging Face embedding model.

    ```python
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding

    embed_model = HuggingFaceEmbedding(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        device=0 
    )
    ```

### Data Ingestion

- Read and process documents in chunks.

    ```python
    from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext

    def data_ingestion(chunk_size, chunk_overlap):
        documents = SimpleDirectoryReader(DATA_DIR).load_data()
        storage_context = StorageContext.from_defaults()
        index = VectorStoreIndex.from_documents(documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        index.storage_context.persist(persist_dir=PERSIST_DIR)
    ```

### Query Handling

- Handle and process user queries.

    ```python
    from llama_index.core import load_index_from_storage, get_response_synthesizer, ChatPromptTemplate

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
        response_synthesizer = get_response_synthesizer(response_mode="compact")
        query_engine = index.as_query_engine(response_synthesizer=response_synthesizer)
        response = query_engine.query(query)
        context_str = response.response
        full_prompt = text_qa_template.format(context_str=context_str, query_str=query)
        answer = generate_text(full_prompt, max_new_tokens, do_sample, top_p, top_k)
        return answer, lang
    ```

### Text Generation

- Generate text based on user queries.

    ```python
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
    ```

### Streamlit Interface

- Build the Streamlit interface for user interaction.

    ```python
    import streamlit as st

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
    ```

## Technical Details

- **Models**: Utilizes `mistralai/Mistral-7B-Instruct-v0.3` for text generation and `sentence-transformers/all-MiniLM-L6-v2` for embeddings.
- **Libraries**: Built with `transformers`, `streamlit`, `llama-index`, and `langdetect`.
- **Storage**: Persistent storage with `VectorStoreIndex` for efficient document retrieval.
- **Hardware**: Optimized for CUDA-capable GPUs, with fallback to CPU.

## Contributing

Fork the repository, create a new branch, and submit a pull request with detailed commit messages.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.