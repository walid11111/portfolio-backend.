import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory

# Load environment variables (ensure GROQ_API_KEY is set in .env)
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("GROQ_API_KEY not found in environment variables")

def get_rag_chain():
    # Path setup for vectorstore (built from portfolio.json)
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    vectorstore_path = os.path.join(BASE_DIR, "vectorstore")
    
    # Verify path exists and is a directory
    if not os.path.exists(vectorstore_path):
        raise FileNotFoundError(f"Vectorstore directory not found at {vectorstore_path}. Run build_embeddings.py locally first.")
    if not os.path.isdir(vectorstore_path):
        raise ValueError(f"Vectorstore path {vectorstore_path} is not a directory. Expected index.faiss and index.pkl inside.")
    
    print(f"Loading vectorstore from: {vectorstore_path}")  # For debugging in Vercel logs

    # Load FAISS vectorstore with precomputed embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )  # Minimal embedding model for FAISS deserialization
    vectorstore = FAISS.load_local(
        vectorstore_path,
        embeddings,
        allow_dangerous_deserialization=True
    )

    # Connect to Groq LLM with a versatile model
    llm = ChatGroq(
        api_key=api_key,
        model_name="llama-3.3-70b-versatile",
        temperature=0.3
    )

    # Enhanced prompt template: Unchanged
    prompt_template = """
You are a professional AI assistant for Walid Khan's portfolio website. Answer the question directly and concisely based **only** on the provided context from Walid's resume/portfolio and the conversation history. Do not include introductory phrases like "Hello, I'm an AI assistant" or "Introduction to Walid Khan." Begin the response immediately with the relevant information (e.g., "Walid Khan is an AI Engineer..."). Follow these guidelines:

- **Accuracy**: Use only the provided context and chat history. Do not invent, assume, or add information beyond what is given.
- **Clarity**: Structure responses using Markdown: `#` for main headings, `##` for subheadings, `-` for lists, and `**bold**` for emphasis.
- **Conciseness**: Provide direct answers without unnecessary elaboration or repetition.
- **Intent Analysis**:
  - For list requests (e.g., projects, skills), enumerate **all** relevant items from the context without summarizing or omitting any.
  - For follow-up questions, use chat history to provide relevant details (e.g., expand on a specific project mentioned earlier).
  - For unclear or out-of-scope queries, respond exactly: *Sorry, that information isn't in my knowledge base from Walid's portfolio. Could you rephrase or ask about something else, like projects or skills?*
- **Professional Tone**: Keep responses polite, professional, and focused, ending with an offer to assist further if appropriate.

**Chat History**:
{chat_history}

**Context**:
{context}

**Question**:
{question}

**Answer**:
"""
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "chat_history", "question"]
    )

    # Add conversation memory with input_key matching the chain's expected input
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        input_key="question"
    )

    # Build Conversational Retrieval-Augmented QA chain with memory
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 10}),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt},
        return_source_documents=False
    )

    return chain