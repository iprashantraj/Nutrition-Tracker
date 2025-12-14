import streamlit as st
import os

# --- SQLite Fix for Streamlit Cloud (ChromaDB) ---
try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import tempfile
from PIL import Image

# --- Configuration & Setup ---
st.set_page_config(page_title="Advanced Nutrition Tracker", page_icon="🥗", layout="wide")

# Load environment variables (for Developer Key)
load_dotenv()
DEV_API_KEY = os.getenv("GOOGLE_API_KEY")

# --- CSS Styling for Premium Look ---
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .stButton>button {
        width: 100%;
        background-color: #ff4b4b;
        color: white;
        border-radius: 10px;
        height: 50px;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #ff6b6b;
        border: none;
    }
    .metric-card {
        background-color: #262730;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    h1, h2, h3 {
        color: #ffffff;
    }
    </style>
""", unsafe_allow_html=True)

# --- Sidebar: Hybrid Auth System ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2921/2921822.png", width=100)
    st.title("Settings")
    
    st.markdown("### 🔑 API Access")
    user_api_key = st.text_input("Enter your Google API Key", type="password", help="Get one from aistudio.google.com")
    
    auth_status = st.empty()
    
    # Initialize Session State for Demo Count
    if 'demo_count' not in st.session_state:
        st.session_state.demo_count = 0
    
    final_api_key = None
    
    if user_api_key:
        final_api_key = user_api_key
        auth_status.success("Using User Key ✅")
    else:
        if st.session_state.demo_count < 2:
            final_api_key = DEV_API_KEY
            remaining = 2 - st.session_state.demo_count
            auth_status.info(f"Using Demo Key (Free tries left: {remaining}) ⚠️")
        else:
            auth_status.error("Demo limit exceeded! Please enter your own API Key.")
            final_api_key = None

    st.markdown("---")
    st.markdown("### 📚 Database Check")
    if os.path.exists("fssai_chroma_db"):
        st.success("FSSAI Database Found ✅")
    else:
        st.warning("FSSAI Database Missing ❌")
        if st.button("Build Database (First Time Only)"):
            with st.spinner("Building vector database from PDFs..."):
                try:
                    # Logic to build DB
                    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
                    pdf_dir = "data/fssai"
                    documents = []
                    if os.path.exists(pdf_dir):
                        for file in os.listdir(pdf_dir):
                            if file.endswith(".pdf"):
                                loader = PyPDFLoader(os.path.join(pdf_dir, file))
                                documents.extend(loader.load())
                        
                        text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=500)
                        docs = text_splitter.split_documents(documents)
                        vector_db = Chroma.from_documents(docs, embeddings, persist_directory="fssai_chroma_db")
                        st.success("Database Built Successfully! Refresh page.")
                    else:
                        st.error("Project/fssai directory not found!")
                except Exception as e:
                    st.error(f"Error: {e}")

# --- Main App Functions ---

# --- Main App Functions ---

def analyze_image_with_gemini(image, api_key):
    """Uses Gemini 1.5 Flash Vision to extract nutrition info."""
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key)
    
    prompt = """
    You are an expert nutritionist. Look at this food label image.
    1. Extract all nutritional information (Calories, Sugar, Fat, Protein, Ingredients).
    2. Identify the product name.
    3. Return the data in valid JSON format.
    """
    
    from langchain.schema.messages import HumanMessage
    
    # Convert PIL image to bytes for LangChain
    import io
    import base64
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

    message = HumanMessage(
        content=[
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": f"data:image/jpeg;base64,{img_str}"}
        ]
    )
    
    try:
        response = llm.invoke([message])
        return response.content
    except Exception as e:
        return f"Error: {str(e)}"

def get_fssai_context(query, api_key):
    """Retrieves relevant FSSAI regulations using RAG (Manual Implementation)."""
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_db = Chroma(persist_directory="fssai_chroma_db", embedding_function=embeddings)
    
    retriever = vector_db.as_retriever(search_kwargs={"k": 2})
    
    # 1. Retrieve Documents
    docs = retriever.invoke(query)
    context_text = "\\n\\n".join([doc.page_content for doc in docs])
    
    if not context_text:
        return "No specific FSSAI regulations found for this query."

    # 2. Generate Answer
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key)
    
    rag_prompt = f"""
    You are an expert on FSSAI (Food Safety and Standards Authority of India) regulations.
    Use the following legal context to answer the user's query about food compliance.
    
    Context:
    {context_text}
    
    Query: {query}
    
    Answer:
    """
    
    try:
        response = llm.invoke(rag_prompt)
        return response.content
    except Exception as e:
        return f"Retrieval Error: {str(e)}"

# --- Main UI Layout ---
st.title("🥗 Intelligent Nutrition Tracker")
st.markdown("### scan. analyze. eat smart.")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📸 Upload Product Label")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Label', use_container_width=True)
        
        analyze_btn = st.button("🚀 Analyze Nutrition", disabled=(final_api_key is None))
        
        if analyze_btn:
            if final_api_key:
                # Increment usage if using demo key
                if not user_api_key:
                    st.session_state.demo_count += 1
                
                with st.spinner("🤖 Vision AI is reading the label..."):
                    # Step 1: Vision Extraction
                    raw_data = analyze_image_with_gemini(image, final_api_key)
                    st.session_state['raw_data'] = raw_data
                
                with st.spinner("⚖️ Consulting FSSAI Regulations..."):
                    # Step 2: FSSAI RAG
                    fssai_insight = get_fssai_context(f"Check regulations for: {raw_data}", final_api_key)
                    st.session_state['fssai_insight'] = fssai_insight
                    
                st.success("Analysis Complete!")

with col2:
    st.subheader("📊 Health Report")
    
    if 'raw_data' in st.session_state:
        with st.expander("📝 Extracted Data (Raw)", expanded=False):
            st.write(st.session_state['raw_data'])
            
    if 'fssai_insight' in st.session_state:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("#### 🏛️ FSSAI Compliance Check")
        st.write(st.session_state['fssai_insight'])
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Final Verdict Logic (Simple LLM Call)
        if final_api_key:
             llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=final_api_key)
             verdict_prompt = f"""
             Based on the nutrition data: {st.session_state['raw_data']}
             And FSSAI rules: {st.session_state['fssai_insight']}
             
             Give a final Health Verdict (Healthy/Unhealthy), Rating (1-10), and 3 bullet points of advice.
             Format as Markdown.
             """
             verdict = llm.invoke(verdict_prompt).content
             st.markdown("### 🏆 Final Verdict")
             st.markdown(verdict)
