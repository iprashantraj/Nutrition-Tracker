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
import tempfile
from PIL import Image
import json
import hashlib
import datetime
import uuid
import pandas as pd
import altair as alt
import re

# --- Usage Tracking Mechanics ---
USAGE_FILE = "usage_tracker.json"

def get_user_id():
    """Generates a persistent ID using URL query parameters."""
    # Check if ID exists in URL
    if "uid" in st.query_params:
        return st.query_params["uid"]
    
    # Check session state as backup (before URL update propagates)
    if "user_id_session" in st.session_state:
        # Sync to URL
        st.query_params["uid"] = st.session_state.user_id_session
        return st.session_state.user_id_session

    # Generate new ID
    new_id = str(uuid.uuid4())
    st.query_params["uid"] = new_id
    st.session_state.user_id_session = new_id
    return new_id

def load_usage():
    if not os.path.exists(USAGE_FILE):
        return {}
    try:
        with open(USAGE_FILE, "r") as f:
            return json.load(f)
    except:
        return {}

def save_usage(data):
    try:
        with open(USAGE_FILE, "w") as f:
            json.dump(data, f)
            f.flush()
            os.fsync(f.fileno())
    except Exception as e:
        log_debug(f"Save failed: {e}")

def get_usage_count(user_id):
    data = load_usage()
    return data.get(user_id, 0)

def increment_usage(user_id):
    data = load_usage()
    data[user_id] = data.get(user_id, 0) + 1
    save_usage(data)



# --- Configuration & Setup ---
st.set_page_config(page_title="Advanced Nutrition Tracker", page_icon="🥗", layout="wide")


# Load environment variables (for Developer Key)
load_dotenv()
DEV_API_KEY = os.getenv("GOOGLE_API_KEY")

# --- CSS Styling for Premium Look ---
# --- CSS Styling for Premium Look ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif;
    }

    /* Adaptive Backgrounds */
    .main {
        background-color: var(--background-color);
    }
    
    /* Custom Card Style - Adaptive */
    .metric-card {
        background: var(--secondary-background-color);
        padding: 24px;
        border-radius: 16px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
        border-left: 4px solid #10b981; /* Emerald Green Accent */
    }

    /* Primary Button */
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #10b981 0%, #059669 100%);
        color: white;
        border-radius: 12px;
        height: 55px;
        font-weight: 600;
        font-size: 1.1rem;
        border: none;
        box-shadow: 0 4px 14px rgba(16, 185, 129, 0.4);
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(16, 185, 129, 0.6);
    }

    /* Headings - Adaptive */
    h1, h2, h3 {
        color: var(--text-color) !important;
        font-weight: 700 !important;
    }
    
    /* Upload Box Border - Adaptive */
    [data-testid="stFileUploader"] {
        border-radius: 16px;
        border: 2px dashed var(--text-color);
        opacity: 0.7;
        padding: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# --- Sidebar: Hybrid Auth System ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2921/2921822.png", width=100)
    st.title("Settings")
    st.caption("App Version: v1.0.2") # Removed Debug Mode text
    
    st.markdown("### 🔑 API Access")
    user_api_key = st.text_input("Enter your Google API Key", type="password", help="Get one from aistudio.google.com")
    
    auth_status = st.empty()
    
    # Initialize Session State (Just for UI consistency, but valid source is file)
    user_id = get_user_id()
    current_usage = get_usage_count(user_id)
    
    final_api_key = None
    
    if user_api_key:
        final_api_key = user_api_key
        auth_status.success("Using User Key ✅")
    else:
        if current_usage < 2:
            final_api_key = DEV_API_KEY
            remaining = 2 - current_usage
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
    llm = ChatGoogleGenerativeAI(model="gemini-flash-latest", google_api_key=api_key)
    
    prompt = """
    You are an expert nutritionist. Look at this food label image.
    1. Extract all nutritional information (Calories, Sugar, Fat, Protein, Ingredients).
    2. Identify the product name.
    3. Return the data in valid JSON format.
    """
    
    from langchain_core.messages import HumanMessage
    
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
    llm = ChatGoogleGenerativeAI(model="gemini-flash-latest", google_api_key=api_key)
    
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
st.markdown('<div style="text-align: center; margin-bottom: 30px;">', unsafe_allow_html=True)
st.title("🥗 Intelligent Nutrition Tracker")
st.markdown('<h3 style="opacity: 0.7; font-weight: 300;">scan. analyze. eat smart.</h3>', unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

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
                    increment_usage(user_id)
                    # Verify immediate update
                    if get_usage_count(user_id) >= 2:
                         st.rerun()
                
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

        # Try to parse JSON
        try:
            # Clean JSON if it has markdown backticks
            cleaned_json = st.session_state['raw_data'].replace("```json", "").replace("```", "").strip()
            nutrition_data = json.loads(cleaned_json)
            
            # --- Display Nutrition Dashboard ---
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown("### 📊 Nutrition Overview")
            
            # 1. Key Metrics Row
            mc1, mc2, mc3 = st.columns(3)
            with mc1:
                st.metric("Calories", nutrition_data.get("Calories", "N/A"))
            with mc2:
                st.metric("Protein", nutrition_data.get("Protein", "0g"))
            with mc3:
                st.metric("Fat", nutrition_data.get("Fat", "0g"))
            
            st.markdown("---")
            
            # 2. Macro Chart using Altair
            # Prepare data for chart
            def clean_value(val):
                # Remove 'g' and convert to float
                return float(re.sub(r'[^\d.]', '', str(val))) if val else 0

            macros = {
                "Carbs": clean_value(nutrition_data.get("Carbohydrates", nutrition_data.get("Carbs", "0g"))),
                "Protein": clean_value(nutrition_data.get("Protein", "0g")),
                "Fat": clean_value(nutrition_data.get("Fat", "0g"))
            }
            
            df_macros = pd.DataFrame([
                {"Macro": k, "Value": v} for k, v in macros.items() if v > 0
            ])
            
            if not df_macros.empty:
                base = alt.Chart(df_macros).encode(
                    theta=alt.Theta("Value", stack=True)
                )
                pie = base.mark_arc(outerRadius=120).encode(
                    color=alt.Color("Macro", scale=alt.Scale(scheme="category10")),
                    order=alt.Order("Value", sort="descending"),
                    tooltip=["Macro", "Value"]
                )
                text = base.mark_text(radius=140).encode(
                    text=alt.Text("Value", format=".1f"),
                    order=alt.Order("Value", sort="descending"),
                    color=alt.value("var(--text-color)")  
                )
                st.altair_chart(pie + text, use_container_width=True)
            else:
                st.info("Insufficient data for Macro Chart")
                
            st.markdown('</div>', unsafe_allow_html=True)
            
            with st.expander("📝 View Raw Data Source"):
                st.json(nutrition_data)

        except Exception as e:
            st.error(f"Could not parse nutrition data. Showing raw text.")
            st.code(st.session_state['raw_data'][:500])
            print(f"JSON Parse Error: {e}")
            
    if 'fssai_insight' in st.session_state:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("#### 🏛️ FSSAI Compliance Check")
        st.write(st.session_state['fssai_insight'])
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Final Verdict Logic (Simple LLM Call)
        if final_api_key:
             llm = ChatGoogleGenerativeAI(model="gemini-flash-latest", google_api_key=final_api_key)
             verdict_prompt = f"""
             Based on the nutrition data: {st.session_state['raw_data']}
             And FSSAI rules: {st.session_state['fssai_insight']}
             
             Return a valid JSON object with the following keys:
             - "verdict": (String) "Healthy" or "Unhealthy" or "Moderate"
             - "score": (Integer) Health Rating from 1 to 10
             - "explanation": (String) Brief summary interpretation
             - "advice": (List of Strings) 3 actionable bullet points
             """
             try:
                 verdict_response = llm.invoke(verdict_prompt).content
                 # Clean potential markdown
                 cleaned_verdict = verdict_response.replace("```json", "").replace("```", "").strip()
                 verdict_data = json.loads(cleaned_verdict)
                 
                 st.markdown("### 🏆 Final Verdict")
                 
                 # --- Health Score Gauge UI ---
                 score = verdict_data.get("score", 5)
                 color = "#10b981" if score >= 8 else "#f59e0b" if score >= 5 else "#ef4444"
                 
                 st.markdown(f"""
                 <div style="background-color: var(--secondary-background-color); border-radius: 16px; padding: 20px; text-align: center; border: 2px solid {color}; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                     <h2 style="margin:0; color: var(--text-color); opacity: 0.8;">Health Score</h2>
                     <h1 style="font-size: 4rem; color: {color}; margin: 10px 0;">{score}/10</h1>
                     <h3 style="color: {color};">{verdict_data.get('verdict', 'Unknown')}</h3>
                     <p style="color: var(--text-color);">{verdict_data.get('explanation', '')}</p>
                 </div>
                 """, unsafe_allow_html=True)
                 
                 st.markdown("#### 💡 Expert Advice")
                 for tip in verdict_data.get("advice", []):
                     st.info(f"• {tip}")
                 
                 # Celebration for healthy food
                 if score >= 8:
                     st.balloons()

                     
             except Exception as e:
                 st.error("Could not generate structured verdict.")
                 st.write(e)
