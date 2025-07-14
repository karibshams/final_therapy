import os
import streamlit as st
import asyncio
from datetime import datetime
from typing import Dict
from main import EmothriveAI, EmothriveBackendInterface, TherapyType
from pdf_processor import PDFDocument, PDFProcessor
from openai import OpenAIEmbeddings
from prompt import TherapyType

# Load environment variables
from dotenv import load_dotenv
load_dotenv()  # This loads the .env file

# Page configuration
st.set_page_config(
    page_title="Emothrive AI Testing Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 2rem;
    }
    .response-box {
        background-color: #e3f2fd;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .user-message {
        background-color: #e8f5e9;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        text-align: right;
    }
    .ai-message {
        background-color: #f3e5f5;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'ai_engine' not in st.session_state:
    st.session_state.ai_engine = None
    st.session_state.backend_interface = None
    st.session_state.conversation_history = []
    st.session_state.initialized = False


def initialize_ai_engine():
    """Initialize the AI engine with API keys from .env"""
    try:
        openai_key = os.getenv("OPENAI_API_KEY")
        
        if not openai_key:
            raise ValueError("OpenAI API key not found in environment variables")

        with st.spinner("Initializing AI Engine..."):
            # Create AI engine without ElevenLabs (skip TTS/STT)
            ai_engine = EmothriveAI(
                openai_api_key=openai_key,
                elevenlabs_api_key=None,  # Skip ElevenLabs
                pdf_folder=os.getenv("PDF_FOLDER_PATH", "./pdf/"),  # Load from .env
                default_therapy_type=TherapyType.GENERAL,
                enable_crisis_detection=True
            )
            
            # Create backend interface
            backend_interface = EmothriveBackendInterface(ai_engine)
            
            # Store in session state
            st.session_state.ai_engine = ai_engine
            st.session_state.backend_interface = backend_interface
            st.session_state.initialized = True
            
            st.success("✅ AI Engine initialized successfully!")
            return True
            
    except Exception as e:
        st.error(f"❌ Initialization failed: {str(e)}")
        return False


async def test_text_response(message: str, therapy_type: str = None):
    """Test text-based response"""
    if not st.session_state.backend_interface:
        st.error("AI Engine not initialized!")
        return None
    
    request = {
        "message": message,
        "message_type": "text",
        "therapy_type": therapy_type,
        "return_audio": False,  # No audio response
        "audio_format": "mp3",  # Not used in this case
        "user_context": {
            "mood": "neutral",
            "urgency": "medium"
        }
    }
    
    response = await st.session_state.backend_interface.process_message(request)
    return response


def display_response(response: Dict):
    """Display API response in a formatted way"""
    if response['success']:
        resp_data = response['response']
        
        # Display text response
        st.markdown("### 🤖 AI Response")
        st.markdown(f"<div class='response-box'>{resp_data['text']}</div>", 
                   unsafe_allow_html=True)
        
        # Display metadata
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Therapy Type", resp_data.get('therapy_type', 'N/A'))
        with col2:
            st.metric("Confidence", f"{resp_data.get('confidence', 0):.2%}")
        with col3:
            st.metric("Processing Time", f"{resp_data.get('processing_time', 0):.2f}s")
        with col4:
            st.metric("Context Sources", len(resp_data.get('context_sources', [])))
        
        # Display context sources
        if resp_data.get('context_sources'):
            with st.expander("📚 Context Sources Used"):
                for source in resp_data['context_sources']:
                    st.write(f"- **{source['source']}** ({source['therapy_type']})")
    else:
        st.error(f"❌ Error: {response.get('error', 'Unknown error')}")


def main():
    st.markdown("<h1 class='main-header'>🧠 Emothrive AI Testing Dashboard</h1>", 
               unsafe_allow_html=True)
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Keys
        st.subheader("API Keys")
        openai_key = st.text_input("OpenAI API Key", type="password", 
                                  value=os.getenv("OPENAI_API_KEY", ""))
        
        if st.button("Initialize AI Engine", type="primary"):
            if openai_key:
                initialize_ai_engine()
            else:
                st.error("Please provide the OpenAI API key")
        
    # Main content area
    if not st.session_state.initialized:
        st.warning("⚠️ Please initialize the AI Engine using the sidebar configuration")
        return

    # Chat Interface
    st.header("Interactive Chat Testing")
    
    user_input = st.text_input("Your message:", key="chat_input")
    selected_therapy = st.selectbox("Therapy Type", options=[t.name for t in TherapyType], key="chat_therapy")
    
    if st.button("Send Message", type="primary") or user_input:
        if user_input:
            # Add user message to history
            st.session_state.conversation_history.append({
                "role": "user",
                "content": user_input,
                "timestamp": datetime.now()
            })
            
            # Get AI response
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            response = loop.run_until_complete(
                test_text_response(user_input, selected_therapy)
            )
            
            if response and response['success']:
                # Add AI response to history
                st.session_state.conversation_history.append({
                    "role": "assistant",
                    "content": response['response']['text'],
                    "timestamp": datetime.now(),
                    "metadata": response['response']
                })
            
            # Display response
            display_response(response)
    
    # Display conversation history
    if st.session_state.conversation_history:
        st.subheader("Conversation History")
        for msg in reversed(st.session_state.conversation_history[-10:]):
            if msg['role'] == 'user':
                st.markdown(f"<div class='user-message'><b>You:</b> {msg['content']}</div>", 
                           unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='ai-message'><b>AI:</b> {msg['content']}</div>", 
                           unsafe_allow_html=True)

if __name__ == "__main__":
    main()
