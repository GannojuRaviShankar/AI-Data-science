import streamlit as st
import json
import os
import time
import datetime
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate

# ✅ Load Environment Variables Securely
load_dotenv()
API_KEY = os.getenv("Google_api_key")
if not API_KEY:
    st.error("⚠️ Google GenAI API key is missing! Add it to `.env`.")
    st.stop()

# ✅ Configure LangChain AI Model
model = ChatGoogleGenerativeAI(model="gemini-1.5-pro", google_api_key=API_KEY)

# ✅ System Prompt for AI
SYSTEM_PROMPT = """
You are an AI Data Science Tutor.
- Offer **ML model suggestions, hyperparameter tuning, and dataset recommendations**.
- Explain **concepts with examples and code snippets** when needed.
- Format responses using **headings, bullet points, and markdown formatting**.
"""

# ✅ Load and Save Chat History
CHAT_HISTORY_FILE = "chat_history.json"

def load_chat_history():
    try:
        with open(CHAT_HISTORY_FILE, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []

def save_chat_history():
    with open(CHAT_HISTORY_FILE, "w") as f:
        json.dump(st.session_state.chat_history, f, indent=4)

# ✅ AI Response Generation (Using LangChain)
def get_ai_response(user_input):
    try:
        prompt = ChatPromptTemplate.from_template(f"{SYSTEM_PROMPT}\n\nQuestion: {user_input}")
        response = model.invoke(prompt.format_messages()) 
        
        if not response or not response.content:
            st.error("⚠️ No response generated. Check your API configuration and prompt format.")
            return "⚠️ No valid response from AI."
        
        return response.content
    except Exception as e:
        st.error(f"🚨 API Error: {str(e)}") 
        return f"⚠️ API Error: {str(e)}"

# ✅ Initialize Session States
def initialize_session_states():
    if "dark_mode" not in st.session_state:
        st.session_state.dark_mode = False
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = load_chat_history()
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    if "live_typing" not in st.session_state:
        st.session_state.live_typing = True
    if "last_active" not in st.session_state:
        st.session_state.last_active = time.time()

initialize_session_states()

# ✅ Session Timeout (Auto-Logout)
SESSION_TIMEOUT = 300  # 5 min
if time.time() - st.session_state.last_active > SESSION_TIMEOUT:
    st.session_state.logged_in = False
    st.rerun()

st.session_state.last_active = time.time()

# ✅ Streamlit Page Config
st.set_page_config(page_title="AI Data Science Tutor", page_icon="🧠", layout="wide")

# ✅ User Authentication
def authenticate_user():
    if not st.session_state.logged_in:
        st.title("🔑 Login to AI Data Science Tutor")
        username = st.text_input("Enter your username:")
        role = st.selectbox("Select Role:", ["User", "Admin", "Student", "Employee"])

        if st.button("Login"):
            if not username:
                st.warning("Please enter your username to proceed.")
            else:
                st.session_state.logged_in = True
                st.session_state.username = username
                st.session_state.role = role
                st.rerun()
        st.stop()

authenticate_user()

# ✅ Role-Based Access Control
st.sidebar.subheader("🔑 User Info")
st.sidebar.write(f"👋 Welcome, **{st.session_state.username}** ({st.session_state.role})")

if st.session_state.role == "Admin":
    st.sidebar.write("📊 **Admin Dashboard** – You have full access.")
elif st.session_state.role == "Student":
    st.sidebar.write("📚 **Student Mode** – Focus on learning!")

# ✅ Sidebar Configuration
with st.sidebar:
    st.title("⚙️ Settings")

    # Toggle for Live Typing
    st.toggle("Live Typing", key="live_typing")

    # Button to Clear Chat History
    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []
        save_chat_history()
        st.sidebar.success("✅ Chat history cleared!")
        st.rerun()

    # About Section
    st.subheader("ℹ️ About")
    st.write(
        """
        **AI Data Science Tutor**  
        Powered by Gemini 1.5 Pro using LangChain.  
        Get answers to machine learning, data science, and AI-related questions with examples and code snippets.
        """
    )

    # Instructions Section
    st.subheader("📖 Instructions")
    st.write(
        """
        - Type your question in the chat box below.  
        - The AI will respond with detailed explanations and code examples when applicable.  
        - Use "Live Typing" to see real-time responses.  
        - Click "Clear Chat History" to reset the conversation.  
        """
    )


# ✅ Chat Interface
def handle_user_input():
    st.title("🧠 AI Data Science Tutor")
    user_input = st.chat_input("Ask an AI-powered question...")

    if user_input:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.session_state.chat_history.append(("user", user_input, timestamp))

        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            response = get_ai_response(user_input)
            if response:
                if st.session_state.live_typing:
                    response_text = ""
                    for word in response.split():
                        response_text += word + " "
                        time.sleep(0.02)
                        response_placeholder.markdown(response_text)
                else:
                    response_placeholder.markdown(response)

                st.session_state.chat_history.append(("assistant", response, timestamp))
                save_chat_history()
                st.rerun()

# ✅ Limit Chat History to Last 50 Messages
MAX_CHAT_HISTORY = 50
st.session_state.chat_history = st.session_state.chat_history[-MAX_CHAT_HISTORY:]

handle_user_input()

# ✅ Display Chat History
def display_chat_history():
    st.subheader("📜 Chat History")
    for role, msg, timestamp in st.session_state.chat_history:
        role_display = "👤 **User:**" if role == "user" else "🤖 **AI:**"
        with st.chat_message(role):
            st.markdown(f"**[{timestamp}] {role_display}** {msg}", unsafe_allow_html=True)

display_chat_history()
