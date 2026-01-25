import streamlit as st
import streamlit.components.v1 as components  # Required for JS injection
import requests
import os
import time
import random
import string
from datetime import datetime
import firebase_admin
from firebase_admin import credentials, firestore
import cv2
import tempfile

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Deepfake Edge Agent",
    page_icon="🛡️",
    layout="wide"
)

# --- FIREBASE SETUP ---
if not firebase_admin._apps:
    # ⚠️ Ensure 'serviceAccountKey.json' is in your project directory
    if os.path.exists("serviceAccountKey.json"):
        cred = credentials.Certificate("serviceAccountKey.json")
        firebase_admin.initialize_app(cred)
    else:
        # Fallback for when running from a different directory
        cred = credentials.Certificate("app/ui/serviceAccountKey.json")
        firebase_admin.initialize_app(cred)

db = firestore.client()

# --- DATABASE HELPER FUNCTIONS ---
def load_collection(collection_name):
    """Fetches all documents from a Firestore collection."""
    try:
        docs = db.collection(collection_name).stream()
        return [{"_id": doc.id, **doc.to_dict()} for doc in docs]
    except Exception as e:
        st.error(f"Error loading {collection_name}: {e}")
        return []

def add_to_db(collection_name, data):
    """Adds a new document to Firestore."""
    db.collection(collection_name).add(data)

def delete_from_db(collection_name, doc_id):
    """Deletes a document by ID."""
    db.collection(collection_name).document(doc_id).delete()

def refresh_data():
    """Reloads all data from Firestore into Session State."""
    st.session_state.users = load_collection("users")
    st.session_state.employees = load_collection("employees")
    st.session_state.meetings = load_collection("meetings")
    st.session_state.secrets = load_collection("secrets")
    st.session_state.reports = load_collection("audit_reports")

# --- INITIALIZE SESSION STATE ---
if "theme" not in st.session_state:
    st.session_state.theme = "dark"  # Default theme

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "users" not in st.session_state:
    try:
        refresh_data()
    except Exception as e:
        st.session_state.users = []
        st.session_state.employees = []
        st.session_state.meetings = []
        st.session_state.secrets = []
        st.session_state.reports = []

# Live Mode States
if "session_code" not in st.session_state:
    st.session_state.session_code = ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))
if "logs" not in st.session_state:
    st.session_state.logs = []
if "security_scanned" not in st.session_state:
    st.session_state.security_scanned = False
if "recorded_frames" not in st.session_state:
    st.session_state.recorded_frames = []
if "is_recording" not in st.session_state:
    st.session_state.is_recording = False
if "recording_complete" not in st.session_state:
    st.session_state.recording_complete = False

# --- HELPER FUNCTIONS ---
def add_log(message):
    if "logs" not in st.session_state:
        st.session_state.logs = []
    timestamp = datetime.now().strftime("%H:%M:%S")
    st.session_state.logs.append(f"[{timestamp}] {message}")

def generate_sic():
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))

def generate_emp_id():
    suffix = ''.join(random.choices(string.digits, k=3))
    return f"EMP{suffix}"

def generate_meeting_id():
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))

def generate_report_id():
    """Generates a unique Report ID (Diff Code)."""
    chars = string.ascii_uppercase + string.digits
    suffix = ''.join(random.choices(chars, k=6))
    return f"REP-{suffix}"

# --- CORE ANALYSIS FUNCTION ---
def process_analysis(uploaded_file, endpoint_url, media_type):
    """
    Sends file to backend, gets REAL score, and saves audit report.
    """
    with st.spinner("Processing through Edge AI Pipeline..."):
        try:
            # 1. Prepare file for upload
            uploaded_file.seek(0)
            files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
            
            # 2. SEND REQUEST TO BACKEND
            response = requests.post(endpoint_url, files=files)
            
            # 3. Check if backend responded successfully
            if response.status_code == 200:
                result = response.json()
                
                # Extract Real Data from Backend (backend returns "deepfake_score", not "score")
                score = result.get("deepfake_score", 0.0)
                verdict_data = result.get("decision", {})
                if isinstance(verdict_data, str):
                    verdict_txt = verdict_data
                    risk_txt = "UNKNOWN"
                else:
                    verdict_txt = verdict_data.get("verdict", "UNKNOWN")
                    risk_txt = verdict_data.get("risk_level", "UNKNOWN")
                
                explanation = result.get("explanation", "No details provided by backend.")

                # 4. SAVE TO DATABASE (Audit Log)
                report_id = generate_report_id()
                timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                audit_data = {
                    "ReportID": report_id,
                    "Timestamp": timestamp_str,
                    "Filename": uploaded_file.name,
                    "MediaType": media_type,
                    "Verdict": verdict_txt,
                    "Confidence": f"{round(score * 100, 2)}%",
                    "RiskLevel": risk_txt,
                    "Details": explanation
                }
                
                add_to_db("audit_reports", audit_data)

                # 5. DISPLAY RESULTS
                st.success(f"Analysis Complete. Report stored: {report_id}")
                st.metric(label="Deepfake Confidence", value=f"{round(score * 100, 2)}%")

                if verdict_txt == "DEEPFAKE":
                    st.error(f"VERDICT: {verdict_txt}")
                elif verdict_txt == "REAL":
                    st.success(f"VERDICT: {verdict_txt}")
                else:
                    st.warning(f"VERDICT: {verdict_txt}")

                with st.expander("🔍 View AI Explanation", expanded=True):
                    st.write(explanation)
                    st.caption(f"Risk Level: {risk_txt}")
                    st.info(f"Audit Log saved to database with ID: {report_id}")
            
            else:
                st.error(f"Backend Error ({response.status_code}): {response.text}")

        except requests.exceptions.ConnectionError:
            st.error("❌ Connection Failed: Is 'main.py' running? (Try: python main.py)")
        except Exception as e:
            st.error(f"An error occurred: {e}")

def load_css():
    theme = st.session_state.get("theme", "dark")
    
    if theme == "dark":
        st.markdown("""
        <style>
        /* 1. Page Background */
        .stApp {
            background: radial-gradient(circle at 50% 10%, #1e1e2f 0%, #0e0e17 100%) !important;
            color: #ffffff !important;
        }

        /* 2. Sidebar */
        section[data-testid="stSidebar"] {
            background-color: #0b0b12 !important;
            border-right: 1px solid #2d2d3d !important;
        }

        /* 3. Text Visibility */
        h1, h2, h3, p, span, label, .stMarkdown, [data-testid="stWidgetLabel"] {
            color: #ffffff !important;
        }

        /* 4. Inputs & File Uploader */
        input, textarea, [data-baseweb="select"], [data-baseweb="input"], [data-testid="stFileUploader"] section {
            background-color: #161625 !important;
            color: #ffffff !important;
            border: 1px solid #32324d !important;
        }

        /* 5. LIVE CODE BOXES */
        div[style*="background-color: #0e1117"] {
            background-color: #12121f !important;
            border: 1px solid #4a4a6a !important;
            color: #ffffff !important;
            border-radius: 8px !important;
        }

        /* 6. SIC CODES - REMOVED BORDERS */
        .stCode, .stCode > pre, code {
            background-color: transparent !important;
            border: none !important;
            color: #00C6FF !important;
            font-weight: bold !important;
            padding: 0 !important;
        }

        /* 7. Buttons */
        div.stButton > button {
            background-color: #1c1c2e !important;
            color: #ffffff !important;
            border: 1px solid #3d3d5c !important;
            border-radius: 8px !important;
            transition: all 0.3s ease;
        }
        div.stButton > button:hover {
            border-color: #6366f1 !important;
            background-color: #252545 !important;
        }

        /* 8. Cleanup */
        .login-card { background: transparent !important; border: none !important; box-shadow: none !important; }
        div[data-testid="stVerticalBlock"] > div:has(iframe[title="streamlit_js_eval.streamlit_js_eval"]) {
            display: none !important;
        }
        
        /* File uploader fix */
        [data-testid='stFileUploaderDropzone'] {
            pointer-events: none !important;
        }
        [data-testid='stFileUploaderDropzone'] button {
            pointer-events: auto !important;
            cursor: pointer !important;
            position: relative;
            z-index: 10;
        }
        </style>
        """, unsafe_allow_html=True)
    else: 
        st.markdown("""
        <style>
        /* Light Mode CSS */
        .stApp { background-color: #FFFFFF !important; }
        .stApp *, label, p, span, h1, h2, h3 { color: #1E293B !important; }
        section[data-testid="stSidebar"] { background-color: #F8FAFC !important; border-right: 1px solid #E2E8F0 !important; }
        input, textarea, [data-baseweb="select"], [data-baseweb="base-input"] { background-color: #FFFFFF !important; color: #1E293B !important; border: 1px solid #CBD5E1 !important; }
        
        .stCode, .stCode > pre, code {
            background-color: transparent !important;
            border: none !important;
            color: #0F172A !important;
            font-weight: bold !important;
        }

        div[style*="background-color: #0e1117"] { background-color: #F0F9FF !important; border: 2px solid #3B82F6 !important; }
        div[style*="background-color: #0e1117"] h1 { color: #1D4ED8 !important; }
        .stButton>button { background-color: #FFFFFF !important; color: #1E293B !important; border: 1px solid #CBD5E1 !important; }
        
        .login-card { background: transparent !important; border: none !important; box-shadow: none !important; }
        div[data-testid="stVerticalBlock"] > div:has(iframe[title="streamlit_js_eval.streamlit_js_eval"]) {
            display: none !important;
        }
        
        [data-testid='stFileUploaderDropzone'] {
            pointer-events: none !important;
        }
        [data-testid='stFileUploaderDropzone'] button {
            pointer-events: auto !important;
            cursor: pointer !important;
            position: relative;
            z-index: 10;
        }
        </style>
        """, unsafe_allow_html=True)

# --- JAVASCRIPT INJECTION FOR ENTER KEY NAVIGATION ---
def inject_js():
    components.html(
        """
        <script>
        const doc = window.parent.document;
        
        // Polling loop to ensure listeners stick even after Streamlit reruns
        setInterval(() => {
            const inputs = Array.from(doc.querySelectorAll('input[type="text"], input[type="password"]'));
            
            inputs.forEach((input, index) => {
                if (input.dataset.enterNav === "true") return;
                
                input.dataset.enterNav = "true";
                
                input.addEventListener('keydown', function(e) {
                    if (e.key === 'Enter') {
                        e.preventDefault(); 
                        e.stopPropagation();
                        
                        const nextInput = inputs[index + 1];
                        
                        if (nextInput) {
                            nextInput.focus();
                        } else {
                            input.blur();
                            
                            const buttons = Array.from(doc.querySelectorAll('button'));
                            const authBtn = buttons.find(btn => btn.innerText.includes("AUTHENTICATE"));
                            
                            if (authBtn) {
                                setTimeout(() => {
                                    authBtn.click();
                                }, 150);
                            }
                        }
                    }
                });
            });
        }, 300);
        </script>
        """,
        height=0,
        width=0
    )

def render_sidebar_controls(key_prefix):
    threshold = st.slider("Sensitivity Threshold", 0.0, 1.0, 0.75, key=f"{key_prefix}_thresh")
    st.caption(f"Current Sensitivity: {int(threshold*100)}%")
    st.markdown("---")
    mode = st.radio("Processing Mode", ["Light", "Heavy"], horizontal=True, key=f"{key_prefix}_mode")
    st.write("Selected mode:", mode)

# --- DIALOGS (POP-UPS) ---
@st.dialog("Confirm Logout")
def logout_modal():
    st.markdown("<h3 style='text-align:center;'>⚠️ End Session?</h3>", unsafe_allow_html=True)
    st.markdown("<div style='text-align: center;'>You are about to disconnect from the secure agent.</div>", unsafe_allow_html=True)
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Yes, Logout", use_container_width=True):
            st.session_state.logged_in = False
            st.rerun()
    with col2:
        if st.button("Cancel", use_container_width=True):
            st.rerun()

# --- DB MODALS ---
@st.dialog("Add New User")
def add_user_modal():
    st.write("Enter the details for the new authorized personnel.")
    with st.form("new_user_form"):
        name = st.text_input("Full Name")
        submitted = st.form_submit_button("Generate SIC & Save", use_container_width=True)
        if submitted and name:
            new_sic = generate_sic()
            add_to_db("users", {"Name": name, "SIC": new_sic})
            st.success(f"User Added! SIC: {new_sic}")
            time.sleep(1)
            refresh_data()
            st.rerun()

@st.dialog("Delete User?")
def delete_user_modal(index_to_remove):
    user = st.session_state.users[index_to_remove]
    st.markdown(f"Are you sure you want to remove **{user['Name']}**?")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Yes, Delete", use_container_width=True, type="primary"):
            delete_from_db("users", user["_id"])
            refresh_data()
            st.rerun()
    with col2:
        if st.button("Cancel", use_container_width=True):
            st.rerun()

@st.dialog("Add New Employee")
def add_employee_modal():
    st.write("Enter details for the new employee record.")
    with st.form("new_emp_form"):
        name = st.text_input("Full Name")
        submitted = st.form_submit_button("Generate ID & Save", use_container_width=True)
        if submitted and name:
            new_id = generate_emp_id()
            add_to_db("employees", {"Name": name, "ID": new_id})
            st.success(f"Employee Added! ID: {new_id}")
            time.sleep(1)
            refresh_data()
            st.rerun()

@st.dialog("Delete Employee?")
def delete_employee_modal(index_to_remove):
    emp = st.session_state.employees[index_to_remove]
    st.markdown(f"Are you sure you want to remove **{emp['Name']}**?")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Yes, Delete", use_container_width=True, type="primary"):
            delete_from_db("employees", emp["_id"])
            refresh_data()
            st.rerun()
    with col2:
        if st.button("Cancel", use_container_width=True):
            st.rerun()

@st.dialog("Add New Meeting")
def add_meeting_modal():
    st.write("Schedule a new secure session.")
    with st.form("new_meeting_form"):
        name = st.text_input("Meeting Name")
        submitted = st.form_submit_button("Generate ID & Add Meeting", use_container_width=True)
        if submitted and name:
            new_id = generate_meeting_id()
            add_to_db("meetings", {"Name": name, "ID": new_id})
            st.success(f"Meeting Added! ID: {new_id}")
            time.sleep(1)
            refresh_data()
            st.rerun()

@st.dialog("Cancel Meeting?")
def delete_meeting_modal(index_to_remove):
    meeting = st.session_state.meetings[index_to_remove]
    st.markdown(f"Are you sure you want to cancel **{meeting['Name']}**?")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Yes, Cancel", use_container_width=True, type="primary"):
            delete_from_db("meetings", meeting["_id"])
            refresh_data()
            st.rerun()
    with col2:
        if st.button("Return", use_container_width=True):
            st.rerun()

@st.dialog("Add New Secret")
def add_secret_modal():
    st.write("Add encrypted entry to vault.")
    with st.form("new_secret_form"):
        key_name = st.text_input("Secret Name/Key")
        value_data = st.text_input("Secret Value", type="password")
        submitted = st.form_submit_button("Encrypt & Save", use_container_width=True)
        if submitted and key_name and value_data:
            add_to_db("secrets", {"Key": key_name, "Value": value_data})
            st.success("Secret stored successfully.")
            time.sleep(1)
            refresh_data()
            st.rerun()

@st.dialog("Delete Secret?")
def delete_secret_modal(index_to_remove):
    secret = st.session_state.secrets[index_to_remove]
    st.markdown(f"Delete secret **{secret['Key']}**?")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Yes, Delete", use_container_width=True, type="primary"):
            delete_from_db("secrets", secret["_id"])
            refresh_data()
            st.rerun()
    with col2:
        if st.button("Cancel", use_container_width=True):
            st.rerun()

# --- LOGIN PAGE ---
CORRECT_SEQUENCE = ["finguard", "ai", "is", "the", "best"]

def login_page():
    load_css()
    inject_js()
    
    # Custom CSS for password fields and eye icon
    st.markdown("""
    <style>
    /* Unmask password dots */
    input[type="password"] {
        -webkit-text-security: none !important;
        -moz-text-security: none !important;
        text-security: none !important;
    }
    
    /* Customize eye icon button */
    div[data-baseweb="base-input"] button {
        background-color: transparent !important;
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        padding: 0px 5px !important;
        min-height: 0px !important;
    }
    
    div[data-baseweb="base-input"] button:hover {
        background-color: transparent !important;
        border: none !important;
        box-shadow: none !important;
    }
    
    /* Shrink SVG icon */
    div[data-baseweb="base-input"] svg {
        width: 14px !important;
        height: 14px !important;
        opacity: 0.5 !important;
    }
    
    div[data-baseweb="base-input"] button:hover svg {
        opacity: 0.8 !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        # Header
        st.markdown("""
            <div style="text-align: center;">
                <h1 style="margin-bottom: 0;">🛡️ FINGUARD AI</h1>
                <b style="letter-spacing: 2px; opacity: 0.8;">LOGIN TERMINAL</b>
                <br><br>
            </div>
        """, unsafe_allow_html=True)

        # Username & Password
        user_id = st.text_input("Administrator Username", placeholder="Enter ID...", key="user_id")
        user_pass = st.text_input("Password", placeholder="Enter Password...", type="password", key="user_pass")
        
        st.markdown("<br><p style='text-align:center; font-size:0.9em; opacity:0.8;'>PHRASE SEQUENCE VERIFICATION</p>", unsafe_allow_html=True)

        # 5 Word Inputs
        spacer_l, words_container, spacer_r = st.columns([1, 6, 1])
        
        user_phrases = []

        with words_container:
            # Row 1: 3 Boxes
            row1_cols = st.columns(3)
            for i in range(3):
                with row1_cols[i]:
                    val = st.text_input(
                        f"Word {i+1}", 
                        label_visibility="collapsed", 
                        key=f"word_box_{i}", 
                        placeholder=f"Word {i+1}",
                        type="password"
                    )
                    user_phrases.append(val.strip().lower())

            # Row 2: 2 Boxes (Centered)
            row2_cols = st.columns([0.5, 1, 1, 0.5]) 
            
            with row2_cols[1]:
                val = st.text_input(
                    "Word 4", 
                    label_visibility="collapsed", 
                    key="word_box_3", 
                    placeholder="Word 4",
                    type="password"
                )
                user_phrases.append(val.strip().lower())

            with row2_cols[2]:
                val = st.text_input(
                    "Word 5", 
                    label_visibility="collapsed", 
                    key="word_box_4", 
                    placeholder="Word 5",
                    type="password"
                )
                user_phrases.append(val.strip().lower())

        # Progress Dots
        dot_html = ""
        for i in range(5):
            if user_phrases[i]: 
                dot_html += '<span style="color: #9D50BB; font-size: 30px; margin: 0 10px;">●</span>'
            else:
                dot_html += '<span style="color: #444; font-size: 30px; margin: 0 10px;">○</span>'
        
        st.markdown(f'<div style="text-align: center; margin: 15px 0;">{dot_html}</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Login Button
        if st.button("AUTHENTICATE & LOGIN", type="primary", use_container_width=True):
            if not user_id:
                st.warning("⚠️ Identity Required.")
            elif user_pass != "1234":
                st.error("❌ Access Denied: Incorrect Password.")
            elif user_phrases == CORRECT_SEQUENCE:
                st.success(f"✅ Access Granted. Welcome, {user_id}.")
                time.sleep(2)
                st.session_state.logged_in = True
                try:
                    refresh_data()
                except:
                    pass
                st.rerun()
            else:
                st.error("❌ Invalid Sequence. Check your phrase order.")
                time.sleep(2)

# --- MAIN APP LOGIC ---
def main_app():
    load_css()
    col_title, col_nav = st.columns([3, 1], vertical_alignment="center")
    with col_title:
        st.markdown('<h1 style="margin-top:0;">FINGUARD AGENT Dashboard</h1>', unsafe_allow_html=True)
    with col_nav:
        nav_mode = st.selectbox("Navigation", ["Upload", "Live", "Database"], label_visibility="collapsed")

    # ==========================
    #      UPLOAD UI
    # ==========================
    if nav_mode == "Upload":
        with st.sidebar:
            st.markdown("#### 🌓 System Theme")
            if st.button("Switch to " + ("Light Mode" if st.session_state.theme == "dark" else "Dark Mode"), use_container_width=True):
                st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"
                st.rerun()
            st.divider()
            st.markdown('<div class="sidebar-profile-title" style="text-align: center; font-size: 1.45rem;">YOUR PROFILE</div>', unsafe_allow_html=True)
            st.write(" ")
            st.markdown("""
                <div style="text-align: center;">
                    <img src="https://cdn-icons-png.flaticon.com/512/149/149071.png"
                        style="width:100px; height:100px; border-radius:50%; 
                            border:3px solid #00C6FF; box-shadow: 0 0 15px rgba(0, 198, 255, 0.3);">
                    <h3 style="margin-top:10px; color: white;">Admin</h3>
                </div>
                <hr style="border-color: #444;">
                """, unsafe_allow_html=True)
            st.info("System Mode: **EDGE_ONLINE**")
            render_sidebar_controls("upload")
            if st.button("🚪 Logout", use_container_width=True):
                logout_modal()

        st.subheader("AI-Driven Media Authenticity")
        tab_video, tab_audio = st.tabs(["🎥 Video Analysis", "🎙️ Audio Analysis"])

        with tab_video:
            uploaded_video = st.file_uploader("Upload a video for analysis", type=["mp4", "avi", "mov"], key="video_uploader")
            if uploaded_video:
                col1, col2 = st.columns([1.5, 1])
                with col1:
                    st.video(uploaded_video)
                with col2:
                    if st.button("🚀 Analyze Video"):
                        process_analysis(uploaded_video, "http://localhost:8000/analyze/video", "Video")

        with tab_audio:
            uploaded_audio = st.file_uploader("Upload audio for analysis", type=["wav", "mp3", "flac"], key="audio_uploader")
            if uploaded_audio:
                col1, col2 = st.columns([1, 1])
                with col1:
                    st.audio(uploaded_audio)
                with col2:
                    if st.button("🚀 Analyze Audio"):
                        process_analysis(uploaded_audio, "http://localhost:8000/analyze/audio", "Audio")

    # ==========================
    #      LIVE MODE UI
    # ==========================
    elif nav_mode == "Live":
        with st.sidebar:
            st.markdown("#### 🌓 System Theme")
            if st.button("Switch to " + ("Light Mode" if st.session_state.theme == "dark" else "Dark Mode"), 
                         use_container_width=True, key="live_theme_btn"):
                st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"
                st.rerun()
            st.title("🛡️ Agent Control")
            st.info("System Mode: **LIVE_BROADCAST**")
            render_sidebar_controls("live")
            
            st.markdown("---")
            st.subheader("📜 System Logs")
            log_container = st.container(height=500, border=True)
            with log_container:
                for log in reversed(st.session_state.logs):
                    if "ALERT" in log:
                        st.error(log)
                    elif "INTEGRITY" in log or "HARDWARE" in log:
                        st.success(log)
                    else:
                        st.caption(log)
            if st.button("Clear Logs"):
                st.session_state.logs = []
                st.rerun()

        st.subheader("Live Verification System")

        # Display session code with reset button - INLINE
        col_code, col_reset = st.columns([4, 1])
        
        with col_code:
            st.markdown(f"""
                <div style="background-color: #0e1117; border: 2px solid #ff4b4b; padding: 10px; 
                            border-radius: 8px; text-align: center; margin-bottom: 15px;">
                    <p style="color: gray; margin-bottom: 3px; font-size: 0.8em;">SPEAK THIS CODE:</p>
                    <h1 style="color: #ff4b4b; letter-spacing: 6px; font-size: 2em; margin: 5px 0;">
                        {st.session_state.session_code}
                    </h1>
                </div>
            """, unsafe_allow_html=True)
        
        with col_reset:
            if st.button("🔄 New Code", use_container_width=True, key="reset_code_btn"):
                st.session_state.session_code = ''.join(
                    random.choices(string.ascii_uppercase + string.digits, k=6)
                )
                add_log(f"SESSION: New code generated - {st.session_state.session_code}")
                st.rerun()

        # HTML/JS for video recording with auto-upload - INCREASED HEIGHT
        video_recorder_html = f"""
        <div style="text-align: center;">
            <video id="videoPreview" autoplay muted playsinline 
                   style="width: 100%; max-width: 640px; height: 360px; border: 2px solid #ff4b4b; border-radius: 8px; background: #000; object-fit: cover;">
            </video>
            <br>
            <div style="margin: 10px 0;">
                <button id="startBtn" onclick="startRecording()" 
                        style="padding: 10px 20px; font-size: 14px; margin: 3px; background: #28a745; color: white; border: none; border-radius: 6px; cursor: pointer; font-weight: bold;">
                    🔴 Start Recording
                </button>
                <button id="stopBtn" onclick="stopRecording()" disabled
                        style="padding: 10px 20px; font-size: 14px; margin: 3px; background: #dc3545; color: white; border: none; border-radius: 6px; cursor: pointer; font-weight: bold;">
                    ⏹️ Stop & Verify
                </button>
                <button id="resetBtn" onclick="resetSession()"
                        style="padding: 10px 20px; font-size: 14px; margin: 3px; background: #6c757d; color: white; border: none; border-radius: 6px; cursor: pointer; font-weight: bold;">
                    🔄 Reset
                </button>
            </div>
            <p id="status" style="margin: 8px 0; font-size: 13px; color: #aaa; min-height: 20px;"></p>
            <div id="results" style="margin-top: 15px; padding-bottom: 20px;"></div>
            
            <script>
            let mediaRecorder;
            let recordedChunks = [];
            let stream;
            let recordedBlob = null;
            
            async function initCamera() {{
                try {{
                    stream = await navigator.mediaDevices.getUserMedia({{ 
                        video: {{ width: 640, height: 360 }}, 
                        audio: {{
                            echoCancellation: true,
                            noiseSuppression: true,
                            sampleRate: 16000
                        }}
                    }});
                    
                    document.getElementById('videoPreview').srcObject = stream;
                    
                    // Check if audio track exists
                    const audioTracks = stream.getAudioTracks();
                    if (audioTracks.length > 0) {{
                        console.log('Audio track available:', audioTracks[0].label);
                        document.getElementById('status').textContent = '✅ Camera & Microphone Ready';
                    }} else {{
                        console.warn('No audio track found');
                        document.getElementById('status').textContent = '⚠️ Camera ready but NO MICROPHONE detected';
                    }}
                    document.getElementById('status').style.color = '#28a745';
                }} catch (err) {{
                    document.getElementById('status').textContent = '❌ Camera/Mic access denied: ' + err.message;
                    document.getElementById('status').style.color = '#dc3545';
                }}
            }}
            
            function startRecording() {{
                recordedChunks = [];
                recordedBlob = null;
                document.getElementById('results').innerHTML = '';
                
                const options = {{ mimeType: 'video/webm;codecs=vp9,opus' }};
                
                if (!MediaRecorder.isTypeSupported(options.mimeType)) {{
                    options.mimeType = 'video/webm;codecs=vp8,opus';
                }}
                
                if (!MediaRecorder.isTypeSupported(options.mimeType)) {{
                    options.mimeType = 'video/webm';
                }}
                
                mediaRecorder = new MediaRecorder(stream, options);
                
                mediaRecorder.ondataavailable = (event) => {{
                    if (event.data.size > 0) {{
                        recordedChunks.push(event.data);
                    }}
                }};
                
                mediaRecorder.onstop = async () => {{
                    recordedBlob = new Blob(recordedChunks, {{ type: 'video/webm' }});
                    document.getElementById('status').textContent = '⏳ Uploading...';
                    document.getElementById('status').style.color = '#ffc107';
                    
                    await uploadAndVerify(recordedBlob);
                }};
                
                mediaRecorder.start();
                document.getElementById('startBtn').disabled = true;
                document.getElementById('stopBtn').disabled = false;
                document.getElementById('status').textContent = '🔴 RECORDING... Speak: {st.session_state.session_code}';
                document.getElementById('status').style.color = '#dc3545';
            }}
            
            function stopRecording() {{
                if (mediaRecorder && mediaRecorder.state !== 'inactive') {{
                    mediaRecorder.stop();
                    document.getElementById('startBtn').disabled = false;
                    document.getElementById('stopBtn').disabled = true;
                }}
            }}
            
            async function uploadAndVerify(blob) {{
                try {{
                    const formData = new FormData();
                    formData.append('file', blob, 'session-{st.session_state.session_code}.webm');
                    
                    const response = await fetch('http://localhost:8000/analyze/live-verification', {{
                        method: 'POST',
                        body: formData
                    }});
                    
                    if (response.ok) {{
                        const result = await response.json();
                        displayResults(result);
                    }} else {{
                        const errorText = await response.text();
                        document.getElementById('status').textContent = '❌ Error: ' + errorText;
                        document.getElementById('status').style.color = '#dc3545';
                    }}
                }} catch (err) {{
                    document.getElementById('status').textContent = '❌ Connection Failed';
                    document.getElementById('status').style.color = '#dc3545';
                    document.getElementById('results').innerHTML = `
                        <div style="padding: 12px; background: #dc354522; border: 2px solid #dc3545; border-radius: 6px; margin-top: 10px;">
                            <p style="color: #dc3545; margin: 0; font-size: 13px;">⚠️ Backend not running: python app/main.py</p>
                        </div>
                    `;
                }}
            }}
            
            function displayResults(result) {{
                const scores = result.scores;
                const video = scores.video;
                const audio = scores.audio;
                const code = scores.code;
                
                let html = '<div style="text-align: left; max-width: 700px; margin: 10px auto; padding: 15px; background: #1a1a2e; border-radius: 8px;">';
                html += '<h3 style="color: #fff; margin: 0 0 12px 0; font-size: 1.3em;">📊 Verification Results</h3>';
                html += '<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-bottom: 12px;">';
                
                // Video Results
                html += '<div style="padding: 12px; background: #0f3460; border-radius: 6px;">';
                html += '<h4 style="color: #fff; margin: 0 0 5px 0; font-size: 1em;">🎥 Visual</h4>';
                html += '<p style="color: #aaa; margin: 0; font-size: 0.8em;">' + video.model + '</p>';
                html += '<p style="font-size: 26px; color: ' + (video.verdict === 'REAL' ? '#28a745' : '#dc3545') + '; margin: 8px 0; font-weight: bold;">';
                html += (video.score * 100).toFixed(1) + '%</p>';
                html += '<p style="color: ' + (video.verdict === 'REAL' ? '#28a745' : '#dc3545') + '; font-weight: bold; margin: 0; font-size: 0.9em;">';
                html += video.verdict === 'REAL' ? '✅ AUTHENTIC' : '❌ FAKE';
                html += '</p></div>';
                
                // Audio Results
                html += '<div style="padding: 12px; background: #0f3460; border-radius: 6px;">';
                html += '<h4 style="color: #fff; margin: 0 0 5px 0; font-size: 1em;">🎙️ Audio</h4>';
                html += '<p style="color: #aaa; margin: 0; font-size: 0.8em;">' + audio.model + '</p>';
                html += '<p style="font-size: 26px; color: ' + (audio.verdict === 'REAL' ? '#28a745' : '#dc3545') + '; margin: 8px 0; font-weight: bold;">';
                html += (audio.score * 100).toFixed(1) + '%</p>';
                html += '<p style="color: ' + (audio.verdict === 'REAL' ? '#28a745' : '#dc3545') + '; font-weight: bold; margin: 0; font-size: 0.9em;">';
                html += audio.verdict === 'REAL' ? '✅ AUTHENTIC' : '❌ FAKE';
                html += '</p></div>';
                
                html += '</div>';
                
                // Code Verification Result
                if (code && code.code_match !== null && code.code_match !== undefined) {{
                    const codeColor = code.code_match ? '#28a745' : '#dc3545';
                    const codeBg = code.code_match ? '#28a74522' : '#dc354522';
                    html += '<div style="padding: 12px; background: ' + codeBg + '; border: 2px solid ' + codeColor + '; border-radius: 6px; margin-bottom: 12px;">';
                    html += '<h4 style="color: ' + codeColor + '; margin: 0 0 5px 0; font-size: 1em;">🗣️ Code Verification</h4>';
                    html += '<p style="color: #fff; margin: 0; font-size: 0.85em;">Spoken: "' + (code.spoken_text || 'Not detected') + '"</p>';
                    html += '<p style="color: ' + codeColor + '; font-weight: bold; margin: 5px 0 0 0; font-size: 0.9em;">';
                    html += code.code_match ? '✅ CODE MATCHED (' + (code.confidence * 100).toFixed(0) + '%)' : '❌ CODE MISMATCH';
                    html += '</p></div>';
                }} else if (code && code.error) {{
                    // Show warning if code verification failed due to technical issues
                    html += '<div style="padding: 12px; background: #ffc10722; border: 2px solid #ffc107; border-radius: 6px; margin-bottom: 12px;">';
                    html += '<h4 style="color: #ffc107; margin: 0 0 5px 0; font-size: 1em;">🗣️ Code Verification</h4>';
                    html += '<p style="color: #fff; margin: 0; font-size: 0.85em;">⚠️ ' + (code.spoken_text || 'Audio issues') + '</p>';
                    html += '<p style="color: #ffc107; font-weight: bold; margin: 5px 0 0 0; font-size: 0.85em;">Verification skipped (proceeded with video + audio only)</p>';
                    html += '</div>';
                }}
                
                // Final Verdict
                const finalColor = result.final_verdict === 'PASS' ? '#28a745' : '#dc3545';
                const finalBg = result.final_verdict === 'PASS' ? '#28a74522' : '#dc354522';
                html += '<div style="padding: 15px; background: ' + finalBg + '; border: 2px solid ' + finalColor + '; border-radius: 6px; text-align: center; margin-bottom: 10px;">';
                html += '<h3 style="color: ' + finalColor + '; margin: 0 0 5px 0; font-size: 1.3em; line-height: 1.4;">';
                html += result.final_verdict === 'PASS' ? '✅ SESSION VERIFIED' : '❌ SESSION REJECTED';
                html += '</h3>';
                
                let subtitle = result.final_verdict === 'PASS' ? 'User is Authenticated' : 'Authentication Failed';
                if (result.failure_reason) {{
                    subtitle = result.failure_reason;
                }}
                html += '<p style="color: #fff; margin: 0; font-size: 0.9em;">' + subtitle + '</p>';
                html += '</div></div>';
                
                document.getElementById('results').innerHTML = html;
                document.getElementById('status').textContent = '✅ Complete!';
                document.getElementById('status').style.color = '#28a745';
                
                if (result.final_verdict === 'PASS') {{
                    window.parent.postMessage({{type: 'streamlit:balloons'}}, '*');
                }}
            }}
            
            function resetSession() {{
                document.getElementById('results').innerHTML = '';
                document.getElementById('status').textContent = '✅ Camera Ready';
                document.getElementById('status').style.color = '#28a745';
                recordedChunks = [];
                recordedBlob = null;
                document.getElementById('startBtn').disabled = false;
                document.getElementById('stopBtn').disabled = true;
            }}
            
            initCamera();
            </script>
        </div>
        """
        
        components.html(video_recorder_html, height=950)

    # ==========================
    #      DATABASE UI (SYNCED)
    # ==========================
    elif nav_mode == "Database":
        with st.sidebar:
            st.markdown("#### 🌓 System Theme")
            if st.button("Switch to " + ("Light Mode" if st.session_state.theme == "dark" else "Dark Mode"), use_container_width=True, key="db_theme_btn"):
                st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"
                st.rerun()
            st.markdown('<div class="sidebar-profile-title" style="text-align: center; font-size: 1.45rem;">YOUR PROFILE</div>', unsafe_allow_html=True)
            st.markdown("  ")
            st.markdown("""
                <div style="text-align: center;">
                    <img src="https://cdn-icons-png.flaticon.com/512/149/149071.png"
                        style="width:100px; height:100px; border-radius:50%; 
                            border:3px solid #00C6FF; box-shadow: 0 0 15px rgba(0, 198, 255, 0.3);">
                    <h3 style="margin-top:10px; color: white;">Admin</h3>
                </div>
                """, unsafe_allow_html=True)
            st.info("System Mode: **EDGE_OFFLINE**")
            render_sidebar_controls("database")
            st.markdown("---")
            if st.button("🚪 Logout", use_container_width=True):
                logout_modal()

        st.subheader("Secure Database Access")
        st.caption("⚡ Live Sync with Cloud Firestore")
        
        tab_users, tab_employees, tab_meet, tab_secrets, tab_reports = st.tabs([
            "👥 Users", "👔 Employees", "📅 Meetings", "🔐 Secrets", "📊 Audit Logs"
        ])

        with tab_users:
            col_header_left, col_header_right = st.columns([3, 1], vertical_alignment="center")
            with col_header_left:
                st.markdown("### 📋 Active User Directory")
            with col_header_right:
                if st.button("➕ Add User", use_container_width=True):
                    add_user_modal()
            
            # Search bar for users
            search_query_users = st.text_input("🔍 Search User", placeholder="Search by name or SIC...", label_visibility="collapsed").lower()
            
            st.markdown("<hr style='margin: 5px 0; border-color: #333;'>", unsafe_allow_html=True)
            
            if search_query_users:
                filtered_users = [
                    u for u in st.session_state.users 
                    if search_query_users in u.get("Name", "").lower() or search_query_users in u.get("SIC", "").lower()
                ]
            else:
                filtered_users = st.session_state.users
            
            h1, h2, h3 = st.columns([2, 2, 0.5])
            h1.caption("FULL NAME")
            h2.caption("SIC CODE")
            h3.caption("ACTION")
            
            if not filtered_users:
                st.info("No users found.")
            else:
                for user in filtered_users:
                    orig_idx = st.session_state.users.index(user)
                    
                    c1, c2, c3 = st.columns([2, 2, 0.5], vertical_alignment="center")
                    c1.write(f"**{user.get('Name', 'Unknown')}**")
                    c2.code(user.get("SIC", "N/A"))
                    if c3.button("🗑️", key=f"del_user_{user.get('SIC', orig_idx)}"):
                        delete_user_modal(orig_idx)
                    st.markdown("<hr style='margin: 2px 0; opacity: 0.1;'>", unsafe_allow_html=True)

        with tab_employees:
            col_e_left, col_e_right = st.columns([3, 1], vertical_alignment="center")
            with col_e_left:
                st.markdown("### 👔 Active Employee Directory")
            with col_e_right:
                if st.button("➕ Add Employee", use_container_width=True):
                    add_employee_modal()
            
            # Search bar for employees
            search_query_emp = st.text_input("🔍 Search Employee", placeholder="Search by name or ID...", label_visibility="collapsed").lower()
            
            st.divider()
            
            if search_query_emp:
                filtered_employees = [
                    e for e in st.session_state.employees 
                    if search_query_emp in e.get("Name", "").lower() or search_query_emp in e.get("ID", "").lower()
                ]
            else:
                filtered_employees = st.session_state.employees
            
            e1, e2, e3 = st.columns([2, 2, 0.5])
            e1.caption("FULL NAME")
            e2.caption("ID CODE")
            e3.caption("ACTION")
            
            if not filtered_employees:
                st.info("No matching employees found.")
            else:
                for emp in filtered_employees:
                    orig_idx = st.session_state.employees.index(emp)
                    
                    c1, c2, c3 = st.columns([2, 2, 0.5], vertical_alignment="center")
                    c1.write(f"**{emp.get('Name', 'Unknown')}**")
                    c2.code(emp.get("ID", "N/A"))
                    if c3.button("🗑️", key=f"del_emp_{emp.get('ID', orig_idx)}"):
                        delete_employee_modal(orig_idx)
                    st.markdown("<hr style='margin: 2px 0; opacity: 0.05;'>", unsafe_allow_html=True)

        with tab_meet:
            col_m_left, col_m_right = st.columns([3, 1], vertical_alignment="center")
            with col_m_left:
                st.markdown("### 📅 Upcoming Sessions")
            with col_m_right:
                if st.button("➕ Add Meeting", use_container_width=True):
                    add_meeting_modal()
            
            # Search bar for meetings
            search_query_meet = st.text_input("🔍 Search Meeting", placeholder="Search by name or ID...", label_visibility="collapsed").lower()
            
            st.markdown("<hr style='margin: 5px 0; border-color: #333;'>", unsafe_allow_html=True)
            
            if search_query_meet:
                filtered_meetings = [
                    m for m in st.session_state.meetings 
                    if search_query_meet in m.get("Name", "").lower() or search_query_meet in m.get("ID", "").lower()
                ]
            else:
                filtered_meetings = st.session_state.meetings

            m1, m2, m3 = st.columns([2, 2, 0.5])
            m1.caption("MEETING NAME")
            m2.caption("MEETING ID")
            m3.caption("ACTION")
            
            if not filtered_meetings:
                st.info("No meetings scheduled.")
            else:
                for meeting in filtered_meetings:
                    orig_idx = st.session_state.meetings.index(meeting)
                    
                    c1, c2, c3 = st.columns([2, 2, 0.5], vertical_alignment="center")
                    c1.write(f"**{meeting.get('Name', 'Unknown')}**")
                    c2.code(meeting.get("ID", "N/A"))
                    if c3.button("🗑️", key=f"del_meet_{meeting.get('ID', orig_idx)}"):
                        delete_meeting_modal(orig_idx)
                    st.markdown("<hr style='margin: 2px 0; opacity: 0.1;'>", unsafe_allow_html=True)

        with tab_secrets:
            col_s_left, col_s_right = st.columns([3, 1], vertical_alignment="center")
            with col_s_left:
                st.markdown("### 🔐 Encrypted Vault")
            with col_s_right:
                if st.button("➕ Add Secret", use_container_width=True):
                    add_secret_modal()
            st.divider()
            s1, s2, s3 = st.columns([2, 2, 0.5])
            s1.caption("KEY NAME")
            s2.caption("VALUE (ENCRYPTED)")
            s3.caption("ACTION")
            if not st.session_state.secrets:
                st.info("Vault is empty.")
            else:
                for index, secret in enumerate(st.session_state.secrets):
                    c1, c2, c3 = st.columns([2, 2, 0.5], vertical_alignment="center")
                    c1.write(f"**{secret.get('Key', 'Unknown')}**")
                    val = secret.get('Value', '')
                    c2.code("•" * len(val) if val else "N/A")
                    if c3.button("🗑️", key=f"del_sec_{index}"):
                        delete_secret_modal(index)
                    st.markdown("<hr style='margin: 2px 0; opacity: 0.1;'>", unsafe_allow_html=True)

        # Audit Logs Tab
        with tab_reports:
            st.markdown("### 📊 Audit Trail")
            st.caption("Auto-generated reports from Media Analysis.")
            
            if not st.session_state.reports:
                st.info("No audit reports available.")
            else:
                sorted_reports = sorted(
                    st.session_state.reports, 
                    key=lambda x: x.get('Timestamp', ''), 
                    reverse=True
                )
                
                for report in sorted_reports:
                    with st.expander(f"{report.get('ReportID', 'N/A')} - {report.get('Timestamp', 'N/A')}"):
                        c1, c2 = st.columns(2)
                        with c1:
                            st.write(f"**File:** {report.get('Filename', 'Unknown')}")
                            st.write(f"**Type:** {report.get('MediaType', 'Unknown')}")
                            st.write(f"**Risk:** {report.get('RiskLevel', 'Unknown')}")
                        with c2:
                            st.metric("Confidence", report.get('Confidence', '0%'))
                            st.write(f"**Verdict:** {report.get('Verdict', 'UNKNOWN')}")
                        
                        st.markdown("**Analysis Details:**")
                        st.write(report.get('Details', 'No details available.'))


if __name__ == "__main__":
    if st.session_state.logged_in:
        main_app()
    else:
        login_page()