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
import json # Added for Integrity Check parsing

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
                
                # Extract Real Data from Backend
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

# --- STYLESHEET ---
def load_css():
    theme = st.session_state.get("theme", "dark")

    if theme == "dark":
        st.markdown("""
        <style>

        /* Background */
        .stApp {
            background-image:
                radial-gradient(circle at 20% 20%, rgba(79, 172, 254, 0.15), transparent 40%),
                radial-gradient(circle at 80% 30%, rgba(255, 45, 85, 0.12), transparent 45%),
                radial-gradient(circle at 50% 80%, rgba(0, 242, 254, 0.10), transparent 50%),
                linear-gradient(135deg, #060712, #0a0e1a 50%, #05060d) !important;
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            color: #ffffff !important;
        }

        /* Subtle particle texture overlay */
        .stApp::before {
            content: "";
            position: fixed;
            inset: 0;
            pointer-events: none;
            background-image: radial-gradient(rgba(255,255,255,0.16) 1px, transparent 1px);
            background-size: 70px 70px;
            opacity: 0.05;
            z-index: 0;
        }

        .stApp > div {
            position: relative;
            z-index: 1;
        }

        /* Glass + glow cards */
        .neon-card {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 18px;
            border: 1px solid rgba(255,255,255,0.12);
            padding: 28px;
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            box-shadow:
                0 0 25px rgba(79,172,254,0.12),
                0 0 35px rgba(255,45,85,0.10);
        }

        .neon-border {
            border: 1px solid rgba(79,172,254,0.35);
            box-shadow: 0 0 18px rgba(79,172,254,0.25);
        }

        /* Sidebar */
        section[data-testid="stSidebar"] {
            background-color: rgba(10, 12, 18, 0.85) !important;
            border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            height: 100vh !important;
            overflow: hidden !important;
        }

        section[data-testid="stSidebar"] > div {
            height: 100% !important;
            overflow-y: auto !important;
            overflow-x: hidden !important;
        }

        /* Text */
        h1, h2, h3, p, span, label, .stMarkdown, [data-testid="stWidgetLabel"] {
            color: #ffffff !important;
        }

        /* --- UPDATED INPUT STYLING (FIXED) --- */
        
        /* 1. Main Input Container (The "Pill") */
        div[data-baseweb="base-input"], div[data-baseweb="select"] > div {
            background-color: rgba(255, 255, 255, 0.05) !important;
            border: 1px solid rgba(255, 255, 255, 0.12) !important;
            border-radius: 14px !important;
            box-shadow: none !important;
            transition: all 0.25s ease;
            overflow: hidden;
            
            /* Remove right padding so border ends exactly after content */
            padding-right: 0px !important; 
        }
        
        /* 2. FORCE REMOVE RED OUTLINE (Override Validation Errors) */
        /* This forces the border to be gray even if Streamlit thinks it's invalid */
        div[data-baseweb="base-input"], 
        div[data-baseweb="base-input"][class*="error"], 
        div[data-baseweb="base-input"] * {
            border-color: rgba(255, 255, 255, 0.12) !important;
        }

        /* 3. FOCUS STATE - Blue Glow (Only when active) */
        div[data-baseweb="base-input"]:focus-within {
            border: 1px solid rgba(79,172,254,0.9) !important;
            box-shadow: 0 0 16px rgba(79,172,254,0.55) !important;
            border-color: rgba(79,172,254,0.9) !important;
        }

        /* 4. Input Field Transparency */
        input, textarea, [data-baseweb="input"] input {
            background-color: transparent !important;
            border: none !important;
            box-shadow: none !important;
            color: #ffffff !important;
        }
        
        /* 5. Eye Icon Container - Tightening */
        div[data-baseweb="base-input"] > div:last-child {
            background-color: transparent !important;
            padding-right: 12px !important; /* Small padding for visual balance */
            margin-right: 0px !important;   /* Ensure no extra margin extends the box */
            border: none !important;
        }

        /* 6. "Press Enter" Text - Positioned Safely to Left */
        [data-testid="InputInstructions"] {
            color: rgba(255, 255, 255, 0.5) !important;
            font-size: 0.65rem !important;  /* Reduced font size */
            visibility: visible !important;
            position: absolute !important;
            bottom: 0px !important;
            right: 60px !important; /* Changed to 60px as requested */
            pointer-events: none;
            white-space: nowrap;
        }

        /* File Uploader specific */
        [data-testid="stFileUploader"] section {
            background-color: rgba(22, 22, 37, 0.65) !important;
        }

        input::placeholder {
            color: rgba(255,255,255,0.38) !important;
        }

        /* Code text */
        .stCode, .stCode > pre, code {
            background-color: transparent !important;
            border: none !important;
            color: #00C6FF !important;
            font-weight: bold !important;
            padding: 0 !important;
        }

        /* Buttons (simple, transparent) */
        div.stButton > button {
            background: transparent !important;
            color: rgba(255, 255, 255, 0.92) !important;
            border: 1px solid rgba(255,255,255,0.18) !important;
            border-radius: 14px !important;
            box-shadow: none !important;
            transition: all 0.25s ease;
        }

        div.stButton > button:hover {
            background: rgba(255,255,255,0.06) !important;
            border-color: rgba(79,172,254,0.55) !important;
            box-shadow: 0 0 14px rgba(79,172,254,0.18) !important;
            transform: scale(1.01);
        }

        /* Live page box */
        .session-container {
            background: rgba(19, 22, 28, 0.75) !important;
            backdrop-filter: blur(12px);
            border: 1px solid rgba(255, 255, 255, 0.10) !important;
            border-top: 2px solid #FF4B4B !important;
            border-radius: 12px;
            padding: 15px 20px !important;
            max-width: 320px !important;
            margin: 0 auto 20px auto;
            box-shadow:
                0 10px 30px rgba(0, 0, 0, 0.55),
                0 0 25px rgba(255, 75, 75, 0.12);
            text-align: center;
        }

        .session-label {
            color: rgba(255, 255, 255, 0.5) !important;
            font-size: 0.65rem !important;
            text-transform: uppercase;
            letter-spacing: 2px;
        }

        .session-code-text {
            font-family: 'JetBrains Mono', 'Fira Code', monospace !important;
            font-size: 2.2rem !important;
            font-weight: 900 !important;
            letter-spacing: 8px !important;
            background: -webkit-linear-gradient(45deg, #FF4B4B, #FF914D);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-shadow: 0 0 30px rgba(255, 75, 75, 0.25);
            margin: 8px 0 !important;
            text-transform: uppercase;
        }

        /* Cleanup */
        .login-card {
            background: transparent !important;
            border: none !important;
            box-shadow: none !important;
        }

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

    else:
        st.markdown("""
        <style>

        .stApp {
            background-color: #ffffff !important;
        }

        .stApp *, label, p, span, h1, h2, h3 {
            color: #1E293B !important;
        }

        section[data-testid="stSidebar"] {
            background-color: #F8FAFC !important;
            border-right: 1px solid #E2E8F0 !important;

            height: 100vh !important;
            overflow: hidden !important;
        }

        section[data-testid="stSidebar"] > div {
            height: 100% !important;
            overflow-y: auto !important;
            overflow-x: hidden !important;
        }

        input, textarea, [data-baseweb="select"], [data-baseweb="base-input"] {
            background-color: #ffffff !important;
            color: #1E293B !important;
            border: 1px solid #CBD5E1 !important;
            border-radius: 12px !important;
        }

        .stCode, .stCode > pre, code {
            background-color: transparent !important;
            border: none !important;
            color: #0F172A !important;
            font-weight: bold !important;
        }

        .stButton>button {
            background: transparent !important;
            color: #1E293B !important;
            border: 1px solid #CBD5E1 !important;
            border-radius: 12px !important;
            box-shadow: none !important;
        }

        .stButton>button:hover {
            background: rgba(15, 23, 42, 0.04) !important;
            border-color: rgba(79, 172, 254, 0.55) !important;
        }

        .login-card {
            background: transparent !important;
            border: none !important;
            box-shadow: none !important;
        }

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

        .session-container {
            background: rgba(19, 22, 28, 0.9) !important;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(0, 0, 0, 0.08) !important;
            border-top: 2px solid #FF4B4B !important;
            border-radius: 12px;
            padding: 15px 20px !important;
            max-width: 320px !important;
            margin: 0 auto 20px auto;
            box-shadow: 0 8px 30px rgba(0, 0, 0, 0.15);
            text-align: center;
        }

        .session-code-text {
            font-family: 'JetBrains Mono', 'Fira Code', monospace !important;
            font-size: 2.2rem !important;
            font-weight: 900 !important;
            letter-spacing: 8px !important;
            color: #0F172A !important;
            margin: 8px 0 !important;
            text-transform: uppercase;
        }

        .session-label {
            color: rgba(30, 41, 59, 0.65) !important;
            font-size: 0.65rem !important;
            text-transform: uppercase;
            letter-spacing: 2px;
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
    threshold = st.slider("Low Sensitivty", 0.0, 1.0, 0.75, key=f"{key_prefix}_lowthresh")
    st.caption(f"Current Sensitivity: {int(threshold*100)}%")
    threshold1 = st.slider("High Sensitivity", 0.0, 1.0, 0.75, key=f"{key_prefix}_highthresh")
    st.caption(f"Current Sensitivity: {int(threshold1*100)}%")
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
    with st.form("new_user_form", clear_on_submit=True):
        st.subheader("Register New User")
        st.info("Additional details are stored securely for verification purposes.")
        name = st.text_input("Full Name")
        mobile = st.text_input("Mobile Number",help="Enter 10-digit number")
        email = st.text_input("Email ID")
        address = st.text_area("Address")

        submitted = st.form_submit_button("Generate SIC & Save", use_container_width=True)
        if submitted:
            if name and mobile and email:
                new_sic = generate_sic()
                user_data = {
                    "Name": name,
                    "Mobile": mobile,
                    "Email": email,
                    "Address": address,
                    "SIC": new_sic,
                    "Created_At": time.strftime("%Y-%m-%d %H:%M:%S")
                }
                add_to_db("users", user_data)
                st.success(f"User Added! SIC: {new_sic}")
                st.info(f"**Name:** {name} | **SIC:** {new_sic}")
                time.sleep(1)
                refresh_data()
                st.rerun()
            else:
                st.error("Please fill in all the required fields.")

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
    with st.form("new_emp_form", clear_on_submit=True):
        st.write("Enter details for the new employee record.")
        name = st.text_input("Full Name")
        email = st.text_input("Official Email ID")
        
        # Ensures they appear on the same line (Side-by-Side)
        col1, col2 = st.columns(2)
        with col1:
            dept = st.selectbox("Department", ["IT", "HR", "Sales", "Operations", "Finance", "Other"])
        with col2:
            designation = st.text_input("Designation")
        
        mobile = st.text_input("Contact Number")

        submitted = st.form_submit_button("Generate ID & Save", use_container_width=True)

        if submitted:
            if name and email and dept:
                new_id = generate_emp_id()
                emp_data = {
                    "Name": name,
                    "ID": new_id,
                    "Email": email,
                    "Department": dept,
                    "Designation": designation,
                    "Mobile": mobile,
                    "Status": "Active",
                    "Onboarding_Date": time.strftime("%Y-%m-%d")
                }
                add_to_db("employees", emp_data)
                st.success(f"✅ Employee Added!")
                st.info(f"**Name:** {name} | **ID:** {new_id}")
                
                time.sleep(1)
                refresh_data()
                st.rerun()
            else:
                st.error("Please fill in the required fields.")

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

    st.markdown("""
    <style>

    /* layout spacing */
    .login-wrapper {
        padding-top: 10px;
    }

    /* sidebar panel */
    .sidebar-box {
        background: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid rgba(255, 255, 255, 0.10) !important;
        border-radius: 16px !important;
        padding: 28px 20px !important;
        height: 80vh;
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        box-shadow:
            0 0 22px rgba(79,172,254,0.10),
            0 0 22px rgba(255,45,85,0.08);
    }

    .sidebar-text {
        font-family: 'Segoe UI', sans-serif;
        font-style: italic !important;
        font-size: 1rem !important;
        color: rgba(255, 255, 255, 0.67) !important;
        line-height: 1.55 !important;
        margin-bottom: 18px !important;
    }

    .sidebar-box h3, .sidebar-box h4 {
        font-size: 1.2rem !important;
        margin-bottom: 10px !important;
        letter-spacing: 1px !important;
        color: rgba(255,255,255,0.8) !important;
        text-shadow: 0 0 10px rgba(79,172,254,0.22);
    }

    /* login header */
    .login-title h1 {
        margin-bottom: 2px !important;
        letter-spacing: 1px;
        text-shadow:
            0 0 18px rgba(79,172,254,0.25),
            0 0 18px rgba(255,45,85,0.18);
    }

    .login-title b {
        opacity: 0.75;
        letter-spacing: 2px;
    }

    /* phrase section label */
    .phrase-title {
        text-align: center;
        font-size: 0.9em;
        opacity: 0.78;
        letter-spacing: 1.5px;
        margin-top: 14px;
        margin-bottom: 10px;
        text-transform: uppercase;
    }

    /* make column padding compact */
    div[data-testid="column"] {
        padding: 6px !important;
    }

    /* remove grey backgrounds around input adornments (eye button area) */
    div[data-testid="stInputAdornment"],
    div[data-baseweb="base-input"] > div:last-child {
        background-color: transparent !important;
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
    }

    /* eye icon button background fully transparent */
    div[data-baseweb="base-input"] button {
        background-color: transparent !important;
        background: transparent !important;
        box-shadow: none !important;
        border: none !important;
    }

    /* remove divider line inside inputs */
    div[data-baseweb="base-input"] > div {
        border-right: none !important;
    }

    /* eye icon color */
    svg[class*="StyledIcon"] {
        fill: rgba(255,255,255,0.8) !important;
        opacity: 0.75;
    }

    /* progress dots glow */
    .dot-filled {
        color: #9D50BB;
        font-size: 30px;
        margin: 0 10px;
        text-shadow: 0 0 18px rgba(157, 80, 187, 0.55);
    }
    .dot-empty {
        color: rgba(255,255,255,0.22);
        font-size: 30px;
        margin: 0 10px;
    }

    /* ensure the neon card feels like the screenshot */
    .neon-card {
        padding: 34px !important;
        box-shadow:
            0 0 28px rgba(79,172,254,0.14),
            0 0 28px rgba(255,45,85,0.12),
            0 18px 55px rgba(0,0,0,0.55);
    }
    
    </style>
    """, unsafe_allow_html=True)

    col1, space, col2 = st.columns([1.5, 0.2, 3])

    with col1:
        st.markdown(f"""
            <div class="sidebar-box">
                <h3>Instructions</h3>
                <p class="sidebar-text">
                    1. Authenticate your username.<br>
                    2. Input the password.<br>
                    3. Complete the 5-phrase sequence verification.<br>
                    4. System access is logged and monitored.
                </p>
                <h3>About</h3>
                <p class="sidebar-text" style="text-align: justify;">
                    FinGuard AI is an agentic AI security system developed by Team OverClocked. 
                    It is designed to autonomously detect, analyse, and respond to deepfake videos 
                    and AI-generated audio in real time while integrating a Secure Identification 
                    Code (SIC) for enhanced and tighter security with minimal human intervention.
                </p>
            </div>
        """, unsafe_allow_html=True)

    with col2:
        
        st.markdown("""
            <div class="login-title" style="text-align: center;">
                <h1>🛡️ FINGUARD AI</h1>
                <b>LOGIN TERMINAL</b>
            </div>
            <br>
        """, unsafe_allow_html=True)

        user_id = st.text_input("Administrator Username", placeholder="Enter Username...", key="user_id")
        user_pass = st.text_input("Password", placeholder="Enter Password...", type="password", key="user_pass")

        st.markdown(f"<div class='phrase-title'>PHRASE SEQUENCE VERIFICATION</div>", unsafe_allow_html=True)

        spacer_l, words_container, spacer_r = st.columns([1, 10, 1])

        user_phrases = []

        with words_container:
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

        dot_html = ""
        for i in range(5):
            if user_phrases[i]:
                dot_html += '<span class="dot-filled">●</span>'
            else:
                dot_html += '<span class="dot-empty">○</span>'

        st.markdown(f'<div style="text-align:center; margin: 14px 0 12px 0;">{dot_html}</div>', unsafe_allow_html=True)

        

        if st.button("AUTHENTICATE & LOGIN", type="primary", use_container_width=True):
            if not user_id:
                st.warning("⚠️ Identity Required.")
            elif user_pass != "1234":
                st.error("❌ Access Denied: Incorrect Password.")
            elif user_phrases == CORRECT_SEQUENCE:
                st.success(f"✅ Access Granted. Welcome, {user_id}.")
                time.sleep(1)
                st.session_state.logged_in = True
                try:
                    refresh_data()
                except:
                    pass
                st.rerun()
            else:
                st.error("❌ Invalid Sequence. Check your phrase order.")
                time.sleep(1)


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
            # Styling from app1.py sidebar
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
    #      LIVE MODE UI (Combined)
    # ==========================
    elif nav_mode == "Live":
        # --- INTEGRITY CHECK DEPENDENCIES ---
        try:
            from streamlit_js_eval import streamlit_js_eval
            import json 
            DEPENDENCIES_INSTALLED = True
        except ImportError:
            DEPENDENCIES_INSTALLED = False

        with st.sidebar:
            st.markdown("#### 🌓 System Theme")
            if st.button("Switch Theme", use_container_width=True, key="live_theme_btn"):
                st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"
                st.rerun()
            st.title("🛡️ Agent Control")
            st.info("System Mode: **LIVE_BROADCAST**")
            render_sidebar_controls("live") 
            
            st.markdown("---")
            st.subheader("📜 System Logs")
            log_container = st.container(height=400, border=True)
            with log_container:
                for log in reversed(st.session_state.logs):
                    if "ALERT" in log or "SPOOF" in log:
                        st.error(log)
                    elif "INTEGRITY" in log:
                        st.success(log)
                    elif "VM DETECTED" in log:
                        st.warning(log)
                    elif "HARDWARE" in log:
                        st.info(log)
                    else:
                        st.caption(log)
            if st.button("Clear Logs", use_container_width=True):
                st.session_state.logs = []
                st.rerun()
            if st.button("🚪 Logout", use_container_width=True):
                logout_modal()

        if not DEPENDENCIES_INSTALLED:
            st.error("Missing dependencies. Run: pip install streamlit-js-eval")
        else:
            # --- MILITARY-GRADE FINGERPRINTING SCRIPT (from app1.py) ---
            fingerprint_script = """
            (async function() {
                var details = {
                    cores: navigator.hardwareConcurrency || 'Unknown',
                    memory: navigator.deviceMemory || 'Unknown',
                    platform: navigator.platform || 'Unknown',
                    webdriver: navigator.webdriver || false,
                    screen_res: window.screen.width + "x" + window.screen.height,
                    timezone: Intl.DateTimeFormat().resolvedOptions().timeZone,
                    gpu: 'Unknown',
                    connection: 'Unknown',
                    battery: 'No Battery/Desktop',
                    audio_fp: 'Unknown'
                };
                
                try {
                    var canvas = document.createElement("canvas");
                    var gl = canvas.getContext("webgl") || canvas.getContext("experimental-webgl");
                    if (gl) {
                        var debugInfo = gl.getExtension("WEBGL_debug_renderer_info");
                        if (debugInfo) {
                            details.gpu = gl.getParameter(debugInfo.UNMASKED_RENDERER_WEBGL);
                        }
                    }
                } catch(e) {}

                if (navigator.connection) {
                    details.connection = navigator.connection.effectiveType || 'Unknown';
                }

                try {
                    var audioCtx = new (window.AudioContext || window.webkitAudioContext)();
                    details.audio_fp = audioCtx.state; 
                    audioCtx.close();
                } catch(e) { details.audio_fp = "Not Supported"; }

                try {
                    if (navigator.getBattery) {
                        let batt = await navigator.getBattery();
                        details.battery = (batt.level * 100) + "% " + (batt.charging ? "(Charging)" : "(On Battery)");
                    }
                } catch(e) {}
                
                return JSON.stringify(details);
            })()
            """
            
            js_data_str = streamlit_js_eval(
                js_expressions=fingerprint_script, 
                want_output=True, 
                key="security_scan_ultra"
            )

            # Parse JSON
            device_data = None
            if js_data_str:
                try:
                    device_data = json.loads(js_data_str)
                except:
                    device_data = None

            if device_data:
                # Extract
                d_cores = device_data.get('cores', 'Unknown')
                d_mem = device_data.get('memory', 'Unknown')
                d_gpu = device_data.get('gpu', 'Unknown')
                d_batt = device_data.get('battery', 'Unknown')
                d_conn = device_data.get('connection', 'Unknown')
                d_auto = device_data.get('webdriver')
                
                if not st.session_state.security_scanned:
                    # -- INTEGRITY LOGIC --
                    
                    # 1. LOG CORES & RAM
                    add_log(f"HARDWARE: CPU Cores: {d_cores} | RAM: ~{d_mem} GB")
                    
                    # 2. VM/Bot Check (GPU)
                    if "SwiftShader" in d_gpu or "llvmpipe" in d_gpu:
                        add_log(f"⚠️ VM DETECTED: Renderer '{d_gpu}' is virtual.")
                    else:
                        add_log(f"HARDWARE: GPU '{d_gpu}' verified.")

                    # 3. Power Source Check
                    if "No Battery" in d_batt:
                         add_log("INFO: Device appears to be a Desktop or Server (No Battery).")
                    else:
                         add_log(f"POWER: Battery detected at {d_batt}.")

                    # 4. Network
                    add_log(f"NETWORK: Connection type '{d_conn}'.")

                    # 5. Automation
                    if d_auto:
                        add_log("🚨 SECURITY ALERT: Browser Automation Detected!")
                    else:
                        add_log("INTEGRITY: Browser environment is native.")

                    st.session_state.security_scanned = True


        # --- MAIN HEADER ---
        st.subheader("Live Verification Portal")

        # --- TABS FOR INTEGRITY CHECK UI ---
        tab_verify, tab_security = st.tabs(["🎥 Live Broadcast", "🔒 Security Status"])

        with tab_verify:
            st.markdown("----")
            
            st.markdown("""
                <div style="text-align: center; padding: 15px; border-radius: 12px; background: rgba(255, 255, 255, 0.05); border: 1px solid rgba(255, 255, 255, 0.1); margin-bottom: 20px;">
                    <div style="font-weight: bold; font-size: 1.2rem; margin-bottom: 12px; color: #ff4b4b; letter-spacing: 1px;">
                        📢 INSTRUCTIONS:
                    </div>
                    <div style="font-size: 1rem; color: rgba(255, 255, 255, 0.9); margin-bottom: 6px;">
                        1. Sit in a well lit environment and press <b>START</b>.
                    </div>
                    <div style="font-size: 1rem; color: rgba(255, 255, 255, 0.9);">
                        2. Look directly into the camera and speak the code shown below loudly and clearly.
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
            st.markdown("----")


            # --- SESSION CODE SECTION ---
            with st.container():
                # Styling from app1.py
                st.markdown(f"""
                    <div class="session-container">
                        <div class="session-label">Active Session Challenge</div>
                        <div class="session-code-text">{st.session_state.session_code}</div>
                    </div>
                """, unsafe_allow_html=True)
            
                col_spacer_l, col_btn, col_spacer_r = st.columns([2, 2, 2])
                with col_btn:
                    if st.button("🔄 Generate New Code", use_container_width=True, key="reset_code_btn"):
                        st.session_state.session_code = ''.join(
                            random.choices(string.ascii_uppercase + string.digits, k=6)
                        )
                        add_log(f"SESSION: New code generated - {st.session_state.session_code}")
                        st.rerun()

            st.markdown("<br>", unsafe_allow_html=True)

            # --- VIDEO RECORDER HTML (Styling from app1.py) ---
            video_recorder_html = """
            <div class="video-wrapper">
                <div class="video-frame">
                    <div id="recordingIndicator" class="rec-dot hidden"></div>
                    <video id="videoPreview" autoplay muted playsinline></video>
                    <div id="videoOverlay" class="overlay">
                        <div id="statusText">Initialize Camera...</div>
                    </div>
                </div>

                <div class="controls-bar">
                    <button id="startBtn" onclick="startRecording()" class="btn-control start">
                        <span>🔴</span> Start
                    </button>
                    <button id="stopBtn" onclick="stopRecording()" disabled class="btn-control stop">
                        <span>⏹️</span> Stop & Verify
                    </button>
                    <button id="resetBtn" onclick="resetSession()" class="btn-control reset">
                        <span>🔄</span> Reset
                    </button>
                </div>
                
                <div id="results" class="results-area"></div>
            </div>
            
            <script>
                let mediaRecorder;
                let recordedChunks = [];
                let stream;
                let recordedBlob = null;
                let audioContext, analyser, dataArray, source;
                let animationId;
                let isVisualizerRunning = false;
                
                async function initCamera() {
                    try {
                        // Try to get both video and audio
                        stream = await navigator.mediaDevices.getUserMedia({ 
                            video: { width: 640, height: 360 }, 
                            audio: true 
                        });
                        
                        document.getElementById('videoPreview').srcObject = stream;
                        
                        // --- AUDIO REACTIVE SETUP (Safe Mode) ---
                        try {
                            setupAudioVisualizer(stream);
                        } catch (e) {
                            console.log("Audio visualizer failed to load (minor issue):", e);
                        }
                        
                        updateStatus('✅ Camera & Mic Ready', 'ready');
                        
                    } catch (err) {
                        console.error("Camera Error:", err);
                        updateStatus('❌ Camera Access Denied', 'error');
                        document.getElementById('videoOverlay').innerHTML = '<div style="color:red; padding:20px;">Camera Access Denied.<br>Please allow permissions in your browser.</div>';
                    }
                }
                
                function setupAudioVisualizer(stream) {
                    if (!window.AudioContext && !window.webkitAudioContext) return;
                    
                    audioContext = new (window.AudioContext || window.webkitAudioContext)();
                    analyser = audioContext.createAnalyser();
                    analyser.fftSize = 64; 
                    
                    source = audioContext.createMediaStreamSource(stream);
                    source.connect(analyser);
                    
                    dataArray = new Uint8Array(analyser.frequencyBinCount);
                    
                    if (!isVisualizerRunning) {
                        isVisualizerRunning = true;
                        visualize();
                    }
                }
                
                function visualize() {
                    if (!analyser) return;
                    
                    animationId = requestAnimationFrame(visualize);
                    analyser.getByteFrequencyData(dataArray);

                    // Calculate average volume
                    let sum = 0;
                    for(let i = 0; i < dataArray.length; i++) {
                        sum += dataArray[i];
                    }
                    let average = sum / dataArray.length;

                    // Map volume (0-255) to glow pixels (0-60px)
                    let glowSize = (average / 150) * 50; 
                    if (glowSize > 60) glowSize = 60;
                    
                    const videoFrame = document.querySelector('.video-frame');
                    if (videoFrame) {
                        const isRecording = videoFrame.classList.contains('recording');
                        
                        if (isRecording) {
                            // PULSING RED (Recording)
                            videoFrame.style.boxShadow = `0 0 ${20 + glowSize}px rgba(255, 75, 75, ${0.4 + (average/255)})`;
                            videoFrame.style.borderColor = `rgba(255, 75, 75, ${0.8 + (average/500)})`;
                        } else {
                            // IDLE - NO GLOW (Static)
                            videoFrame.style.boxShadow = 'none';
                            videoFrame.style.borderColor = "#333";
                        }
                    }
                }
                
                function updateStatus(text, type) {
                    const el = document.getElementById('statusText');
                    if(el) {
                        el.textContent = text;
                        if(type === 'recording') el.style.color = '#ff4b4b';
                        else if(type === 'success') el.style.color = '#00e676';
                        else el.style.color = 'white';
                    }
                }

                function startRecording() {
                    if (!stream) return;
                    
                    // Resume Audio Context if suspended (Browser policy)
                    if (audioContext && audioContext.state === 'suspended') {
                        audioContext.resume();
                    }

                    recordedChunks = [];
                    recordedBlob = null;
                    document.getElementById('results').innerHTML = '';
                    document.getElementById('recordingIndicator').classList.remove('hidden');
                    document.querySelector('.video-frame').classList.add('recording');
                    
                    const options = { mimeType: 'video/webm;codecs=vp9,opus' };
                    try {
                        mediaRecorder = new MediaRecorder(stream, options);
                    } catch (e) {
                        // Fallback for Safari/other browsers
                        mediaRecorder = new MediaRecorder(stream);
                    }
                    
                    mediaRecorder.ondataavailable = (event) => {
                        if (event.data.size > 0) recordedChunks.push(event.data);
                    };
                    
                    mediaRecorder.onstop = async () => {
                        document.getElementById('recordingIndicator').classList.add('hidden');
                        document.querySelector('.video-frame').classList.remove('recording');
                        
                        // Reset Glow
                        const videoFrame = document.querySelector('.video-frame');
                        if(videoFrame) {
                            videoFrame.style.boxShadow = 'none';
                            videoFrame.style.borderColor = '#333';
                        }

                        recordedBlob = new Blob(recordedChunks, { type: 'video/webm' });
                        updateStatus('⏳ Uploading & Analyzing...', 'neutral');
                        await uploadAndVerify(recordedBlob);
                    };
                    
                    mediaRecorder.start();
                    toggleButtons(true);
                    updateStatus('🔴 RECORDING... Speak Code: __SESSION_CODE__', 'recording');
                }
                
                function stopRecording() {
                    if (mediaRecorder && mediaRecorder.state !== 'inactive') {
                        mediaRecorder.stop();
                        toggleButtons(false);
                    }
                }
                
                function toggleButtons(isRecording) {
                    document.getElementById('startBtn').disabled = isRecording;
                    document.getElementById('stopBtn').disabled = !isRecording;
                    document.getElementById('startBtn').style.opacity = isRecording ? 0.5 : 1;
                    document.getElementById('stopBtn').style.opacity = !isRecording ? 0.5 : 1;
                }
                
                async function uploadAndVerify(blob) {
                    try {
                        const formData = new FormData();
                        formData.append('file', blob, 'session-__SESSION_CODE__.webm');
                        
                        const response = await fetch('http://localhost:8000/analyze/live-verification', {
                            method: 'POST',
                            body: formData
                        });
                        
                        if (response.ok) {
                            const result = await response.json();
                            displayResults(result);
                            updateStatus('✅ Verification Complete', 'success');
                        } else {
                            updateStatus('❌ Server Error', 'error');
                        }
                    } catch (err) {
                        updateStatus('❌ Connection Failed', 'error');
                        document.getElementById('results').innerHTML = `<div style="color:#ff4b4b; text-align:center; padding:10px;">Ensure Backend is running on Port 8000</div>`;
                    }
                }
                
                function displayResults(result) {
                    const scores = result.scores;
                    const video = scores.video;
                    const audio = scores.audio;
                    const isPass = result.final_verdict === 'PASS';
                    
                    // --- FIX STARTS HERE ---
                    // 1. Safely access the nested code object
                    const codeData = (scores && scores.code) ? scores.code : {};
                    
                    // 2. Extract the spoken code (Fixes "N/A" issue)
                    const detectedSIC = codeData.spoken_code || "N/A";
                    
                    // 3. Calculate Percentage Correctness (Confidence * 100)
                    let sicAccuracy = "0.0%";
                    if (codeData.confidence !== undefined && codeData.confidence !== null) {
                        sicAccuracy = (codeData.confidence * 100).toFixed(1) + "%";
                    }
                    // --- FIX ENDS HERE ---
                    
                    let html = `
                    <div style="margin-top:20px; display:grid; grid-template-columns:1fr 1fr; gap:10px; color: white;">
                        <div style="background:#13161c; padding:15px; border-radius:10px; border-left:4px solid ${video.verdict === 'REAL' ? '#00e676' : '#ff4b4b'}">
                            <div style="font-size:0.8em; opacity:0.7">VISUAL ANALYSIS</div>
                            <div style="font-size:1.4em; font-weight:bold">${(video.score * 100).toFixed(1)}%</div>
                            <div style="color:${video.verdict === 'REAL' ? '#00e676' : '#ff4b4b'}">${video.verdict}</div>
                        </div>
                        <div style="background:#13161c; padding:15px; border-radius:10px; border-left:4px solid ${audio.verdict === 'REAL' ? '#00e676' : '#ff4b4b'}">
                            <div style="font-size:0.8em; opacity:0.7">AUDIO PATTERNS</div>
                            <div style="font-size:1.4em; font-weight:bold">${(audio.score * 100).toFixed(1)}%</div>
                            <div style="color:${audio.verdict === 'REAL' ? '#00e676' : '#ff4b4b'}">${audio.verdict}</div>
                        </div>
                    </div>
                    
                    <div style="margin-top:15px; padding:15px; background:${isPass ? 'rgba(0,230,118,0.1)' : 'rgba(255,75,75,0.1)'}; border:2px solid ${isPass ? '#00e676' : '#ff4b4b'}; border-radius:12px; text-align:center;">
                        <h2 style="margin:0; color:${isPass ? '#00e676' : '#ff4b4b'}">${isPass ? 'ACCESS GRANTED' : 'ACCESS DENIED'}</h2>
                        <p style="margin:5px 0 0 0; opacity:0.8; color: ${isPass ? '#00e676' : '#ff4b4b'}">${isPass ? 'Identity Verified Successfully' : 'Security Threat Detected'}</p>
                        
                        <p style="margin:10px 0 0 0; font-family: monospace; font-size: 1.1em; color: white;">
                            Detected SIC: <strong>${detectedSIC}</strong> 
                            <span style="font-size:0.8em; opacity:0.7; color:#bbb; margin-left:8px;">
                                (${sicAccuracy} Match)
                            </span>
                        </p>
                    </div>
                    `;

                    document.getElementById('results').innerHTML = html;
                    if (isPass) window.parent.postMessage({type: 'streamlit:balloons'}, '*');
                }
                
                function resetSession() {
                    document.getElementById('results').innerHTML = '';
                    updateStatus('✅ Camera Ready', 'ready');
                    toggleButtons(false);
                }
                
                // Start Camera
                initCamera();
            </script>

            <style>
                /* COMPONENT CSS */
                :root {
                    --primary: #FF4B4B;
                    --success: #00e676;
                }

                .video-wrapper {
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                    font-family: 'Segoe UI', sans-serif;
                    width: 100%;
                }

                /* VIDEO FRAME STYLING */
                .video-frame {
                    position: relative;
                    width: 100%;
                    max-width: 640px;
                    aspect-ratio: 16/9;
                    background: #000;
                    border-radius: 16px;
                    overflow: hidden;
                    box-shadow: 0 10px 30px rgba(0,0,0,0.5);
                    border: 1px solid #333;
                    transition: box-shadow 0.05s ease, border-color 0.1s ease;
                }
                
                video {
                    width: 100%;
                    height: 100%;
                    object-fit: cover;
                    transform: scaleX(-1); /* Mirror effect */
                }

                /* PULSING REC DOT */
                .rec-dot {
                    position: absolute;
                    top: 20px;
                    right: 20px;
                    width: 15px;
                    height: 15px;
                    background-color: var(--primary);
                    border-radius: 50%;
                    z-index: 10;
                    box-shadow: 0 0 0 0 rgba(255, 75, 75, 1);
                    animation: pulse-red 2s infinite;
                }
                .rec-dot.hidden { display: none; }
                
                @keyframes pulse-red {
                    0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(255, 75, 75, 0.7); }
                    70% { transform: scale(1); box-shadow: 0 0 0 10px rgba(255, 75, 75, 0); }
                    100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(255, 75, 75, 0); }
                }

                /* STATUS OVERLAY */
                .overlay {
                    position: absolute;
                    bottom: 0;
                    left: 0;
                    right: 0;
                    padding: 10px;
                    background: linear-gradient(to top, rgba(0,0,0,0.8), transparent);
                    color: white;
                    text-align: center;
                    font-weight: 500;
                }

                /* CONTROLS BAR */
                .controls-bar {
                    margin-top: 20px;
                    background: #21262d;
                    padding: 8px;
                    border-radius: 50px;
                    display: flex;
                    gap: 10px;
                    box-shadow: 0 4px 15px rgba(0,0,0,0.2);
                    border: 1px solid #30363d;
                }

                .btn-control {
                    border: none;
                    padding: 10px 24px;
                    border-radius: 40px;
                    font-size: 14px;
                    font-weight: 600;
                    cursor: pointer;
                    transition: all 0.2s;
                    display: flex;
                    align-items: center;
                    color: white;
                }
                
                .btn-control.start { background: #238636; }
                .btn-control.start:hover { background: #2ea043; }
                
                .btn-control.stop { background: var(--primary); }
                .btn-control.stop:hover { background: #ff5b5b; }
                
                .btn-control.reset { background: #30363d; border: 1px solid #6e7681; }
                .btn-control.reset:hover { background: #484f58; }

                .btn-control:disabled {
                    opacity: 0.5;
                    cursor: not-allowed;
                }

                .btn-control span { margin-right: 8px; font-size: 1.2em; }
            </style>
            """.replace("__SESSION_CODE__", st.session_state.session_code)
            
            components.html(video_recorder_html, height=750)

        with tab_security:
            st.subheader("Forensic Metadata")
            if device_data:
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.caption("DEVICE IDENTITY")
                    st.write(f"**CPU Cores:** {d_cores}")
                    st.write(f"**RAM:** ~{d_mem} GB")
                    st.write(f"**GPU:** {d_gpu}")
                with c2:
                    st.caption("ENVIRONMENT")
                    st.write(f"**Battery:** {d_batt}")
                    st.write(f"**Net Type:** {d_conn}")
                    st.write(f"**Timezone:** {device_data.get('timezone')}")
                with c3:
                    st.caption("SECURITY")
                    st.write(f"**Automation:** {'DETECTED 🚨' if device_data.get('webdriver') else 'Clean'}")
                    st.write(f"**Audio State:** {device_data.get('audio_fp')}")
                
                with st.expander("View Raw JSON Payload"):
                    st.json(device_data)
            else:
                st.warning("Synchronizing with hardware... please wait.")

    # ==========================
    #      DATABASE UI (Functionality from app.py)
    # ==========================
    elif nav_mode == "Database":
        with st.sidebar:
            st.markdown("#### 🌓 System Theme")
            if st.button("Switch to " + ("Light Mode" if st.session_state.theme == "dark" else "Dark Mode"), use_container_width=True, key="db_theme_btn"):
                st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"
                st.rerun()
            # Styling from app1.py
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