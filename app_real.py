import re
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import sys
import matplotlib.pyplot as plt


from nlp_utils import NLPUtils
from generate_real_time_dataset import TravelAIEngine
from db import init_db, migrate_users_table
from auth import signup, login
from history import save_search, load_user_history,delete_history_item , clear_user_history

from streamlit_webrtc import webrtc_streamer, WebRtcMode
import speech_recognition as sr

class SpeechProcessor:
    def __init__(self):
        self.text = ""

    def recv(self, frame):
        return frame


# -------------------------------------------------
# IMPORT LOCAL MODULES
# -------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

# -------------------------------------------------
# PATHS
# -------------------------------------------------
MODEL_PATH = "travel_budget_model.joblib"
FEATURE_PATH = "trained_feature_columns.joblib"
DATA_PATH = "india_comprehensive_dataset_2026.csv"

OUTPUT_DIR = os.path.join(BASE_DIR, "training_outputs")
PLOT_DIR = os.path.join(OUTPUT_DIR, "plots")
LOG_FILE = os.path.join(OUTPUT_DIR, "train_logs.txt")

# -------------------------------------------------
# INIT
# -------------------------------------------------
init_db()
migrate_users_table()   # 👈 THIS LINE FIXES IT

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="2026 India Travel AI",
    layout="wide",
    page_icon="🤖"
)

st.markdown("""
<style>
header {visibility: visible;}
[data-testid="stToolbar"] {display: flex;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
.block-container {
    padding-top: 2.2rem !important; /* safe for toolbar */
    padding-bottom: 2rem !important;
}

h1 {
    margin-top: 0.5rem !important;
    margin-bottom: 0.6rem !important;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# SESSION STATE
# -------------------------------------------------
if "user_id" not in st.session_state:
    st.session_state.user_id = None
# -------------------------------------------------
# LOAD SYSTEM
# -------------------------------------------------
@st.cache_resource
def load_system():
    return NLPUtils(), TravelAIEngine(
        serp_key=os.getenv("718b1b6b6d16e7565670055ff2d7b6608aaa40c072e4ab0a21937a49311f6171"),
        aviation_key=os.getenv("caaeb0963625efa4e8d9196cf2eda3b3")
    )

nlp_tool, gen = load_system()

# -------------------------------------------------
# AUTH UI
# -------------------------------------------------
def auth_page():
    
    st.title("🔐 Travel AI Login")
    tab1, tab2 = st.tabs(["Sign In", "Sign Up"])
    with tab1:
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if st.button("Login"):
            ok, user_id = login(username, password)
            if ok:
                st.session_state.user_id = user_id
                st.session_state.username = username
                st.success("Logged in successfully")
                st.rerun()
            else:
                st.error("Invalid credentials")
    with tab2:
        u = st.text_input("New Username")
        e = st.text_input("Email")
        p = st.text_input("New Password", type="password")

        if st.button("Create Account"):
            ok, msg = signup(u, e, p)
            if ok:
                st.success(msg)
            else:
                st.error(msg)
# -------------------------------------------------
# AUTH GUARD
# -------------------------------------------------
if st.session_state.user_id is None:
    auth_page()
    st.stop()

# -------------------------------------------------
# COST BREAKDOWN LOGIC (ERROR-PROOF)
# -------------------------------------------------
def calculate_cost_breakdown(df, destination, days, people, rain_flag):

    df = df.copy()
    df["Destination"] = df["Destination"].astype(str).str.lower()
    destination = destination.lower()

    # Safer destination match
    dest_df = df[df["Destination"].str.contains(destination, na=False)].copy()

    if dest_df.empty:
        dest_df = df.copy()

    if "Grand_Total" not in dest_df.columns:
        raise ValueError("Dataset must contain 'Grand_Total' column")

    dest_df.loc[:, "Grand_Total"] = pd.to_numeric(
        dest_df["Grand_Total"], errors="coerce"
    )

    avg_total = dest_df["Grand_Total"].mean()

    # Ratios
    transport_ratio = 0.30
    hotel_ratio = 0.35
    food_ratio = 0.20
    local_ratio = 0.15

    days = max(1, days)
    people = max(1, people)

    transport = avg_total * transport_ratio * people
    hotel = avg_total * hotel_ratio * days
    food = avg_total * food_ratio * days * people
    local = avg_total * local_ratio * days

    weather_adjustment = 1.10 if rain_flag else 1.00
    total = (transport + hotel + food + local) * weather_adjustment

    return {
        "Transport": round(transport, 2),
        "Hotel": round(hotel, 2),
        "Food": round(food, 2),
        "Local Travel": round(local, 2),
        "Final Budget": round(total, 2)
    }

def process_user_query(query: str):
        if not query:
            return

        entities = nlp_tool.extract_entities(query)
        entities["travel_mode"] = entities.get("travel_mode") or "Flight"

        if not entities.get("destination"):
           st.error("Destination not detected")
           return

        df_live = pd.read_csv(DATA_PATH)

        weather_str = gen.fetch_live_weather(entities["destination"])
        rain_flag = 1 if "rain" in weather_str.lower() else 0

        breakdown = calculate_cost_breakdown(
            df_live,
            entities["destination"],
            entities["days"],
            entities["people"],
            rain_flag
        )

        # Store session
        st.session_state["breakdown"] = breakdown
        st.session_state["entities"] = entities
        st.session_state["weather_str"] = weather_str
        st.session_state["rain_flag"] = rain_flag

        # Save history
        save_search(
           st.session_state.user_id,
           {
              "Source": entities.get("source", ""),
              "Destination": entities.get("destination", ""),
              "Travel_Mode": entities.get("travel_mode", "Flight"),
              "Duration_Days": int(entities.get("days", 1)),
              "Num_People": int(entities.get("people", 1)),
              "Temperature": float(
                  re.search(r"(\d+)", weather_str).group(1)
                  if re.search(r"(\d+)", weather_str) else 25
                ),
              "Rain_Flag": rain_flag,
              "Total_Cost": breakdown["Final Budget"]
            }
        )

# -------------------------------------------------
st.markdown("""
<style>
/* ChatGPT-style sidebar list */
.chat-item {
    padding: 10px 12px;
    margin-bottom: 6px;
    border-radius: 8px;
    cursor: pointer;
    font-size: 14px;
    background-color: #f7f7f8;
}

.chat-item:hover {
    background-color: #eaeaea;
}

.chat-item-active {
    background-color: #dfe3ff;
    font-weight: 600;
}

/* Scroll container */
.chat-history {
    max-height: 320px;
    overflow-y: auto;
}

/* Sidebar divider */
.sidebar-divider {
    margin: 12px 0;
    border-top: 1px solid #ddd;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# SIDEBAR
# -------------------------------------------------

st.sidebar.title("📊 System Status")
st.sidebar.success("Model Found" if os.path.exists(MODEL_PATH) else "Model Missing")
st.sidebar.success("Dataset Found" if os.path.exists(DATA_PATH) else "Dataset Missing")
st.sidebar.markdown("---")

st.sidebar.markdown("### 🕘 Chat History")
st.sidebar.markdown("-------")

if "active_chat" not in st.session_state:
    st.session_state.active_chat = None

history_df = load_user_history(st.session_state.user_id)

if not history_df.empty:
    history_df.columns = history_df.columns.str.lower()
    history_df = history_df.sort_values(by="timestamp", ascending=False)

    for _, row in history_df.iterrows():
        label = f"{row['source']} → {row['destination']}"

        colA, colB = st.sidebar.columns([6, 1])

        with colA:
            if st.button(label, key=f"open_{row['history_id']}", use_container_width=True):
                # ✅ Restore previous search
                st.session_state["entities"] = {
                    "source": row["source"],
                    "destination": row["destination"],
                    "days": int(row["duration_days"]),
                    "people": int(row["num_people"]),
                    "travel_mode": row["travel_mode"]
                }
                st.session_state["replay"] = True
                st.rerun()

        with colB:
            if st.button("🗑", key=f"del_{row['history_id']}"):
                delete_history_item(row["history_id"])
                st.rerun()

    if st.sidebar.button("🧹 Clear All History", use_container_width=True):
        clear_user_history(st.session_state.user_id)
        st.rerun()
else:
    st.sidebar.caption("No chats yet")
# -------------------------------------------------
# SIDEBAR PROFILE (BOTTOM FIXED)
# -------------------------------------------------
if st.session_state.user_id is not None:

    st.sidebar.markdown(
        """
        <style>
        [data-testid="stSidebar"] > div:first-child {
            display: flex;
            flex-direction: column;
            height: 100%;
        }
        .sidebar-bottom {
            margin-top: auto;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    with st.sidebar.container():
        st.markdown('<div class="sidebar-bottom">', unsafe_allow_html=True)

        col1, col2 = st.columns([1, 3])

        with col1:
            st.markdown("<div style='font-size:32px;'>👤</div>", unsafe_allow_html=True)

        with col2:
            st.markdown(
                f"""
                <div style="font-weight:600;">
                {st.session_state.get("username", "User")}
                </div>
                <div style="font-size:12px;color:gray;">Logged in</div>
                """,
                unsafe_allow_html=True
            )

        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.clear()
            st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------------------------
# UI
# -------------------------------------------------
st.markdown(
    """
    <style>
    .hero-img img {
        max-height: 260px;   /* 👈 balanced height */
        width: 100%;
        object-fit: cover;
        border-radius: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="hero-img">
        <img src="https://images.unsplash.com/photo-1677442136019-21780ecad995">
    </div>
    """,
    unsafe_allow_html=True
)


st.markdown(
    "<h1 style='margin-top:0.5rem;'>🤖 2026 India Travel AI Assistant</h1>",
    unsafe_allow_html=True
)

@st.cache_data(ttl=1800)
def enrich_live_dataset(df, _weather_engine):
    df = df.copy()

    temps = []
    rain_flags = []

    for dest in df["Destination"].astype(str):
        try:
            weather = _weather_engine.fetch_live_weather(dest)

            temp_match = re.search(r"(\d+(\.\d+)?)", weather)
            temperature = float(temp_match.group(1)) if temp_match else 25.0

            rain_flag = 1 if "rain" in weather.lower() else 0

        except Exception:
            temperature = 25.0
            rain_flag = 0

        temps.append(temperature)
        rain_flags.append(rain_flag)

    df["Temperature"] = temps
    df["Rain_Flag"] = rain_flags

    return df

st.markdown("<div style='margin-top:-10px;'></div>", unsafe_allow_html=True)

with st.form("travel_form", clear_on_submit=False): # adding mic feature button  inside chat box 

    col_input, col_mic, col_btn = st.columns([6, 1, 2])

    with col_input:
        user_query = st.text_input(
            "Enter travel request",
            placeholder="From Mumbai to Delhi for 4 days with 2 people",
            key="text_query",

        )
    with col_mic:
       webrtc_ctx = webrtc_streamer(
           key="speech",
           mode=WebRtcMode.SENDONLY,
           audio_receiver_size=256,
           media_stream_constraints={"audio": True, "video": False},
        )
    with col_btn:
        submit = st.form_submit_button("Analyze & Forecast")

if webrtc_ctx.state.playing:
    st.info("🎤 Listening... speak now")

if webrtc_ctx.state.playing is False and webrtc_ctx.audio_receiver:
    audio_frames = webrtc_ctx.audio_receiver.get_frames(timeout=1)

    if audio_frames:
        recognizer = sr.Recognizer()

        pcm_data = b"".join(
            frame.to_ndarray().tobytes()
            for frame in audio_frames
        )

        audio_data = sr.AudioData(
            pcm_data,
            sample_rate=48000,
            sample_width=2
        )

        try:
            spoken_text = recognizer.recognize_google(audio_data)

            # ✅ FEED INTO SAME PIPELINE
            st.session_state["text_query"] = spoken_text
            process_user_query(spoken_text)

            st.success(f"🎧 Heard: {spoken_text}")

        except sr.UnknownValueError:
            st.warning("Could not understand audio")

        except Exception as e:
            st.error(f"Voice input failed: {e}")


# -------------------------------------------------
# REPLAY PREVIOUS SEARCH (CRITICAL)
# -------------------------------------------------
if st.session_state.get("replay", False):
    entities = st.session_state["entities"]
    st.session_state["replay"] = False

    df_live = pd.read_csv(DATA_PATH)
    weather_str = gen.fetch_live_weather(entities["destination"])
    rain_flag = 1 if "rain" in weather_str.lower() else 0

    breakdown = calculate_cost_breakdown(
        df_live,
        entities["destination"],
        entities["days"],
        entities["people"],
        rain_flag
    )

    st.session_state["breakdown"] = breakdown
    st.session_state["weather_str"] = weather_str
    st.session_state["rain_flag"] = rain_flag


if submit:
    query = st.session_state.get("text_query", "").strip()
    if query:
        process_user_query(query)

    # -------------------------------------------------
   # st.session_state.pop("breakdown", None)
    #st.session_state.pop("pie_df", None)
    # Extract entities using NLP
    
    entities = nlp_tool.extract_entities(user_query)
    entities["travel_mode"] = entities.get("travel_mode") or  "Flight"   #default ML-safe

    if not entities["destination"]:
        st.error("Destination not detected")
        st.stop()
    # Load live dataset
    df_live = pd.read_csv(DATA_PATH)
    # Weather check
    weather_str = gen.fetch_live_weather(entities["destination"])
    rain_flag = 1 if "rain" in weather_str.lower() else 0

     # Calculate safe cost breakdown
    breakdown = calculate_cost_breakdown(
        df_live,
        entities["destination"],
        entities["days"],
        entities["people"],
        rain_flag
    )

    # ✅ STORE EVERYTHING
    st.session_state["breakdown"] = breakdown
    st.session_state["entities"] = entities
    st.session_state["weather_str"] = weather_str
    st.session_state["rain_flag"] = rain_flag

    # ✅ SAVE SEARCH HISTORY (FIX)
    save_search(
        st.session_state.user_id,
        {
            "Source": entities.get("source", ""),
            "Destination": entities.get("destination", ""),
            "Travel_Mode": entities.get("travel_mode", "Flight"),
            "Duration_Days": int(entities.get("days", 1)),
            "Num_People": int(entities.get("people", 1)),
            "Temperature": float(
                re.search(r"(\d+)", weather_str).group(1)
                if re.search(r"(\d+)", weather_str) else 25
            ),
            "Rain_Flag": rain_flag,
            "Total_Cost": breakdown["Final Budget"]
        }
    )
    
# -------------------------------------------------
# DISPLAY RESULTS (NO RESET ON SLIDER)
# -------------------------------------------------
if "breakdown" in st.session_state:
    breakdown = st.session_state["breakdown"]
    entities = st.session_state["entities"]
    weather_str = st.session_state["weather_str"]
    rain_flag = st.session_state["rain_flag"]

    # ---------------- BUDGET DISPLAY ----------------
    st.subheader("💰 Budget Breakdown")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🚆 Transport", f"₹{breakdown['Transport']:,.0f}")
    c2.metric("🏨 Hotel", f"₹{breakdown['Hotel']:,.0f}")
    c3.metric("🍽 Food", f"₹{breakdown['Food']:,.0f}")
    c4.metric("🚕 Local", f"₹{breakdown['Local Travel']:,.0f}")

    st.metric("✅ Final Budget", f"₹{breakdown['Final Budget']:,.0f}")

    # ---------------- PIE CHART ----------------
    pie_df = pd.DataFrame({
       "Category": ["Transport", "Hotel", "Food", "Local Travel"],
       "Cost": [
           breakdown["Transport"],
           breakdown["Hotel"],
           breakdown["Food"],
           breakdown["Local Travel"]
        ]
    })
     
    # Extract labels & values safely
    labels = pie_df["Category"].tolist()
    values = pie_df["Cost"].tolist()

    chart_size = st.slider(
        "Pie chart size",
         min_value=3,
         max_value=8,
         value=4,
         key="pie_size"
    )

    fig, ax = plt.subplots(figsize=(chart_size, chart_size))
    ax.pie(
       values,
       labels=labels,
       autopct="%1.1f%%",
       startangle=90,
       textprops={"fontsize": 9}
    )
    ax.axis("equal")

    st.pyplot(fig, width="content")


    # ---------------- ML PREDICTION ----------------
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        trained_feature_columns = joblib.load("trained_feature_columns.joblib")
        # SAFE numeric-only prediction
        input_df = pd.DataFrame([{
            "Source": entities["source"],
            "Destination": entities["destination"],
            "Travel_Mode": entities.get("travel_mode", "Unknown"),
            "Transport_Cost": breakdown["Transport"],
            "Duration_Days": entities["days"],
            "Num_People": entities["people"],
            "Temperature": float(
                re.search(r"(\d+)", weather_str).group(1)
                if re.search(r"(\d+)", weather_str) else 25
            ),
            "Rain_Flag": rain_flag
        }])

         # ✅ Categorical safety (only real categorical columns)
        categorical_cols = ["Source", "Destination","Travel_Mode"]

        for col in categorical_cols:
            input_df[col] = input_df[col].astype(str)
            input_df[col] = input_df[col].replace(
                ["nan", "None",  "","Unknown"],"Unknown")

        # ✅ Numeric safety (only real numeric columns)
        numeric_cols = [
           "Transport_Cost",
           "Duration_Days",
           "Num_People",
           "Temperature",
           "Rain_Flag"
        ]
        for col in numeric_cols:
            input_df[col] = pd.to_numeric(input_df[col], errors="coerce").fillna(0)
        
        # ✅ Match training order EXACTLY
        input_df = input_df.reindex(
            columns=trained_feature_columns,
            fill_value="Unknown"
        )

        # ✅ FINAL GUARD
        assert input_df.isnull().sum().sum() == 0 #"Input contains NaN values"
        # 🔮 Prediction
        prediction = model.predict(input_df)

        #st.write("Model expects:", model.feature_names_in_)
        #st.write("Input DF columns:", list(input_df.columns))

        st.metric("🤖 ML Predicted Budget", f"₹{prediction[0]:,.0f}")
        # ---------------- TRAINING VISUALS DISPLAY IN UI DASHBOARD----------------
        st.subheader("📊 Model Training Visualizations")

        plot_files = {
            "Actual vs Predicted Budget": "actual_vs_predicted.png",
            "Top Feature Importance": "feature_importance.png",
            "Weather Impact on Cost": "weather_impact.png",
            "Regional Cost Analysis": "regional_cost.png",
        }

        missing = []

        # 🔽 Optional collapse for clean UI
        with st.expander("📈 View Training Visualizations", expanded=True):
            cols = st.columns(2)  # 👈 2 plots per row
            col_index = 0

            for title, file in plot_files.items():
                path = os.path.join(PLOT_DIR, file)

                if os.path.exists(path):
                    with cols[col_index]:
                        st.markdown(f"**{title}**")

                        # ✅ Fit-size preview, full-resolution zoom
                        st.image(
                            path,
                            width=400,  # 👈 balanced preview size
                           # caption="🔍 Click to zoom (actual resolution)"
                        )

                    col_index = (col_index + 1) % 2  # move to next column
                else:
                    missing.append(file)

        if missing:
           st.warning("⚠️ Missing plot files:\n" + "\n".join(missing))

        # -------------------------------------------------
        # TRAINING LOGS
        # -------------------------------------------------
        st.subheader("🧾 Model Training Logs")

        if os.path.exists(LOG_FILE):
            with open(LOG_FILE, "r") as f:
                raw_logs = f.readlines()

        # ✅ Allow only meaningful log lines
            ALLOWED_KEYWORDS = [
                "Training started",
                "Training rows",
                "Testing rows",
                "R²",
                "R2",
                "Model saved",
                "visualization PNGs generated",
                "Clean dataset saved"
            ]

            filtered_logs = [
                line for line in raw_logs
                if any(key.lower() in line.lower() for key in ALLOWED_KEYWORDS)
            ]

            st.text_area(
                "Training & Dataset Logs (Clean View)",
                "".join(filtered_logs) if filtered_logs else "No training logs available.",
                height=300
            )
        # -------------------------------------------------
        # LIVE FULL DATASET VIEW (REAL-TIME, FIXED)
        # -------------------------------------------------
        st.subheader("📄 Live Travel Dataset (Real-Time)")

        DISPLAY_COLUMNS = [
           "Timestamp",
           "Source",
           "Destination",
           "Travel_Mode",
           "Transport_Cost",
           "Duration_Days",
           "Num_People",
           "Temperature",
           "Rain_Flag",
           "Total_Cost"
        ]

        if os.path.exists(DATA_PATH):
            try:
              df_live = pd.read_csv(DATA_PATH)
              df_live.columns = df_live.columns.str.strip()

              # ---- Normalize ----
              df_live.rename(columns={
                 "Days": "Duration_Days",
                 "People": "Num_People",
                 "Grand_Total": "Total_Cost"
              }, inplace=True)

              # ---- Weather Injection (KEY FIX) ----
              df_live = enrich_live_dataset(df_live, gen)

              # ---- Total Cost FIX ----
              if "Total_Cost" not in df_live.columns or df_live["Total_Cost"].isna().any():
                  df_live["Total_Cost"] = (
                  df_live["Transport_Cost"].fillna(0) * 1.8 +
                  df_live["Duration_Days"].fillna(0) * 2000 +
                  df_live["Num_People"].fillna(0) * 1500
                )

              # ---- Numeric Safety ----
              NUMERIC_COLS = [
                  "Transport_Cost",
                  "Duration_Days",
                  "Num_People",
                  "Temperature",
                  "Rain_Flag",
                  "Total_Cost"
                ]
              for col in NUMERIC_COLS:
                  df_live[col] = pd.to_numeric(df_live[col], errors="coerce").fillna(0)

              # ---- Timestamp Order ----
              df_live["Timestamp"] = pd.to_datetime(df_live["Timestamp"], errors="coerce")

              df_display = (
                 df_live[DISPLAY_COLUMNS]
                 .sort_values(by="Timestamp", ascending=False)
              )

              st.dataframe(
                df_display,
                use_container_width=True,
                height=450
              )

              st.caption(f"📊 Total records loaded: {len(df_display)}")

            except Exception as e:
                st.error(f"❌ Failed to load dataset: {e}")
        else:
            st.warning("⚠️ Live dataset file not found.")

        if st.sidebar.button("🧹 Clear Cache"):
           st.cache_data.clear()
           st.cache_resource.clear()
           st.rerun()
        



        