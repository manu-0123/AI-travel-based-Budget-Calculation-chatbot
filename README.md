🌏 AI-Based Travel Budget Predictor (India – 2026)

An intelligent AI-powered travel budget assistant that uses Natural Language Processing (NLP) and Machine Learning to estimate travel costs for Indian cities based on real-time signals and historical 2026 market data.

🚀 Key Features

🧠 NLP Query Understanding
Parses natural language queries such as:
“From Mumbai to Delhi for 4 days with 2 people”

🌦 Real-Time Context Awareness

Live weather via Open-Meteo API

Weather-aware cost adjustment (rain/storm impact)

📊 Explainable Budget Breakdown

Transport

Hotel

Food

Local Travel

Final AI-estimated budget

🤖 Machine Learning Prediction

Random Forest Regressor

Trained on India-wide 2026 travel cost data

Supports unseen destinations safely

🖥 Interactive Dashboard (Streamlit)

Budget pie chart

Destination popularity insights

Real-time dataset logs

🛠️ Tech Stack

Language: Python 3.10+

Frontend: Streamlit

NLP: spaCy (en_core_web_sm)

ML Model: Scikit-learn (Random Forest Regressor)

Visualization: Matplotlib, Seaborn, Plotly

APIs:

Open-Meteo (Weather)

SerpAPI (Hotel pricing reference – optional)

📦 Installation & Setup
1️⃣ Clone the Repository
git clone https://github.com/your-username/ai-travel-budget-predictor.git
cd ai-travel-budget-predictor

2️⃣ Install Dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm

3️⃣ Generate 2026 Dataset (Required First)
python generate_real_time_dataset.py


This creates:

india_comprehensive_dataset_2026.csv

4️⃣ Train the Machine Learning Model
python train_model.py


Outputs:

travel_budget_model.joblib

Diagnostic visualization PNGs

5️⃣ Run the Streamlit Application
streamlit run app_real.py

📂 Project Structure
├── app_real.py                      # Main Streamlit app
├── nlp_utils.py                     # NLP entity extraction logic
├── generate_real_time_dataset.py    # 2026 dataset generator
├── train_model.py                   # ML training & visualization
├── travel_budget_model.joblib       # Trained ML model
├── india_comprehensive_dataset_2026.csv
├── requirements.txt
└── README.md

AI-Travel based Budget Calculation chatbot/ (to create login in or logout user account)
│
├── app_real.py
├── users.db              👈 NEW (auto-created)
├── auth.py               👈 NEW
├── db.py                 👈 NEW
├── history.py            👈 NEW
└── generate_real_time_dataset.py

📊 2026 Market Assumptions (India)

The AI model is calibrated using realistic 2026 pricing trends:

🍽 Food

₹800 – ₹1,200 per person / day

🚕 Local Travel

₹900 – ₹1,500 per day

🏨 Hotels (per night)

Budget: ₹1,200+

Standard: ₹3,500+

Premium: ₹6,000 – ₹8,500+

✈ Transport

Average domestic flight baseline: ₹8,000 – ₹9,000 (one-way)

🌧 Weather Impact

Rain / Storm: ~10% cost increase

✅ Reliability & Error Safety

✔ Handles missing cities gracefully

✔ Prevents NLP extraction failures

✔ No KeyError / TypeError during prediction

✔ Safe fallback when destination data is unavailable

✔ Model supports unseen destinations

🎓 Ideal For

Academic projects

Technical seminars

Final-year engineering demos

AI + ML portfolio projects

Streamlit cloud deployment

📌 Future Enhancements

Multi-transport mode (train / bus / flight)

Date-based seasonal pricing

User budget preference (low / medium / luxury)

Cloud deployment (Streamlit Cloud / HuggingFace)