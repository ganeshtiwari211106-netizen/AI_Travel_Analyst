# ✈️ AI Travel Analyst — Flight Price Prediction & EDA

A comprehensive Machine Learning solution developed for the MIC AIML Recruitment Challenge (Track 3: 2nd Year Track — Part 1 Exploration + Part 2 Modeling).

🎥 **Project Demo Video (3–5 Min):** [Paste Your Public Google Drive / YouTube Video Link Here]

---

## 📌 Project Overview
**AI Travel Analyst** analyzes historical domestic and international flight pricing data to uncover the primary macroeconomic and itinerary-based cost drivers. The system features a full data cleaning pipeline, exploratory visualizations, a trained Gradient Boosted Decision Tree model, and an interactive Streamlit dashboard for real-time price estimation.

---

## 🎯 Problem Statement
Airfare pricing fluctuates dynamically based on demand curves, booking windows, carrier tiers, and routing efficiency. This project aims to:
1. Identify and visualize key factors dictating flight ticket pricing.
2. Build an optimized machine learning regression model to accurately forecast ticket prices based on passenger booking inputs.

---

## 📂 Dataset Used
The recruitment flight pricing dataset (`flight_pricing_dataset.csv`) contains historical flight records with 18 raw attributes:
* **Identifiers & Dates:** `Flight_ID`, `Departure_Date`, `Departure_Time`, `Arrival_Time`
* **Route & Specs:** `Source`, `Destination`, `Total_Stops`, `Distance_km`, `Duration`, `Aircraft_Type`
* **Passenger & Market Features:** `Airline`, `Travel_Class`, `Days_Before_Departure`, `Season`, `Weekday`, `Booking_Channel`, `Passenger_Count`
* **Target Variable:** `Price` (INR)

---

## 🛠️ Technologies Used
* **Programming Language:** Python 3.10+
* **Data Processing & Pipeline:** Pandas, NumPy, Regex (`re`)
* **Data Visualization:** Matplotlib, Seaborn
* **Machine Learning:** Scikit-Learn (`HistGradientBoostingRegressor`, `RandomForestRegressor`, `LinearRegression`)
* **Model Serialization:** Joblib
* **Web UI Dashboard:** Streamlit
* **Version Control:** Git, GitHub

---

## 🔬 Methodology & Architecture

### 1. Data Cleaning & Preprocessing (Part 1)
* **Format Normalization:** Parsed heterogeneous string duration formats (e.g., `"14h 10m"`, `"609 min"`, `"4.33"`) into standardized decimal hours.
* **Stop Resolution:** Standardized mixed categorical values (`"non-stop"`, `"1 stop"`, `"0"`, `"1"`) into numeric integers (`0`, `1`, `2`).
* **Artifact Stripping:** Removed non-numeric currency prefixes (`"Rs."`, commas) and unit markers (`"km"`, `"days"`).
* **Entity Standardization:** Resolved case-sensitivity discrepancies across categorical entities (`Airline`, `Source`, `Destination`, `Travel_Class`).

### 2. Exploratory Data Analysis (Part 1)
Generated 5 core visualizations:
* **Price Distribution (Histogram):** Captures the long-tailed distribution and secondary density peak for premium seating tiers.
* **Booking Window Dynamics (Line Plot):** Highlights price escalation during the final 7–10 days before departure.
* **Airline Price Variance (Box Plot):** Details median pricing and outlier distribution across budget vs. full-service carriers.
* **Layover Cost Impact (Bar Chart):** Demonstrates average fare progression across direct vs. multi-stop flights.
* **Correlation Heatmap:** Evaluates multi-collinearity across numerical parameters (`Distance_km`, `Duration_Hours`, `Days_Before_Departure`, `Total_Stops`).

### 3. Feature Engineering & Modeling (Part 2)
* Extracted `Departure_Hour` (0–23) from timestamps to reflect peak vs. red-eye pricing patterns.
* Encoded categorical attributes via One-Hot Encoding.
* Implemented **HistGradientBoostingRegressor** to train sequential decision trees on prediction residuals.

---

## 📊 Results & Model Evaluation

| Model Architecture | Technique | R² Score | Mean Absolute Error (MAE) |
| :--- | :--- | :--- | :--- |
| **Linear Regression** | Baseline Single Model | ~0.5200 | ~₹17,200.00 |
| **Random Forest Regressor** | Bagging Ensemble (Parallel Trees) | 0.6315 | ₹14,118.21 |
| **HistGradientBoostingRegressor** | Boosting Ensemble (Sequential Residuals) | **0.6781** | **₹12,793.34** |

### Key Findings
* **Booking Lead Time:** Booking 20–45 days prior to departure secures the lowest average fares; pricing climbs sharply inside the 10-day window.
* **Primary Cost Drivers:** `Travel_Class` (Business/First) and `Distance_km` account for the highest feature variance, followed by `Aircraft_Type` (wide-body vs. narrow-body).

---

## ⚠️ Challenges Faced
* **Heavily Corrupted Data Formats:** Resolving irregular string tokens, mixed duration formats, and unit suffixes across numeric fields.
* **Outlier Sensitivity:** Managing squared-error penalties on high-price international/first-class outliers during model optimization.

---

## 🚀 Future Improvements
* Integrating **SHAP (SHapley Additive exPlanations)** values for localized feature attribution in the UI.
* Deploying real-time web scraping pipelines for live airfare tracking and anomaly alerts.

---

## 💻 Installation & Setup Instructions

```bash
# 1. Clone Repository
git clone [https://github.com/](https://github.com/)<YOUR_GITHUB_USERNAME>/AI_Travel_Analyst.git
cd AI_Travel_Analyst

# 2. Create and Activate Virtual Environment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# 3. Install Required Dependencies
pip install -r requirements.txt

# 4. Run EDA & Model Training Pipeline
python src/eda.py
python src/train_model.py

# 5. Launch the Streamlit App
python -m streamlit run app.py
