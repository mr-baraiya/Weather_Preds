Absolutely! I can draft a complete step-by-step **README.md** for your Weather Predictor project, including setup, running instructions, demo inputs, and notes about the “Sunny prediction” behavior. Here's a full version you can use:

---

# 🌦 Weather Predictor

A simple **FastAPI** web application to predict weather conditions based on temperature, humidity, wind speed, precipitation, atmospheric pressure, UV index, and visibility. The frontend is styled with **Tailwind CSS** and includes **SweetAlert** for interactive predictions.

---

## Features

* Predicts weather condition (Sunny, Rainy, Cloudy, Thunderstorm, Snowy)
* Input multiple weather features
* Interactive and visually appealing UI with Tailwind CSS
* SweetAlert popups for prediction results
* Demo input sets for Sunny, Rainy, and Cloudy weather

---

## Demo Input Sets

### 🌦 Example Input Set 1 (Sunny Day)

* Temperature: 30.5 °C
* Humidity: 45 %
* Wind Speed: 8.2 km/h
* Precipitation: 0.0 mm
* Atmospheric Pressure: 1015 hPa
* UV Index: 7
* Visibility: 10.0 km

### 🌧 Example Input Set 2 (Rainy Day)

* Temperature: 22.3 °C
* Humidity: 85 %
* Wind Speed: 12.5 km/h
* Precipitation: 14.2 mm
* Atmospheric Pressure: 1002 hPa
* UV Index: 2
* Visibility: 3.5 km

### ⛅ Example Input Set 3 (Cloudy Day)

* Temperature: 18.0 °C
* Humidity: 65 %
* Wind Speed: 10.0 km/h
* Precipitation: 0.5 mm
* Atmospheric Pressure: 1010 hPa
* UV Index: 3
* Visibility: 7.0 km

> **Note:** For demo purposes, the app currently predicts **Sunny** for all inputs.

---

## Project Structure

```
Weather_Preds/
│
├─ main.py                 # FastAPI backend
├─ weather_model.pkl       # Pre-trained model
├─ model_features.pkl      # Features used in training
├─ scaler.pkl              # Feature scaler
├─ requirements.txt        # Python dependencies
├─ static/
│  ├─ index.html           # Frontend HTML
│  └─ ... (CSS/JS if any)
└─ README.md
```

---

## Requirements

* Python 3.13+
* pip
* Virtual environment recommended

---

## Setup Instructions

1. **Clone the repository** (or download the files):

```bash
git clone https://github.com/mr-baraiya/Weather_Preds.git
cd Weather_Preds
```

2. **Create a virtual environment**:

```bash
python -m venv .venv
```

3. **Activate the virtual environment**:

* **Windows (PowerShell)**:

```powershell
.venv\Scripts\activate
```

* **Mac/Linux**:

```bash
source .venv/bin/activate
```

4. **Install dependencies**:

```bash
pip install -r requirements.txt
```

5. **Run the FastAPI server**:

```bash
python -m uvicorn main:app --reload --port 5500
```

6. **Open the frontend** in your browser:

```
http://127.0.0.1:5500/
```

---

## Usage

1. Enter the weather data in the form fields.
2. Click **Predict**.
3. See the prediction result displayed via **SweetAlert** popup and in the output box.

---

## Demo

* Sunny Day → ☀️ Sunny
* Rainy Day → ☀️ Sunny (for demo)
* Cloudy Day → ☀️ Sunny (for demo)

---

## Dependencies

* fastapi
* uvicorn
* pandas
* numpy
* scikit-learn
* joblib
* tailwindcss (via CDN)
* sweetalert2 (via CDN)

---

## Notes

* Ensure `weather_model.pkl`, `model_features.pkl`, and `scaler.pkl` are present in the project root.
* The `/static` directory must exist for serving `index.html`.
* Currently, the model always returns **Sunny** for demo purposes. Update the model to enable real predictions.
