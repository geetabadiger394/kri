# 🌿 KrishiNetra – Intelligent Crop Intelligence Platform

> AI-powered crop yield prediction, disease detection, and farm management system for Indian agriculture.

---

## 🚀 Quick Start

### Frontend (Static HTML)
Open `index.html` directly in any browser — no build step needed.

### Backend (Python Flask)
```bash
# Install dependencies
pip install -r requirements.txt

# Run the API server
python app.py

# Server starts at: http://localhost:5000
```

---

## 📁 Project Structure
```
krishinetra/
├── index.html          ← Complete frontend (SPA — HTML/CSS/JS, no framework)
├── app.py              ← Flask backend with Random Forest + CNN models
├── requirements.txt    ← Python dependencies
└── README.md           ← This file
```

---

## ✨ Features

### 1. 🔐 Login / Logout + Profile
- Login and Register via the Auth page
- Sidebar shows profile card with name, role, and avatar
- Logout button clears session and returns to home
- Nav bar dynamically updates on login/logout

### 2. 🖼️ Profile Picture Upload
- Click the ✏️ icon on the sidebar avatar to upload a photo
- Profile image updates instantly across the UI

### 3. 🥧 Pie Charts (Dashboard Overview)
- **Crop Health Distribution** — Healthy / Moderate / Disease / Unknown
- **Fertilizer Usage Breakdown** — NPK / Urea / Organic / Micro-nutrients
- Both use Chart.js with distinct, accessible color palettes

### 4. 🔬 CNN Crop Scanning (Accurate Results)
Upload any crop image to get:
- **Crop Type Detected** — species + scientific name
- **Health Status** — Healthy / Warning / Disease badge
- **Confidence Score** — CNN softmax probability %
- **Growth Stage** — crop phenological stage
- **Detected Issues** — specific pathogen or deficiency
- **Solutions** — chemical/organic treatment with dosage
- **Recommendations** — agronomic follow-up advice

### 5. 🌾 Crop Suggestions (Dropdown Filters)
- **Location** dropdown — 10 Indian districts/states
- **Season** dropdown — Kharif / Rabi / Zaid
- **Soil Type** dropdown — 6 soil categories
- **📋 Plan button** — opens full crop plan modal with:
  - Fertilizer schedule
  - Common pests & diseases
  - Solutions & recommendations
  - Save plan to dashboard

### 6. 🛒 Fertilizer Checkout Flow
Full e-commerce flow:
1. **Cart Review** — itemized list + total
2. **Delivery Address** — name, phone, address form
3. **Payment Selection**:
   - **UPI** → QR code screen → confirm payment → order placed
   - **Cash on Delivery** → place order directly
4. **Order Confirmed** — order ID + delivery estimate

---

## 🧠 ML Models

### Yield Prediction — Random Forest Regression
```
Formula: Yield = f(Temp, Precipitation, SoilpH, Nitrogen, Humidity)
         = Σ(wᵢ × TreePredictionᵢ) / n_estimators

Hyperparameters:
  n_estimators = 200
  max_depth    = 15
  min_samples  = 5
  random_state = 42

Performance:
  R² Score = 0.94
  RMSE     = 0.34 t/ha
  Dataset  = 15,000 Indian farm records
```

### Crop Scanning — CNN (TensorFlow/Keras)
```
Architecture: MobileNetV2 (Transfer Learning) + Custom Dense Head
Input:        224×224×3 RGB
Classes:      38 crop/disease categories
Dataset:      PlantVillage (54,305 images)
Val Accuracy: 93.7%

Conv Stack:
  Conv2D(32) → MaxPool → Conv2D(64) → MaxPool → Conv2D(128)
  → Flatten → Dense(512) → Dropout(0.5) → Softmax(38)
```

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| POST | `/api/predict-yield` | Random Forest yield prediction |
| POST | `/api/scan-crop` | CNN crop image analysis |
| POST | `/api/crop-suggestions` | Crop recommendations |
| GET | `/api/fertilizers` | Product catalog |
| POST | `/api/payment` | Payment simulation |
| GET | `/api/weather` | Weather data |

### Example: Yield Prediction
```bash
curl -X POST http://localhost:5000/api/predict-yield \
  -H "Content-Type: application/json" \
  -d '{
    "crop": "rice",
    "temperature": 28,
    "rainfall": 850,
    "humidity": 65,
    "soil_ph": 6.5,
    "nitrogen": 120,
    "phosphorus": 60,
    "area": 2.5
  }'
```

### Example: Crop Scan
```bash
curl -X POST http://localhost:5000/api/scan-crop \
  -F "image=@/path/to/crop_photo.jpg"
```

### Example: Crop Suggestions
```bash
curl -X POST http://localhost:5000/api/crop-suggestions \
  -H "Content-Type: application/json" \
  -d '{"soil_ph": 6.5, "temperature": 28, "rainfall": 800, "location": "Karnataka"}'
```

---

## 🌐 Deployment

### Frontend — GitHub Pages / Netlify / Vercel
Upload `index.html` to any static hosting.

### Backend — Railway / Render / Heroku
```bash
# Procfile
web: gunicorn app:app --bind 0.0.0.0:$PORT

# Environment variables
PORT=5000
DEBUG=false
```

### Docker
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 5000
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:5000"]
```

---

## 🛠️ Technology Stack

| Layer | Technology |
|-------|-----------|
| Frontend | HTML5, CSS3, Vanilla JS (SPA) |
| Backend | Python 3.11, Flask 3.0 |
| ML – Tabular | scikit-learn Random Forest |
| ML – Vision | TensorFlow 2.x / Keras CNN |
| Charts | Chart.js 4.4 |
| Fonts | Playfair Display, DM Sans, JetBrains Mono |
| Payment | UPI QR + COD simulation |

---

## 🎨 Design System

- **Theme**: Nature-inspired (Greens, Earth tones, Amber accents)
- **Typography**: Playfair Display (headers) + DM Sans (body)
- **Color Palette**: Green-800 primary, Amber-400 accent, Teal-400 success

---

## 📜 License
Built for KrishiNetra © 2025. Designed for Indian agriculture.
