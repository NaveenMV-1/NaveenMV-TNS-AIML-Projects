# 🏦 Loan Approval Predictor (Supervised Machine Learning)

A complete end-to-end Machine Learning web application to evaluate loan applications, assess borrower risk, and predict approval status in real-time.

---

## 📁 Project Architecture & Files

| File | Type | Purpose |
|------|------|---------|
| `requirements.txt` | Configuration | Defines dependencies for training, FastAPI backend, and Streamlit frontend. |
| `loans.csv` | Dataset | 500 applicant records (`income`, `credit_score`, `loan_amount`, `employment_years`, `loan_status`). |
| `dataset.py` | Script | Programmatically generates synthetic applicant data if you ever need new data. |
| `train_model.py` | ML Pipeline | Preprocesses, scales features (`StandardScaler`), trains Random Forest, and exports `.pkl` files. |
| `model.pkl` | Binary Artifact | Saved trained Random Forest classification model. |
| `scaler.pkl` | Binary Artifact | Saved StandardScaler preprocessor. |
| `main.py` | Backend API | FastAPI REST service providing `/predict` endpoint. |
| `app.py` | Frontend UI | Streamlit interactive dashboard with live calculation and dual-mode inference. |

---

## 🚀 How to Run the Project

### Step 1: Install Dependencies
Open your terminal inside this folder and run:
```bash
python -m pip install -r requirements.txt
```

### Step 2: Train the ML Model
Generate `model.pkl` and `scaler.pkl`:
```bash
python train_model.py
```

### Step 3: Run the Backend & Frontend

#### Terminal 1 — Start FastAPI Backend:
```bash
python -m uvicorn main:app --reload --port 8000
```
- API Documentation: [http://localhost:8000/docs](http://localhost:8000/docs)
- Health Check: [http://localhost:8000/health](http://localhost:8000/health)

#### Terminal 2 — Start Streamlit Frontend:
```bash
python -m streamlit run app.py
```
- Web App UI: [http://localhost:8501](http://localhost:8501)

*(Note: `app.py` is equipped with direct model loading fallback, so you can also run it directly even if the FastAPI backend is temporarily stopped!)*
