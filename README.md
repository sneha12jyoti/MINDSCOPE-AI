# MINDSCOPE AI
### Advanced AI-Powered Student Mental Wellness & Lifestyle Intelligence Platform

[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.13-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-18-61DAFB.svg)](https://react.dev/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.7-3178C6.svg)](https://www.typescriptlang.org/)
[![Tailwind CSS](https://img.shields.io/badge/Tailwind-3.4-38B2AC.svg)](https://tailwindcss.com/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.6-F7931E.svg)](https://scikit-learn.org/)
[![SHAP](https://img.shields.io/badge/SHAP-TreeExplainer-FF6F00.svg)](https://shap.readthedocs.io/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)](https://www.docker.com/)
[![Tests](https://img.shields.io/badge/Pytest-42%20Passed-brightgreen.svg)]()

---

## 📌 Executive Overview

**MINDSCOPE AI** is a production-grade, full-stack student mental wellness and lifestyle intelligence platform. It transforms raw academic and digital habits into validated mental wellness predictions, provides complete model interpretability via **SHAP (SHapley Additive exPlanations)**, tracks longitudinal habits, and generates evidence-based micro-habits.

The platform was engineered by systematically auditing, benchmarking, and refactoring the legacy `MENTAL-HEALTH-SCORE` project into an enterprise architecture built on **FastAPI**, **PostgreSQL**, **React 18 + Vite**, **Extra Trees Ensembles**, and **Docker**.

> **Important Ethical & Responsible AI Notice:**
> MINDSCOPE AI is an educational lifestyle analytics research platform designed to foster self-awareness regarding screen habits, sleep, and physical activity. **It is NOT a medical diagnosis tool, psychiatric assessment system, or clinical treatment service.** Integrated 24/7 crisis helplines (such as Tele-MANAS `14416` and `988`) are highlighted for any student reporting elevated strain or distress.

---

## 🏗️ System Architecture

```
                                  +---------------------------------------+
                                  |     React 18 + Vite + Tailwind UI     |
                                  |  - Dashboard & Historical Analytics   |
                                  |  - SHAP Waterfall & Feature Drivers   |
                                  |  - Habit Checklist & Hotlines Hub     |
                                  +-------------------+-------------------+
                                                      |
                                           HTTPS / JWT Bearer
                                                      |
                                                      v
                                  +---------------------------------------+
                                  |         FastAPI Gateway (v2.0)        |
                                  |  - Sliding-Window Rate Limiting (60/m)|
                                  |  - JWT Auth & Refresh Token Rotation  |
                                  |  - Role-Based Access Control (RBAC)   |
                                  +---------+-------------------+---------+
                                            |                   |
                     +----------------------+                   +---------------------+
                     v                                                                v
   +-----------------------------------+                            +-----------------------------------+
   |      ML Engine & Explainability   |                            |     PostgreSQL / SQLAlchemy DB    |
   | - Extra Trees Regressor (n=200)   |                            | - Users & StudentProfiles         |
   | - StandardScaler & Target Encoding|                            | - Predictions & Feature Snapshots |
   | - SHAP TreeExplainer Decomposition|                            | - Recommendations & History       |
   | - PSI & Wasserstein Drift Monitor |                            | - ModelVersions & AuditLogs       |
   +-----------------------------------+                            +-----------------------------------+
```

---

## 🔬 Machine Learning Pipeline & Empirical Benchmarks

### 1. Data Quality & Preprocessing
- **Dataset:** 5,000 student social media and lifestyle observations across 13 features.
- **Deduplication:** Identified and purged duplicate records.
- **Outlier Resolution:** Detected invalid negative values in physical activity (e.g., `-0.4` in legacy code); clamped to `[0.0, 24.0]`.
- **Distribution Skewness:** Applied `FunctionTransformer(np.log1p)` to heavily skewed features (e.g., `Study_Hours`, skewness $= 1.41$).
- **Column Transformers:** Dedicated preprocessing pipelines using `StandardScaler`, `OrdinalEncoder`, `OneHotEncoder`, and defensive `SimpleImputer` fallbacks.

### 2. Multi-Model 5-Fold Cross-Validation Benchmark
All algorithms were evaluated strictly on the training set ($n = 3,998$) using 5-Fold Cross-Validation with identical random state seeds ($42$):

| Candidate Algorithm | 5-Fold CV $R^2$ Mean | 5-Fold CV $R^2$ Std | 5-Fold CV MAE | 5-Fold CV RMSE | Selection Decision |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Extra Trees Regressor** | **0.8590** | **±0.0113** | **0.3559** | **0.4738** | **Selected Best Model** |
| Random Forest Regressor | 0.8431 | ±0.0121 | 0.3802 | 0.5002 | Baseline Ensemble |
| Gradient Boosting Regressor | 0.8385 | ±0.0118 | 0.3854 | 0.5074 | Competitive Ensemble |
| XGBoost Regressor | 0.8354 | ±0.0135 | 0.3881 | 0.5120 | Gradient Boosting Variant |
| Decision Tree Regressor | 0.7697 | ±0.0194 | 0.4491 | 0.6062 | Prone to variance |
| Linear Regression | 0.7206 | ±0.0182 | 0.5132 | 0.6680 | Underfitted baseline |

### 3. Untouched Test Set Verification ($n = 1,000$)
After tuning via `RandomizedSearchCV`, the winner was verified on an untouched test set:
- **Test Set $R^2$:** **0.9186** (vs legacy ~0.87)
- **Mean Absolute Error (MAE):** **0.2607 score points** (24.9% error reduction)
- **Root Mean Squared Error (RMSE):** **0.3811**
- **Train-Test Generalization Gap:** Minimal ($R^2$ drop < 0.04), indicating strong resistance to overfitting.

---

## 🧠 Explainable AI (SHAP)

MindScope AI unpacks every prediction using **SHAP TreeExplainer**:
$$\text{Predicted Score} = E[f(X)] + \sum_{i=1}^{M} \phi_i$$
- **Cohort Base Value ($E[f(X)]$):** **6.2246**
- **Additive Decomposition:** SHAP values plus base value reconstruct the final score within floating-point machine precision.
- **Global Feature Ranking (Mean $|\phi|$):**
  1. `Avg_Daily_Usage_Hours` (Mean $|\phi| = 0.3717$)
  2. `Sleep_Hours_Per_Night` (Mean $|\phi| = 0.2643$)
  3. `Stress_Level` (Mean $|\phi| = 0.2165$)
  4. `Daily_Unlocks` (Mean $|\phi| = 0.1649$)
  5. `Study_Hours` (Mean $|\phi| = 0.1284$)
  6. `Physical_Activity_Hours` (Mean $|\phi| = 0.0912$)

---

## 🛡️ Authentication, Authorization & Security

- **Password Security:** Salted Bcrypt hashing with SHA-256 pre-hashing to prevent truncation attacks.
- **JWT Tokens:**
  - Access tokens: Short-lived (60 min) with unique cryptographic `jti` and `iat` claims.
  - Refresh tokens: Long-lived (7 days) with strict rotation on every refresh call.
- **Role-Based Access Control (RBAC):**
  - `student`: Access to personal assessment history, lifestyle overview, and habit recommendations.
  - `admin`: Access to aggregate telemetry, model registry benchmarks, drift logs, and user directory.
- **Multi-Tenant Data Isolation:** Queries strictly scoped by `user_id`. Parameter tampering returns `404 Not Found`.

---

## 📡 REST API Specification

### Authentication
| Method | Endpoint | Access | Description |
| :--- | :--- | :--- | :--- |
| `POST` | `/api/v1/auth/register` | Public | Register new student or admin account |
| `POST` | `/api/v1/auth/login` | Public | Authenticate with credentials and receive token pair |
| `POST` | `/api/v1/auth/refresh` | Public | Rotate refresh token and issue new access token |
| `POST` | `/api/v1/auth/logout` | Authenticated | Revoke refresh token and end session |

### Predictions & Explainability
| Method | Endpoint | Access | Description |
| :--- | :--- | :--- | :--- |
| `POST` | `/api/v1/predictions` | Student | Compute prediction, SHAP breakdown & recommendations |
| `GET` | `/api/v1/predictions` | Student | List authenticated student's historical predictions |
| `GET` | `/api/v1/predictions/{id}` | Student | Fetch prediction details and SHAP waterfall steps |
| `DELETE`| `/api/v1/predictions/{id}` | Student | Delete historical prediction record |

### Analytics & Recommendations
| Method | Endpoint | Access | Description |
| :--- | :--- | :--- | :--- |
| `GET` | `/api/v1/analytics/overview` | Student | Statistical summary (avg sleep, screen time, habits) |
| `GET` | `/api/v1/analytics/history` | Student | Time-series data points for Recharts charts |
| `GET` | `/api/v1/recommendations` | Student | Filterable list of personalized lifestyle habits |
| `POST` | `/api/v1/recommendations/{id}/complete` | Student | Mark habit recommendation as completed |
| `POST` | `/api/v1/recommendations/{id}/uncomplete` | Student | Reset habit recommendation to active |
| `GET` | `/api/v1/recommendations/crisis-resources` | Public | Regional emergency hotlines directory |

### Administration
| Method | Endpoint | Access | Description |
| :--- | :--- | :--- | :--- |
| `GET` | `/api/v1/admin/overview` | Admin | Aggregate platform telemetry & PSI drift status |
| `GET` | `/api/v1/admin/models` | Admin | List registered ML model versions and benchmark scores |
| `GET` | `/api/v1/users` | Admin | User administration directory with role filtering |

---

## 💻 Local Quickstart

### Prerequisites
- Python 3.11+
- Node.js 18+ & npm
- Git

### 1. Backend Setup
```bash
# Clone the repository
git clone https://github.com/sneha12jyoti/MINDSCOPE-AI.git
cd MINDSCOPE-AI

# Configure local settings and replace the example secrets
cp .env.example .env
# Generate a random key, then set it as SECRET_KEY in .env
python -c "import secrets; print(secrets.token_urlsafe(32))"

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r backend/requirements.txt

# Run database migrations and register the model metadata
alembic upgrade head
python backend/app/db/init_db.py

# Start FastAPI development server
uvicorn backend.app.main:app --reload --port 8000
```
API Documentation will be live at: `http://localhost:8000/docs`

### 2. Frontend Setup
```bash
# Navigate to frontend
cd frontend

# Install dependencies
npm install

# Start Vite dev server
npm run dev
```
Client application will be live at: `http://localhost:5173`

### Account Setup
No default or demo accounts are created. Register an account through the application.

---

## 🐳 Docker Deployment

The platform is fully containerized with Docker Compose orchestrating PostgreSQL, FastAPI, and Nginx:

```bash
# Create .env from .env.example and set unique SECRET_KEY and POSTGRES_PASSWORD values
cp .env.example .env

# Build and launch all services
docker compose up --build -d

# View service health and status
docker compose ps

# View live container logs
docker compose logs -f
```

Services are exposed at:
- **Frontend Dashboard:** `http://localhost` (Port 80)
- **Backend API:** `http://localhost:8000`
- **PostgreSQL Database:** `localhost:5432`

---

## 🧪 Automated Test Suite

MINDSCOPE AI includes 42 automated tests spanning unit, integration, and ML validation:

```bash
# Run complete test suite
pytest -v

# Run with coverage report
pytest --cov=backend --cov=ml_engine
```

**Test Coverage Summary:**
- ✅ `test_phase1_architecture.py`: API root, legacy backward-compatibility, and health endpoints.
- ✅ `test_ml_pipeline.py`: Data loader, outlier clipping, preprocessor imputation, candidate benchmark, and pipeline inference.
- ✅ `test_shap_explainability.py`: TreeExplainer, additive reconstruction, and waterfall formatting.
- ✅ `test_recommendations.py`: Rule triggers (sleep, screen time, high stress), lifecycle status transitions, and hotlines mapping.
- ✅ `test_auth_rbac.py`: Registration, salted bcrypt, JWT expiration, refresh rotation, and RBAC enforcement.
- ✅ `test_database.py`: SQLAlchemy relationships, foreign keys, cascades, unique constraints, and audit logging.
- ✅ `test_drift_monitoring.py`: Population Stability Index (PSI) and Wasserstein Distance algorithms.
- ✅ `test_analytics_admin.py`: Analytics aggregations, time-series endpoints, admin telemetry, and user directory.

---

## 📜 Crisis Lifelines Directory

If you or a student you know is going through a difficult time, free, confidential, 24/7 help is available:

- 🇮🇳 **India:** Tele-MANAS (`14416` or `1800-891-4416`) | Kiran (`1800-599-0019`)
- 🇺🇸 **United States:** Suicide & Crisis Lifeline (`988`) | Crisis Text Line (Text `HOME` to `741741`)
- 🇨🇦 **Canada:** Suicide Crisis Helpline (`988`)
- 🇬🇧 **United Kingdom:** NHS Mental Health (`111`) | Samaritans (`116 123`)
- 🇦🇺 **Australia:** Lifeline (`13 11 14`)
- 🌐 **Worldwide Directory:** [Befrienders Worldwide](https://www.befrienders.org/)

---

## 📄 License
This project is licensed under the MIT License. Developed for research and educational purposes.
