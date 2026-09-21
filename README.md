# Welcome to the Satellite Conjunction Risk Assessment System! 🚀

This tool helps assess potential orbital collisions using a blend of orbital mechanics and advanced statistical models. The system allows users to visualize satellite orbits, calculate collision probabilities, and analyze Conjunction Data Messages (CDMs). It's designed for space agencies, satellite operators, and researchers to improve decision-making and avoid costly or dangerous on-orbit collisions.


## 🚀 Key Features

### User Accounts
- **Registration & Login**: Users can create secure accounts to access the system.
- **Profile Management**: Users can update profile details and manage their account.

### Collision Prediction Functionality
- **Data Input**: Upload satellite information for collision risk assessments.
- **Analytical Predictions**: Physics-based orbital calculations using the NASA CARA
  MATLAB routines (`Pc2D_Foster`, `Pc3D_Hall`). Requires MATLAB — see
  [MATLAB is optional](#matlab-is-optional).
- **Machine Learning Predictions**: Regression and classification models over CDM
  features — see [About the ML models](#about-the-ml-models) for what they do and
  do not tell you.
- **Reports**: Save and manage prediction reports for further analysis.

### Admin Controls
- **User Management**: Admins can manage user accounts, including role assignments.
- **System Monitoring**: Admins can monitor prediction usage and system performance.

## 🛠️ Tech Stack

- **Backend**: Django 5.2 (LTS) with Django REST Framework
- **Orbit Mechanics**: MATLAB integration via MATLAB Engine for Python (optional)
- **Machine Learning**: scikit-learn, XGBoost for predictive modeling
- **Scientific Computing**: NumPy, SciPy, Pandas
- **Frontend**: Next.js 16 (React 19) with TypeScript and Tailwind CSS 4
- **Visualization**: D3.js for globe rendering, Satellite.js for orbit propagation,
  Chart.js and Highcharts for plots
- **Database**: PostgreSQL (default); SQLite available for local work and tests
- **Authentication**: JWT for secure API access

## 📂 Project Structure

```plaintext
Satellite-Conjunction-Risk-Assessment/
│
├── on-orbit-frontend/             # Next.js frontend
│   └── src/lib/api.ts             # API client: base URL, auth, token refresh
│
├── Orbit_Predictor-BackEnd/       # Django backend
│   ├── api/                       # Django app with models, views, serializers, and URLs
│   │   ├── matlab/                # MATLAB routines for physics-based calculations
│   │   ├── matlab_runtime.py      # Lazy, optional MATLAB Engine loader
│   │   ├── ml/                    # Machine learning module
│   │   ├── models/                # Data models including CDM, Collision, and ML models
│   │   ├── management/            # Management commands for data handling and ML training
│   │   ├── tests/                 # Test suite
│   │   └── views/                 # API endpoints
│   └── orbit_predictor/           # Main project configuration files
│
├── create_test_data.py            # Script for generating test collision data
├── requirements.txt               # Python dependencies
├── requirements-matlab.txt        # Optional MATLAB Engine dependency
│
└── README.md                      # Project README
```

## 🚀 Getting Started

### Prerequisites

- **Python 3.10–3.13** for the backend. If you want the MATLAB-backed analytic
  endpoints, use **3.12**: MATLAB Engine for Python supports no higher.
- **Node.js 22+** and **npm** for the Next.js frontend
- **PostgreSQL** (or use `DB_ENGINE=sqlite` for local work)
- **MATLAB R2024b** — optional, only for the analytic probability endpoints
- **Docker** — optional, runs the whole stack

### Setup

1. **Clone the Repository**

   ```bash
   git clone https://github.com/akchaud5/Satellite-Conjunction-Risk-Assessment-System.git
   cd Satellite-Conjunction-Risk-Assessment-System
   ```

2. **Install Dependencies**

   - **Backend**:

     ```bash
     python3.12 -m venv .venv
     source .venv/bin/activate
     pip install -r requirements.txt
     ```

   - **Frontend**:

     ```bash
     cd on-orbit-frontend
     npm ci
     ```

3. **Configure the environment**

   ```bash
   cp Orbit_Predictor-BackEnd/.env.example Orbit_Predictor-BackEnd/.env
   cp on-orbit-frontend/.env.example on-orbit-frontend/.env.local
   ```

   Then edit both. `SECRET_KEY`, `JWT_SECRET_KEY` and `ALLOWED_HOSTS` are
   **required** whenever `DEBUG=False`; the app refuses to start without them
   rather than falling back to an insecure default. Generate a secret key with:

   ```bash
   python -c "from django.core.management.utils import get_random_secret_key; print(get_random_secret_key())"
   ```

4. **Database Setup**

   **PostgreSQL (default)**: create a database named `orbit_predictor` and set
   `DB_NAME` / `DB_USER` / `DB_PASSWORD` / `DB_HOST` / `DB_PORT` in `.env`. See
   [POSTGRES_MIGRATION.md](POSTGRES_MIGRATION.md) for detailed instructions.

   **SQLite (local development and tests)**: set `DB_ENGINE=sqlite` in `.env`.
   No server needed.

5. **Run Migrations and Load Sample Data**

   ```bash
   cd Orbit_Predictor-BackEnd
   python manage.py migrate
   python manage.py seed_cdm_data --file api/sample_data/oct5_data/cdm0.json
   python manage.py seed_cdm_data --file api/sample_data/oct5_data/cdm1.json
   python manage.py seed_cdm_data --file api/sample_data/oct5_data/cdm2.json
   ```

   The bundled sample CDMs are dated 2024-10-05. To shift them to the present for
   a demo, see [Update CDM Dates](#update-cdm-dates).

6. **Inputting CDMs**

   To load CDMs over the API, POST a CDM JSON object to
   `http://localhost:8000/api/cdms/create/`:

    `{
     "CCSDS_CDM_VERS": "{{version}}",
     "CREATION_DATE": "{{creation_date}}",
     "ORIGINATOR": "{{originator}}",
     "MESSAGE_ID": "{{message_id}}",
     "TCA": "{{time_of_closest_approach}}",
     "MISS_DISTANCE": "{{miss_distance}}",
     "COLLISION_PROBABILITY": "{{collision_probability}}",
     "SAT1_OBJECT": "{{sat1_object}}",
     "SAT1_OBJECT_DESIGNATOR": "{{sat1_designator}}",
     "SAT1_CATALOG_NAME": "{{sat1_catalog_name}}",
     "SAT1_OBJECT_NAME": "{{sat1_object_name}}",
     "SAT1_INTERNATIONAL_DESIGNATOR": "{{sat1_intl_designator}}",
     "SAT1_OBJECT_TYPE": "{{sat1_object_type}}",
     "SAT1_OPERATOR_ORGANIZATION": "{{sat1_operator_org}}",
     "SAT1_COVARIANCE_METHOD": "{{sat1_covariance_method}}",
     "SAT1_MANEUVERABLE": "{{sat1_maneuverable}}",
     "SAT1_REFERENCE_FRAME": "{{sat1_reference_frame}}",
     "SAT1_X": "{{sat1_x}}",
     "SAT1_Y": "{{sat1_y}}"
     // continue on with rest of fields
   }`

   `MESSAGE_ID` is required. Without MATLAB installed the CDM is still stored, and
   the response carries a `warning` noting that the analytic probability was not
   computed.

### Running the Project

#### Using Docker

```bash
# Start all services (backend, frontend, PostgreSQL)
docker compose up -d

# View logs
docker compose logs -f

# Stop all services
docker compose down
```

This will start:
- **PostgreSQL database** at `localhost:5432`
- **Django backend** at `http://localhost:8000`
- **Next.js frontend** at `http://localhost:3000`

The containers run Django's development server. For production, serve
`orbit_predictor.wsgi:application` with gunicorn behind a real web server, and
run the frontend with `npm run build && npm run start`.

#### Manual Setup (Development)

Run the backend and frontend in separate terminal windows:

1. **Start the Django backend**:
   ```bash
   cd Orbit_Predictor-BackEnd
   source ../.venv/bin/activate
   python manage.py runserver
   ```

2. **Start the Next.js frontend** (in another terminal):
   ```bash
   cd on-orbit-frontend
   npm run dev
   ```

This will start:
- **Next.js frontend** at `http://localhost:3000`
- **Django backend** at `http://localhost:8000`

The frontend reads the backend URL from `NEXT_PUBLIC_API_URL`
(`on-orbit-frontend/.env.local`), defaulting to `http://localhost:8000`.

### Running the Tests

```bash
cd Orbit_Predictor-BackEnd
DEBUG=true DB_ENGINE=sqlite python manage.py test api
```

Frontend checks:

```bash
cd on-orbit-frontend
npm run lint
npx tsc --noEmit
npm run build
```

### MATLAB is optional

The analytic probability endpoints (`Pc2D_Foster` / `Pc3D_Hall`) need MATLAB
Engine for Python, which ships with MATLAB itself rather than from PyPI. It is
**not** installed by `requirements.txt`, because pinning a machine-specific
wheel path there made `pip install` fail on every machine without MATLAB at that
exact path.

Everything else — CDM management, users, the dashboard, the visualization and
the ML endpoints — runs without it. The endpoints that do need it return
**503** with an explanatory message when it is absent.

To enable them, install the engine matching your MATLAB release:

```bash
pip install -r requirements-matlab.txt
```

See `requirements-matlab.txt` for the per-platform install path. The engine
supports Python 3.9–3.12 only.

### Using the Visualization

1. Create an account and log in
2. Navigate to "Visualization" in the sidebar
3. Select satellites from the dropdown menus (e.g. ISS - 25544 and NOAA-20 - 43013)
4. Click "View Orbital Trajectories" to see the 3D visualization

### Machine Learning Integration

#### About the ML models

Be clear about what these models are. They are trained against
`Collision.probability_of_collision`, which is itself computed by the MATLAB
`Pc2D_Foster` routine from the same CDM state vectors and covariances that are
used as the input features. The models are therefore **surrogates of the
analytic calculation**, not an independent estimate of collision risk: at best
they reproduce `Pc2D_Foster` quickly and without a MATLAB dependency, and they
cannot be more accurate than it.

That is a legitimate and useful thing to have — a fast approximation that runs
where MATLAB does not. It is not "better collision probabilities". Treating the
output as an independent second opinion on risk would be a mistake.

#### ML Capabilities

- **Collision Probability Prediction**: regression onto the analytic probability
- **Risk Classification**: binary high/low-risk classification at a probability threshold
- **Feature Importance Analysis**: which orbital parameters drive the model's output
- **Multiple Algorithms**: Random Forest, Gradient Boosting, XGBoost

#### Using Machine Learning

```bash
cd Orbit_Predictor-BackEnd
source ../.venv/bin/activate

# Train a new collision probability prediction model
python manage.py train_ml_model --model-type collision_probability --algorithm random_forest

# Train with hyperparameter tuning
python manage.py train_ml_model --model-type conjunction_risk --tune
```

Training needs CDMs that already have collision records, so run
`create_test_data.py` or compute collisions first.

#### API Endpoints

- `GET/POST /api/ml/models/`: List and create ML models
- `POST /api/ml/training/`: Start a training job. Returns **202 Accepted**
  immediately with a `training_job_id`; training runs in the background.
- `GET /api/ml/training/<id>/`: Poll a training job for status and metrics
- `POST /api/ml/predict/`: Make predictions using trained models
- `GET /api/ml/predictions/`: View prediction history for CDMs

### Maintaining Data Quality

#### Check Inactive Satellites

Verify and remove CDMs with satellites that are no longer in orbit:

```bash
cd Orbit_Predictor-BackEnd
source ../.venv/bin/activate

# Check for inactive satellites (dry run - no changes made)
python manage.py check_inactive_satellites --dry-run

# Remove CDMs with inactive satellites
python manage.py check_inactive_satellites
```

This feature uses TLE (Two-Line Element) data from multiple sources to verify if
a satellite is still in orbit, helping maintain a clean and accurate database.

#### Update CDM Dates

Make conjunction events appear current by updating their timestamps:

```bash
cd Orbit_Predictor-BackEnd
source ../.venv/bin/activate

# Preview date changes without modifying the database
python manage.py update_cdm_dates --dry-run

# Update all CDM dates to be current
python manage.py update_cdm_dates
```

This exists for demonstrations: the bundled sample data is fixed at 2024-10-05,
and this shifts it to the present so conjunctions appear to be happening now. It
rewrites real timestamps, so do not run it against data you care about.
