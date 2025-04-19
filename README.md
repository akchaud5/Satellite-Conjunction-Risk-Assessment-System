# Welcome to the Satellite Conjunction Risk Assessment System! 🚀

This tool helps assess potential orbital collisions using a blend of orbital mechanics and advanced statistical models. The system allows users to visualize satellite orbits, calculate collision probabilities, and analyze Conjunction Data Messages (CDMs). It's designed for space agencies, satellite operators, and researchers to improve decision-making and avoid costly or dangerous on-orbit collisions.


## 🚀 Key Features

### User Accounts
- **Registration & Login**: Users can create secure accounts to access the system.
- **Profile Management**: Users can update profile details and manage their account.

### Collision Prediction Functionality
- **Data Input**: Upload satellite information for collision risk assessments.
- **Analytical Predictions**: Perform physics-based orbital calculations using MATLAB.
- **Machine Learning Predictions**: Generate enhanced collision predictions using AI/ML models.
- **Reports**: Save and manage prediction reports for further analysis.

### Admin Controls
- **User Management**: Admins can manage user accounts, including role assignments.
- **System Monitoring**: Admins can monitor prediction usage and system performance.

## 🛠️ Tech Stack

- **Backend**: Django with Django REST Framework
- **Orbit Mechanics**: MATLAB integration via MATLAB Engine for Python
- **Machine Learning**: scikit-learn, XGBoost for predictive modeling
- **Scientific Computing**: NumPy, SciPy, Pandas
- **Frontend**: Next.js (React) with TypeScript and Tailwind CSS 
- **Visualization**: D3.js for globe rendering, Satellite.js for orbit calculations
- **Database**: SQLite (local development) / PostgreSQL (production)
- **Authentication**: JWT for secure API access

## 📂 Project Structure

```plaintext
Satellite-Conjunction-Risk-Assessment/
│
├── on-orbit-frontend/             # Next.js frontend
│
├── Orbit_Predictor-BackEnd/       # Django backend
│   ├── api/                       # Django app with models, views, serializers, and URLs
│   │   ├── matlab/                # MATLAB integration for physics-based calculations
│   │   ├── ml/                    # Machine learning module for AI-driven predictions
│   │   ├── models/                # Data models including CDM, Collision, and ML models
│   │   ├── management/            # Management commands for data handling and ML training
│   │   └── views/                 # API endpoints for frontend communication
│   └── orbit_predictor/           # Main project configuration files
│
├── env_py312/                     # Python virtual environment
│
├── test_ml.py                     # ML functionality testing script
├── create_test_data.py            # Script for generating test collision data
│
└── README.md                      # Project README
```

## 🚀 Getting Started

### Prerequisites

- **Python 3.10-3.12** (MATLAB Engine supports up to 3.12) for the backend
- **Node.js** and **npm** for the Next.js frontend
- **MATLAB R2024b** for orbital calculations (required for full functionality)
- **Machine Learning Libraries**: scikit-learn, pandas, joblib, xgboost
- **Java JRE 11** (Amazon Corretto 11 recommended for Apple Silicon)
- **SQLite** is included for local development
- **PostgreSQL** for production database (optional)

### Setup

1. **Clone the Repository**

   ```bash
   git clone https://github.com/akchaud5/Satellite-Conjunction-Risk-Assessment.git
   cd Satellite-Conjunction-Risk-Assessment
   ```

2. **Install Dependencies**

   - **Backend**: Set up and activate the virtual environment, then install Django and other requirements.

     ```bash
     # Use Python 3.12 for MATLAB compatibility
     python3.12 -m venv py312_venv
     source py312_venv/bin/activate
     pip install -r requirements.txt
     ```

   - **MATLAB Engine**: The MATLAB Engine is now automatically installed when you run pip install with the requirements.txt file. Make sure you have MATLAB R2024b installed.

   - **Important Note**: MATLAB Engine for Python only supports Python versions up to 3.12. Do not use Python 3.13 or newer.

   - **Frontend**: Navigate to the `on-orbit-frontend` folder and install dependencies.

     ```bash
     cd on-orbit-frontend
     npm install
     ```

3. **Database Setup**

   **SQLite (Development)**: 
   SQLite is configured by default for local development. No additional setup is required.
   
   **PostgreSQL (Production)**:
   For production environments, the project is configured to use PostgreSQL:
   
   ```bash
   # Using Docker (recommended)
   docker-compose up -d
   
   # Manual setup
   # 1. Install PostgreSQL
   # 2. Create a database: orbit_predictor
   # 3. Configure environment variables in .env file
   # 4. Run migrations: python manage.py migrate
   ```
   
   For detailed instructions on migrating from SQLite to PostgreSQL, see [POSTGRES_MIGRATION.md](POSTGRES_MIGRATION.md).

4. **Inputting CDMs**  

   In order to input CDMs into the DB, you can use a configured endpoint and send in the CDM data as a JSON object. Here's how:

   Assuming your backend is running on port `8000`:
   Send a request to `http://localhost:8000/api/cdms/create/` with your CDM json object. Example:

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


6. **Run DB Migrations and Load Sample Data**

   Set up the database and load sample data:

   ```bash
   cd Orbit_Predictor-BackEnd
   python manage.py migrate
   python manage.py seed_cdm_data --file api/sample_data/oct5_data/cdm0.json
   python manage.py seed_cdm_data --file api/sample_data/oct5_data/cdm1.json
   python manage.py seed_cdm_data --file api/sample_data/oct5_data/cdm2.json
   ```

### Running the Project

#### Using Docker (Recommended for Production)

The easiest way to run the entire stack with PostgreSQL:

```bash
# Start all services (backend, frontend, PostgreSQL)
docker-compose up -d

# View logs
docker-compose logs -f

# Stop all services
docker-compose down
```

This will start:
- **PostgreSQL database** at `localhost:5432`
- **Django backend** at `http://localhost:8000`
- **Next.js frontend** at `http://localhost:3000`

#### Manual Setup (Development)

Run the backend and frontend in separate terminal windows:

1. **Start the Django backend**:
   ```bash
   cd Orbit_Predictor-BackEnd
   source ../py312_venv/bin/activate
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

### Using the Visualization

1. Create an account and log in
2. Navigate to "Visualization" in the sidebar
3. Select satellites from the dropdown menus (e.g., ISS - 25544 and NOAA-20 - 43013)
4. Click "View Orbital Trajectories" to see the 3D visualization

### Machine Learning Integration

The system features a sophisticated machine learning module that enhances collision prediction capabilities:

#### ML Capabilities

- **Collision Probability Prediction**: ML models trained to provide more accurate collision probabilities
- **Risk Classification**: Binary classifiers to categorize conjunctions as high or low risk
- **Feature Importance Analysis**: Identifies which orbital parameters most influence collision risk
- **Multiple Algorithms**: Support for Random Forest, Gradient Boosting, and XGBoost

#### Using Machine Learning

Train and use ML models with the following commands:

```bash
# Train a new collision probability prediction model
cd Orbit_Predictor-BackEnd
source ../env_py312/bin/activate
python manage.py train_ml_model --model-type collision_probability --algorithm random_forest

# Train with hyperparameter tuning
python manage.py train_ml_model --model-type conjunction_risk --tune

# Test ML functionality
cd ..
python test_ml.py
```

#### API Endpoints

The ML functionality is accessible through REST API endpoints:

- `GET/POST /api/ml/models/`: List and create ML models
- `POST /api/ml/predict/`: Make predictions using trained models
- `GET /api/ml/predictions/`: View prediction history for CDMs

### Maintaining Data Quality

The system includes several tools to ensure data quality:

#### Check Inactive Satellites

Verify and remove CDMs with satellites that are no longer in orbit:

```bash
# Check for inactive satellites (dry run - no changes made)
cd Orbit_Predictor-BackEnd
source ../env_py312/bin/activate
python manage.py check_inactive_satellites --dry-run

# Remove CDMs with inactive satellites
python manage.py check_inactive_satellites
```

This feature uses TLE (Two-Line Element) data from multiple sources to verify if a satellite is still in orbit, helping maintain a clean and accurate database.

#### Update CDM Dates

Make conjunction events appear current by updating their timestamps:

```bash
# Preview date changes without modifying the database
cd Orbit_Predictor-BackEnd
source ../env_py312/bin/activate
python manage.py update_cdm_dates --dry-run

# Update all CDM dates to be current
python manage.py update_cdm_dates
```

This is particularly useful for demonstrations and testing, as it makes past or future-dated conjunction events appear to be happening now.

