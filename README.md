# Welcome to the Satellite Conjunction Risk Assessment System! 🚀

This tool helps assess potential orbital collisions using a blend of orbital mechanics and advanced statistical models. The system allows users to visualize satellite orbits, calculate collision probabilities, and analyze Conjunction Data Messages (CDMs). It's designed for space agencies, satellite operators, and researchers to improve decision-making and avoid costly or dangerous on-orbit collisions.


## 🚀 Key Features

### User Accounts
- **Registration & Login**: Users can create secure accounts to access the system.
- **Profile Management**: Users can update profile details and manage their account.

### Collision Prediction Functionality
- **Data Input**: Upload satellite information for collision risk assessments.
- **Prediction Results**: Generate collision predictions based on machine learning models.
- **Reports**: Save and manage prediction reports for further analysis.

### Admin Controls
- **User Management**: Admins can manage user accounts, including role assignments.
- **System Monitoring**: Admins can monitor prediction usage and system performance.

## 🛠️ Tech Stack

- **Backend**: Django with Django REST Framework
- **Orbit Mechanics**: MATLAB integration via MATLAB Engine for Python
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
│   └── orbit_predictor/           # Main project configuration files
│
├── env/                           # Python virtual environment
│
└── README.md                      # Project README
```

## 🚀 Getting Started

### Prerequisites

- **Python 3.10-3.12** (MATLAB Engine supports up to 3.12) for the backend
- **Node.js** and **npm** for the Next.js frontend
- **MATLAB R2024b** for orbital calculations (required for full functionality)
- **Java JRE 11** (Amazon Corretto 11 recommended for Apple Silicon)
- **SQLite** is included for local development
- **PostgreSQL** for production database (optional)

### Setup

1. **Clone the Repository**

   ```bash
   git clone https://github.com/Erik-Cupsa/Satellite-Conjunction-Risk-Assessment.git
   cd Satellite-Conjunction-Risk-Assessment
   ```

2. **Install Dependencies**

   - **Backend**: Set up and activate the virtual environment, then install Django and other requirements.

     ```bash
     python3 -m venv env_py312
     source env_py312/bin/activate
     pip install -r requirements.txt
     ```

   - **MATLAB Engine**: Install the MATLAB Engine for Python (after installing MATLAB R2024b).

     ```bash
     cd /Applications/MATLAB_R2024b.app/extern/engines/python
     python setup.py install
     ```

   - **Frontend**: Navigate to the `on-orbit-frontend` folder and install dependencies.

     ```bash
     cd on-orbit-frontend
     npm install
     ```

3. **Database Setup**

   SQLite is configured by default for local development. No additional setup is required.

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

Run the backend and frontend in separate terminal windows:

1. **Start the Django backend**:
   ```bash
   cd Orbit_Predictor-BackEnd
   source ../env_py312/bin/activate
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

