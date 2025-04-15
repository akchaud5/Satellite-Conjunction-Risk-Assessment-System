# Welcome to the Satellite Conjunction Risk Assessment System! 🚀

This tool helps assess potential orbital collisions using a blend of machine learning and advanced statistical models. The system allows users to input satellite data, calculate collision probabilities, and manage prediction reports. It's designed for space agencies, satellite operators, and researchers to improve decision-making and avoid costly or dangerous on-orbit collisions.

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

- **Backend**: Django
- **Machine Learning**: MATLAB for initial calculations, Python (scikit-learn) for machine learning model development
- **Frontend**: Next.js with D3 for 3D visualization
- **Database**: SQLite (local development) / PostgreSQL (production)

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

- **Python 3.10+** (but less than 3.13) and **Django** for the backend to run **MATLAB**
- **Node.js** and **npm** for the Next.js frontend
- **MATLAB** for initial prediction calculations (optional for extended functionality)
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
     python3 -m venv env
     source env/bin/activate
     pip install -r requirements.txt
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


6. **Run DB Migrations**

   If you modify the schema:

   ```bash
   cd Orbit_Predictor-BackEnd
   python manage.py makemigrations
   python manage.py migrate
   ```

### Running the Project

To run both the Django backend and the Next.js frontend concurrently:

```bash
npm run dev
```

This command will start:
- **Next.js frontend** at `http://localhost:3000`
- **Django backend** at `http://localhost:8000`

