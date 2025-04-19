# Migrating from SQLite to PostgreSQL

This document provides instructions for migrating the On-Orbit Collision Predictor application from SQLite to PostgreSQL.

## Why PostgreSQL?

PostgreSQL offers several advantages over SQLite for production environments:

1. **Concurrency**: PostgreSQL can handle multiple concurrent write operations, while SQLite can only handle one writer at a time.
2. **Scalability**: PostgreSQL can scale to handle very large datasets and high traffic.
3. **Data Types**: PostgreSQL has a richer set of data types and functions.
4. **Backup and Recovery**: PostgreSQL has robust backup and recovery options.
5. **Performance**: For large datasets, PostgreSQL often performs better.

## Getting Started Options

There are two main ways to set up PostgreSQL for this project:

1. **Docker (Recommended)**: The easiest way to get started without installing PostgreSQL
2. **Native Installation**: Installing PostgreSQL directly on your system

## Option 1: Using Docker (Recommended for Testing)

Docker provides the simplest way to test the PostgreSQL setup without installing anything directly on your system.

### Prerequisites

1. Install Docker and Docker Compose:
   - [Docker Desktop for Mac](https://docs.docker.com/desktop/install/mac-install/)
   - [Docker Desktop for Windows](https://docs.docker.com/desktop/install/windows-install/)
   - [Docker Engine for Linux](https://docs.docker.com/engine/install/)

### Steps

1. **Start the Docker containers**:

   ```bash
   cd /path/to/On-Orbit-Collision-Predictor
   docker-compose up -d
   ```

2. **Verify the containers are running**:

   ```bash
   docker-compose ps
   ```

   You should see three services running: `backend`, `db` (PostgreSQL), and `frontend`.

3. **Check the PostgreSQL logs**:

   ```bash
   docker-compose logs db
   ```

   You should see messages indicating the database is ready to accept connections.

4. **Access the application**:
   - Frontend: http://localhost:3000
   - Backend API: http://localhost:8000

## Option 2: Native PostgreSQL Installation

If you prefer to install PostgreSQL directly on your system:

### macOS Installation

1. **Using Homebrew**:

   ```bash
   # Install Homebrew if you don't have it
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   
   # Install PostgreSQL
   brew install postgresql
   
   # Start PostgreSQL service
   brew services start postgresql
   ```

2. **Create a database**:

   ```bash
   # Create a database for the project
   createdb orbit_predictor
   
   # Verify it was created
   psql -l | grep orbit_predictor
   ```

3. **Set environment variables** in your `.env` file:

   ```
   DB_NAME=orbit_predictor
   DB_USER=your_macos_username
   DB_PASSWORD=
   DB_HOST=localhost
   DB_PORT=5432
   ```

### Ubuntu/Debian Installation

1. **Install PostgreSQL**:

   ```bash
   sudo apt update
   sudo apt install postgresql postgresql-contrib
   ```

2. **Create a database**:

   ```bash
   # Login to PostgreSQL
   sudo -u postgres psql
   
   # Create database
   CREATE DATABASE orbit_predictor;
   
   # Create user (optional)
   CREATE USER orbituser WITH PASSWORD 'password';
   
   # Grant privileges
   GRANT ALL PRIVILEGES ON DATABASE orbit_predictor TO orbituser;
   
   # Exit PostgreSQL
   \q
   ```

3. **Set environment variables** in your `.env` file:

   ```
   DB_NAME=orbit_predictor
   DB_USER=orbituser
   DB_PASSWORD=password
   DB_HOST=localhost
   DB_PORT=5432
   ```

### Windows Installation

1. **Download and install** PostgreSQL from the [official website](https://www.postgresql.org/download/windows/)

2. **During installation**:
   - Set a password for the postgres user
   - Keep the default port (5432)
   - Launch Stack Builder at the end if prompted

3. **Create a database**:
   - Open pgAdmin (installed with PostgreSQL)
   - Connect to the server
   - Right-click on "Databases" and select "Create" > "Database..."
   - Name it "orbit_predictor" and save

4. **Set environment variables** in your `.env` file:

   ```
   DB_NAME=orbit_predictor
   DB_USER=postgres
   DB_PASSWORD=your_password
   DB_HOST=localhost
   DB_PORT=5432
   ```

### 3. Set Environment Variables

Create a `.env` file in the backend directory with the following content (customize as needed):

```
DB_NAME=orbit_predictor
DB_USER=postgres
DB_PASSWORD=your-password
DB_HOST=localhost
DB_PORT=5432
```

### 4. Using Docker (Recommended)

The easiest way to migrate is using the provided Docker configuration:

```bash
# Start the Docker containers
docker-compose up -d

# The migration will happen automatically
```

### 5. Manual Migration

If you're not using Docker:

```bash
# Install the psycopg2 library if not already installed
pip install psycopg2-binary

# Run migrations to create the schema in PostgreSQL
python manage.py migrate

# Create a data dump from SQLite (optional, for existing data)
python manage.py dumpdata > data.json

# Load the data into PostgreSQL
python manage.py loaddata data.json
```

## Verification

To verify that the migration was successful:

```bash
# Connect to the PostgreSQL database
psql -U postgres -d orbit_predictor

# List tables
\dt

# Verify table contents
SELECT COUNT(*) FROM api_cdm;
SELECT COUNT(*) FROM api_mlmodel;
```

## Rollback Plan

If you need to revert to SQLite:

1. Change the database settings in `settings.py` back to SQLite
2. Run `python manage.py migrate` to create the schema in SQLite
3. Load your data using `python manage.py loaddata data.json` if you created a dump

## Production Considerations

For production environments:
- Use strong, unique passwords
- Configure backups
- Set up a connection pool
- Consider using a managed PostgreSQL service like AWS RDS or Google Cloud SQL