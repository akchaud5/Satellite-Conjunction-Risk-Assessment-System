#!/usr/bin/env python
"""
SQLite to PostgreSQL Migration Script

This script helps migrate data from SQLite to PostgreSQL for the
On-Orbit Collision Predictor application.

Prerequisites:
- PostgreSQL database set up and running
- SQLite database with existing data
- Django project with migrations applied to both databases

Usage:
1. Configure the settings below
2. Run: python migrate_sqlite_to_postgres.py

Note: This script should be run from the project root directory.
"""

import os
import sys
import json
import subprocess
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
SQLITE_DB_PATH = os.path.join('Orbit_Predictor-BackEnd', 'db.sqlite3')
DUMP_FILE_PATH = 'data_dump.json'
POSTGRES_ENV = {
    'PGUSER': os.getenv('DB_USER', 'postgres'),
    'PGPASSWORD': os.getenv('DB_PASSWORD', ''),
    'PGHOST': os.getenv('DB_HOST', 'localhost'),
    'PGPORT': os.getenv('DB_PORT', '5432'),
    'PGDATABASE': os.getenv('DB_NAME', 'orbit_predictor'),
}

# Django project directory
DJANGO_PROJECT_DIR = 'Orbit_Predictor-BackEnd'

def run_command(command, env=None):
    """Run a shell command and print output"""
    print(f"Running: {command}")
    try:
        env_vars = os.environ.copy()
        if env:
            env_vars.update(env)
            
        result = subprocess.run(
            command, 
            shell=True, 
            check=True, 
            text=True, 
            capture_output=True,
            env=env_vars
        )
        if result.stdout:
            print(result.stdout)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {e}")
        if e.stderr:
            print(f"Error output: {e.stderr}")
        return False

def check_prerequisites():
    """Check that required components are available"""
    # Check SQLite database exists
    if not os.path.exists(SQLITE_DB_PATH):
        print(f"Error: SQLite database not found at {SQLITE_DB_PATH}")
        return False
        
    # Check PostgreSQL connection
    pg_check_cmd = f"cd {DJANGO_PROJECT_DIR} && python -c \"import psycopg2; conn = psycopg2.connect(dbname='{POSTGRES_ENV['PGDATABASE']}', user='{POSTGRES_ENV['PGUSER']}', password='{POSTGRES_ENV['PGPASSWORD']}', host='{POSTGRES_ENV['PGHOST']}', port='{POSTGRES_ENV['PGPORT']}'); print('PostgreSQL connection successful');\""
    if not run_command(pg_check_cmd):
        print("Error: Could not connect to PostgreSQL database")
        return False
        
    return True

def dump_sqlite_data():
    """Dump data from SQLite database"""
    print("\n--- Dumping data from SQLite ---")
    dump_cmd = f"cd {DJANGO_PROJECT_DIR} && python manage.py dumpdata --exclude auth.permission --exclude contenttypes > ../{DUMP_FILE_PATH}"
    if not run_command(dump_cmd):
        print("Error: Failed to dump data from SQLite")
        return False
    
    print(f"Data dumped to {DUMP_FILE_PATH}")
    return True

def load_data_to_postgres():
    """Load data into PostgreSQL database"""
    print("\n--- Loading data into PostgreSQL ---")
    
    # First apply migrations to PostgreSQL
    migrate_cmd = f"cd {DJANGO_PROJECT_DIR} && python manage.py migrate"
    if not run_command(migrate_cmd, POSTGRES_ENV):
        print("Error: Failed to apply migrations to PostgreSQL")
        return False
    
    # Then load the data
    load_cmd = f"cd {DJANGO_PROJECT_DIR} && python manage.py loaddata ../{DUMP_FILE_PATH}"
    if not run_command(load_cmd, POSTGRES_ENV):
        print("Error: Failed to load data into PostgreSQL")
        return False
    
    return True

def verify_migration():
    """Verify that data was migrated correctly"""
    print("\n--- Verifying migration ---")
    verify_cmd = f"cd {DJANGO_PROJECT_DIR} && python manage.py shell -c \"from api.models.cdm import CDM; print(f'CDMs in PostgreSQL: {CDM.objects.count()}');\""
    if not run_command(verify_cmd, POSTGRES_ENV):
        print("Error: Failed to verify data migration")
        return False
    
    return True

def main():
    print("=== SQLite to PostgreSQL Migration ===")
    
    # Check prerequisites
    print("\nChecking prerequisites...")
    if not check_prerequisites():
        print("Error: Prerequisites check failed")
        return 1
    
    # Ask for confirmation
    proceed = input("\nThis will migrate data from SQLite to PostgreSQL. Proceed? (y/n): ")
    if proceed.lower() != 'y':
        print("Migration aborted.")
        return 0
    
    # Dump data from SQLite
    if not dump_sqlite_data():
        return 1
    
    # Load data into PostgreSQL
    if not load_data_to_postgres():
        return 1
    
    # Verify migration
    if not verify_migration():
        return 1
    
    print("\n✅ Migration complete!")
    print(f"Data has been successfully migrated from SQLite to PostgreSQL.")
    print("You can now update your settings to use PostgreSQL as your database.")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())