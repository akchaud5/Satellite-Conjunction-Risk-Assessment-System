# Management Commands

This directory contains custom Django management commands for the Orbital Collision Predictor application.

## Available Commands

### Check for Inactive Satellites

This command checks all satellites in the database against the TLE APIs to verify if they are still active (in orbit).
If inactive satellites are found, it can automatically remove any CDM entries that contain these inactive satellites.

**Usage:**

```bash
# Dry run - show what would be deleted but don't actually delete
python manage.py check_inactive_satellites --dry-run

# Live run - actually delete CDMs containing inactive satellites
python manage.py check_inactive_satellites
```

**How it works:**
1. Retrieves all unique NORAD IDs from the database
2. Checks each NORAD ID against:
   - NASA/Ivanstanojevic TLE API
   - CelesTrak as a fallback
3. Identifies satellites with no TLE data as inactive
4. Removes CDM entries that contain inactive satellites

**When to use:**
- Run periodically (e.g., weekly/monthly) to clean up the database
- Run after importing new CDM data to ensure only active satellites are included
- Run when visualization errors related to specific satellites occur

### Seed CDM Data

Imports CDM data from JSON files into the database.

**Usage:**

```bash
# Import from a specific JSON file
python manage.py seed_cdm_data --file path/to/cdm/file.json
```

**Notes:**
- File must be in the proper CDM JSON format
- Any satellites listed in the imported data should be verified active using the check_inactive_satellites command