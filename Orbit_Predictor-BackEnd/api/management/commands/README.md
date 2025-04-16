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

### Update CDM Dates

Updates the dates for all CDM entries to be relative to the current time. Useful for demonstration purposes to make CDMs appear recent/current instead of showing past or future dates.

**Usage:**

```bash
# Dry run (show what would change without actually changing)
python manage.py update_cdm_dates --dry-run

# Update all CDM dates to be current (shifts from earliest CDM to now)
python manage.py update_cdm_dates

# Update with a specific number of days shift
python manage.py update_cdm_dates --days 3

# Update with reference to a specific start date
python manage.py update_cdm_dates --reference-date 2024-10-05
```

**How it works:**
1. Identifies the reference date (either the earliest CDM date or a specified date)
2. Calculates the time shift needed to bring that date to the current time
3. Applies the same time shift to all CDM dates (creation_date and TCA)

**When to use:**
- After loading sample/demo data that has outdated timestamps
- Before demonstrations to make conjunction events appear to be upcoming
- To test time-based features with current dates instead of past/future dates

### Fix Timezone Issues

Fixes timezone-related issues by shifting all CDM dates by a specific number of hours. Particularly useful for correcting dates that appear to be one day in the future due to timezone conversions.

**Usage:**

```bash
# Dry run (show what would change without actually changing)
python manage.py fix_timezone --dry-run

# Fix dates that appear one day in the future (default: shifts by -24 hours)
python manage.py fix_timezone

# Shift by a custom number of hours
python manage.py fix_timezone --hours -12
```

**How it works:**
1. Shifts all CDM dates (creation_date and TCA) by the specified number of hours
2. By default, shifts dates back by 24 hours to fix dates appearing one day in the future

**When to use:**
- When CDM dates appear to be in the future due to timezone issues
- To adjust date/time displays to match a specific timezone for demonstration purposes