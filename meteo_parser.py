import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import traceback # Added for better error reporting

# --- Constants ---
# Assuming the number of expected data columns (excluding time) is 96
EXPECTED_COLUMNS = 96
# Expected rows for a full day with 1-minute intervals
EXPECTED_ROWS_PER_DAY = 24 * 60 # 1440
# Define known placeholder strings that represent missing data in source files
MISSING_VALUE_PLACEHOLDERS = {'//////', '/////', '....', '...', '..', '-', ''}


# Define column names dictionary (assuming this remains unchanged from the original script)
# Note: Ensure this dictionary is available in the scope where process_meteo_data is called.
# Example: column_names = {0: 'time_hhmm', 1: 'inst_air_temp_port', ... , 95: 'coordinate_status'}
# (Using the full dictionary provided in the prompt)
column_names = {
        0: 'time_hhmm',                         # 1. Время, формат hh:mm
        1: 'inst_air_temp_port',                # 2. Мгновенная температура воздуха левый борт °C
        2: 'mean_air_temp_1h_port',             # 3. Средняя температура воздуха за 1 час левый борт °C
        3: 'max_air_temp_1h_port',              # 4. Максимальная температура воздуха за 1 час левый борт °C
        4: 'min_air_temp_1h_port',              # 5. Минимальная температура воздуха за 1 час левый борт °C
        5: 'inst_air_temp_starboard',           # 6. Мгновенная температура воздуха правый борт °C
        6: 'mean_air_temp_1h_starboard',        # 7. Средняя температура воздуха за 1 час правый борт °C
        7: 'max_air_temp_1h_starboard',         # 8. Максимальная температура воздуха за 1 час правый борт °C
        8: 'min_air_temp_1h_starboard',         # 9. Минимальная температура воздуха за 1 час правый борт °C
        9: 'inst_rel_humidity_port',            # 10. Мгновенная относительная влажность воздуха левый борт %
        10: 'mean_rel_humidity_1h_port',        # 11. Средняя относительная влажность воздуха за 1 час левый борт %
        11: 'max_rel_humidity_1h_port',         # 12. Максимальная относительная влажность воздуха за 1 час левый борт %
        12: 'min_rel_humidity_1h_port',         # 13. Минимальная относительная влажность воздуха за 1 час левый борт %
        13: 'inst_rel_humidity_starboard',      # 14. Мгновенная относительная влажность воздуха правый борт %
        14: 'mean_rel_humidity_1h_starboard',   # 15. Средняя относительная влажность воздуха за 1 час правый борт %
        15: 'max_rel_humidity_1h_starboard',    # 16. Максимальная относительная влажность воздуха за 1 час правый борт %
        16: 'min_rel_humidity_1h_starboard',    # 17. Минимальная относительная влажность воздуха за 1 час правый борт %
        17: 'inst_dewpoint_port',               # 18. Мгновенная точка росы левый борт °C
        18: 'mean_dewpoint_1h_port',            # 19. Средняя точка росы за 1 час левый борт °C
        19: 'max_dewpoint_1h_port',             # 20. Максимальная точка росы за 1 час левый борт °C
        20: 'min_dewpoint_1h_port',             # 21. Минимальная точка росы за 1 час левый борт °C
        21: 'inst_dewpoint_starboard',          # 22. Мгновенная точка росы правый борт °C
        22: 'mean_dewpoint_1h_starboard',       # 23. Средняя точка росы за 1 час правый борт °C
        23: 'max_dewpoint_1h_starboard',        # 24. Максимальная точка росы за 1 час правый борт °C
        24: 'min_dewpoint_1h_starboard',        # 25. Минимальная точка росы за 1 час правый борт °C
        25: 'inst_water_temp',                  # 26. Мгновенная температура воды °C
        26: 'mean_water_temp_1h',               # 27. Средняя температура воды за 1 час °C
        27: 'max_water_temp_1h',                # 28. Максимальная температура воды за 1 час °C
        28: 'min_water_temp_1h',                # 29. Минимальная температура воды за 1 час °C
        29: 'inst_atm_pressure',                # 30. Мгновенное атмосферное давление гПа
        30: 'mean_atm_pressure_1h',             # 31. Среднее атмосферное давление за 1 час гПа
        31: 'max_atm_pressure_1h',              # 32. Максимальное атмосферное давление за 1 час гПа
        32: 'min_atm_pressure_1h',              # 33. Минимальное атмосферное давление за 1 час гПа
        33: 'baric_tendency_3h',                # 34. Барическая тенденция за 3 часа гПа
        34: 'baric_tendency_code_3h',           # 35. Ход барической тенденции за 3 часа, от 0 до 8 (см. КН01С)
        35: 'inst_atm_pressure_sea_level',      # 36. Мгновенное атмосферное давление приведенное к уровню моря гПа
        36: 'mean_atm_pressure_sea_level_1h',   # 37. Среднее атмосферное давление приведенное к уровню моря за 1 час гПа
        37: 'max_atm_pressure_sea_level_1h',    # 38. Максимальное атмосферное давление приведенное к уровню моря за 1 час гПа
        38: 'min_atm_pressure_sea_level_1h',    # 39. Минимальное атмосферное давление приведенное к уровню моря за 1 час гПа
        39: 'inst_total_radiation_port',        # 40. Мгновенная суммарная радиация левый борт W/m²
        40: 'mean_total_radiation_1h_port',     # 41. Средняя суммарная радиация за 1 час левый борт W/m²
        41: 'max_total_radiation_1h_port',      # 42. Максимальная суммарная радиация за 1 час левый борт W/m²
        42: 'min_total_radiation_1h_port',      # 43. Минимальная суммарная радиация за 1 час левый борт W/m²
        43: 'inst_total_radiation_starboard',   # 44. Мгновенная суммарная радиация правый борт W/m²
        44: 'mean_total_radiation_1h_starboard', # 45. Средняя суммарная радиация за 1 час правый борт W/m²
        45: 'max_total_radiation_1h_starboard', # 46. Максимальная суммарная радиация за 1 час правый борт W/m²
        46: 'min_total_radiation_1h_starboard', # 47. Минимальная суммарная радиация за 1 час правый борт W/m²
        47: 'inst_direct_radiation',            # 48. Мгновенная прямая радиация W/m²
        48: 'mean_direct_radiation_1h',         # 49. Средняя прямая радиация за 1 час W/m²
        49: 'max_direct_radiation_1h',          # 50. Максимальная прямая радиация за 1 час W/m²
        50: 'min_direct_radiation_1h',          # 51. Минимальная прямая радиация за 1 час W/m²
        51: 'inst_true_ws_foremast',            # 52. Мгновенная скорость истинного ветра носовая мачта, м/с
        52: 'mean_true_ws_2min_foremast',       # 53. Усредненная скорость истинного ветра за 2 мин носовая мачта, м/с
        53: 'max_true_ws_2min_foremast',        # 54. Максимальная скорость истинного ветра за 2 мин носовая мачта, м/с
        54: 'min_true_ws_2min_foremast',        # 55. Минимальная скорость истинного ветра за 2 мин носовая мачта, м/с
        55: 'mean_true_ws_10min_foremast',      # 56. Усредненная скорость истинного ветра за 10 мин носовая мачта, м/с
        56: 'max_true_ws_10min_foremast',       # 57. Максимальная скорость истинного ветра за 10 мин носовая мачта, м/с
        57: 'min_true_ws_10min_foremast',       # 58. Минимальная скорость истинного ветра за 10 мин носовая мачта, м/с
        58: 'inst_true_ws_aftermast',           # 59. Мгновенная скорость истинного ветра кормовая мачта, м/с
        59: 'mean_true_ws_2min_aftermast',      # 60. Усредненная скорость истинного ветра за 2 мин кормовая мачта, м/с
        60: 'max_true_ws_2min_aftermast',       # 61. Максимальная скорость истинного ветра за 2 мин кормовая мачта, м/с
        61: 'min_true_ws_2min_aftermast',       # 62. Минимальная скорость истинного ветра за 2 мин кормовая мачта, м/с
        62: 'mean_true_ws_10min_aftermast',     # 63. Усредненная скорость истинного ветра за 10 мин кормовая мачта, м/с
        63: 'max_true_ws_10min_aftermast',      # 64. Максимальная скорость истинного ветра за 10 мин кормовая мачта, м/с
        64: 'min_true_ws_10min_aftermast',      # 65. Минимальная скорость истинного ветра за 10 мин кормовая мачта, м/с
        65: 'inst_true_wdir_foremast',          # 66. Мгновенное направление истинного ветра носовая мачта, град
        66: 'mean_true_wdir_2min_foremast',     # 67. Усредненное направление истинного ветра за 2 мин носовая мачта, град
        67: 'max_true_wdir_2min_foremast',      # 68. Максимальное направление истинного ветра за 2 мин носовая мачта, град
        68: 'min_true_wdir_2min_foremast',      # 69. Минимальное направление истинного ветра за 2 мин носовая мачта, град
        69: 'mean_true_wdir_10min_foremast',    # 70. Усредненное направление истинного ветра за 10 мин носовая мачта, град
        70: 'max_true_wdir_10min_foremast',     # 71. Максимальное направление истинного ветра за 10 мин носовая мачта, град
        71: 'min_true_wdir_10min_foremast',     # 72. Минимальное направление истинного ветра за 10 мин носовая мачта, град
        72: 'inst_true_wdir_aftermast',         # 73. Мгновенное направление истинного ветра кормовая мачта, град
        73: 'mean_true_wdir_2min_aftermast',    # 74. Усредненное направление истинного ветра за 2 мин кормовая мачта, град
        74: 'max_true_wdir_2min_aftermast',     # 75. Максимальное направление истинного ветра за 2 мин кормовая мачта, град
        75: 'min_true_wdir_2min_aftermast',     # 76. Минимальное направление истинного ветра за 2 мин кормовая мачта, град
        76: 'mean_true_wdir_10min_aftermast',   # 77. Усредненное направление истинного ветра за 10 мин кормовая мачта, град
        77: 'max_true_wdir_10min_aftermast',    # 78. Максимальное направление истинного ветра за 10 мин кормовая мачта, град
        78: 'min_true_wdir_10min_aftermast',    # 79. Минимальное направление истинного ветра за 10 мин кормовая мачта, град
        79: 'ship_speed',                       # 80. Скорость судна, м/с
        80: 'ship_course',                      # 81. Курс судна, град
        81: 'cloud_base_height_layer1',         # 82. Высота нижней границы 1 слоя, метры
        82: 'cloud_base_height_layer2',         # 83. Высота нижней границы 2 слоя, метры
        83: 'vertical_visibility',              # 84. Вертикальная видимость, метры
        84: 'cloud_base_sensor_status',         # 85. Статус датчика нижней границы облаков
        85: 'wind_unknown',                     # 86. Ветер- ?
        86: 'unknown_param_87',                 # 87. Unknown parameter
        87: 'unknown_param_88',                 # 88. Unknown parameter
        88: 'unknown_param_89',                 # 89. Unknown parameter
        89: 'inst_conductivity',                # 90. Мгновенное значение электропроводности, MS/cm
        90: 'inst_salinity',                    # 91. Мгновенное значение солености, ppt
        91: 'latitude',                         # 92. Широта
        92: 'latitude_direction',               # 93. Направление широты: N – cеверная, S – южная
        93: 'longitude',                        # 94. Долгота
        94: 'longitude_direction',              # 95. Направление долготы: Е – восточная, W – западная
        95: 'coordinate_status'                 # 96. Поле статуса координат: V - Недействительное значение, A - Действительное значение
    }


# --- Helper Function to parse raw lines (similar to clean_and_normalize_line but simpler for refill) ---
def parse_raw_line_simple(line, expected_fields=EXPECTED_COLUMNS + 1):
    """Parses a raw line, splitting by tab and stripping."""
    parts = line.strip().split('\t')
    cleaned_parts = [p.strip() for p in parts]
    # Pad if raw line was shorter than expected (unlikely but possible)
    if len(cleaned_parts) < expected_fields:
        cleaned_parts.extend([''] * (expected_fields - len(cleaned_parts)))
    # Truncate if raw line was longer (also unlikely if original parsing was ok)
    elif len(cleaned_parts) > expected_fields:
        cleaned_parts = cleaned_parts[:expected_fields]
    return cleaned_parts


# --- New Helper Function ---
def fill_missing_from_source(df, raw_data_map, column_map):
    """
    Fills missing values (NaN) in the DataFrame by referencing the original raw data lines.

    Args:
        df (pd.DataFrame): The DataFrame to fill (modified in place).
        raw_data_map (dict): A dictionary mapping (datetime, source_file) tuples
                             to the original raw line string from the text file.
        column_map (dict): Dictionary mapping original column index (int) to
                           the final column name (str) in the DataFrame.
                           Example: {1: 'inst_air_temp_port', ...}
    """
    print("\n--- Starting Missing Value Fill from Source ---")
    fill_count = 0
    rows_checked = 0
    rows_modified = 0
    # Create an inverse map: column name -> original index
    # Exclude index 0 ('time_hhmm') as it's replaced by 'datetime'
    name_to_index_map = {name: idx for idx, name in column_map.items() if idx > 0 and idx <= EXPECTED_COLUMNS}
    # Get list of columns that correspond to original data fields (indices 1 to 96)
    data_columns = [col for col in df.columns if col in name_to_index_map]

    # Iterate only through rows that have at least one NaN in the relevant data columns
    rows_with_nan_indices = df[df[data_columns].isnull().any(axis=1)].index
    print(f"Found {len(rows_with_nan_indices)} rows with potential missing values to check.")

    if len(rows_with_nan_indices) == 0:
        print("No rows with missing values found in data columns. Skipping fill process.")
        return # No work to do

    for idx in rows_with_nan_indices:
        rows_checked += 1
        row = df.loc[idx] # Get the row data
        row_key = (row['datetime'], row['source_file'])
        row_modified_flag = False

        if row_key not in raw_data_map:
            # This might happen if the raw map wasn't populated correctly or keys don't match
            # print(f"   Warning: Raw data not found for row index {idx} ({row['datetime']}, {row['source_file']}). Skipping fill for this row.")
            continue

        raw_line = raw_data_map[row_key]
        # Parse the raw line using the simple parser
        raw_fields_cleaned = parse_raw_line_simple(raw_line, expected_fields=EXPECTED_COLUMNS + 1)

        # --- Optional: Count Comparison (as requested) ---
        # Count non-placeholder values in the *original* raw fields (indices 1 to EXPECTED_COLUMNS)
        raw_non_empty_count = sum(1 for i, field in enumerate(raw_fields_cleaned)
                                  if 0 < i <= EXPECTED_COLUMNS and field not in MISSING_VALUE_PLACEHOLDERS)

        # Count non-NaN values in the *current* DataFrame row for the data columns
        df_non_empty_count = row[data_columns].notna().sum()

        # --- Fill Logic ---
        # Decide whether to fill based on count mismatch or simply presence of NaN
        # Using the count mismatch as requested:
        if raw_non_empty_count != df_non_empty_count:
            # print(f"   Mismatch detected for row index {idx} ({row_key}): Raw non-empty={raw_non_empty_count}, DF non-empty={df_non_empty_count}. Attempting fill.")

            for col_name in data_columns:
                # Check if the current value in the DataFrame is NaN
                if pd.isna(row[col_name]):
                    original_index = name_to_index_map.get(col_name)
                    if original_index is None:
                        # Should not happen if data_columns is derived correctly
                        # print(f"    Internal Error: No original index found for column '{col_name}'.")
                        continue

                    # Check if the original index is valid for the parsed raw fields
                    if original_index < len(raw_fields_cleaned):
                        raw_value_str = raw_fields_cleaned[original_index]

                        # Check if the raw value is NOT a known placeholder
                        if raw_value_str not in MISSING_VALUE_PLACEHOLDERS:
                            try:
                                # Attempt to convert the raw string value to numeric
                                numeric_value = pd.to_numeric(raw_value_str)
                                # Update the DataFrame directly using .loc
                                df.loc[idx, col_name] = numeric_value
                                fill_count += 1
                                row_modified_flag = True
                                # print(f"      Filled row {idx}, col '{col_name}' with value '{numeric_value}' from source '{raw_value_str}'")
                            except (ValueError, TypeError):
                                # If conversion fails, it wasn't a valid number - leave as NaN
                                # print(f"      Could not convert source value '{raw_value_str}' for col '{col_name}' to numeric. Leaving NaN.")
                                pass
                        # else:
                            # print(f"      Source value for col '{col_name}' ('{raw_value_str}') is a placeholder. Leaving NaN.")
                    # else:
                        # print(f"    Warning: Original index {original_index} out of bounds for raw fields (len={len(raw_fields_cleaned)}) for row {idx}.")
            if row_modified_flag:
                rows_modified += 1
        # else:
            # print(f"   Counts match for row index {idx} ({row_key}): Raw={raw_non_empty_count}, DF={df_non_empty_count}. No fill attempted based on count.")


    print(f"--- Missing Value Fill Summary ---")
    print(f"Checked {rows_checked} rows with potential missing values.")
    print(f"Compared raw/DF counts and attempted fill on rows where counts mismatched.")
    print(f"Filled a total of {fill_count} individual missing values.")
    print(f"Modified {rows_modified} rows.")
    print(f"--- Finished Missing Value Fill from Source ---")
    # The DataFrame 'df' is modified in place, no need to return unless desired
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import traceback # Added for better error reporting

# --- Assume column_names and fill_missing_from_source are defined elsewhere ---
# --- Assume constants EXPECTED_COLUMNS, MISSING_VALUE_PLACEHOLDERS are defined ---
# --- Assume helper function clean_and_normalize_line is defined elsewhere ---

# --- Modified process_meteo_data Function (Integrating user's wind block) ---
def process_meteo_data(meteo_dir, output_path, output_file="RAE70_meteo_data_processed.csv", output_file2="RAE70_winds.csv"):
    """
    Process meteorological data files, handle data types, combine with existing data,
    remove duplicates, and create/update combined data files including a specific wind subset.

    Args:
        meteo_dir: Directory containing meteorological data files (.txt)
        output_path: Path for output CSV files
        output_file: Name of the full processed meteo data file (default: "RAE70_meteo_data_processed.csv")
        output_file2: Name of the wind data subset file (default: "RAE70_winds.csv")

    Returns:
        DataFrame with the processed wind data subset, or empty DataFrame if no data processed.
    """
    # Construct full paths for output files
    csv_file = os.path.join(output_path, output_file)
    winds_file = os.path.join(output_path, output_file2)

    # Initialize dataframes for existing data
    existing_df = pd.DataFrame()
    existing_timestamps = set() # To store existing (datetime, source_file) tuples

    # --- Check and load existing combined meteo data file ---
    if os.path.exists(csv_file):
        try:
            print(f"Loading existing meteo data from {csv_file}...")
            existing_df = pd.read_csv(csv_file, sep=";", parse_dates=['datetime'], dayfirst=False, low_memory=False)
            if pd.api.types.is_datetime64_any_dtype(existing_df['datetime']):
                 existing_df['datetime'] = existing_df['datetime'].dt.tz_localize(None) # Ensure timezone naive for comparison
            print(f"Loaded {len(existing_df)} rows from existing meteo data file.")
            if 'datetime' in existing_df.columns and 'source_file' in existing_df.columns:
                # Ensure datetime is parsed correctly before creating the set
                existing_df['datetime'] = pd.to_datetime(existing_df['datetime'])
                existing_timestamps = set(existing_df[['datetime', 'source_file']].itertuples(index=False, name=None))
                print(f"Found {len(existing_timestamps)} unique datetime/source file combinations.")
            else:
                 print("Warning: 'datetime' or 'source_file' column missing in existing file. Cannot reliably check for duplicates.")
                 existing_timestamps = set()
        except Exception as e:
            print(f"Error loading or parsing existing meteo data file {csv_file}: {str(e)}")
            print("Will proceed as if creating a new file.")
            existing_df = pd.DataFrame()
            existing_timestamps = set()
    else:
        print(f"No existing meteo data file found at {csv_file}. Will create a new one.")

    # List to hold dataframes from newly processed files
    new_data_list = []
    # Dictionary to store raw lines (used by fill_missing_from_source, but kept here for context)
    raw_lines_map = {}

    # Define reasonable date boundaries
    max_valid_date = datetime(2050, 1, 1)
    min_valid_date = datetime(1950, 1, 1)

    # --- Process meteo files ---
    print(f"\nProcessing text files in directory: {meteo_dir}")
    processed_files_count = 0
    skipped_files_count = 0
    files_in_dir = [f for f in os.listdir(meteo_dir) if os.path.isfile(os.path.join(meteo_dir, f))]

    if not files_in_dir:
        print(f"Warning: No files found in the specified meteo directory: {meteo_dir}")
        # Return empty wind_df if no input files
        return pd.DataFrame()

    for filename in sorted(files_in_dir): # Sort for consistent processing order
        if filename.endswith('.txt') and not filename.startswith('temp_'):
            file_path = os.path.join(meteo_dir, filename)
            print(f"--- Processing {filename} ---")

            try:
                # --- 1. Parse Date from Filename ---
                try:
                    date_str = os.path.splitext(filename)[0]
                    try:
                        base_date = datetime.strptime(date_str, '%d%m%y') # DDMMYY
                    except ValueError:
                        try:
                           base_date = datetime.strptime(date_str, '%Y%m%d') # YYYYMMDD
                        except ValueError:
                            try:
                                base_date = datetime.strptime(date_str, '%Y-%m-%d') # YYYY-MM-DD
                            except ValueError:
                                 print(f"   Warning: Could not parse date from filename '{filename}'. Skipping file.")
                                 skipped_files_count += 1
                                 continue # Skip this file

                    if not (min_valid_date <= base_date <= max_valid_date):
                        print(f"   Warning: Date from filename {filename} ({base_date.date()}) is outside the valid range ({min_valid_date.date()} - {max_valid_date.date()}). Skipping file.")
                        skipped_files_count += 1
                        continue
                except Exception as e:
                    print(f"   Error extracting date from filename {filename}: {e}. Skipping file.")
                    skipped_files_count += 1
                    continue

                # --- 2. Read and Clean Lines ---
                file_data = []
                file_timestamps = []
                line_count = 0
                valid_line_count = 0
                invalid_time_count = 0

                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    for line_num, line in enumerate(f):
                        line_count += 1
                        original_line = line # Keep the original line
                        if not line.strip(): continue

                        # Use the *original* clean_and_normalize_line function here
                        # This function handles initial parsing, padding, placeholder replacement to NaN
                        cleaned_fields = clean_and_normalize_line(line, expected_fields=EXPECTED_COLUMNS + 1) # +1 for time

                        if cleaned_fields:
                            try:
                                time_str = cleaned_fields[0]
                                hours, minutes = map(int, time_str.split(':'))
                                current_datetime = base_date + timedelta(hours=hours, minutes=minutes)
                                current_datetime = current_datetime.replace(tzinfo=None) # Ensure timezone naive

                                if not (min_valid_date <= current_datetime <= max_valid_date):
                                    invalid_time_count += 1
                                    continue

                                row_key = (current_datetime, filename)

                                if row_key in existing_timestamps:
                                    continue # Skip duplicate based on existing processed data

                                # Store raw line (even though fill function is excluded from this snippet)
                                raw_lines_map[row_key] = original_line

                                # Store data (excluding original time string) and timestamp
                                file_data.append(cleaned_fields[1:]) # Store data fields only
                                file_timestamps.append(current_datetime)
                                valid_line_count += 1

                            except (ValueError, IndexError) as e:
                                invalid_time_count += 1
                                # print(f"   Skipping line {line_num+1} due to time parse error: {e}. Line: '{line[:50]}...'")
                                continue
                        # else: Line was skipped by clean_and_normalize_line

                print(f"   Read {line_count} lines. Found {valid_line_count} valid data lines.")
                if invalid_time_count > 0:
                     print(f"   Skipped {invalid_time_count} lines due to invalid/out-of-range time or duplicates already processed.")

                # --- 3. Create DataFrame for the file ---
                if file_data:
                    # Use numeric indices matching EXPECTED_COLUMNS count
                    df_file = pd.DataFrame(file_data, columns=range(EXPECTED_COLUMNS))
                    df_file['datetime'] = file_timestamps
                    df_file['source_file'] = filename
                    new_data_list.append(df_file)
                    processed_files_count += 1
                    print(f"   Added {len(df_file)} new rows from {filename}.")
                elif valid_line_count == 0 and line_count > 0:
                     print(f"   No valid data rows found in {filename} after cleaning/filtering.")
                     skipped_files_count += 1
                else:
                     print(f"   No data added from {filename} (possibly all duplicates or empty).")
                     if line_count > 0 and valid_line_count == 0: # Count as skipped if lines existed but none were valid/new
                         skipped_files_count += 1


            except Exception as e:
                print(f"   !!! Critical Error processing {filename}: {str(e)} !!!")
                traceback.print_exc()
                skipped_files_count += 1

    print(f"\n--- File Processing Summary ---")
    print(f"Successfully processed and potentially added data from: {processed_files_count} files.")
    print(f"Skipped or found no new data in: {skipped_files_count} files.")
    # print(f"Stored raw lines for {len(raw_lines_map)} unique timestamps for potential refill.") # Commented out as refill func excluded

    # --- Combine new data ---
    new_data_df = pd.DataFrame()
    if new_data_list:
        new_data_df = pd.concat(new_data_list, ignore_index=True)
        print(f"\nCombined {len(new_data_df)} new rows from processed files.")

        # --- 4. Data Type Conversion for New Data ---
        print("Converting data types for new rows...")
        # Convert columns 0 to EXPECTED_COLUMNS-1 to numeric, coercing errors
        data_cols_indices = list(range(EXPECTED_COLUMNS))
        for col_idx in data_cols_indices:
            if col_idx in new_data_df.columns: # Check if column exists
                 # Use pd.to_numeric for robust conversion
                 new_data_df[col_idx] = pd.to_numeric(new_data_df[col_idx], errors='coerce')
            # else: Column might be missing if all files had issues? Should not happen normally.

        # --- 5. Column Renaming for New Data ---
        print("Renaming columns for new rows...")
        # Create the renaming map: DataFrame index i -> column_names key i+1
        # Ensure column_names dictionary is accessible here
        # **** IMPORTANT: Make sure 'column_names' dictionary is defined in the scope ****
        # Example: column_names = {1: 'col_name_1', 2: 'col_name_2', ...}
        rename_map = {i: column_names[i+1] for i in range(EXPECTED_COLUMNS) if (i+1) in column_names}
        new_data_df.rename(columns=rename_map, inplace=True)
        # Verify expected columns exist after rename
        expected_new_cols = list(rename_map.values()) + ['datetime', 'source_file']
        missing_cols = [col for col in expected_new_cols if col not in new_data_df.columns]
        if missing_cols:
            print(f"Warning: After renaming, the following expected columns are missing: {missing_cols}")

    else:
        print("\nNo new data rows were added from any file.")
        # If no new data, the existing data (if any) is the final data.
        if existing_df.empty:
            print("No existing data and no new data. Cannot create output files.")
            return pd.DataFrame() # Return empty dataframe

    # --- 6. Combine Existing and New Data ---
    print("Combining new data with existing data...")
    if not existing_df.empty and not new_data_df.empty:
        # Ensure columns match before concatenation, crucial if existing_df schema differs slightly
        # Align columns, fill missing ones with NaN
        combined_df = pd.concat([existing_df, new_data_df], ignore_index=True, sort=False)
        print(f"Combined DataFrame size before deduplication: {len(combined_df)} rows.")
    elif not new_data_df.empty:
        combined_df = new_data_df
        print("Using only newly processed data (no existing file or it was empty).")
    else: # Only existing_df has data
        combined_df = existing_df
        print("Using only existing data (no new files processed or no new rows found).")


    # --- 7. Sort and Deduplicate ---
    if not combined_df.empty:
        print("Sorting and removing duplicates...")
        # Ensure datetime is the correct type before sorting
        combined_df['datetime'] = pd.to_datetime(combined_df['datetime'])
        # Sort primarily by datetime, then by source_file (in case of exact time overlap from different files)
        combined_df.sort_values(by=['datetime', 'source_file'], inplace=True)
        # Remove duplicates based on the unique key, keeping the last occurrence
        initial_rows = len(combined_df)
        combined_df.drop_duplicates(subset=['datetime', 'source_file'], keep='last', inplace=True)
        rows_removed = initial_rows - len(combined_df)
        if rows_removed > 0:
             print(f"Removed {rows_removed} duplicate rows based on datetime and source_file.")
        print(f"Final combined DataFrame size: {len(combined_df)} rows.")
    else:
        print("Combined DataFrame is empty, skipping sort/deduplication.")
        return pd.DataFrame() # Nothing to save or return

    # --- 8. Save Combined Output File --- (Moved saving main file before wind extraction)
    try:
        print(f"\nSaving combined meteo data to {csv_file}...")
        # Ensure output directory exists
        os.makedirs(output_path, exist_ok=True)
        combined_df.to_csv(csv_file, sep=";", index=False, date_format='%Y-%m-%d %H:%M:%S') # Standard date format
        print("Combined data saved successfully.")
    except Exception as e:
        print(f"!!! Error saving combined meteo data file {csv_file}: {str(e)} !!!")
        traceback.print_exc()


    # --- 9. Create and save wind data subset (User Provided Block) ---
    wind_df = pd.DataFrame()
    if not combined_df.empty:
        print(f"\nCreating wind data subset...")
        # Mapping from combined_df column names to desired wind_df column names
        wind_column_mapping = {
            'inst_true_ws_foremast': 'inst_true_ws',
            'mean_true_ws_2min_foremast': 'mean_true_ws_2min',
            'mean_true_ws_10min_foremast': 'mean_true_ws_10min',
            'inst_true_wdir_foremast': 'inst_true_wdir',
            'mean_true_wdir_2min_foremast': 'mean_true_wdir_2min',
            'mean_true_wdir_10min_foremast': 'mean_true_wdir_10min'
        }
        # Initialize wind_df
        wind_df = pd.DataFrame()

        # Add datetime column
        if 'datetime' in combined_df.columns:
            wind_df['meteo_datetime'] = combined_df['datetime']
        else:
             print("   Warning: 'datetime' column not found in combined_df for wind subset.")

        # Add wind columns based on mapping
        found_wind_cols = 0
        for orig_col_name, new_col_name in wind_column_mapping.items():
            if orig_col_name in combined_df.columns:
                wind_df[new_col_name] = combined_df[orig_col_name]
                found_wind_cols += 1
            else:
                # Add the column even if missing in source, filled with NaN
                print(f"   Warning: Wind column '{orig_col_name}' not found in the combined data. Adding column '{new_col_name}' with NaNs.")
                wind_df[new_col_name] = np.nan

        # Add source_file column
        if 'source_file' in combined_df.columns:
            wind_df['source_file'] = combined_df['source_file']
        else:
            print("   Warning: 'source_file' column not found in combined_df for wind subset.")
            wind_df['source_file'] = np.nan # Add source file column with NaN if missing

        print(f"Created wind subset with {len(wind_df)} rows and {found_wind_cols} mapped wind columns (out of {len(wind_column_mapping)} possible).")

        # Save the wind subset file
        if not wind_df.empty:
             try:
                print(f"Saving wind data subset ({len(wind_df)} rows) to {winds_file}...")
                os.makedirs(output_path, exist_ok=True) # Ensure output directory exists
                wind_df.to_csv(winds_file, sep=";", index=False, date_format='%Y-%m-%d %H:%M:%S')
                print(f"Successfully saved wind data subset.")
             except Exception as e:
                print(f"   !!! Error saving wind data subset to {winds_file}: {str(e)} !!!")
                traceback.print_exc() # Print stack trace for saving errors
        else:
             print("Wind DataFrame is empty, nothing to save.")
    else:
        print("\nCombined data is empty, cannot create wind data subset.")

    # --- 10. Return Wind DataFrame ---
    print("\n--- process_meteo_data function finished ---")
    return wind_df

def clean_and_normalize_line(line, expected_fields=EXPECTED_COLUMNS + 1):
    """
    Cleans a single line of data, handling extra whitespace and attempting
    to normalize the number of fields by padding/truncating. Returns a list
    of fields or None if the line is fundamentally invalid (e.g., bad time).
    """
    # Split by tabs first
    parts = line.strip().split('\t')
    # Remove empty strings resulting from multiple tabs, and strip whitespace
    cleaned_parts = [p.strip() for p in parts if p.strip()]

    # Basic check: Must have at least time and one data point
    if len(cleaned_parts) < 2:
        if line.strip(): # Avoid warning for completely blank lines
            print(f"Warning: Line '{line[:50]}...' has too few parts ({len(cleaned_parts)}) after cleaning, skipping.")
        return None

    # Check time format in the first part
    try:
        time_str = cleaned_parts[0]
        hours, minutes = map(int, time_str.split(':'))
        if not (0 <= hours <= 23 and 0 <= minutes <= 59):
            raise ValueError("Invalid hour or minute value")
    except (ValueError, IndexError):
        # Allow lines that don't start with time, they might be headers or junk
        # print(f"Warning: Invalid or missing time format '{cleaned_parts[0]}' in line '{line[:50]}...', skipping.")
        return None # Skip lines with invalid time format in the first column

    # --- Field Count Adjustment ---
    current_fields = len(cleaned_parts)
    if current_fields > expected_fields:
        # print(f"Warning: Line starting '{cleaned_parts[0]}' has {current_fields} fields (expected {expected_fields}). Truncating.")
        cleaned_parts = cleaned_parts[:expected_fields]
    elif current_fields < expected_fields:
        # print(f"Warning: Line starting '{cleaned_parts[0]}' has {current_fields} fields (expected {expected_fields}). Padding with NaN.")
        cleaned_parts.extend([np.nan] * (expected_fields - current_fields))

    # Replace common non-numeric placeholders like '/////', '-', etc., with NaN
    # Start from index 1 (skip time)
    for i in range(1, len(cleaned_parts)):
        part = cleaned_parts[i]
        if isinstance(part, str):
            # Check for common placeholders or potential issues
            if part in ['//////', '/////', '....', '...', '..', '-']:
                 # Check if '-' is part of a negative number (look ahead)
                is_negative_number = False
                if part == '-':
                    if i + 1 < len(cleaned_parts):
                        next_part = str(cleaned_parts[i+1]).strip()
                        if next_part and (next_part[0].isdigit() or next_part[0] == '.'):
                            # Likely part of a negative number like "- 1.23" split by mistake
                            # We might need more sophisticated joining logic if this is common
                            # For now, assume standalone '-' is NaN unless clearly followed by number part
                            pass # Let potential number conversion handle it later? Or treat as NaN?
                            # Let's be conservative: treat standalone '-' as NaN for now.
                            cleaned_parts[i] = np.nan
                        else:
                             cleaned_parts[i] = np.nan
                    else: # '-' is the last element
                        cleaned_parts[i] = np.nan
                else: # '/////', etc.
                    cleaned_parts[i] = np.nan
            elif part == 'NaN': # Handle explicit 'NaN' strings
                 cleaned_parts[i] = np.nan

    return cleaned_parts
# --- Placeholder for the required helper function (implement separately) ---


# --- Add this new helper function ---
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import traceback # Added for better error reporting

# --- Constants ---
# Assuming the number of expected data columns (excluding time) is 96
EXPECTED_COLUMNS = 96
# Expected rows for a full day with 1-minute intervals
EXPECTED_ROWS_PER_DAY = 24 * 60 # 1440
# Define known placeholder strings that represent missing data in source files
MISSING_VALUE_PLACEHOLDERS = {'//////', '/////', '....', '...', '..', '-', ''}


# Define column names dictionary (assuming this remains unchanged from the original script)
# Note: Ensure this dictionary is available in the scope where process_meteo_data is called.
# Example: column_names = {0: 'time_hhmm', 1: 'inst_air_temp_port', ... , 95: 'coordinate_status'}
# (Using the full dictionary provided in the prompt)
column_names = {
        0: 'time_hhmm',                         # 1. Время, формат hh:mm
        1: 'inst_air_temp_port',                # 2. Мгновенная температура воздуха левый борт °C
        2: 'mean_air_temp_1h_port',             # 3. Средняя температура воздуха за 1 час левый борт °C
        3: 'max_air_temp_1h_port',              # 4. Максимальная температура воздуха за 1 час левый борт °C
        4: 'min_air_temp_1h_port',              # 5. Минимальная температура воздуха за 1 час левый борт °C
        5: 'inst_air_temp_starboard',           # 6. Мгновенная температура воздуха правый борт °C
        6: 'mean_air_temp_1h_starboard',        # 7. Средняя температура воздуха за 1 час правый борт °C
        7: 'max_air_temp_1h_starboard',         # 8. Максимальная температура воздуха за 1 час правый борт °C
        8: 'min_air_temp_1h_starboard',         # 9. Минимальная температура воздуха за 1 час правый борт °C
        9: 'inst_rel_humidity_port',            # 10. Мгновенная относительная влажность воздуха левый борт %
        10: 'mean_rel_humidity_1h_port',        # 11. Средняя относительная влажность воздуха за 1 час левый борт %
        11: 'max_rel_humidity_1h_port',         # 12. Максимальная относительная влажность воздуха за 1 час левый борт %
        12: 'min_rel_humidity_1h_port',         # 13. Минимальная относительная влажность воздуха за 1 час левый борт %
        13: 'inst_rel_humidity_starboard',      # 14. Мгновенная относительная влажность воздуха правый борт %
        14: 'mean_rel_humidity_1h_starboard',   # 15. Средняя относительная влажность воздуха за 1 час правый борт %
        15: 'max_rel_humidity_1h_starboard',    # 16. Максимальная относительная влажность воздуха за 1 час правый борт %
        16: 'min_rel_humidity_1h_starboard',    # 17. Минимальная относительная влажность воздуха за 1 час правый борт %
        17: 'inst_dewpoint_port',               # 18. Мгновенная точка росы левый борт °C
        18: 'mean_dewpoint_1h_port',            # 19. Средняя точка росы за 1 час левый борт °C
        19: 'max_dewpoint_1h_port',             # 20. Максимальная точка росы за 1 час левый борт °C
        20: 'min_dewpoint_1h_port',             # 21. Минимальная точка росы за 1 час левый борт °C
        21: 'inst_dewpoint_starboard',          # 22. Мгновенная точка росы правый борт °C
        22: 'mean_dewpoint_1h_starboard',       # 23. Средняя точка росы за 1 час правый борт °C
        23: 'max_dewpoint_1h_starboard',        # 24. Максимальная точка росы за 1 час правый борт °C
        24: 'min_dewpoint_1h_starboard',        # 25. Минимальная точка росы за 1 час правый борт °C
        25: 'inst_water_temp',                  # 26. Мгновенная температура воды °C
        26: 'mean_water_temp_1h',               # 27. Средняя температура воды за 1 час °C
        27: 'max_water_temp_1h',                # 28. Максимальная температура воды за 1 час °C
        28: 'min_water_temp_1h',                # 29. Минимальная температура воды за 1 час °C
        29: 'inst_atm_pressure',                # 30. Мгновенное атмосферное давление гПа
        30: 'mean_atm_pressure_1h',             # 31. Среднее атмосферное давление за 1 час гПа
        31: 'max_atm_pressure_1h',              # 32. Максимальное атмосферное давление за 1 час гПа
        32: 'min_atm_pressure_1h',              # 33. Минимальное атмосферное давление за 1 час гПа
        33: 'baric_tendency_3h',                # 34. Барическая тенденция за 3 часа гПа
        34: 'baric_tendency_code_3h',           # 35. Ход барической тенденции за 3 часа, от 0 до 8 (см. КН01С)
        35: 'inst_atm_pressure_sea_level',      # 36. Мгновенное атмосферное давление приведенное к уровню моря гПа
        36: 'mean_atm_pressure_sea_level_1h',   # 37. Среднее атмосферное давление приведенное к уровню моря за 1 час гПа
        37: 'max_atm_pressure_sea_level_1h',    # 38. Максимальное атмосферное давление приведенное к уровню моря за 1 час гПа
        38: 'min_atm_pressure_sea_level_1h',    # 39. Минимальное атмосферное давление приведенное к уровню моря за 1 час гПа
        39: 'inst_total_radiation_port',        # 40. Мгновенная суммарная радиация левый борт W/m²
        40: 'mean_total_radiation_1h_port',     # 41. Средняя суммарная радиация за 1 час левый борт W/m²
        41: 'max_total_radiation_1h_port',      # 42. Максимальная суммарная радиация за 1 час левый борт W/m²
        42: 'min_total_radiation_1h_port',      # 43. Минимальная суммарная радиация за 1 час левый борт W/m²
        43: 'inst_total_radiation_starboard',   # 44. Мгновенная суммарная радиация правый борт W/m²
        44: 'mean_total_radiation_1h_starboard', # 45. Средняя суммарная радиация за 1 час правый борт W/m²
        45: 'max_total_radiation_1h_starboard', # 46. Максимальная суммарная радиация за 1 час правый борт W/m²
        46: 'min_total_radiation_1h_starboard', # 47. Минимальная суммарная радиация за 1 час правый борт W/m²
        47: 'inst_direct_radiation',            # 48. Мгновенная прямая радиация W/m²
        48: 'mean_direct_radiation_1h',         # 49. Средняя прямая радиация за 1 час W/m²
        49: 'max_direct_radiation_1h',          # 50. Максимальная прямая радиация за 1 час W/m²
        50: 'min_direct_radiation_1h',          # 51. Минимальная прямая радиация за 1 час W/m²
        51: 'inst_true_ws_foremast',            # 52. Мгновенная скорость истинного ветра носовая мачта, м/с
        52: 'mean_true_ws_2min_foremast',       # 53. Усредненная скорость истинного ветра за 2 мин носовая мачта, м/с
        53: 'max_true_ws_2min_foremast',        # 54. Максимальная скорость истинного ветра за 2 мин носовая мачта, м/с
        54: 'min_true_ws_2min_foremast',        # 55. Минимальная скорость истинного ветра за 2 мин носовая мачта, м/с
        55: 'mean_true_ws_10min_foremast',      # 56. Усредненная скорость истинного ветра за 10 мин носовая мачта, м/с
        56: 'max_true_ws_10min_foremast',       # 57. Максимальная скорость истинного ветра за 10 мин носовая мачта, м/с
        57: 'min_true_ws_10min_foremast',       # 58. Минимальная скорость истинного ветра за 10 мин носовая мачта, м/с
        58: 'inst_true_ws_aftermast',           # 59. Мгновенная скорость истинного ветра кормовая мачта, м/с
        59: 'mean_true_ws_2min_aftermast',      # 60. Усредненная скорость истинного ветра за 2 мин кормовая мачта, м/с
        60: 'max_true_ws_2min_aftermast',       # 61. Максимальная скорость истинного ветра за 2 мин кормовая мачта, м/с
        61: 'min_true_ws_2min_aftermast',       # 62. Минимальная скорость истинного ветра за 2 мин кормовая мачта, м/с
        62: 'mean_true_ws_10min_aftermast',     # 63. Усредненная скорость истинного ветра за 10 мин кормовая мачта, м/с
        63: 'max_true_ws_10min_aftermast',      # 64. Максимальная скорость истинного ветра за 10 мин кормовая мачта, м/с
        64: 'min_true_ws_10min_aftermast',      # 65. Минимальная скорость истинного ветра за 10 мин кормовая мачта, м/с
        65: 'inst_true_wdir_foremast',          # 66. Мгновенное направление истинного ветра носовая мачта, град
        66: 'mean_true_wdir_2min_foremast',     # 67. Усредненное направление истинного ветра за 2 мин носовая мачта, град
        67: 'max_true_wdir_2min_foremast',      # 68. Максимальное направление истинного ветра за 2 мин носовая мачта, град
        68: 'min_true_wdir_2min_foremast',      # 69. Минимальное направление истинного ветра за 2 мин носовая мачта, град
        69: 'mean_true_wdir_10min_foremast',    # 70. Усредненное направление истинного ветра за 10 мин носовая мачта, град
        70: 'max_true_wdir_10min_foremast',     # 71. Максимальное направление истинного ветра за 10 мин носовая мачта, град
        71: 'min_true_wdir_10min_foremast',     # 72. Минимальное направление истинного ветра за 10 мин носовая мачта, град
        72: 'inst_true_wdir_aftermast',         # 73. Мгновенное направление истинного ветра кормовая мачта, град
        73: 'mean_true_wdir_2min_aftermast',    # 74. Усредненное направление истинного ветра за 2 мин кормовая мачта, град
        74: 'max_true_wdir_2min_aftermast',     # 75. Максимальное направление истинного ветра за 2 мин кормовая мачта, град
        75: 'min_true_wdir_2min_aftermast',     # 76. Минимальное направление истинного ветра за 2 мин кормовая мачта, град
        76: 'mean_true_wdir_10min_aftermast',   # 77. Усредненное направление истинного ветра за 10 мин кормовая мачта, град
        77: 'max_true_wdir_10min_aftermast',    # 78. Максимальное направление истинного ветра за 10 мин кормовая мачта, град
        78: 'min_true_wdir_10min_aftermast',    # 79. Минимальное направление истинного ветра за 10 мин кормовая мачта, град
        79: 'ship_speed',                       # 80. Скорость судна, м/с
        80: 'ship_course',                      # 81. Курс судна, град
        81: 'cloud_base_height_layer1',         # 82. Высота нижней границы 1 слоя, метры
        82: 'cloud_base_height_layer2',         # 83. Высота нижней границы 2 слоя, метры
        83: 'vertical_visibility',              # 84. Вертикальная видимость, метры
        84: 'cloud_base_sensor_status',         # 85. Статус датчика нижней границы облаков
        85: 'wind_unknown',                     # 86. Ветер- ?
        86: 'unknown_param_87',                 # 87. Unknown parameter
        87: 'unknown_param_88',                 # 88. Unknown parameter
        88: 'unknown_param_89',                 # 89. Unknown parameter
        89: 'inst_conductivity',                # 90. Мгновенное значение электропроводности, MS/cm
        90: 'inst_salinity',                    # 91. Мгновенное значение солености, ppt
        91: 'latitude',                         # 92. Широта
        92: 'latitude_direction',               # 93. Направление широты: N – cеверная, S – южная
        93: 'longitude',                        # 94. Долгота
        94: 'longitude_direction',              # 95. Направление долготы: Е – восточная, W – западная
        95: 'coordinate_status'                 # 96. Поле статуса координат: V - Недействительное значение, A - Действительное значение
    }


# --- Helper Function to parse raw lines (similar to clean_and_normalize_line but simpler for refill) ---
def parse_raw_line_simple(line, expected_fields=EXPECTED_COLUMNS + 1):
    """Parses a raw line, splitting by tab and stripping."""
    parts = line.strip().split('\t')
    cleaned_parts = [p.strip() for p in parts]
    # Pad if raw line was shorter than expected (unlikely but possible)
    if len(cleaned_parts) < expected_fields:
        cleaned_parts.extend([''] * (expected_fields - len(cleaned_parts)))
    # Truncate if raw line was longer (also unlikely if original parsing was ok)
    elif len(cleaned_parts) > expected_fields:
        cleaned_parts = cleaned_parts[:expected_fields]
    return cleaned_parts


# --- New Helper Function ---
def fill_missing_from_source(df, raw_data_map, column_map):
    """
    Fills missing values (NaN) in the DataFrame by referencing the original raw data lines.

    Args:
        df (pd.DataFrame): The DataFrame to fill (modified in place).
        raw_data_map (dict): A dictionary mapping (datetime, source_file) tuples
                             to the original raw line string from the text file.
        column_map (dict): Dictionary mapping original column index (int) to
                           the final column name (str) in the DataFrame.
                           Example: {1: 'inst_air_temp_port', ...}
    """
    print("\n--- Starting Missing Value Fill from Source ---")
    fill_count = 0
    rows_checked = 0
    rows_modified = 0
    # Create an inverse map: column name -> original index
    # Exclude index 0 ('time_hhmm') as it's replaced by 'datetime'
    name_to_index_map = {name: idx for idx, name in column_map.items() if idx > 0 and idx <= EXPECTED_COLUMNS}
    # Get list of columns that correspond to original data fields (indices 1 to 96)
    data_columns = [col for col in df.columns if col in name_to_index_map]

    # Iterate only through rows that have at least one NaN in the relevant data columns
    rows_with_nan_indices = df[df[data_columns].isnull().any(axis=1)].index
    print(f"Found {len(rows_with_nan_indices)} rows with potential missing values to check.")

    if len(rows_with_nan_indices) == 0:
        print("No rows with missing values found in data columns. Skipping fill process.")
        return # No work to do

    for idx in rows_with_nan_indices:
        rows_checked += 1
        row = df.loc[idx] # Get the row data
        row_key = (row['datetime'], row['source_file'])
        row_modified_flag = False

        if row_key not in raw_data_map:
            # This might happen if the raw map wasn't populated correctly or keys don't match
            # print(f"   Warning: Raw data not found for row index {idx} ({row['datetime']}, {row['source_file']}). Skipping fill for this row.")
            continue

        raw_line = raw_data_map[row_key]
        # Parse the raw line using the simple parser
        raw_fields_cleaned = parse_raw_line_simple(raw_line, expected_fields=EXPECTED_COLUMNS + 1)

        # --- Optional: Count Comparison (as requested) ---
        # Count non-placeholder values in the *original* raw fields (indices 1 to EXPECTED_COLUMNS)
        raw_non_empty_count = sum(1 for i, field in enumerate(raw_fields_cleaned)
                                  if 0 < i <= EXPECTED_COLUMNS and field not in MISSING_VALUE_PLACEHOLDERS)

        # Count non-NaN values in the *current* DataFrame row for the data columns
        df_non_empty_count = row[data_columns].notna().sum()

        # --- Fill Logic ---
        # Decide whether to fill based on count mismatch or simply presence of NaN
        # Using the count mismatch as requested:
        if raw_non_empty_count != df_non_empty_count:
            # print(f"   Mismatch detected for row index {idx} ({row_key}): Raw non-empty={raw_non_empty_count}, DF non-empty={df_non_empty_count}. Attempting fill.")

            for col_name in data_columns:
                # Check if the current value in the DataFrame is NaN
                if pd.isna(row[col_name]):
                    original_index = name_to_index_map.get(col_name)
                    if original_index is None:
                        # Should not happen if data_columns is derived correctly
                        # print(f"    Internal Error: No original index found for column '{col_name}'.")
                        continue

                    # Check if the original index is valid for the parsed raw fields
                    if original_index < len(raw_fields_cleaned):
                        raw_value_str = raw_fields_cleaned[original_index]

                        # Check if the raw value is NOT a known placeholder
                        if raw_value_str not in MISSING_VALUE_PLACEHOLDERS:
                            try:
                                # Attempt to convert the raw string value to numeric
                                numeric_value = pd.to_numeric(raw_value_str)
                                # Update the DataFrame directly using .loc
                                df.loc[idx, col_name] = numeric_value
                                fill_count += 1
                                row_modified_flag = True
                                # print(f"      Filled row {idx}, col '{col_name}' with value '{numeric_value}' from source '{raw_value_str}'")
                            except (ValueError, TypeError):
                                # If conversion fails, it wasn't a valid number - leave as NaN
                                # print(f"      Could not convert source value '{raw_value_str}' for col '{col_name}' to numeric. Leaving NaN.")
                                pass
                        # else:
                            # print(f"      Source value for col '{col_name}' ('{raw_value_str}') is a placeholder. Leaving NaN.")
                    # else:
                        # print(f"    Warning: Original index {original_index} out of bounds for raw fields (len={len(raw_fields_cleaned)}) for row {idx}.")
            if row_modified_flag:
                rows_modified += 1
        # else:
            # print(f"   Counts match for row index {idx} ({row_key}): Raw={raw_non_empty_count}, DF={df_non_empty_count}. No fill attempted based on count.")


    print(f"--- Missing Value Fill Summary ---")
    print(f"Checked {rows_checked} rows with potential missing values.")
    print(f"Compared raw/DF counts and attempted fill on rows where counts mismatched.")
    print(f"Filled a total of {fill_count} individual missing values.")
    print(f"Modified {rows_modified} rows.")
    print(f"--- Finished Missing Value Fill from Source ---")
    # The DataFrame 'df' is modified in place, no need to return unless desired


def parse_raw_line_simple(line, expected_fields=EXPECTED_COLUMNS + 1):
    """
    Parses a raw line simply by splitting by tab and stripping whitespace.
    Pads with empty strings or truncates if necessary to match expected_fields.
    Does NOT convert placeholders to NaN.

    Args:
        line (str): The raw line from the text file.
        expected_fields (int): The total number of fields expected (including time).

    Returns:
        list: A list of strings representing the fields.
    """
    # Split by tabs first
    parts = line.strip().split('\t')
    # Just strip whitespace from each part
    cleaned_parts = [p.strip() for p in parts]

    # Adjust field count by padding with '' or truncating
    current_fields = len(cleaned_parts)
    if current_fields < expected_fields:
        cleaned_parts.extend([''] * (expected_fields - current_fields))
    elif current_fields > expected_fields:
        cleaned_parts = cleaned_parts[:expected_fields]

    return cleaned_parts

# --- Add this new function ---
# --- Optimized fill_missing_from_source function ---
import pandas as pd
import numpy as np
from time import perf_counter # To measure performance

def fill_missing_from_source(df, raw_data_map, column_map):
    """
    Optimized function to fill missing values (NaN) in the DataFrame by
    referencing the original raw data lines using vectorized operations.

    Args:
        df (pd.DataFrame): The DataFrame to fill (modified in place).
        raw_data_map (dict): A dictionary mapping (datetime, source_file) tuples
                             to the original raw line string from the text file.
        column_map (dict): Dictionary mapping original column index (int, 1-based for data)
                           to the final column name (str) in the DataFrame.
                           Example: {1: 'inst_air_temp_port', ...}
    """
    start_time = perf_counter()
    print("\n--- Starting Optimized Missing Value Fill from Source ---")

    # --- 1. Identify relevant columns and create index map ---
    name_to_index_map = {name: idx for idx, name in column_map.items() if idx > 0 and idx <= EXPECTED_COLUMNS}
    data_columns = [col for col in df.columns if col in name_to_index_map]

    if not data_columns:
        print("Warning: No data columns found for filling. Check column_map and DataFrame columns.")
        return

    # Use MISSING_VALUE_PLACEHOLDERS defined globally
    global MISSING_VALUE_PLACEHOLDERS

    # --- 2. Identify rows with any NaNs in data columns ---
    nan_rows_mask = df[data_columns].isnull().any(axis=1)
    rows_with_nan_indices = df.index[nan_rows_mask]
    total_rows_to_check = len(rows_with_nan_indices)

    print(f"Found {total_rows_to_check} rows with potential missing values in data columns to check.")
    if total_rows_to_check == 0:
        print("No rows with missing values found. Skipping fill process.")
        return

    # --- 3. Create subset DataFrame and map raw lines ---
    # Create a copy to avoid SettingWithCopyWarning when modifying later
    df_subset = df.loc[rows_with_nan_indices].copy()

    # Create the key for mapping (ensure datetime is naive)
    df_subset['datetime_naive'] = pd.to_datetime(df_subset['datetime']).dt.tz_localize(None)
    map_keys = list(zip(df_subset['datetime_naive'], df_subset['source_file']))

    # Map raw lines - use .get to handle potential missing keys gracefully
    df_subset['raw_line'] = [raw_data_map.get(key) for key in map_keys]
    df_subset.drop(columns=['datetime_naive'], inplace=True) # Remove temporary key column

    # Filter out rows where the raw line couldn't be found
    original_subset_len = len(df_subset)
    df_subset.dropna(subset=['raw_line'], inplace=True)
    rows_dropped_no_raw = original_subset_len - len(df_subset)
    if rows_dropped_no_raw > 0:
        print(f"   Skipped {rows_dropped_no_raw} rows because their raw source line was not found in the map.")

    if df_subset.empty:
        print("No rows remaining after filtering for missing raw lines. Skipping fill.")
        return

    print(f"Processing {len(df_subset)} rows with available raw source lines.")

    # --- 4. Iterate through COLUMNS and apply vectorized fill ---
    fill_count_total = 0
    modified_cols_count = 0

    # Split all raw lines once (potentially memory intensive but vectorized)
    # Pad with extra columns in case some lines are unexpectedly long, handle errors
    try:
        # Estimate max columns needed, add buffer. +2 for time and potential extra field.
        max_raw_cols = EXPECTED_COLUMNS + 2
        raw_split = df_subset['raw_line'].str.split('\t', n=max_raw_cols-1, expand=True)
        # Rename columns to match potential indices (0 to max_raw_cols-1)
        raw_split.columns = range(raw_split.shape[1])
    except Exception as e:
        print(f"Error during vectorized split of raw lines: {e}. Aborting fill.")
        return


    for col_name in data_columns:
        start_col_time = perf_counter()
        original_index = name_to_index_map.get(col_name)
        if original_index is None:
            # print(f"    Internal Error: No original index found for column '{col_name}'.")
            continue

        # Check if the required index exists in the split raw data
        if original_index not in raw_split.columns:
            # print(f"    Warning: Original index {original_index} (for {col_name}) not found in split raw data columns. Skipping column.")
            continue

        # --- Vectorized Extraction ---
        # Select the column corresponding to the original index from the split data
        raw_values_series = raw_split[original_index].astype(str) # Ensure string type

        # --- Vectorized Cleaning & Conversion ---
        # Replace placeholders. Use regex=False for literal replacement if needed.
        cleaned_values = raw_values_series.replace(list(MISSING_VALUE_PLACEHOLDERS), np.nan)
        # Convert to numeric
        numeric_source_values = pd.to_numeric(cleaned_values, errors='coerce')

        # --- Vectorized Filling ---
        # Identify where the target column in the subset is NaN AND the source value is NOT NaN
        nan_mask = df_subset[col_name].isnull()
        source_valid_mask = numeric_source_values.notnull()
        fill_mask = nan_mask & source_valid_mask

        # Count how many values will be filled for this column
        fill_count_col = fill_mask.sum()

        if fill_count_col > 0:
            # Apply the fill using np.where or .loc
            # Using .loc is often clearer for direct assignment based on a boolean mask
            df_subset.loc[fill_mask, col_name] = numeric_source_values[fill_mask]
            fill_count_total += fill_count_col
            modified_cols_count += 1
            # print(f"  Filled {fill_count_col} values in column '{col_name}'. Time: {perf_counter() - start_col_time:.2f}s")


    # --- 5. Update Original DataFrame ---
    print(f"\nUpdating original DataFrame with filled values...")
    update_start_time = perf_counter()
    # Update works efficiently based on matching indices
    df.update(df_subset[[col for col in data_columns if col in df_subset.columns]]) # Ensure only existing columns are updated
    update_time = perf_counter() - update_start_time
    print(f"DataFrame update completed in {update_time:.2f}s.")

    # --- Summary ---
    end_time = perf_counter()
    print(f"--- Optimized Missing Value Fill Summary ---")
    print(f"Checked {total_rows_to_check} rows with potential missing values.")
    print(f"Processed {len(df_subset)} rows that had corresponding raw source lines.")
    print(f"Filled a total of {fill_count_total} individual missing values across {modified_cols_count} columns.")
    print(f"Total time for fill operation: {end_time - start_time:.2f} seconds.")
    print(f"--- Finished Optimized Missing Value Fill ---")

# Make sure the rest of your script (imports, constants, other functions, __main__) remains the same.
# Ensure EXPECTED_COLUMNS and MISSING_VALUE_PLACEHOLDERS are defined globally.

# --- Rewrite the entire process_meteo_data function ---
def process_meteo_data(meteo_dir, output_path, output_file="RAE70_meteo_data_processed.csv", output_file2="RAE70_winds.csv"):
    """
    Process meteorological data files, handle data types, combine with existing data,
    remove duplicates, fill missing column values from source, and create/update
    combined data files including a specific wind subset.

    Args:
        meteo_dir: Directory containing meteorological data files (.txt)
        output_path: Path for output CSV files
        output_file: Name of the full processed meteo data file (default: "RAE70_meteo_data_processed.csv")
        output_file2: Name of the wind data subset file (default: "RAE70_winds.csv")

    Returns:
        DataFrame with the processed wind data subset, or empty DataFrame if no data processed.
    """
    # Construct full paths for output files
    csv_file = os.path.join(output_path, output_file)
    winds_file = os.path.join(output_path, output_file2)

    # Initialize dataframes for existing data
    existing_df = pd.DataFrame()
    existing_timestamps = set() # To store existing (datetime, source_file) tuples

    # --- Check and load existing combined meteo data file ---
    if os.path.exists(csv_file):
        try:
            print(f"Loading existing meteo data from {csv_file}...")
            existing_df = pd.read_csv(csv_file, sep=";", parse_dates=['datetime'], dayfirst=False, low_memory=False)
            if 'datetime' in existing_df.columns and pd.api.types.is_datetime64_any_dtype(existing_df['datetime']):
                 existing_df['datetime'] = existing_df['datetime'].dt.tz_localize(None) # Ensure timezone naive for comparison
            print(f"Loaded {len(existing_df)} rows from existing meteo data file.")
            if 'datetime' in existing_df.columns and 'source_file' in existing_df.columns:
                # Ensure datetime is parsed correctly before creating the set
                existing_df['datetime'] = pd.to_datetime(existing_df['datetime']).dt.tz_localize(None)
                existing_timestamps = set(existing_df[['datetime', 'source_file']].itertuples(index=False, name=None))
                print(f"Found {len(existing_timestamps)} unique datetime/source file combinations.")
            else:
                 print("Warning: 'datetime' or 'source_file' column missing in existing file. Cannot reliably check for duplicates.")
                 existing_timestamps = set()
        except Exception as e:
            print(f"Error loading or parsing existing meteo data file {csv_file}: {str(e)}")
            print("Will proceed as if creating a new file.")
            existing_df = pd.DataFrame()
            existing_timestamps = set()
    else:
        print(f"No existing meteo data file found at {csv_file}. Will create a new one.")

    # List to hold dataframes from newly processed files
    new_data_list = []
    # *** NEW: Dictionary to store raw lines for refill ***
    raw_lines_map = {}

    # Define reasonable date boundaries
    max_valid_date = datetime(2050, 1, 1)
    min_valid_date = datetime(1950, 1, 1)

    # --- Process meteo files ---
    print(f"\nProcessing text files in directory: {meteo_dir}")
    processed_files_count = 0
    skipped_files_count = 0
    files_in_dir = [f for f in os.listdir(meteo_dir) if os.path.isfile(os.path.join(meteo_dir, f))]

    if not files_in_dir:
        print(f"Warning: No files found in the specified meteo directory: {meteo_dir}")
        # Return empty wind_df if no input files
        return pd.DataFrame()

    for filename in sorted(files_in_dir): # Sort for consistent processing order
        if filename.endswith('.txt') and not filename.startswith('temp_'):
            file_path = os.path.join(meteo_dir, filename)
            print(f"--- Processing {filename} ---")

            try:
                # --- 1. Parse Date from Filename ---
                try:
                    date_str = os.path.splitext(filename)[0]
                    try:
                        base_date = datetime.strptime(date_str, '%d%m%y') # DDMMYY
                    except ValueError:
                        try:
                           base_date = datetime.strptime(date_str, '%Y%m%d') # YYYYMMDD
                        except ValueError:
                            try:
                                base_date = datetime.strptime(date_str, '%Y-%m-%d') # YYYY-MM-DD
                            except ValueError:
                                 print(f"   Warning: Could not parse date from filename '{filename}'. Skipping file.")
                                 skipped_files_count += 1
                                 continue # Skip this file

                    if not (min_valid_date <= base_date <= max_valid_date):
                        print(f"   Warning: Date from filename {filename} ({base_date.date()}) is outside the valid range ({min_valid_date.date()} - {max_valid_date.date()}). Skipping file.")
                        skipped_files_count += 1
                        continue
                except Exception as e:
                    print(f"   Error extracting date from filename {filename}: {e}. Skipping file.")
                    skipped_files_count += 1
                    continue

                # --- 2. Read and Clean Lines ---
                file_data = []
                file_timestamps = []
                line_count = 0
                valid_line_count = 0
                invalid_time_count = 0

                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    for line_num, line in enumerate(f):
                        line_count += 1
                        original_line = line.strip() # Keep the original line (stripped)
                        if not original_line: continue

                        # Use the *original* clean_and_normalize_line function here
                        # This function handles initial parsing, padding, placeholder replacement to NaN
                        cleaned_fields = clean_and_normalize_line(line, expected_fields=EXPECTED_COLUMNS + 1) # +1 for time

                        if cleaned_fields:
                            try:
                                time_str = cleaned_fields[0]
                                hours, minutes = map(int, time_str.split(':'))
                                current_datetime = base_date + timedelta(hours=hours, minutes=minutes)
                                current_datetime = current_datetime.replace(tzinfo=None) # Ensure timezone naive

                                if not (min_valid_date <= current_datetime <= max_valid_date):
                                    invalid_time_count += 1
                                    continue

                                row_key = (current_datetime, filename)

                                # *** NEW: Store raw line BEFORE skipping duplicates ***
                                # Store the stripped original line for potential refill later
                                raw_lines_map[row_key] = original_line

                                if row_key in existing_timestamps:
                                    # print(f"   Skipping duplicate row: {row_key}") # Optional: Verbose logging
                                    continue # Skip duplicate based on existing processed data

                                # Store data (excluding original time string) and timestamp for the new DataFrame row
                                file_data.append(cleaned_fields[1:]) # Store data fields only
                                file_timestamps.append(current_datetime)
                                valid_line_count += 1

                            except (ValueError, IndexError) as e:
                                invalid_time_count += 1
                                # print(f"   Skipping line {line_num+1} due to time parse error: {e}. Line: '{line[:50]}...'")
                                continue
                        # else: Line was skipped by clean_and_normalize_line (e.g., invalid time)

                print(f"   Read {line_count} lines. Found {valid_line_count} potentially new data lines.")
                if invalid_time_count > 0:
                     print(f"   Skipped {invalid_time_count} lines due to invalid/out-of-range time.")

                # --- 3. Create DataFrame for the file ---
                if file_data:
                    # Use numeric indices matching EXPECTED_COLUMNS count
                    df_file = pd.DataFrame(file_data, columns=range(EXPECTED_COLUMNS))
                    df_file['datetime'] = file_timestamps
                    df_file['source_file'] = filename
                    new_data_list.append(df_file)
                    processed_files_count += 1
                    print(f"   Added {len(df_file)} new rows from {filename}.")
                elif valid_line_count == 0 and line_count > 0:
                     print(f"   No valid & new data rows found in {filename} after cleaning/filtering.")
                     # Only count as skipped if it had lines but none were valid/new
                     if line_count > 0: skipped_files_count += 1
                else:
                     print(f"   No data added from {filename} (possibly all duplicates or empty).")
                     # Only count as skipped if it had lines but none were valid/new
                     if line_count > 0 and valid_line_count == 0:
                         skipped_files_count += 1


            except Exception as e:
                print(f"   !!! Critical Error processing {filename}: {str(e)} !!!")
                traceback.print_exc()
                skipped_files_count += 1

    print(f"\n--- File Processing Summary ---")
    print(f"Successfully processed and potentially added data from: {processed_files_count} files.")
    print(f"Skipped or found no new data in: {skipped_files_count} files.")
    print(f"Stored raw lines for {len(raw_lines_map)} unique timestamps for potential refill.") # Report map size

    # --- Combine new data ---
    new_data_df = pd.DataFrame()
    if new_data_list:
        new_data_df = pd.concat(new_data_list, ignore_index=True)
        print(f"\nCombined {len(new_data_df)} new rows from processed files.")

        # --- 4. Data Type Conversion for New Data ---
        print("Converting data types for new rows...")
        # Convert columns 0 to EXPECTED_COLUMNS-1 to numeric, coercing errors
        data_cols_indices = list(range(EXPECTED_COLUMNS))
        for col_idx in data_cols_indices:
            if col_idx in new_data_df.columns: # Check if column exists
                 # Use pd.to_numeric for robust conversion
                 new_data_df[col_idx] = pd.to_numeric(new_data_df[col_idx], errors='coerce')
            # else: Column might be missing if all files had issues? Should not happen normally.

        # --- 5. Column Renaming for New Data ---
        print("Renaming columns for new rows...")
        # Create the renaming map: DataFrame index i -> column_names key i+1
        # Ensure column_names dictionary is accessible here
        # **** IMPORTANT: Make sure 'column_names' dictionary is defined in the scope ****
        # Check if column_names is defined and is a dictionary
        if 'column_names' not in globals() or not isinstance(column_names, dict):
             raise NameError("The 'column_names' dictionary is not defined or accessible.")

        rename_map = {i: column_names[i+1] for i in range(EXPECTED_COLUMNS) if (i+1) in column_names}
        new_data_df.rename(columns=rename_map, inplace=True)
        # Verify expected columns exist after rename
        expected_new_cols = list(rename_map.values()) + ['datetime', 'source_file']
        missing_cols = [col for col in expected_new_cols if col not in new_data_df.columns]
        if missing_cols:
            print(f"Warning: After renaming, the following expected columns are missing: {missing_cols}")


# --- Continuing the process_meteo_data function ---

    else:
        print("\nNo new data rows were added from any file.")
        # If no new data, the existing data (if any) is the final data.
        if existing_df.empty:
            print("No existing data and no new data. Cannot create output files.")
            return pd.DataFrame() # Return empty dataframe

    # --- 6. Combine Existing and New Data ---
    print("\nCombining new data with existing data (if any)...")
    if not existing_df.empty and not new_data_df.empty:
        # Ensure columns match before concatenation, crucial if existing_df schema differs slightly
        # Align columns, fill missing ones with NaN. Use outer join logic.
        combined_df = pd.concat([existing_df, new_data_df], ignore_index=True, sort=False)
        print(f"Combined DataFrame size before deduplication: {len(combined_df)} rows.")
    elif not new_data_df.empty:
        combined_df = new_data_df
        print("Using only newly processed data (no existing file or it was empty).")
    else: # Only existing_df has data
        combined_df = existing_df
        print("Using only existing data (no new files processed or no new rows found).")


    # --- 7. Sort and Deduplicate ---
    if not combined_df.empty:
        print("\nSorting and removing duplicates...")
        # Ensure datetime is the correct type before sorting/deduplicating
        combined_df['datetime'] = pd.to_datetime(combined_df['datetime']).dt.tz_localize(None)

        # Sort primarily by datetime, then by source_file (helps keep='last' be consistent)
        combined_df.sort_values(by=['datetime', 'source_file'], inplace=True)

        # Remove duplicates based on the unique key, keeping the last occurrence
        # (implicitly keeps the entry from the latest processed file if times overlap)
        initial_rows = len(combined_df)
        combined_df.drop_duplicates(subset=['datetime', 'source_file'], keep='last', inplace=True)
        rows_removed = initial_rows - len(combined_df)
        if rows_removed > 0:
             print(f"Removed {rows_removed} duplicate rows based on datetime and source_file.")
        print(f"Combined DataFrame size after deduplication: {len(combined_df)} rows.")
    else:
        print("Combined DataFrame is empty, skipping sort/deduplication.")
        return pd.DataFrame() # Nothing to save or return

    # --- 8. *** NEW: Fill Missing Column Values from Source *** ---
    if not combined_df.empty:
        # Ensure column_names dictionary is available
        if 'column_names' not in globals() or not isinstance(column_names, dict):
             raise NameError("The 'column_names' dictionary is not defined or accessible for refill step.")
        # Ensure raw_lines_map is available
        if 'raw_lines_map' not in locals():
             raise NameError("The 'raw_lines_map' dictionary is not defined or accessible for refill step.")

        # Call the function to fill NaNs by comparing with raw data
        fill_missing_from_source(combined_df, raw_lines_map, column_names)
        # The combined_df is modified in place by the function call.
    else:
        print("Combined DataFrame is empty, skipping missing value fill step.")


    # --- 9. Save Combined Output File ---
    if not combined_df.empty:
        try:
            print(f"\nSaving final combined meteo data to {csv_file}...")
            # Ensure output directory exists
            os.makedirs(output_path, exist_ok=True)
            # Use standard ISO format for dates, ensure consistent separator
            combined_df.to_csv(csv_file, sep=";", index=False, date_format='%Y-%m-%d %H:%M:%S')
            print("Combined data saved successfully.")
        except Exception as e:
            print(f"!!! Error saving combined meteo data file {csv_file}: {str(e)} !!!")
            traceback.print_exc()
    else:
        print("\nCombined DataFrame is empty after processing, not saving main file.")


    # --- 10. Create and save wind data subset (User Provided Block - Adapted) ---
    wind_df = pd.DataFrame() # Initialize empty wind dataframe
    if not combined_df.empty:
        print(f"\nCreating wind data subset...")
        # Mapping from combined_df column names to desired wind_df column names
        wind_column_mapping = {
            'inst_true_ws_foremast': 'inst_true_ws',
            'mean_true_ws_2min_foremast': 'mean_true_ws_2min',
            'mean_true_ws_10min_foremast': 'mean_true_ws_10min',
            'inst_true_wdir_foremast': 'inst_true_wdir',
            'mean_true_wdir_2min_foremast': 'mean_true_wdir_2min',
            'mean_true_wdir_10min_foremast': 'mean_true_wdir_10min'
        }
        # Initialize wind_df columns based on mapping keys + datetime + source_file
        wind_df_cols = ['meteo_datetime'] + list(wind_column_mapping.values()) + ['source_file']
        wind_df = pd.DataFrame(columns=wind_df_cols) # Create with correct columns

        # Add datetime column
        if 'datetime' in combined_df.columns:
            wind_df['meteo_datetime'] = combined_df['datetime']
        else:
             print("   Warning: 'datetime' column not found in combined_df for wind subset. 'meteo_datetime' will be empty.")
             # Column already exists, will be filled with NaN by default

        # Add wind columns based on mapping
        found_wind_cols_count = 0
        for orig_col_name, new_col_name in wind_column_mapping.items():
            if orig_col_name in combined_df.columns:
                # Assign data from combined_df to the pre-existing column in wind_df
                wind_df[new_col_name] = combined_df[orig_col_name]
                found_wind_cols_count += 1
            else:
                # Column already exists in wind_df, will remain filled with NaN
                print(f"   Warning: Wind column '{orig_col_name}' not found in the combined data. Column '{new_col_name}' will contain NaNs.")

        # Add source_file column
        if 'source_file' in combined_df.columns:
            wind_df['source_file'] = combined_df['source_file']
        else:
            print("   Warning: 'source_file' column not found in combined_df for wind subset. 'source_file' will contain NaNs.")
            # Column already exists, will remain filled with NaN

        print(f"Created wind subset with {len(wind_df)} rows. Found data for {found_wind_cols_count}/{len(wind_column_mapping)} mapped wind columns.")

        # Save the wind subset file
        if not wind_df.empty:
             try:
                print(f"Saving wind data subset ({len(wind_df)} rows) to {winds_file}...")
                os.makedirs(output_path, exist_ok=True) # Ensure output directory exists
                wind_df.to_csv(winds_file, sep=";", index=False, date_format='%Y-%m-%d %H:%M:%S')
                print(f"Successfully saved wind data subset.")
             except Exception as e:
                print(f"   !!! Error saving wind data subset to {winds_file}: {str(e)} !!!")
                traceback.print_exc() # Print stack trace for saving errors
        else:
             # This case should ideally not be reached if combined_df was not empty,
             # but check just in case wind_df creation failed unexpectedly.
             print("Wind DataFrame is empty, nothing to save.")
    else:
        print("\nCombined data is empty, cannot create wind data subset.")

    # --- 11. Return Wind DataFrame ---
    print("\n--- process_meteo_data function finished ---")
    return wind_df

# --- End of process_meteo_data function ---

# Note: Ensure the rest of your script (constants, helper functions like
# clean_and_normalize_line, column_names dictionary, and the __main__ block)
# remains as it was, including the import statements.

# --- Placeholder for the required helper function (implement separately) ---
def clean_and_normalize_line(line, expected_fields=EXPECTED_COLUMNS + 1):
    """
    Cleans a single line of data, handling extra whitespace and attempting
    to normalize the number of fields by padding/truncating. Returns a list
    of fields or None if the line is fundamentally invalid (e.g., bad time).
    """
    # Split by tabs first
    parts = line.strip().split('\t')
    # Remove empty strings resulting from multiple tabs, and strip whitespace
    cleaned_parts = [p.strip() for p in parts if p.strip()]

    # Basic check: Must have at least time and one data point
    if len(cleaned_parts) < 2:
        if line.strip(): # Avoid warning for completely blank lines
            print(f"Warning: Line '{line[:50]}...' has too few parts ({len(cleaned_parts)}) after cleaning, skipping.")
        return None

    # Check time format in the first part
    try:
        time_str = cleaned_parts[0]
        hours, minutes = map(int, time_str.split(':'))
        if not (0 <= hours <= 23 and 0 <= minutes <= 59):
            raise ValueError("Invalid hour or minute value")
    except (ValueError, IndexError):
        # Allow lines that don't start with time, they might be headers or junk
        # print(f"Warning: Invalid or missing time format '{cleaned_parts[0]}' in line '{line[:50]}...', skipping.")
        return None # Skip lines with invalid time format in the first column

    # --- Field Count Adjustment ---
    current_fields = len(cleaned_parts)
    if current_fields > expected_fields:
        # print(f"Warning: Line starting '{cleaned_parts[0]}' has {current_fields} fields (expected {expected_fields}). Truncating.")
        cleaned_parts = cleaned_parts[:expected_fields]
    elif current_fields < expected_fields:
        # print(f"Warning: Line starting '{cleaned_parts[0]}' has {current_fields} fields (expected {expected_fields}). Padding with NaN.")
        cleaned_parts.extend([np.nan] * (expected_fields - current_fields))

    # Replace common non-numeric placeholders like '/////', '-', etc., with NaN
    # Start from index 1 (skip time)
    for i in range(1, len(cleaned_parts)):
        part = cleaned_parts[i]
        if isinstance(part, str):
            # Check for common placeholders or potential issues
            if part in ['//////', '/////', '....', '...', '..', '-']:
                 # Check if '-' is part of a negative number (look ahead)
                is_negative_number = False
                if part == '-':
                    if i + 1 < len(cleaned_parts):
                        next_part = str(cleaned_parts[i+1]).strip()
                        if next_part and (next_part[0].isdigit() or next_part[0] == '.'):
                            # Likely part of a negative number like "- 1.23" split by mistake
                            # We might need more sophisticated joining logic if this is common
                            # For now, assume standalone '-' is NaN unless clearly followed by number part
                            pass # Let potential number conversion handle it later? Or treat as NaN?
                            # Let's be conservative: treat standalone '-' as NaN for now.
                            cleaned_parts[i] = np.nan
                        else:
                             cleaned_parts[i] = np.nan
                    else: # '-' is the last element
                        cleaned_parts[i] = np.nan
                else: # '/////', etc.
                    cleaned_parts[i] = np.nan
            elif part == 'NaN': # Handle explicit 'NaN' strings
                 cleaned_parts[i] = np.nan

    return cleaned_parts
    

# --- Main Execution Block (Should remain as in your previous version) ---
if __name__ == "__main__":

    print("Script started. Using hardcoded absolute paths.")
    print(f"Current date and time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    try:
        # --- Define Absolute Paths (User Provided) ---
        # Make sure these paths are correct for your environment
        meteo_input_dir = "/path_to_meteo_directory/"
        output_dir = "/output_directory/"

        # Define output filenames
        meteo_output_filename = "RAE70_meteo_data_processed.csv"
        winds_output_filename = "RAE70_winds.csv"

        # Construct full paths for output files
        meteo_output_full_path = os.path.join(output_dir, meteo_output_filename)
        winds_output_full_path = os.path.join(output_dir, winds_output_filename)

        print(f"\nUsing Input directory for meteo .txt files: {meteo_input_dir}")
        print(f"Using Output directory for processed files: {output_dir}")
        print(f"Target output file for combined meteo data: {meteo_output_full_path}")
        print(f"Target output file for winds data: {winds_output_full_path}")


        # --- Validate Input Path ---
        if not os.path.isdir(meteo_input_dir):
            print("\n" + "="*30 + " ERROR " + "="*30)
            print(f"Input directory for meteo files not found or is not a directory.")
            print(f"Please ensure the directory exists: '{meteo_input_dir}'")
            print("="*67)
            exit(1) # Exit the script if the input directory is missing
        else:
            print(f"\nFound input directory: {meteo_input_dir}")

            # --- Ensure Output Directory Exists ---
            try:
                # exist_ok=True prevents an error if the directory already exists
                os.makedirs(output_dir, exist_ok=True)
                print(f"Ensured output directory exists: {output_dir}")
            except OSError as e:
                print("\n" + "="*30 + " ERROR " + "="*30)
                print(f"Could not create or access output directory: {output_dir}")
                print(f"Error details: {e}")
                print("Please check permissions or manually create the directory.")
                print("="*67)
                exit(1) # Exit if output directory cannot be created/accessed

            # --- Call the Processing Function ---
            print("\nStarting meteo data processing...")
            # Pass the absolute paths and desired filenames to the function
            processed_wind_result = process_meteo_data(
                meteo_dir=meteo_input_dir,
                output_path=output_dir,
                output_file=meteo_output_filename,
                output_file2=winds_output_filename
            )

            # --- Final Report ---
            print("\n" + "="*30 + " Script Finished " + "="*30)
            # Check if the expected files now exist
            meteo_file_exists = os.path.exists(meteo_output_full_path)
            winds_file_exists = os.path.exists(winds_output_full_path)

            # Check if the processing function returned a non-empty DataFrame
            # (indicating some data was processed or loaded)
            processed_successfully = not processed_wind_result.empty if isinstance(processed_wind_result, pd.DataFrame) else False

            if meteo_file_exists or winds_file_exists:
                 print(f"Processing complete. Output files status:")
                 if meteo_file_exists:
                     print(f"  [FOUND] Combined meteo data: {meteo_output_full_path}")
                 else:
                     print(f"  [MISSING] Combined meteo data: {meteo_output_full_path}")

                 if winds_file_exists:
                     print(f"  [FOUND] Wind data subset:    {winds_output_full_path}")
                 else:
                      print(f"  [MISSING] Wind data subset:    {winds_output_full_path}")

                 if not processed_successfully and (meteo_file_exists or winds_file_exists):
                      print("\nNote: Output files were found, but the script didn't process/add new data in this run (or an issue occurred). Files might be from a previous execution.")

            else: # No processing happened and files don't exist
                 print("Processing finished, but no output data files were generated or found.")
                 print(f"(Checked for input in: {meteo_input_dir})")
                 print(f"(Attempted output to: {output_dir})")
            print("="*77)

    except Exception as e:
        print("\n" + "="*30 + " CRITICAL ERROR " + "="*30)
        print("An unexpected error occurred during script execution:")
        traceback.print_exc() # Print detailed traceback
        print("="*78)
        exit(1) # Exit with error status