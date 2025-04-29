
# RAE70 Akademik Fedorov Meteorological Data Processor

[![Language](https://img.shields.io/badge/Language-Python-blue.svg)](https://www.python.org/)
[![Data Format](https://img.shields.io/badge/Output-CSV-lightgrey.svg)](https://en.wikipedia.org/wiki/Comma-separated_values)

## Overview

This Python script processes raw meteorological data files (`.txt`) collected during the 70th Russian Antarctic Expedition (RAE70) aboard the research vessel (R/V) "Akademik Fedorov". It cleans the data, standardizes the format, renames columns to meaningful English names, handles missing values, and saves the processed data into two CSV files.

The script is designed to be compatible with the data format used for meteorological observations on the R/V Akademik Fedorov since 2009.

## Features

*   **Reads Raw Data:** Processes `.txt` files from a specified input directory. Expects filenames to contain the date (formats like `DDMMYY.txt`, `YYYYMMDD.txt`, `YYYY-MM-DD.txt` are handled).
*   **Data Cleaning:**
    *   Handles various whitespace inconsistencies.
    *   Identifies and replaces common missing value placeholders (`//////`, `////`, `....`, etc.) with `NaN`.
    *   Normalizes the number of columns per row (padding with `NaN` or truncating if necessary).
*   **Timestamping:** Creates a proper `datetime` column by combining the date parsed from the filename and the time (`hh:mm`) from the data row.
*   **Column Renaming:** Renames the original numeric columns (1-96) to descriptive English names based on a predefined dictionary (e.g., `inst_air_temp_port`, `latitude`, `longitude`).
*   **Data Type Conversion:** Converts data columns to appropriate numeric types, handling conversion errors by setting values to `NaN`.
*   **Missing Value Refinement:** Includes a step (`fill_missing_from_source`) to re-examine rows with `NaN` values. If the original raw text line contained a valid numeric value where the DataFrame now has `NaN` (due to initial parsing discrepancies), this step attempts to fill the `NaN` with the value from the source line.
*   **Data Aggregation:** Combines data from multiple input files.
*   **Incremental Updates:** Loads previously processed data (if `RAE70_meteo_data_processed.csv` exists) and appends only new data, avoiding duplicates based on `datetime` and `source_file`.
*   **Output Generation:**
    *   Saves all processed meteorological parameters to `RAE70_meteo_data_processed.csv`.
    *   Saves a subset of wind parameters (from the foremast sensor) to `RAE70_winds.csv`.
*   **Error Handling:** Includes basic error handling and logging during file processing.

## Data Format

*   **Input:** Raw text files (`.txt`) with tab-separated values. The first column is expected to be time (`hh:mm`), followed by 96 data columns. Filenames should represent the date.
*   **Output:** Semicolon-separated CSV files (`;`).
    *   `RAE70_meteo_data_processed.csv`: Contains all 96 meteorological parameters with standardized column names, plus `datetime` and `source_file` columns.
    *   `RAE70_winds.csv`: Contains a selection of wind parameters (`inst_true_ws`, `mean_true_ws_2min`, `mean_true_ws_10min`, `inst_true_wdir`, `mean_true_wdir_2min`, `mean_true_wdir_10min`), plus `meteo_datetime` (renamed from `datetime`) and `source_file`.

## Dependencies

*   Python 3.x
*   pandas
*   numpy

You can install the required libraries using pip:
```bash
pip install pandas numpy

