# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 08:32:11 2024

This program processes CSV data containing coordinate and time information for pole points and weather data.
It provides functionality to:
    - Convert time values in the CSV file to formatted date and time.
    - Convert CSV data to PNEZD format suitable for Civil 3D.
    - Allow user interaction to select input and output files and specify a time range.
    - Find the nearest neighbors for each pole point from time points within a specified time range.
    - Combine weather data with the nearest points based on matching time.

Usage:
    1. Run the program.
    2. Select a CSV file containing time points.
    3. Select a CSV file containing pole points.
    4. Specify a time range using the GUI sliders.
    5. The program finds the nearest neighbors for each pole point from the time points within the specified time range.
    6. Select an Excel file containing the weather data.
    7. The program combines the weather data with the nearest points based on matching time and saves the combined data to a CSV file.

Functionality:
    - get_last_sunday: Returns the datetime of the last Sunday at 12 AM relative to the provided date.
    - epoch_seconds_to_time: Converts seconds since epoch to a readable time in various formats.
    - convert_to_est: Converts a datetime object from UTC to EST.
    - format_date_time: Formats datetime objects in various modes for display.
    - coords_to_points: Parses coordinate data from CSV and writes it in PNEZD format.
    - read_coords_from_csv: Reads specified columns from a CSV file and returns them as a numpy array.
    - find_nearest_neighbors: Finds the nearest neighbors for pole points from a set of time points within a specified time range.
    - write_nearest_neighbors_to_csv: Writes nearest neighbors to a CSV file.
    - select_time_range: Creates a PySimpleGUI interface for the user to select a time range using sliders.
    - parse_weather_data_times: Parses the 'Time' column in a pandas DataFrame as datetime.time objects.
    - read_weather_data: Reads the weather data from a CSV or Excel file into a pandas DataFrame.
    - combine_data: Combines weather data and points data based on matching time.
    - get_last_version_number: Retrieves the highest version number for a given base filename.
    - main: Main function to execute the script.

Author:
    Alex Dering
"""

import re
import os
import tkinter as tk
from tkinter import filedialog
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import PySimpleGUI as sg
from pytz import utc, timezone 
import random
import string

def get_last_sunday(date_input=None):
    """
    Returns the datetime of the last Sunday at 12 AM relative to the provided date.
    If no date is provided, defaults to today's date.
    """
    if date_input is None:
        date_input = datetime.today()
    else:
        try:
            date_input = datetime.strptime(date_input, '%m/%d/%Y')
        except ValueError:
            print("Please provide the date in 'MM/DD/YYYY' format. Using this past Sunday.")
            date_input = datetime.today()
    
    last_sunday = date_input - timedelta(days=date_input.weekday() + 1)
    last_sunday_midnight = last_sunday.replace(hour=0, minute=0, second=0, microsecond=0)
    return last_sunday_midnight

def epoch_seconds_to_time(seconds, epoch=get_last_sunday(), time_mode=3):
    """
    Converts seconds since epoch (default: last Sunday, 12AM) into a readable time
    Compatible with same output modes as format_date_time()
    
    :param min_time: The earliest time in the dataset.
    :param max_time: The latest time in the dataset.
    :return: Selected start and end times.
    """
    sunday = epoch
    delta = timedelta(seconds=seconds)
    time_dt = sunday + delta
    return format_date_time(time_dt, mode=time_mode, to_est=True)

def convert_to_est(date_time_utc):
    """
    Converts a datetime object from UTC to EST.

    :param date_time_utc: The datetime object in UTC
    :return: The datetime object in EST
    """
    # Define the time zones
    utc_zone = utc
    est_zone = timezone('US/Eastern')
    
    # Localize the input datetime to UTC
    date_time_utc = utc_zone.localize(date_time_utc)
    
    # Convert to EST
    date_time_est = date_time_utc.astimezone(est_zone)
    return date_time_est

def format_date_time(date_time, mode=3, to_est=False, custom_format="%I:%M %p"):
    """
    Formats datetime objects, has several modes
        0     ... Returns in format "MM/DD/YYYY, HH:MM:SS.SS"
        1     ... Returns in Day of the week 24 hour format 
        2     ... Returns "MM/DD/YYYY, HH:MM AM/PM"
        3     ... Returns "HH:MM:SS" 24 hour time (default)
        4     ... [Special mode for CSV output] 12 hour format with semicolons
        other ... Custom
    """
    if isinstance(date_time, str):
        for fmt in ["%H:%M:%S", "%I;%M;%S %p", "%I:%M %p"]:
            try:
                date_time = datetime.strptime(date_time, fmt)
                break
            except ValueError:
                continue
        else:
            raise ValueError("Time data does not match any formats")
                
    if to_est:
        date_time = convert_to_est(date_time)
        
    try:
        if mode == 0:
            return date_time.strftime("%m/%d/%Y, %H:%M:%S.%f")
        elif mode == 1:
            return date_time.strftime("%A, %H:%M:%S")
        elif mode == 2:
            return date_time.strftime("%m/%d/%Y, %I:%M %p")
        elif mode == 3:
            return date_time.strftime("%H:%M:%S")
        elif mode == 4:
            return date_time.strftime("%I;%M;%S %p")
        else:
            return date_time.strftime(custom_format)
    except AttributeError as ae:
        print(f"Attribute error in strftime: {ae}")
        return str(date_time)

def coords_to_points(input_csv, output_csv, elevation=False, offset=152000):
    """
    Parses coordinate data (x, y) from CSV file and writes it in PNEZD format to a new CSV file.
    
    :param input_csv: Path to the input CSV file with coordinate data (required)
    :param output_csv: Path to the output CSV file to write PNEZD data (required)
    :param elevation=False: Use elevation (no/yes), else 0.0 for all.
    
    :return: A pandas DataFrame
    """
    try:
        # Read the input CSV file into a DataFrame
        df = pd.read_csv(input_csv)
        
        # Ensure columns exist
        cols = ["Pole Point #", "Nearest Y", "Nearest X", "Time"]
        if elevation:
            cols.append("Elevation")
        
        for col in cols:
            if col not in df.columns:
                raise ValueError(f"Column index {col} not found in the input CSV.")
        
        # Prepare output DataFrame
        output_df = pd.DataFrame(columns=["Point Number", "Northing", "Easting", "Elevation", "Time"])
        # Convert time string to datetime object
        #date_input = input("Enter date of data collection (MM/DD/YYYY) ... ")
        
        rows_list = []
        point_number = offset
        for idx, row in df.iterrows():
            
            point_num = point_number
            easting = float(row["Nearest X"])
            northing = float(row["Nearest Y"])
            time_seconds = float(row["Time"])
            
            point_number += 1
            
            # Convert time in seconds to datetime object
            time = epoch_seconds_to_time(time_seconds, epoch=get_last_sunday())
            formatted_time = format_date_time(time, mode=4)
            
            elevation_value = float(row["Elevation"]) if elevation else 0.0
            
            # Append row to list as dictionary (or list)
            rows_list.append({"Point Number": point_num,
                              "Northing": northing,
                              "Easting": easting,
                              "Elevation": elevation_value,
                              "Time": formatted_time})
                
        output_df = pd.DataFrame(rows_list)
            
        # Write output DataFrame to CSV
        output_df.to_csv(output_csv, index=False, header=False)
        
        return output_df
    
    except FileNotFoundError:
        print(f"Error: File '{input_csv}' not found.")
        return []

def read_coords_from_csv(file_path, cols=['Easting', 'Northing'], chunksize=50000, encoding='utf-8'):
    """
    Reads specified columns from a CSV file and returns them as a numpy array of coordinates.
    
    :param file_path: Path to the input CSV file.
    :param cols: List of column names to read from the CSV file.
    :param chunksize: Number of rows to read at a time (useful for large files).
    :param encoding: Encoding of the CSV file.
    :return: Numpy array of coordinates.
    """
    points = []
    rows = 0
    
    print(f"> Preparing to read {os.path.split(file_path)[1]}, columns {cols}")
    
    coords_file = pd.read_csv(file_path, usecols=[0])
    total_rows = len(coords_file)
    try:
        for chunk in pd.read_csv(file_path, usecols=cols, chunksize=chunksize, encoding=encoding):
            rows += len(chunk)
            print(f"> ...... Processing [{rows}/{total_rows}]")
            # Strip leading and trailing whitespace
            chunk[cols] = chunk[cols].apply(lambda x: x.str.strip() if x.dtype == 'object' else x)
            # Remove apostrophes at the end of strings
            chunk[cols] = chunk[cols].apply(lambda x: x.str.rstrip("'") if x.dtype == 'object' else x)
            # Convert to numpy array and append to points
            points.append(chunk.to_numpy(dtype=float))
        # print(points)
        return np.concatenate(points)
    except (KeyError, ValueError):
        print("> Couldn't find 'Easting' and 'Northing' columns, using columns 1 and 2 as defaults.")
        
        # Retry with positional indexing (columns 1 and 2)
        points = []
        rows = 0
        for chunk in pd.read_csv(file_path, usecols=[1, 2], chunksize=chunksize, encoding=encoding):
            rows += len(chunk)
            print(f"> ...... Processing [{rows}/{total_rows}] (using default columns)")
            # Strip leading and trailing whitespace
            chunk = chunk.apply(lambda x: x.str.strip() if x.dtype == 'object' else x)
            # Remove apostrophes at the end of strings
            chunk = chunk.apply(lambda x: x.str.rstrip("'") if x.dtype == 'object' else x)
            # Convert to numpy array and append to points
            points.append(chunk.to_numpy(dtype=float))
        return np.concatenate(points)

def find_nearest_neighbors(pole_points, time_points, start_time=None, end_time=None, min_distance_threshold=10, min_time_threshold=1, nn_file='nearest-neighbors.csv'):
    """
    Finds the nearest neighbors for pole points from a set of time points within a specified time range.
    (Finds time at each pole)
    
    :param pole_points: Numpy array of pole points.
    :param time_points: Numpy array of time points.
    :param start_time: Start time for filtering time points.
    :param end_time: End time for filtering time points.
    :param min_distance_threshold: Minimum distance threshold to consider points as too close.
    :return: List of nearest neighbors.
    """
    nearest_neighbors = []
    
    def write_nearest_neighbors_to_csv(nearest_neighbors, output_file_path):
        # Writes nearest neighbors to CSV, uses this file to gather point information
        df = pd.DataFrame(nearest_neighbors, columns=['Pole Point #', 'Pole X', 'Pole Y', 'Nearest X', 'Nearest Y', 'Distance', 'Time'])
        df.to_csv(output_file_path, index=False)
        return df
    
    for pole in pole_points:
        # Extract x and y coordinates from poles
        y_pole, x_pole = pole[0:2]
        
        # Filter time_points based on the specified time range
        if start_time is not None and end_time is not None:
            time_filtered_points = time_points[(time_points[:, 2] >= start_time) & (time_points[:, 2] <= end_time)]
        else:
            time_filtered_points = time_points
        
        # Check if there are any time points in the specified range
        if time_filtered_points.size == 0:
            print("> Time points empty. Exiting...")
            return
        
        # Calculate distances between csv_point and all points
        distances = np.linalg.norm(time_filtered_points[:, :2] - np.array([x_pole, y_pole]), axis=1)
        # Find index of the nearest point
        min_distance_index = np.argmin(distances)
        
        # Get the nearest neighbor and its distance
        nearest_neighbor = time_filtered_points[min_distance_index]
        x_time, y_time, time = nearest_neighbor[:3]
        
        pole_num = pole[0]
        nearest_distance = distances[min_distance_index]
        
        # Check if the nearest neighbor is too close to any already added neighbors
        too_close = any(
            np.linalg.norm(np.array([x_time, y_time]) - np.array([n[3], n[4]])) < min_distance_threshold and
            abs(time - n[6]) < min_time_threshold
            for n in nearest_neighbors
        )
        if not too_close:
            # Append all information to the list
            nearest_neighbors.append([pole_num, x_pole, y_pole, x_time, y_time, nearest_distance, time])
        
        
    print(f"> Start time selected: {epoch_seconds_to_time(start_time)}\n> End time selected: {epoch_seconds_to_time(end_time)}")
        
    write_nearest_neighbors_to_csv(nearest_neighbors, nn_file)
    return nearest_neighbors

def select_time_range(min_time, max_time, sunday=get_last_sunday()):
    """
    Creates a PySimpleGUI interface for the user to select a time range using sliders.
    Also includes text boxes for entering column letters for Time, Temperature, Speed, and Direction columns.

    :param min_time: The earliest time in the dataset.
    :param max_time: The latest time in the dataset.
    :return: Selected start and end times, and the column indices for Time, Temperature, Speed, and Direction.
    """
    def column_letter_to_index(letter):
        """
        Converts a column letter (e.g., 'A', 'B', 'C') to a column index (1-based).
        """
        return string.ascii_uppercase.index(letter.upper())
    
    earliest_time = epoch_seconds_to_time(min_time)
    latest_time = epoch_seconds_to_time(max_time)
    sg.theme('LightBlue')

    layout = [
        [sg.Text(f'Earliest Time: {earliest_time}')],
        [sg.Text(f'Latest Time: {latest_time}')],
        [sg.Text('Start Time:'), sg.Slider(range=(min_time, max_time), orientation='h', size=(40, 15), key='start_time', enable_events=True)],
        [sg.Text('End Time:'), sg.Slider(range=(min_time, max_time), orientation='h', size=(40, 15), key='end_time', enable_events=True)],
        [sg.Text('Start Time Display:', size=(20, 1), key='start_time_display')],
        [sg.Text('End Time Display:', size=(20, 1), key='end_time_display')],
        [sg.Text('Time Column:'), sg.InputText(key='time_column'), sg.Text('default: 1', size=(10, 1))],
        [sg.Text('Temperature Column:'), sg.InputText(key='temperature_column'), sg.Text('default: 2', size=(10, 1))],
        [sg.Text('Speed Column:'), sg.InputText(key='speed_column'), sg.Text('default: 7', size=(10, 1))], 
        [sg.Text('Direction Column:'), sg.InputText(key='direction_column'), sg.Text('default: 8', size=(10, 1))],
        [sg.Button('Submit')]
    ]

    window = sg.Window('Select Time Range', layout)

    while True:
        event, values = window.read()
        if event == sg.WINDOW_CLOSED or event == 'Submit':
            break
        if event == 'start_time':
            window['start_time_display'].update(epoch_seconds_to_time(values['start_time']))
        if event == 'end_time':
            window['end_time_display'].update(epoch_seconds_to_time(values['end_time']))

    window.close()

    # Convert column letters to indices, use defaults if inputs are blank
    time_column_index = column_letter_to_index(values['time_column']) if values['time_column'] else 1
    temperature_column_index = column_letter_to_index(values['temperature_column']) if values['temperature_column'] else 2
    speed_column_index = column_letter_to_index(values['speed_column']) if values['speed_column'] else 7
    direction_column_index = column_letter_to_index(values['direction_column']) if values['direction_column'] else 8
    
    column_list = [time_column_index, temperature_column_index, speed_column_index, direction_column_index]
    
    return values['start_time'], values['end_time'], column_list
    
def parse_weather_data_times(df):
    """
    Parse and convert the 'Time' column in the DataFrame to datetime.time objects.
    
    :param df: pandas DataFrame containing a 'Time' column with strings.
    :return: DataFrame with the 'Time' column parsed as datetime.time objects.
    """
    def convert_to_time(time_str):
        formats = [
            '%I:%M %p', '%I:%M%p', '%I:%M:%S %p', '%I:%M:%S%p',
            '%H:%M', '%H:%M:%S',  # 24-hour formats
            '%I:%M a', '%I:%M p',  # Different AM/PM indicators
            '%I:%Ma', '%I:%Mp'
        ]
        for fmt in formats:
            try:
                return datetime.strptime(time_str, fmt).time()
            except ValueError:
                continue
        raise ValueError(f"> Time data '{time_str}' does not match expected formats.")

    df['time'] = df['time'].apply(convert_to_time)
    return df

def find_header_row(file_path, sheet_name=None, looking_for="Time"):
    print("> Searching for 'Time' column...")
    
    if file_path.endswith('.xlsx') or file_path.endswith('.xls'):
        # Read a small portion of the file to find the header row
        if sheet_name:
            df = pd.read_excel(file_path, sheet_name=sheet_name, nrows=20, header=None)
        else:
            df = pd.read_excel(file_path, nrows=20, header=None)
    elif file_path.endswith('.csv'):
        df = pd.read_csv(file_path, nrows=20, header=None)
    else:
        raise ValueError(f"> Unsupported file format '{os.path.split(file_path)[1]}'")
    
    for idx, row in df.iterrows():
        # Normalize row values to lower case and strip whitespace
        normalized_row = [str(value).strip().lower() for value in row.values]
        
        # Check if the keyword is in the normalized row
        if looking_for.lower() in normalized_row:
            print(f"> Header found in row {idx + 1}: {row.values}")
            return idx
    
    raise ValueError("> No header row containing 'Time' found")
    
def read_weather_data(weather_data_file, data_columns=[1, 2, 7, 8]):
    # Finds which row contains the column headers by searching first 20 rows for the 'Time' label
    header = find_header_row(weather_data_file)
    print(f"Header found at row {header}")

    # Read the weather file with specified column types
    try:
        df = pd.read_excel(weather_data_file, usecols=data_columns, header=header, parse_dates=True)
    except ValueError:
        df = pd.read_csv(weather_data_file, usecols=data_columns, header=header, parse_dates=True)

    # Normalize column names
    df.columns = df.columns.str.strip().str.lower()  # Normalize column names

    print(f"> Reading '{os.path.split(weather_data_file)[1]}'...\n{df}")
    print(f"Columns: {df.columns.tolist()}")  # Print columns for debugging

    # Find names of required columns and add to list
    [time, out, spd, wdir] = df.columns
    required_columns = [time, out, spd, wdir]

    for col in required_columns:
        if col not in df.columns:
            print(f"> Missing column: {col}")
            continue

    # Check if 'time' column exists before parsing
    if time not in df.columns:
        print(f"> 'Time' column not found in DataFrame. Available columns: {df.columns.tolist()}")
        return None

    if not isinstance(df[time][header], datetime):
        df = parse_weather_data_times(df)

    # Convert 'Speed' and 'Out' columns to float and round to 1 decimal place
    try:
        df[spd] = df[spd].astype(float).round(1)
        df[out] = df[out].astype(float).round(1)
    except ValueError as ve:
        print(ve)
        pass

    return df

def combine_data(weather_df, points_df, output_file="pole-weather-data-times.csv"):
    """
    Combines weather data and points data based on matching time.

    :param weather_df: The dataframe containing weather data
    :param points_df: The dataframe containing points data
    :param output_file: The path to save the combined dataframe
    :return: The combined dataframe
    """
    def normalize_column_names(df):
        df.columns = [re.sub(r'\s*\(.*?\)', '', col).strip().lower() for col in df.columns]
        return df
    
    if weather_df is None:
        raise ValueError("weather_df is None, ensure the dataframe is properly loaded.")

    if points_df is None:
        raise ValueError("points_df is None, ensure the dataframe is properly loaded.")

    # Normalize column names to remove extensions and standardize them
    weather_df = normalize_column_names(weather_df)
    points_df = normalize_column_names(points_df)

    if 'time' not in weather_df.columns:
        raise ValueError("Missing 'Time' column in {weather_df}.")

    if 'time' not in points_df.columns:
        raise ValueError("Missing 'Time' column in {points_df}.")

    # Format the 'Time' column in both DataFrames
    points_df['formatted_time'] = points_df['time'].apply(
        lambda x: format_date_time(x, mode=5, to_est=False, custom_format="%I:%M %p")
    )
    weather_df['formatted_time'] = weather_df['time'].apply(
        lambda x: format_date_time(x, mode=5, to_est=False, custom_format="%I:%M %p")
    )

    # Merge DataFrames based on 'Formatted Time'
    combined_df = pd.merge(points_df, weather_df, on='formatted_time', how='inner')

    # Drop duplicate 'Time' column if present
    if 'time_y' in combined_df.columns:
        combined_df = combined_df.drop(columns=['time_y'])

    # Save combined DataFrame to CSV
    combined_df.to_csv(output_file, index=False, header=True)
    return combined_df

def get_last_version_number(base_name):
    # Define the pattern for matching the versioned filenames
    pattern = re.compile(rf'{re.escape(base_name)}-(\d+)\.csv')

    # Initialize the highest version number found
    highest_version = 0

    # Scan the current directory for files
    for file_name in os.listdir('.'):
        match = pattern.match(file_name)
        if match:
            version = int(match.group(1))
            if version > highest_version:
                highest_version = version

    return highest_version

def main():
    """
    Main function to execute the script.
    """
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    
    root.attributes("-topmost", True)
    
    # Select input CSV file (full time points file)
    time_points_file = filedialog.askopenfilename(
        parent=root,
        title="Select time points CSV file",
        filetypes=(("CSV Files", "*.csv"), ("All Files", "*.*")),
    
    )
    
    pole_points_file = filedialog.askopenfilename(
        parent=root,
        title="Select pole points CSV file",
        filetypes=(("CSV Files", "*.csv"), ("All Files", "*.*"))
    )
    
    # Read time points into a Dataframe
    time_df = pd.read_csv(time_points_file)
    
    # Find and convert min/max times
    min_time = time_df["TIME"].min()
    max_time = time_df["TIME"].max()
    
    # Define tweak knobs
    pole_cols = ['Easting', 'Northing'] # 'Point Number'
    time_cols = ['X', 'Y', 'TIME']
    
    # Dynamic file naming
    if os.path.exists('nearest-points.csv'):
        base_name_np = 'nearest-points'
        # Get the highest version number for nearest-neighbors
        last_version = get_last_version_number(base_name_np)
        
        # Determine the new version number
        new_version = last_version + 1
        
        # Construct the new filenames
        np_file = f'{base_name_np}-{new_version}.csv'
    else:
        np_file = 'nearest-points.csv'


    pole_points = read_coords_from_csv(pole_points_file, cols=pole_cols)
    
    
    # Read coordinate data from time points file
    time_points = read_coords_from_csv(time_points_file, cols=time_cols, chunksize=50000)
    
    # Select desired time frame
    start_time, end_time, data_cols = select_time_range(min_time, max_time)
    
    if start_time is None or end_time is None:
        print("> Time range selection was cancelled. \n> Exiting...")
        return
    
    # Find the nearest neighbors for each point in pole_points, searches time_points
    find_nearest_neighbors(pole_points, time_points, start_time, end_time, min_distance_threshold=10, min_time_threshold=1)
  
    nearest_points_df = coords_to_points('nearest-neighbors.csv', np_file, elevation=False)
    
    # Ask for the weather data file
    weather_data_file = filedialog.askopenfilename(
        parent=root,
        title="Select Excel file containing the weather data",
        filetypes=(("CSV Files", "*.csv"), ("Excel Files", "*.xlsx"))
    )
    
    weather_df = read_weather_data(weather_data_file, data_columns=data_cols)
    
    full_df = combine_data(weather_df, nearest_points_df)
    
    print(f"""               
> ____________________________
> ........  COMPLETE  ........
> ____________________________
>             
> ....Previewing dataframe....
>            
{full_df}
> ______________________________________________________________________________________
> Output file legend:
> [ nearest-neighbors.csv ] : Helper file created by this program to organize and collect data
>                             -- not very useful to humans.       
>
> [ nearest-points.csv ] : Output file created by this program to be imported to C3D as a points file (PNEZD)
>                          + If you run into problems importing the file to Civil 3D, check to make sure all rows
>                            in the description column are a string (not a time object).
>                            Excel does this automatically sometimes.
>
> [ pole-weather-data-times.csv ] : Final output file created by this program meant to automate 
>                                   the compilation the of the weather data deliverable.
""")
    
if __name__ == "__main__":
    
    art = [
"""
                                     __     -=====|=====-
                                ____   -  *>=====[_]L)
                                    __          -'-`-
   WX ~ [ Automator ] ~ 2024                         ___
                                             ___..--'  .`.
   Author:                          ___...--'     -  .` `.`.
     @alex.dering          ___...--' _      -  _   .` -   `.`.
                  ___...--'  -       _   -       .`  `. - _ `.`.
           __..--'_______________ -         _  .`  _   `.   - `.`.
        .`    _ /\    -        .`      _     .`__________`. _  -`.`.
      .` -   _ /  \_     -   .`  _         .` |ColliersEng|`.   - `.`.
    .`-    _  /   /\   -   .`        _   .`   |___________|  `. _   `.`.
  .`________ /__ /_ \____.`____________.`     ___       ___  - `._____`|
    |   -  __  -|    | - |  ____  |   | | _  |   |  _  |   |  _ |
    | _   |  |  | -  |   | |.--.| |___| |    |___|     |___|    |
    |     |--|  |    | _ | |'--'| |---| |   _|---|     |---|_   |
    |   - |__| _|  - |   | |.--.| |   | |    |   |_  _ |   |    |
 ---``--._      |    |   |=|'--'|=|___|=|====|___|=====|___|====|
 -- . ''  ``--._| _  |  -|_|.--.|_______|_______________________|
`--._           '--- |_  |:|'--'|:::::::|:::::::::::::::::::::::|
_____`--._ ''      . '---'``--._|:::::::|:::::::::::::::::::::::|
----------`--._          ''      ``--.._|:::::::::::::::::::::::|
`--._ _________`--._'        --     .   ''-----.................'
     `--._----------`--._.  _           -- . :''           -    ''
          `--._ _________`--._ :'              -- . :''      -- . ''
 -- . ''       `--._ ---------`--._   -- . :''
          :'        `--._ _________`--._:'  -- . ''      -- . ''
  -- . ''     -- . ''    `--._----------`--._      -- . ''     -- . ''
                              `--._ _________`--._
 -- . ''           :'              `--._ ---------`--._-- . ''    -- . ''
          -- . ''       -- . ''         `--._ _________`--._   -- . ''
:'                 -- . ''          -- . ''  `--._----------`--._
""",
"""
{} 
||   .,,;;;;;;,,.. 
||.;;;;;;*;;;;;;;*;;, ..,,;;;;;;%%%%%, 
||';*;;;;;;;;*;;;;;;,::*::;;;*;;%%%%%%>>%%%%%,                 .; 
|| ';;;;;*;;;;;;;;*;;,:::::*;;;;;@@@##>>%%%%%%,        ..,,;%%%%' 
||  ;*;;;;;;;;*;;;;;;,::*;:;;;;*;@@@@##ooO0@@##>>%%%%%%%%%%%%%%' 
||  ;;;;;;*;;;;;;;;*;;,:;:::*;;;;%%%%%%ooO0@@##>>%%%%%%%%%%a@@' 
||  ;;*;;;;;;;;;*;;;;;,::*;::;;;*;%%%%%%>>%%%%%%ooO@@@@@@@@@@@ 
||  ;;;;;;*;;;;;;;;*;;,:::::;*;;;;@@@@##>>%%%%%%%ooO@@@@@@@@%% 
||  ;;*;;;;;;;;;*;;;;;;,::*;:;;;*;@@@@@##ooO0@@##>>%%%%%%%%%%% 
||  ;;;;;;;*;;;;;;;*;;;,:::::*;;;;;%%%%%%ooO0@@@##>>%%%%%%%%a@, 
||  ;;;*;;;;;;;;*;;;;;;,::*:;;;;;*;%%%%%%%>>%%%%%%%%ooO@@@@@@@@ 
||  ;;;;;;;*;;;;;;;;*;;;,::::;*;;;;@@@@@##>>%%%%%%%%%ooO@@@@@%%' 
||  ;;*;;;;;;;;*;;;;;;;;,::*:;;;:;*;@@@@@##ooO0@@@@##>>%%%%%%%% 
||  ;;;;;;;*;;;;;;*;;;;*;,::::;*;;;;;%%%%%%ooO00@@@@##>>%%%%%a@ 
||  ;*;;a@@@#######@@@@@a,:::*;;;;;;*;%%%%%%>>%%%%%%%%%ooO@@@@@, 
||  ;;@@@@@@#######@@@@@##ooO00@@@@@@@@@@@##>>%%%%%%%%%%ooO@@@%% 
||  a@@@%%%%%%%%%%%%%%%%%%ooO00@@@@@@@@@@@@##ooO0@@@@##>>%%%%%%% 
||  @@%%%%%%%%%%%%%%%%%%%%%>>%%%%%%%%%%%%%%%%ooO00@@@@##>>%%%a@@ 
||  %%%%a@@##########@@@@##>>%%%%%%%%%%%%%%%%%>>%%%%%%%%%ooO@@@a 
||  %%@@@@@##########@@@@@##ooO0@@@@@@@@@@@@##>>%%%%%%%%%%ooO@%% 
||  a@@@%%%%%%%%%%%%%%%%%%%%ooO0@@@@@@@@@@@@@##ooO0@@@@##>>%%%%%. 
||  @@%%%%%%%%%%%%%%%%%%%%%%%>>%%%%%%%%%%%%%%%%ooO0@@@@@##>>%%%a@ 
||  %%%%a@@############@@@@##>>%%%%%%%%%%%%%%%%%>>%%%%%%%%%%ooO@@a 
||  %%@@@@@############@@@@@##ooO0@@@@@@@@@@@@##>>%%%%%%%%%%%ooO%% 
||  a@@@%%%%%%%%%%%%%%%%%%%%%%ooO0@@@@@@@@@@@@@##ooO0@@@@##%>>%%%% 
||  @@%%%%%%%%%%%%%%%%%%%%%%%%%>>%%%%%%%%%%%%%%%%ooO0@@@@@##>>%%a@ 
|| .%%%'                        `>%%%%%%%%%%%%%%%%>>%%%%%%%%%ooO@@, 
||.%%                                             `>%%%%%%%%%ooO%%% 
||'       WX ~ [ Automator ] ~ 2024                          `%%%%% 
||        @author alex.dering                                 `%%' 
||        
""",
"""     
        :++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++:
      [###########################################################################################
     +############################################################################################]
     =############################################################################################]
     =############################################################################################]
     =############################################################################################]
     =###############]>==>)}##################[<####}<############################################]
     =############+  )####)   <############*    #]    ############################################]
     =##########}  -########[ =##############   ###-  ###   }#####################################]
     =#########}   ##########] ##############   ###*  ###}<}######################################]
     =#########-  +################+   :]####   ###+  ###[*:[###[-   +####]:)#  :##*   +##########]
     =#########   -##############   ###   ###   ###+  ##)   [##: )##]  [#    -]=[} +### +#########]
     =#########>   }############+  }####   ##   ###+  ###<  [#>      ::)##-  ####[    *}##########]
     =##########    }#########}#:  #####   ##   ###+  ###<  [#*  [########=  ######<    -#########]
     =###########-   :######< <#[  *###)  ]##   ###*  ###>  [##   ]###> ##=  ####) ####  #########]
     =#############>        )#####-  :  =##<     -:    ::    *##-     <#]     =###   + +##########]
     =############################################################################################]
     =############################################################################################]
     =############################################################################################]
     =############################################################################################]
     =############################################################################################]
     =############################################################################################]
     ----------------------------------------------------------------------------------------------
     ----------------------------------------------------------------------------------------------
     ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
     ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
     +[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[]
      
      WX ~ [ Automator ] ~ 2024
      @author alex.dering
"""]
      
    # Choose random text art
    rand = random.randint(0, 2)
    random_art = art[rand]
    input(f"""
{random_art}
> [ Welcome ] 
> .... If you already know what you're doing just press ENTER, 
> .... otherwise please consult the README.md file before proceeding.
> .... WARNING: Please close all files that you are passing to the program before running
>                   .============================.
>                   |  Press ENTER to proceed... |
>                   '============================'
> """)
    main()
    input("> Press enter to finish...")
    
    
    
    
    
    
    
    