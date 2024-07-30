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
    
