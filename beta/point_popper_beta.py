import csv
from datetime import datetime, timedelta
import tkinter as tk
from tkinter import filedialog
import pytz
import os

# §

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
            print("> Please provide the date in 'MM/DD/YYYY' format. Using this past Sunday.")
            date_input = datetime.today()
    
    last_sunday = date_input - timedelta(days=date_input.weekday() + 1)
    last_sunday_midnight = last_sunday.replace(hour=0, minute=0, second=0, microsecond=0)
    return last_sunday_midnight

def seconds_to_datetime(seconds, epoch=get_last_sunday(), time_mode=4):
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
            raise ValueError("> Time data does not match any formats")
                
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
        print(f"> Attribute error in strftime: {ae}")
        return str(date_time)

def convert_to_est(date_time_utc):
    """
    Converts a datetime object from UTC to EST.

    :param date_time_utc: The datetime object in UTC
    :return: The datetime object in EST
    """
    # Define the time zones
    utc_zone = pytz.utc
    est_zone = pytz.timezone('US/Eastern')
    
    # Localize the input datetime to UTC
    date_time_utc = utc_zone.localize(date_time_utc)
    
    # Convert to EST
    date_time_est = date_time_utc.astimezone(est_zone)
    return date_time_est

def extract_every_x_row(input_csv, output_csv, interval):
    with open(input_csv, 'r') as infile:
        reader = csv.reader(infile)
        header = next(reader)  # Read the header row
        
        # Write the header row to the output file
        with open(output_csv, 'w', newline='') as outfile:
            writer = csv.writer(outfile)
            
            for i, row in enumerate(reader, start=1):
                if i % interval == 0:
                    row.append(i)
                    writer.writerow(row)

def coords_to_points(input_csv, output_csv, p_col=0, n_col=4, e_col=3, z_col=None, d_col=2):
    points = []
    with open(input_csv, 'r') as infile, open(output_csv, 'w', newline='') as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)
        # header = next(reader)  # Read the header row
        
        # Write the Civil 3D header to the output file
        # writer.writerow(["Point Number", "Northing", "Easting", "Elevation", "Time"])
        
        for i, row in enumerate(reader, start=1):
            try:
                # Extract Lon (X) and Lat (Y) coordinates from columns A and B
                # point_num = str(row[p_col])
                easting = float(row[e_col])  # Longitude, Easting
                northing = float(row[n_col]) # Latitude, Northing
                desc = float(row[d_col])     # Desc
                elevation = 0.0
                if z_col != None:
                    elevation = float(row[z_col]) # Elevation

                # Write the point data to the output file
                writer.writerow([i, northing, easting, elevation, seconds_to_datetime(desc)])
                
                # Append the point to the points list (if needed)
                points.append((i, northing, easting, elevation, seconds_to_datetime(desc)))
            except (IndexError, ValueError) as e:
                print(f"> Skipping row due to error: {row}. Error: {e}")
                continue
        return points

def main(test=False):
    
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    
    # Select input CSV file
    input_csv = filedialog.askopenfilename(
        title="Select Input CSV File",
        filetypes=(("CSV Files", "*.csv"), ("All Files", "*.*"))
    )
    
    if not input_csv:
        print("> No input file selected. Exiting.")
        return
    
    # Row selection interval
    if not test:
        interval_str = input("> Select row interval? (Press enter for default: 10,000) ...... ")
        try:
            interval = int(interval_str) if interval_str else 10000
        except ValueError:
            interval = 10000
    else:
        interval = 10000
    
    # Name/locate output CSV file
    output_csv = f"{str.rstrip(input_csv, '.csv')}_{interval}.csv"
    
    # Extract every [interval] row (Default: 10,000)
    extract_every_x_row(input_csv, output_csv, interval)
    
    elev_col = input("> Elevation column #? (count from column A starting at 0. Default: 9) ... ")
    time_col = input("> Time column #? (Default: 2) ... ")
    if not elev_col : elev_col = 9
    else : elev_col = int(elev_col)
    
    if not time_col : time_col = 2
    else : time_col = int(time_col)
     
    # For ORU Demo
    # coords_to_points(output_csv, f'{os.path.split(input_csv)[0]}/cogo_points_{interval}.csv', p_col=10, n_col=1, e_col=0, z_col=elev_col, d_col=time_col)
    
    # For NS Demo
    coords_to_points(output_csv, f'{os.path.split(input_csv)[0]}/cogo_points_{interval}.csv', p_col=11, n_col=1, e_col=0, z_col=2, d_col=10)
    
    print(f"Point data saved to cogo_points_{interval}.csv")
    
if __name__ == "__main__":
    main()