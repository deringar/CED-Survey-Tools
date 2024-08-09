import pandas as pd
import tkinter as tk
from tkinter import filedialog

def parse_weather_data(file_path):
    if file_path.endswith('.xlsx'):
        df = pd.read_excel(file_path)
    elif file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    else:
        raise ValueError("Unsupported file format")

    # Strip whitespace from column names
    df.columns = df.columns.str.strip()
    print("Column names:", df.columns)  # Debugging line

    # Substrings to search for in column names
    substrings = ['Time', 'Temp', 'Out', 'Wind', 'Speed', 'Dir']

    # Find the relevant columns based on substrings
    selected_columns = []
    for substring in substrings:
        for column in df.columns:
            if substring.lower() in column.lower() and column not in selected_columns:
                selected_columns.append(column)

    print("Selected columns before filtering:", selected_columns)  # Debugging line

    # Filter dataframe to only include the selected columns
    df_filtered = df[selected_columns] if selected_columns else None
    
    if df_filtered is None:
        print("No matching columns found.")
        
    return df_filtered

# Example usage
root = tk.Tk()
root.withdraw()  # Hide the root window

root.attributes("-topmost", True)

# Select input CSV file (full time points file)
time_points_file = filedialog.askopenfilename(
    parent=root,
    title="Select time points CSV file",
    filetypes=(("CSV Files", "*.csv"), ("All Files", "*.*")),
)

print(parse_weather_data(time_points_file))
