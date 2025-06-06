import pandas as pd 
import numpy as np
from datetime import datetime
import os



def rename_column(data, old_name, new_name):
    """
        Renames a column in a DataFrame.
        Args:
            data (pd.DataFrame): The DataFrame to modify.
            old_name (str): The current name of the column.
            new_name (str): The new name for the column.
        
        Returns:
            data (pd.DataFrame): The DataFrame with the column renamed.
    """

    try:
        data = data.rename(columns={old_name: new_name})
        print(f"Column '{old_name}' successfully renamed to '{new_name}'.")
        return data
    except KeyError:
        print(f"Error: Column '{old_name}' not found in the DataFrame.")
    

def remove_duplicates(data):
    """
    Removes duplicate rows from a DataFrame.
    
    Args:
        data (pd.DataFrame): The DataFrame to remove duplicates from.
    
    Returns:
        data (pd.DataFrame): The DataFrame with duplicates removed.
    """
    try:
        print(f"Number of Duplicated values: {data.duplicated().sum()}")
        data = data.drop_duplicates()
        print("Duplicates removed successfully.")
        return data
    except Exception as e:
        print(f"Error removing duplicates: {e}")


def handling_missing_data(data, max_missing_percentage=5.0):
    """
    Handles missing data in a DataFrame by checking columns with missing values above a threshold.
    Columns with missing values exceeding the threshold are dropped, while others are imputed.
    
    Args:
        data (pd.DataFrame): The input DataFrame containing potential missing data.
        max_missing_percentage (float, optional): Maximum allowed percentage of missing values
            in a column before it is dropped. Defaults to 5.0.
    
    Returns:
        pd.DataFrame: The DataFrame with missing data handled (dropped or imputed).
    """
    try:
        # Validate input
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame.")
        if data.empty:
            raise ValueError("Input DataFrame is empty.")
        if not 0 <= max_missing_percentage <= 100:
            raise ValueError("max_missing_percentage must be between 0 and 100.")

        # Calculate the percentage of missing values in each column
        missing_percentage = data.isnull().mean() * 100
        
        # Detailed report of missing values
        print("\nMissing Values Report:")
        print("-" * 50)
        print(missing_percentage[missing_percentage > 0].sort_values(ascending=False)
              .to_string(header=True, index=True, name="Missing Percentage (%)"))
        
        # Identify columns with missing values above the threshold
        columns_to_drop = missing_percentage[missing_percentage > max_missing_percentage].index
        
        # Drop columns exceeding the threshold
        if len(columns_to_drop) > 0:
            print(f"\nDropping {len(columns_to_drop)} columns with >{max_missing_percentage}% missing values: {list(columns_to_drop)}")
            data = data.drop(columns=columns_to_drop)
        else:
            print(f"\nNo columns exceed {max_missing_percentage}% missing values.")
        
        # Handle remaining missing values (imputation)
        for column in data.columns:
            if data[column].isnull().sum() > 0:
                if np.issubdtype(data[column].dtype, np.number):
                    # Impute numerical columns with median
                    median_value = data[column].median()
                    data[column] = data[column].fillna(median_value)
                    print(f"Imputed missing values in '{column}' (numerical) with median: {median_value}")
                else:
                    # Impute categorical columns with mode
                    mode_value = data[column].mode()[0]
                    data[column] = data[column].fillna(mode_value)
                    print(f"Imputed missing values in '{column}' (categorical) with mode: {mode_value}")
        
        # Final check for any remaining missing values
        if data.isnull().sum().sum() == 0:
            print("\nAll missing values handled successfully.")
        else:
            print("\nWarning: Some missing values still remain.")
        
        return data

    except TypeError as te:
        print(f"Type error: {te}")
        return None
    except ValueError as ve:
        print(f"Value error: {ve}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

def merge_datasets(*args):
    """
    Merges multiple DataFrames into a single DataFrame.
    
    Args:
        *args: Variable number of DataFrames to merge.
    
    Returns:
        pd.DataFrame: The merged DataFrame.
    """
    try:
        merged_data = pd.concat(args, ignore_index=False)
        print("Datasets merged successfully.")
        return merged_data
    except Exception as e:
        print(f"Error merging datasets: {e}")
        return None
        
def load_data(file_path):
    """
    Loads data from a CSV file into a DataFrame. 
    
    Args: 
        file_path (str): The path to the CSV file.
    
    Returns:
        pd.DataFrame: The loaded DataFrame.
    """
    try:
        data = pd.read_csv(file_path)
        print("Data loaded successfully.")
        return data
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
    except Exception as e:
        print(f"Error loading data: {e}")
        return None
    
def normalize_dates(data, date_column, output_format="%Y-%m-%d"):
    """
    Normalizes dates in a specified DataFrame column to a consistent format (default: YYYY-MM-DD).
    
    Args:
        data (pd.DataFrame): The input DataFrame containing the date column.
        date_column (str): The name of the column containing dates.
        output_format (str, optional): The desired output date format. Defaults to "%Y-%m-%d".
    
    Returns:
        pd.DataFrame: The DataFrame with normalized dates in the specified column.
    """
    try:
        # Validate input
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame.")
        if date_column not in data.columns:
            raise ValueError(f"Column '{date_column}' not found in DataFrame.")
        if data.empty:
            raise ValueError("Input DataFrame is empty.")
        
        # Create a copy to avoid modifying the original DataFrame
        data = data.copy()
        
        # Attempt to convert dates to datetime, handling multiple formats
        data[date_column] = pd.to_datetime(
            data[date_column],
            errors='coerce',  # Convert invalid dates to NaT
            infer_datetime_format=True
        )
        
        # Format valid dates to the desired output format
        data[date_column] = data[date_column].dt.strftime(output_format)
        
        # Report any rows with invalid dates (NaT)
        invalid_dates = data[data[date_column].isna()]
        if not invalid_dates.empty:
            print(f"Warning: {len(invalid_dates)} rows with invalid dates converted to NaN:")
            print(invalid_dates[[date_column]])
        
        print(f"\nDates in '{date_column}' normalized to {output_format} format.")
        return data

    except TypeError as te:
        print(f"Type error: {te}")
        return None
    except ValueError as ve:
        print(f"Value error: {ve}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

def save_to_csv(data, file_path, index=False, encoding='utf-8'):
    """
    Saves a DataFrame to a CSV file with specified options.
    
    Args:
        data (pd.DataFrame): The input DataFrame to save.
        file_path (str): The path where the CSV file will be saved (e.g., 'output.csv').
        index (bool, optional): Whether to include the DataFrame index in the CSV. Defaults to False.
        encoding (str, optional): Encoding for the CSV file. Defaults to 'utf-8'.
    
    Returns:
        bool: True if the save operation was successful, False otherwise.
    """
    try:
        # Validate input
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame.")
        if data.empty:
            raise ValueError("Input DataFrame is empty.")
        if not isinstance(file_path, str) or not file_path.endswith('.csv'):
            raise ValueError("file_path must be a string ending with '.csv'.")
        
        # Ensure the directory exists
        directory = os.path.dirname(file_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")
        
        # Save DataFrame to CSV
        data.to_csv(file_path, index=index, encoding=encoding)
        print(f"DataFrame successfully saved to '{file_path}' at {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        return True

    except TypeError as te:
        print(f"Type error: {te}")
        return False
    except ValueError as ve:
        print(f"Value error: {ve}")
        return False
    except PermissionError:
        print(f"Permission error: Unable to write to '{file_path}'. Check permissions.")
        return False
    except Exception as e:
        print(f"An unexpected error occurred while saving CSV: {e}")
        return False
    
    