import pandas as pd 
import numpy as np


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
        data = data.drop_duplicates()
        print("Duplicates removed successfully.")
        return data
    except Exception as e:
        print(f"Error removing duplicates: {e}")

def handling_missing_data(data, max_missing_percentage=0.05):
    """
    Handles missing data in a DataFrame.
    
    Args:
        data (pd.DataFrame): The dataframe containing missing data.
        max_missing_percentage (float, optional): _description_. Defaults to 0.05.
    
    Returns:
        data (pd.DataFrame): The dataframe with missing data handled.
    """
    
    try:
        # Calculate the percentage of missing values in each column
        missing_percentage = data.isnull().mean() * 100
        
        # Identify columns with missing values above the threshold
        columns_to_drop = missing_percentage[missing_percentage > max_missing_percentage].index
        
        # Drop the identified columns
        data = data.drop(columns=columns_to_drop)
        print(f"Missing values handled Successfully.")
        return data        
    
    except Exception as e:
        print(f"Error handling missing data: {e}")
    

        


    