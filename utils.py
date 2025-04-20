import streamlit as st
import pandas as pd
import numpy as np
import pickle
import io
import json
import nbformat
from nbformat.reader import NotJSONError

def load_pickle_file(uploaded_file):
    """
    Load data from uploaded pickle file
    
    Parameters:
    -----------
    uploaded_file : UploadedFile
        File uploaded through Streamlit's file_uploader
        
    Returns:
    --------
    object
        Object loaded from pickle file
    """
    try:
        # Read file content
        file_content = uploaded_file.read()
        
        # Load pickle data from bytes
        data = pickle.loads(file_content)
        
        return data
    except Exception as e:
        st.error(f"Error loading pickle file: {str(e)}")
        return None

def save_dataframe_to_pickle(df, filename="data.pkl"):
    """
    Save DataFrame to pickle file and provide download link
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame to save
    filename : str
        Name of the file to be downloaded
    """
    try:
        # Create pickle data in memory
        pickle_data = pickle.dumps(df)
        
        # Create download button
        st.download_button(
            label="Download data as pickle",
            data=pickle_data,
            file_name=filename,
            mime="application/octet-stream"
        )
    except Exception as e:
        st.error(f"Error saving DataFrame to pickle: {str(e)}")

def detect_data_type(data):
    """
    Detect the type of data loaded from pickle file
    
    Parameters:
    -----------
    data : object
        Object loaded from pickle file
        
    Returns:
    --------
    str
        Type of data detected
    """
    if isinstance(data, pd.DataFrame):
        return "DataFrame"
    elif isinstance(data, pd.Series):
        return "Series"
    elif isinstance(data, dict):
        return "Dictionary"
    elif isinstance(data, list) or isinstance(data, tuple):
        return "List/Tuple"
    elif isinstance(data, np.ndarray):
        return "NumPy Array"
    elif hasattr(data, 'predict') or hasattr(data, 'fit'):
        return "ML Model"
    else:
        return "Unknown"

def convert_to_dataframe(data):
    """
    Try to convert data to DataFrame if possible
    
    Parameters:
    -----------
    data : object
        Data to convert
        
    Returns:
    --------
    pandas.DataFrame or None
        Converted DataFrame or None if conversion is not possible
    """
    try:
        if isinstance(data, pd.DataFrame):
            return data
        
        elif isinstance(data, pd.Series):
            return pd.DataFrame(data)
        
        elif isinstance(data, dict):
            return pd.DataFrame.from_dict(data)
        
        elif isinstance(data, (list, tuple)):
            return pd.DataFrame(data)
        
        elif isinstance(data, np.ndarray):
            return pd.DataFrame(data)
        
        else:
            return None
    
    except Exception as e:
        st.error(f"Error converting to DataFrame: {str(e)}")
        return None

def check_for_missing_values(df):
    """
    Check for missing values in DataFrame
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame to check
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with missing value information
    """
    # Calculate missing values count and percentage
    missing_count = df.isnull().sum()
    missing_percentage = (missing_count / len(df)) * 100
    
    # Create DataFrame with missing value information
    missing_info = pd.DataFrame({
        'Column': missing_count.index,
        'Missing Count': missing_count.values,
        'Missing Percentage': missing_percentage.values
    })
    
    # Sort by missing count in descending order
    missing_info = missing_info.sort_values('Missing Count', ascending=False)
    
    return missing_info

def get_data_summary(df):
    """
    Get summary of the DataFrame
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame to summarize
        
    Returns:
    --------
    dict
        Dictionary containing summary information
    """
    summary = {
        'shape': df.shape,
        'columns': df.columns.tolist(),
        'dtypes': df.dtypes.to_dict(),
        'missing_values': check_for_missing_values(df),
        'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
        'categorical_columns': df.select_dtypes(include=['object', 'category']).columns.tolist(),
        'date_columns': df.select_dtypes(include=['datetime', 'datetime64']).columns.tolist()
    }
    
    return summary

def load_jupyter_notebook(uploaded_file):
    """
    Load data from uploaded Jupyter notebook file (.ipynb)
    
    Parameters:
    -----------
    uploaded_file : UploadedFile
        Jupyter notebook file uploaded through Streamlit's file_uploader
        
    Returns:
    --------
    tuple
        (dataframes, code_cells) - extracted DataFrames and code cells from the notebook
    """
    try:
        # Read file content as string
        content = uploaded_file.getvalue().decode("utf-8")
        
        # Parse notebook content
        notebook = nbformat.reads(content, as_version=4)
        
        # Extract code cells and DataFrames
        code_cells = []
        dataframes = []
        dataframe_names = []
        
        # Define a namespace for execution
        namespace = {'pd': pd, 'np': np}
        
        # Process cells
        for cell in notebook.cells:
            if cell.cell_type == 'code':
                code = cell.source
                code_cells.append(code)
                
                # Look for DataFrame definitions in the code
                if 'pd.DataFrame' in code or 'pandas.DataFrame' in code or '=' in code:
                    try:
                        # Extract potential DataFrame variable names using heuristics
                        lines = code.split('\n')
                        assign_lines = [line for line in lines if '=' in line and not line.strip().startswith('#')]
                        
                        # Try to extract DataFrames by executing the cell code
                        exec(code, namespace)
                        
                        # Search for DataFrames in the namespace
                        for var_name, var_value in namespace.items():
                            if isinstance(var_value, pd.DataFrame) and var_name not in ['pd', 'np']:
                                dataframes.append(var_value)
                                dataframe_names.append(var_name)
                    except Exception as e:
                        # Skip errors in cell execution
                        pass
        
        return {
            'dataframes': dataframes,
            'dataframe_names': dataframe_names,
            'code_cells': code_cells,
            'notebook': notebook
        }
        
    except NotJSONError:
        st.error("Invalid notebook format. Please upload a valid .ipynb file.")
        return None
    except Exception as e:
        st.error(f"Error loading Jupyter notebook: {str(e)}")
        return None

def extract_code_from_notebook(notebook_data):
    """
    Extract code from Jupyter notebook
    
    Parameters:
    -----------
    notebook_data : dict
        Dictionary containing notebook data
        
    Returns:
    --------
    str
        Extracted code as a single string
    """
    if not notebook_data or 'code_cells' not in notebook_data:
        return ""
    
    # Join code cells with separators for readability
    code = "\n\n# " + "-" * 40 + "\n\n".join(notebook_data['code_cells'])
    return code
