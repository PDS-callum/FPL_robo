"""
Utility functions for file operations and data loading.
Consolidates common patterns used across the codebase.
"""

import pandas as pd
import os
from typing import Optional, Union, List


def robust_csv_reader(url_or_path: str, **kwargs) -> Optional[pd.DataFrame]:
    """
    Robust CSV reader that handles various encoding and parsing issues.
    
    This function tries multiple strategies to read CSV files that may have
    encoding issues or malformed lines, which is common with external data sources.
    
    Parameters:
    -----------
    url_or_path : str
        URL or file path to the CSV file
    **kwargs : dict
        Additional arguments to pass to pd.read_csv
        
    Returns:
    --------
    pd.DataFrame or None
        DataFrame if successful, None if all strategies fail
    """
    # Define strategies in order of preference
    strategies = [
        # Strategy 1: Standard UTF-8
        {'encoding': 'utf-8'},
        # Strategy 2: Latin-1 encoding
        {'encoding': 'latin-1'},
        # Strategy 3: UTF-8 with bad line skipping (modern pandas)
        {'encoding': 'utf-8', 'on_bad_lines': 'skip'},
        # Strategy 4: UTF-8 with bad line skipping (older pandas)
        {'encoding': 'utf-8', 'error_bad_lines': False, 'warn_bad_lines': True},
        # Strategy 5: Quote handling
        {'encoding': 'utf-8', 'quoting': 1, 'skipinitialspace': True},
        # Strategy 6: Python engine (most forgiving)
        {'encoding': 'utf-8', 'engine': 'python', 'on_bad_lines': 'skip'}
    ]
    
    for i, strategy in enumerate(strategies, 1):
        try:
            # Merge strategy with user-provided kwargs
            read_kwargs = {**strategy, **kwargs}
            return pd.read_csv(url_or_path, **read_kwargs)
        except (UnicodeDecodeError, pd.errors.ParserError):
            continue
        except TypeError as e:
            # Handle version-specific parameter differences
            if 'on_bad_lines' in str(e) and 'on_bad_lines' in read_kwargs:
                # Try without on_bad_lines for older pandas versions
                read_kwargs_fallback = read_kwargs.copy()
                del read_kwargs_fallback['on_bad_lines']
                try:
                    return pd.read_csv(url_or_path, **read_kwargs_fallback)
                except:
                    continue
            continue
        except Exception as e:
            if i == len(strategies):
                print(f"All CSV reading strategies failed for {url_or_path}: {e}")
            continue
    
    return None


def ensure_directory_exists(path: str) -> str:
    """
    Ensure a directory exists, creating it if necessary.
    
    Parameters:
    -----------
    path : str
        Directory path to create
        
    Returns:
    --------
    str
        The directory path
    """
    os.makedirs(path, exist_ok=True)
    return path


def get_file_paths(directory: str, extension: str = None, 
                   pattern: str = None) -> List[str]:
    """
    Get all file paths in a directory with optional filtering.
    
    Parameters:
    -----------
    directory : str
        Directory to search
    extension : str, optional
        File extension to filter by (e.g., '.csv', '.json')
    pattern : str, optional
        Pattern that filenames must contain
        
    Returns:
    --------
    List[str]
        List of file paths
    """
    if not os.path.exists(directory):
        return []
    
    files = []
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        
        # Skip directories
        if os.path.isdir(filepath):
            continue
            
        # Apply extension filter
        if extension and not filename.endswith(extension):
            continue
            
        # Apply pattern filter
        if pattern and pattern not in filename:
            continue
            
        files.append(filepath)
    
    return sorted(files)


def safe_file_operation(operation, filepath: str, *args, **kwargs):
    """
    Safely perform a file operation with error handling.
    
    Parameters:
    -----------
    operation : callable
        File operation function (e.g., pd.DataFrame.to_csv, json.dump)
    filepath : str
        Path to the file
    *args, **kwargs
        Arguments to pass to the operation
        
    Returns:
    --------
    bool
        True if operation succeeded, False otherwise
    """
    try:
        # Ensure directory exists
        directory = os.path.dirname(filepath)
        if directory:
            ensure_directory_exists(directory)
        
        # Perform operation
        operation(filepath, *args, **kwargs)
        return True
    except Exception as e:
        print(f"File operation failed for {filepath}: {e}")
        return False


def convert_to_json_serializable(obj):
    """
    Convert objects with numpy/pandas types to JSON-serializable format.
    
    Parameters:
    -----------
    obj : any
        Object to convert
        
    Returns:
    --------
    any
        JSON-serializable version of the object
    """
    if hasattr(obj, 'item'):  # numpy scalar
        return obj.item()
    elif hasattr(obj, 'to_dict'):  # pandas Series/DataFrame
        return obj.to_dict()
    elif isinstance(obj, dict):
        return {k: convert_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]
    else:
        try:
            # Try to convert using standard JSON serialization
            import json
            json.dumps(obj)
            return obj
        except (TypeError, ValueError):
            # If all else fails, convert to string
            return str(obj)
