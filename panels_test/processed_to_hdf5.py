"""
save_to_hdf5.py

Creates an HDF5 file from processed SHREC data, saving both raw data
and pre-computed histograms for quick interactive exploration.
"""

import pandas as pd
import numpy as np
import h5py
import os
import time
from datetime import datetime

# Configuration
INPUT_FOLDER = 'processed_data/'
OUTPUT_FILE = 'shrec_results.h5'

def save_dataframe_to_hdf(group, df, name):
    """Save a pandas DataFrame to an HDF5 group."""
    if len(df) == 0:
        print(f"  Skipping empty dataframe: {name}")
        return
    
    # Create a subgroup for this dataframe
    subgroup = group.create_group(name)
    
    # Store metadata
    subgroup.attrs['rows'] = len(df)
    subgroup.attrs['columns'] = list(df.columns)
    
    # Store each column
    for col in df.columns:
        # Handle different data types appropriately
        if pd.api.types.is_numeric_dtype(df[col]):
            # For numeric data, store as is
            subgroup.create_dataset(col, data=df[col].values)
        else:
            # For string/object data, store as strings
            # Convert to fixed-length strings for HDF5 compatibility
            string_data = df[col].astype(str).values
            subgroup.create_dataset(col, data=string_data, dtype=h5py.string_dtype())

def compute_histograms(df, hist_group, prefix):
    """Compute and store common histograms for a dataframe."""
    # Skip if dataframe is empty
    if len(df) == 0:
        return
    
    # Create a group for this set of histograms
    group = hist_group.create_group(prefix)
    
    # 1. Store time range information
    group.attrs['time_min'] = df['t'].min()
    group.attrs['time_max'] = df['t'].max()
    group.attrs['duration_seconds'] = df['t'].max() - df['t'].min()
    
    # 2. For DSSD data, create energy histograms
    if 'xE' in df.columns and 'yE' in df.columns:
        # X Energy histogram
        counts, edges = np.histogram(df['xE'], bins=200, range=(0, 11000))
        xe_group = group.create_group('energy_x')
        xe_group.create_dataset('counts', data=counts)
        xe_group.create_dataset('bin_edges', data=edges)
        
        # Y Energy histogram
        counts, edges = np.histogram(df['yE'], bins=200, range=(0, 11000))
        ye_group = group.create_group('energy_y')
        ye_group.create_dataset('counts', data=counts)
        ye_group.create_dataset('bin_edges', data=edges)
        
        # X vs Y strip heat map
        if 'x' in df.columns and 'y' in df.columns:
            h, xedges, yedges = np.histogram2d(
                df['x'], df['y'], 
                bins=[174, 60], 
                range=[[0, 174], [0, 60]]
            )
            xy_group = group.create_group('xy_heatmap')
            xy_group.create_dataset('counts', data=h)
            xy_group.create_dataset('x_edges', data=xedges)
            xy_group.create_dataset('y_edges', data=yedges)
    
    # 3. Event rate over time (1-second bins)
    time_min = df['t'].min()
    time_max = df['t'].max()
    bins = np.arange(np.floor(time_min), np.ceil(time_max) + 1, 1)  # 1-second bins
    
    counts, edges = np.histogram(df['t'], bins=bins)
    rate_group = group.create_group('event_rate')
    rate_group.create_dataset('counts', data=counts)
    rate_group.create_dataset('bin_edges', data=edges)
    rate_group.attrs['bin_width_seconds'] = 1.0
    
    # 4. For DSSD with event_type, store counts by type
    if 'event_type' in df.columns:
        event_types = df['event_type'].unique()
        counts_dict = df['event_type'].value_counts().to_dict()
        
        type_group = group.create_group('event_types')
        # Store as attributes for easy access
        for event_type, count in counts_dict.items():
            type_group.attrs[event_type] = count
        
        # Also store energy histograms by event type
        if 'xE' in df.columns:
            by_type_group = group.create_group('energy_by_type')
            for event_type in event_types:
                subset = df[df['event_type'] == event_type]
                if len(subset) > 0:
                    counts, edges = np.histogram(subset['xE'], bins=200, range=(0, 11000))
                    et_group = by_type_group.create_group(event_type)
                    et_group.create_dataset('counts', data=counts)
                    et_group.create_dataset('bin_edges', data=edges)


def inspect_hdf5(file_path):
    """
    Comprehensive inspector for HDF5 file structure and content
    
    Parameters:
    -----------
    file_path : str
        Path to the HDF5 file to inspect
    """
    def print_attrs(obj):
        """Print attributes of an HDF5 object"""
        if hasattr(obj, 'attrs'):
            print("  Attributes:")
            for key, value in obj.attrs.items():
                print(f"    {key}: {value}")
    
    def explore_group(name, obj):
        """Recursively explore HDF5 groups and datasets"""
        if isinstance(obj, h5py.Group):
            print(f"\nGroup: {name}")
            print_attrs(obj)
        
        elif isinstance(obj, h5py.Dataset):
            print(f"\nDataset: {name}")
            print(f"  Shape: {obj.shape}")
            print(f"  Dtype: {obj.dtype}")
            
            # Try to print a sample of the data
            try:
                if obj.size > 0:
                    # For small datasets, print all data
                    if obj.size <= 20:
                        print("  Data:")
                        print(obj[:])
                    else:
                        # For larger datasets, print first and last few elements
                        print("  First 10 elements:")
                        print(obj[:10])
                        print("  Last 10 elements:")
                        print(obj[-10:])
            except Exception as e:
                print(f"  Could not display data: {e}")
            
            print_attrs(obj)
    
    # Open the HDF5 file
    with h5py.File(file_path, 'r') as f:
        print(f"HDF5 File Inspector: {file_path}")
        print("=" * 50)
        
        # Print file-level attributes
        print("\nFile-level Attributes:")
        print_attrs(f)
        
        # Recursively explore the file structure
        print("\nFile Structure:")
        f.visititems(explore_group)

def preview_dataset(file_path, group_name, dataset_name):
    """
    Preview a specific dataset in the HDF5 file
    
    Parameters:
    -----------
    file_path : str
        Path to the HDF5 file
    group_name : str
        Name of the group containing the dataset
    dataset_name : str
        Name of the dataset to preview
    """
    with h5py.File(file_path, 'r') as f:
        # Access the specific dataset
        dataset = f[group_name][dataset_name]
        
        print(f"Dataset Preview: {group_name}/{dataset_name}")
        print("=" * 50)
        print(f"Shape: {dataset.shape}")
        print(f"Dtype: {dataset.dtype}")
        
        # Try to print a sample of the data
        try:
            if dataset.size > 0:
                # For small datasets, print all data
                if dataset.size <= 20:
                    print("\nFull Data:")
                    print(dataset[:])
                else:
                    # For larger datasets, print first and last few elements
                    print("\nFirst 10 elements:")
                    print(dataset[:10])
                    print("\nLast 10 elements:")
                    print(dataset[-10:])
        except Exception as e:
            print(f"Could not display data: {e}")


def main():
    """Main function to save processed data to HDF5."""
    print(f"Starting HDF5 export at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    start_time = time.time()
    
    # Check if input folder exists
    if not os.path.exists(INPUT_FOLDER):
        print(f"Error: Input folder {INPUT_FOLDER} does not exist.")
        return
    
    # Load the processed data
    data = {}
    
    # DSSD clean events
    dssd_path = os.path.join(INPUT_FOLDER, 'dssd_clean_events.csv')
    if os.path.exists(dssd_path):
        print(f"Loading DSSD data from {dssd_path}...")
        data['dssd'] = pd.read_csv(dssd_path)
        print(f"Loaded {len(data['dssd'])} DSSD events")
    else:
        data['dssd'] = pd.DataFrame()
        print(f"Warning: DSSD file not found at {dssd_path}")
    
    # PPAC events
    ppac_path = os.path.join(INPUT_FOLDER, 'ppac_events.csv')
    if os.path.exists(ppac_path):
        print(f"Loading PPAC data from {ppac_path}...")
        data['ppac'] = pd.read_csv(ppac_path)
        print(f"Loaded {len(data['ppac'])} PPAC events")
    else:
        data['ppac'] = pd.DataFrame()
        print(f"Warning: PPAC file not found at {ppac_path}")
    
    # Rutherford events
    ruth_path = os.path.join(INPUT_FOLDER, 'rutherford_events.csv')
    if os.path.exists(ruth_path):
        print(f"Loading Rutherford data from {ruth_path}...")
        data['rutherford'] = pd.read_csv(ruth_path)
        print(f"Loaded {len(data['rutherford'])} Rutherford events")
    else:
        data['rutherford'] = pd.DataFrame()
        print(f"Warning: Rutherford file not found at {ruth_path}")
    
    # Create the HDF5 file
    print(f"\nCreating HDF5 file: {OUTPUT_FILE}")
    with h5py.File(OUTPUT_FILE, 'w') as f:
        # Store metadata
        f.attrs['creation_date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        f.attrs['input_folder'] = INPUT_FOLDER
        
        # Create groups for each type of data
        raw_data = f.create_group('raw_data')
        histograms = f.create_group('histograms')
        
        # Save raw data
        print("Saving raw data...")
        for data_type, df in data.items():
            print(f"  Processing {data_type} data...")
            save_dataframe_to_hdf(raw_data, df, data_type)
        
        # Compute and save histograms
        print("\nComputing and saving histograms...")
        for data_type, df in data.items():
            print(f"  Computing histograms for {data_type} data...")
            compute_histograms(df, histograms, data_type)
    
    # Calculate file size
    file_size_mb = os.path.getsize(OUTPUT_FILE) / (1024 * 1024)
    
    # Print summary
    end_time = time.time()
    print(f"\nExport complete!")
    print(f"Total time: {end_time - start_time:.2f} seconds")
    print(f"HDF5 file size: {file_size_mb:.2f} MB")
    print(f"File saved to: {os.path.abspath(OUTPUT_FILE)}")

if __name__ == "__main__":
    main()
    # Inspect the entire HDF5 file
    inspect_hdf5('shrec_results.h5')
    
    # Preview specific datasets (modify as needed)
    print("\n\nPreview of a specific dataset:")
    preview_dataset('shrec_results.h5', 'raw_data/dssd', 't')
    preview_dataset('shrec_results.h5', 'histograms/dssd/energy_x', 'counts')