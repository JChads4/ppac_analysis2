#!/usr/bin/env python3

import sys
import traceback
import h5py
import numpy as np
import matplotlib.pyplot as plt

# Ensure Panel is imported
try:
    import panel as pn
except ImportError as e:
    print(f"Failed to import Panel: {e}")
    sys.exit(1)

# Explicitly enable Panel extension
pn.extension(template='bootstrap')

class SHRECDataExplorer:
    def __init__(self, hdf5_path='shrec_results.h5'):
        """
        Initialize the data explorer with the HDF5 file
        
        Parameters:
        -----------
        hdf5_path : str
            Path to the HDF5 file containing SHREC data
        """
        self.hdf5_path = hdf5_path
        
        # Identify available datasets and their columns
        self.datasets = self._get_available_datasets()
        
        # Create widgets
        self.dataset_select = pn.widgets.Select(
            name='Select Dataset', 
            options=list(self.datasets.keys())
        )
        
        self.column_select = pn.widgets.Select(
            name='Select Column', 
            options=[]
        )
        
        self.bin_slider = pn.widgets.IntSlider(
            name='Number of Bins', 
            start=10, 
            end=200, 
            value=50
        )
        
        # Create plot area using Matplotlib pane
        self.plot = pn.pane.Matplotlib(self._create_histogram(), height=400)
        
        # Link dataset selection to column selection
        self.dataset_select.param.watch(self._update_columns, 'value')
        
        # Bind widgets to update plot
        self.dataset_select.param.watch(self._update_plot, 'value')
        self.column_select.param.watch(self._update_plot, 'value')
        self.bin_slider.param.watch(self._update_plot, 'value')
    
    def _get_available_datasets(self):
        """
        Retrieve available datasets from the HDF5 file
        
        Returns:
        --------
        dict
            Dictionary of datasets with their columns
        """
        datasets = {}
        with h5py.File(self.hdf5_path, 'r') as f:
            for group_name in ['raw_data', 'histograms']:
                if group_name in f:
                    for dataset_name, dataset in f[group_name].items():
                        # Get columns if available
                        if 'columns' in dataset.attrs:
                            columns = list(dataset.attrs['columns'])
                        else:
                            # For non-struct datasets, list the keys
                            columns = list(dataset.keys())
                        
                        # Create full path for each dataset
                        full_path = f"{group_name}/{dataset_name}"
                        datasets[full_path] = columns
        return datasets
    
    def _update_columns(self, event):
        """
        Update column selection based on selected dataset
        """
        columns = self.datasets.get(event.new, [])
        self.column_select.options = columns
        if columns:
            self.column_select.value = columns[0]
    
    def _create_histogram(self):
        """
        Create histogram for the selected dataset and column
        
        Returns:
        --------
        matplotlib Figure
        """
        # Check if selections are valid
        if not self.dataset_select.value or not self.column_select.value:
            plt.figure(figsize=(10, 6))
            plt.title("Select a dataset and column")
            plt.xlabel("Value")
            plt.ylabel("Frequency")
            return plt.gcf()
        
        # Load data
        with h5py.File(self.hdf5_path, 'r') as f:
            # Split the path
            group, dataset = self.dataset_select.value.split('/', 1)
            column = self.column_select.value
            
            # Handle different data structures
            try:
                # Try to read as a dataset column
                data = f[group][dataset][column][:]
            except Exception:
                # Fallback to reading the entire dataset if column not found
                data = f[group][dataset][:]
        
        # Create histogram
        plt.figure(figsize=(10, 6))
        plt.hist(data, bins=self.bin_slider.value, edgecolor='black')
        plt.title(f"Histogram of {self.dataset_select.value} - {self.column_select.value}")
        plt.xlabel(self.column_select.value)
        plt.ylabel("Frequency")
        plt.tight_layout()
        return plt.gcf()
    
    def _update_plot(self, event=None):
        """
        Update histogram based on selections
        """
        # Clear previous plot
        plt.close('all')
        
        # Create new plot
        new_plot = self._create_histogram()
        
        # Update panel
        self.plot.object = new_plot
    
    def create_layout(self):
        """
        Create the full dashboard layout
        
        Returns:
        --------
        pn.Column
            Full dashboard layout
        """
        return pn.Column(
            pn.pane.Markdown("# SHREC Data Explorer"),
            pn.pane.Markdown("## Interactive Histogram Visualization"),
            pn.Row(
                pn.Column(
                    self.dataset_select,
                    self.column_select,
                    self.bin_slider
                ),
                self.plot
            ),
            height=600, 
            width=1000
        )

def main():
    """
    Main function to create and serve the Panel application
    """
    print("Starting SHREC Data Explorer")
    print("=" * 30)
    
    try:
        # Create the explorer
        explorer = SHRECDataExplorer()
        
        # Create the app layout
        app = explorer.create_layout()
        
        print("App created successfully")
        print("Attempting to serve...")
        
        # Serve the app
        pn.serve(app, port=5007, show=True, verbose=True)
    
    except Exception as e:
        print("ERROR: Failed to start SHREC Data Explorer")
        print(f"Error type: {type(e)}")
        print(f"Error message: {e}")
        print("Traceback:")
        traceback.print_exc()

if __name__ == "__main__":
    main()