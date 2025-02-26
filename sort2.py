"""
shrec_processor.py

Complete processing pipeline for SHREC, PPAC, and Rutherford detector data
with memory-efficient processing.
"""

import pandas as pd
import numpy as np
import os
import time
from datetime import datetime
import warnings

from shrec_utils2 import (
    process_shrec_data, 
    extract_ppac_data, 
    extract_rutherford_data, 
    detmerge,
    mapimp,  # Add these
    mapboxE, 
    mapboxW, 
    mapboxT, 
    mapboxB, 
    mapveto   # Make sure all mapping functions are imported
)


# Disable warnings
warnings.filterwarnings('ignore', category=pd.errors.SettingWithCopyWarning)

# Default folder locations
DATA_FOLDER = '../ppac_data/'
SHREC_MAP = os.path.join(DATA_FOLDER, 'r238_shrec_map.xlsx')
SHREC_CALIBRATION = os.path.join(DATA_FOLDER, 'r238_calibration_v0_copy-from-r237.txt')
OUTPUT_FOLDER = 'processed_data/'

def get_output_paths(output_folder):
    """Generate standard output file paths."""
    os.makedirs(output_folder, exist_ok=True)
    return {
        'dssd_clean': os.path.join(output_folder, 'dssd_non_vetoed_events.csv'),
        'ppac': os.path.join(output_folder, 'ppac_events.csv'),
        'rutherford': os.path.join(output_folder, 'rutherford_events.csv'),
        'all_events': os.path.join(output_folder, 'all_events_merged.csv'),
        'summary': os.path.join(output_folder, 'processing_summary.csv'),
        'log': os.path.join(output_folder, 'processing_log.txt')
    }

def log_message(message, log_file=None):
    """Log a message to console and optionally to a file."""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_line = f"[{timestamp}] {message}"
    print(log_line)
    if log_file:
        with open(log_file, 'a') as f:
            f.write(log_line + '\n')

def load_file_list(file_list_path):
    """Reads the file paths from a text file and returns a list of CSV file paths."""
    with open(file_list_path, 'r') as f:
        files = [line.strip() for line in f if line.strip()]
    return files

def process_file(csv_file, output_paths, shrec_map_path, calibration_path, save_all_events=False, ecut=50):
    """
    Process a single CSV file and extract all detector data.
    """
    try:
        log_message(f"Processing {csv_file}...", output_paths['log'])
        t_start = time.time()
        
        # Read the data
        raw_df = pd.read_csv(csv_file)
        log_message(f"Read {len(raw_df)} raw events", output_paths['log'])
        
        # Extract PPAC and Rutherford data 
        ppac_data = extract_ppac_data(raw_df)
        ruth_data = extract_rutherford_data(raw_df)
        
        # Process SHREC data
        shrec_results = process_shrec_data(
            csv_file, 
            shrec_map_path, 
            calibration_path, 
            ecut=ecut
        )
        
        # Extract veto events
        veto_events = shrec_results.get('veto', pd.DataFrame())
        
        # Process with detmerge to find vetoed events
        dssd_regions = ['imp', 'boxE', 'boxW', 'boxT', 'boxB']
        for region in dssd_regions:
            if region in shrec_results and len(shrec_results[region]) > 0:
                log_message(f"Finding veto coincidences for {region} events...", output_paths['log'])
                shrec_results[region] = detmerge(shrec_results[region], veto_events)
                
                # Count vetoed and non-vetoed events
                if 'is_vetoed' in shrec_results[region].columns:
                    vetoed_count = shrec_results[region]['is_vetoed'].sum()
                    log_message(f"  {vetoed_count} {region} events are vetoed", output_paths['log'])
        
        # Create output dataframes
        # 1. All DSSD events merged (for debugging)
        if save_all_events:
            all_dssd = []
            for region in dssd_regions:
                if region in shrec_results and len(shrec_results[region]) > 0:
                    all_dssd.append(shrec_results[region])
            
            if all_dssd:
                all_dssd_df = pd.concat(all_dssd, ignore_index=True)
                all_dssd_df = all_dssd_df.sort_values(by='t').reset_index(drop=True)
                
                # Append to all events file
                write_header = not os.path.exists(output_paths['all_events'])
                all_dssd_df.to_csv(output_paths['all_events'], mode='a', header=write_header, index=False)
                log_message(f"Appended {len(all_dssd_df)} events to all events file", output_paths['log'])
        
        # 2. Clean DSSD events (non-vetoed)
        clean_dssd = []
        for region in dssd_regions:
            if region in shrec_results and len(shrec_results[region]) > 0:
                # Filter for non-vetoed events
                clean_events = shrec_results[region][~shrec_results[region]['is_vetoed']].copy()
                # Add event_type column
                clean_events['event_type'] = region
                clean_events = clean_events.drop('is_vetoed', axis=1)
                clean_dssd.append(clean_events)
        
        if clean_dssd:
            clean_dssd_df = pd.concat(clean_dssd, ignore_index=True)
            clean_dssd_df = clean_dssd_df.sort_values(by='t').reset_index(drop=True)
            
            # Append to clean events file
            write_header = not os.path.exists(output_paths['dssd_clean'])
            clean_dssd_df.to_csv(output_paths['dssd_clean'], mode='a', header=write_header, index=False)
            log_message(f"Appended {len(clean_dssd_df)} clean events to DSSD clean file", output_paths['log'])
        
        # 3. PPAC data
        if len(ppac_data) > 0:
            write_header = not os.path.exists(output_paths['ppac'])
            ppac_data.to_csv(output_paths['ppac'], mode='a', header=write_header, index=False)
            log_message(f"Appended {len(ppac_data)} events to PPAC file", output_paths['log'])
        
        # 4. Rutherford data
        if len(ruth_data) > 0:
            write_header = not os.path.exists(output_paths['rutherford'])
            ruth_data.to_csv(output_paths['rutherford'], mode='a', header=write_header, index=False)
            log_message(f"Appended {len(ruth_data)} events to Rutherford file", output_paths['log'])
        
        # Calculate summary statistics
        summary = {
            'filename': os.path.basename(csv_file),
            'total_events': sum(len(df) for df in shrec_results.values()) + len(ppac_data) + len(ruth_data)
        }
        
        # Add counts for each detector region
        for region in ['imp', 'boxE', 'boxW', 'boxT', 'boxB', 'veto']:
            if region in shrec_results:
                summary[f'{region}_events'] = len(shrec_results[region])
                if region != 'veto' and 'is_vetoed' in shrec_results[region].columns:
                    summary[f'{region}_vetoed'] = shrec_results[region]['is_vetoed'].sum()
                    summary[f'{region}_clean'] = len(shrec_results[region]) - shrec_results[region]['is_vetoed'].sum()
        
        # Add PPAC and Rutherford counts
        summary['ppac_events'] = len(ppac_data)
        summary['ruth_events'] = len(ruth_data)
        
        # Add processing duration
        t_end = time.time()
        summary['processing_time_seconds'] = t_end - t_start
        
        # Append to summary file
        summary_df = pd.DataFrame([summary])
        write_header = not os.path.exists(output_paths['summary'])
        summary_df.to_csv(output_paths['summary'], mode='a', header=write_header, index=False)
        
        log_message(f"Processing complete for {csv_file}. Time: {t_end - t_start:.1f} seconds", output_paths['log'])
        return summary
        
    except Exception as e:
        log_message(f"Error processing {csv_file}: {str(e)}", output_paths['log'])
        return {
            'filename': os.path.basename(csv_file),
            'total_events': 0,
            'error': str(e)
        }

def main():
    """Main function to process all files."""
    # Configuration variables
    file_list_path = 'files_to_sort.txt'
    data_folder = '../ppac_data/'
    shrec_map_path = os.path.join(data_folder, 'r238_shrec_map.xlsx')
    calibration_path = os.path.join(data_folder, 'r238_calibration_v0_copy-from-r237.txt')
    output_folder = 'processed_data/'
    energy_cut = 50
    save_all_events = False
    clear_outputs = True
    
    # Set up output paths
    output_paths = get_output_paths(output_folder)
    
    # Clear existing output files if requested
    if clear_outputs:
        for path in output_paths.values():
            if os.path.exists(path):
                os.remove(path)
                log_message(f"Removed existing file: {path}")
    
    # Initialize log file
    log_message(f"Starting SHREC data processing", output_paths['log'])
    
    # Load list of files to process
    try:
        csv_files = load_file_list(file_list_path)
        log_message(f"Found {len(csv_files)} files to process", output_paths['log'])
    except Exception as e:
        log_message(f"Error loading file list: {str(e)}", output_paths['log'])
        return
    
    # Process each file
    total_events = 0
    
    for i, csv_file in enumerate(csv_files):
        log_message(f"\nProcessing file {i+1}/{len(csv_files)}: {csv_file}", output_paths['log'])
        
        try:
            # Process this file
            summary = process_file(
                csv_file, 
                output_paths,
                shrec_map_path,
                calibration_path,
                save_all_events=save_all_events,
                ecut=energy_cut
            )
            
            # Update statistics
            if 'total_events' in summary:
                total_events += summary['total_events']
            
            # Print progress
            log_message(f"Progress: {i+1}/{len(csv_files)} files processed", output_paths['log'])
            log_message(f"Running total: {total_events} events", output_paths['log'])
            
        except Exception as e:
            log_message(f"Error processing file {csv_file}: {str(e)}", output_paths['log'])
    
    # Print final summary
    log_message("\nProcessing complete!", output_paths['log'])
    log_message(f"Total files processed: {len(csv_files)}", output_paths['log'])
    log_message(f"Total events processed: {total_events}", output_paths['log'])
    
    # Try to read file size information
    for name, path in output_paths.items():
        if name != 'log' and os.path.exists(path):
            try:
                file_size_mb = os.path.getsize(path) / (1024 * 1024)
                log_message(f"{name} file size: {file_size_mb:.1f} MB", output_paths['log'])
            except:
                pass

if __name__ == "__main__":
    main()