"""
shrec_processor.py

Complete processing pipeline for SHREC, PPAC, and Rutherford detector data
with memory-efficient implementation and veto coincidence filtering.
"""

import pandas as pd
import numpy as np
import os
import gc  # Garbage collector
import psutil  # For memory monitoring (pip install psutil)
import time
from tqdm import tqdm
import warnings
import argparse
from datetime import datetime

warnings.filterwarnings('ignore', category=pd.errors.SettingWithCopyWarning)

# Default folder locations (can be overridden with command line arguments)
DATA_FOLDER = '../ppac_data/'
SHREC_MAP = os.path.join(DATA_FOLDER, 'r238_shrec_map.xlsx')
SHREC_CALIBRATION = os.path.join(DATA_FOLDER, 'r238_calibration_v0_copy-from-r237.txt')
OUTPUT_FOLDER = 'processed_data/'

# Memory settings
MAX_MEMORY_MB = 5120  # 5 GB default (adjust based on your system)
MEMORY_THRESHOLD_MB = MAX_MEMORY_MB * 0.9  # Threshold to trigger garbage collection

# Output file paths
def get_output_paths(output_folder):
    """Generate standard output file paths."""
    os.makedirs(output_folder, exist_ok=True)
    return {
        'dssd_clean': os.path.join(output_folder, 'dssd_clean_events.csv'),
        'ppac': os.path.join(output_folder, 'ppac_events.csv'),
        'rutherford': os.path.join(output_folder, 'rutherford_events.csv'),
        'all_events': os.path.join(output_folder, 'all_events_merged.csv'),
        'summary': os.path.join(output_folder, 'processing_summary.csv'),
        'log': os.path.join(output_folder, 'processing_log.txt')
    }

def get_memory_mb():
    """Get current memory usage in MB."""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    return memory_info.rss / 1024 / 1024

def check_memory():
    """Check if memory usage is above threshold and force garbage collection if needed."""
    memory_mb = get_memory_mb()
    if memory_mb > MEMORY_THRESHOLD_MB:
        log_message(f"Memory usage high ({memory_mb:.1f} MB). Forcing garbage collection...")
        gc.collect()
        log_message(f"Memory after GC: {get_memory_mb():.1f} MB")

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

def process_region_chunk(chunk_df, region_func, shrec_map, calibration_path, event_type_name, ecut=50):
    """
    Process a single detector region from a chunk of raw dataframe.
    
    Parameters:
    -----------
    chunk_df : pd.DataFrame
        Chunk of raw data containing 'board' and 'channel' columns
    region_func : function
        Mapping function for this detector region (e.g., mapimp)
    shrec_map : dict
        Dictionary of mapping DataFrames
    calibration_path : str
        Path to calibration file
    event_type_name : str
        Name of the event type (e.g., 'imp', 'boxE')
    ecut : float
        Energy cut threshold
        
    Returns:
    --------
    pd.DataFrame or None
        Processed data or None if no events found
    """
    try:
        # Apply the mapping function to get x and y data
        x_data, y_data = region_func(chunk_df, shrec_map)
        
        # Skip if no data
        if len(x_data) == 0 or len(y_data) == 0:
            return None
        
        # Apply basic energy cut early to reduce data size
        x_data = x_data[x_data['energy'] >= ecut].copy()
        y_data = y_data[y_data['energy'] >= ecut].copy()
        
        if len(x_data) == 0 or len(y_data) == 0:
            return None
            
        # Convert timetags to time
        x_data['t'] = np.round(x_data['timetag'] * 1e-12, 6)
        y_data['t'] = np.round(y_data['timetag'] * 1e-12, 6)
        x_data['t2'] = np.round(x_data['timetag'] * 1e-12, 5)
        y_data['t2'] = np.round(y_data['timetag'] * 1e-12, 5)
        
        # Load calibration data (only once and reuse)
        calfile = pd.read_csv(calibration_path, sep='\t')
        cal_cols = ['board', 'channel', 'm', 'b']
        cal_subset = calfile[cal_cols].copy()
        
        # Join calibration columns
        x_data = pd.merge(x_data, cal_subset, on=['board', 'channel'], how='left')
        y_data = pd.merge(y_data, cal_subset, on=['board', 'channel'], how='left')
        
        # Compute calibrated energy
        x_data['calE'] = x_data['m'] * (x_data['energy'] - x_data['b'])
        y_data['calE'] = y_data['m'] * (y_data['energy'] - y_data['b'])
        
        # Drop unneeded columns
        x_data = x_data.drop(['energy', 'm', 'b'], axis=1)
        y_data = y_data.drop(['energy', 'm', 'b'], axis=1)
        
        # Check memory before merging
        check_memory()
        
        # Process in smaller time-based chunks if too large
        if len(x_data) > 100000 or len(y_data) > 100000:
            # Find min and max times
            x_times = x_data['t'].sort_values().unique()
            y_times = y_data['t'].sort_values().unique()
            
            # Get merged set of unique times
            all_times = np.sort(np.unique(np.concatenate([x_times, y_times])))
            
            # If we have a lot of times, chunk them
            if len(all_times) > 500:
                chunk_size = len(all_times) // 10  # Split into 10 time chunks
                time_chunks = [all_times[i:i + chunk_size] for i in range(0, len(all_times), chunk_size)]
                
                # Process each time chunk separately
                df_results = []
                for i, time_chunk in enumerate(time_chunks):
                    print(f"  Processing time chunk {i+1}/{len(time_chunks)} for {event_type_name}...")
                    
                    # Get min/max time for this chunk
                    min_t = time_chunk[0]
                    max_t = time_chunk[-1]
                    
                    # Filter data for this time range
                    x_chunk = x_data[(x_data['t'] >= min_t) & (x_data['t'] <= max_t)]
                    y_chunk = y_data[(y_data['t'] >= min_t) & (y_data['t'] <= max_t)]
                    
                    # Skip if either is empty
                    if len(x_chunk) == 0 or len(y_chunk) == 0:
                        continue
                    
                    # Process time chunk
                    result = process_time_chunk(x_chunk, y_chunk, event_type_name)
                    if result is not None and len(result) > 0:
                        df_results.append(result)
                    
                    # Clean up
                    del x_chunk, y_chunk
                    check_memory()
                
                # Combine results from all time chunks
                if df_results:
                    return pd.concat(df_results, ignore_index=True)
                else:
                    return None
            else:
                # Small enough to process directly
                return process_time_chunk(x_data, y_data, event_type_name)
        else:
            # Small enough to process directly
            return process_time_chunk(x_data, y_data, event_type_name)
        
    except Exception as e:
        print(f"Error processing {event_type_name} chunk: {str(e)}")
        return None

def process_time_chunk(x_data, y_data, event_type_name):
    """Process a single time chunk of x and y data."""
    try:
        # Merge on exact time match first
        dfxy1 = pd.merge(x_data, y_data, on='t', suffixes=('_x', '_y'))
        check_memory()
        
        # Then merge on rounded time
        dfxy2 = pd.merge(x_data, y_data, on='t2', suffixes=('_x', '_y'))
        check_memory()
        
        # Fix column names in dfxy1
        if 't2_y' in dfxy1.columns:
            dfxy1 = dfxy1.drop(['t2_y'], axis=1)
        if 't2_x' in dfxy1.columns:
            dfxy1 = dfxy1.rename(columns={'t2_x': 't2'})
            
        # Fix column names in dfxy2
        if 't_y' in dfxy2.columns:
            dfxy2 = dfxy2.drop(['t_y'], axis=1)
        if 't_x' in dfxy2.columns:
            dfxy2 = dfxy2.rename(columns={'t_x': 't'})
        
        # Combine, drop duplicates
        dfxy = pd.concat([dfxy1, dfxy2], ignore_index=True)
        dfxy = dfxy.drop_duplicates()
        if 't2' in dfxy.columns:
            dfxy = dfxy.drop(['t2'], axis=1)
        
        # Free memory from intermediate dataframes
        del dfxy1, dfxy2
        check_memory()
        
        if len(dfxy) == 0:
            return None
        
        # Rename columns as needed
        new_columns = {
            'board_x': 'xboard', 
            'channel_x': 'xchan', 
            'timetag_x': 'tagx', 
            'flags_x': 'xflag', 
            'no_file': 'nfile', 
            'id_x': 'xid', 
            'strip_x': 'x', 
            'calE_x': 'xstripE',
            'board_y': 'yboard', 
            'channel_y': 'ychan', 
            'timetag_y': 'tagy', 
            'flags_y': 'yflag', 
            'no_file_y': 'nfiley', 
            'id_y': 'yid', 
            'strip_y': 'y', 
            'calE_y': 'ystripE'
        }
        
        # Rename only columns that exist
        cols_to_rename = {k: v for k, v in new_columns.items() if k in dfxy.columns}
        dfxy = dfxy.rename(columns=cols_to_rename)
        
        # Keep only essential columns 
        essential_cols = ['t', 'x', 'y', 'xstripE', 'ystripE', 'tagx', 'tagy', 'nfile']
        keep_cols = [col for col in essential_cols if col in dfxy.columns]
        dfxy = dfxy[keep_cols]
        
        # Time difference cut
        dfxy['tdelta'] = dfxy['tagx'] - dfxy['tagy']
        dfxy = dfxy.loc[dfxy['tdelta'].abs() < 400000]
        
        if len(dfxy) == 0:
            return None
            
        # Calculate multiplicities
        x_counts = dfxy.groupby(['t', 'x']).size().reset_index(name='nX')
        y_counts = dfxy.groupby(['t', 'y']).size().reset_index(name='nY')
        
        dfxy = pd.merge(dfxy, x_counts, on=['t', 'x'], how='left')
        dfxy = pd.merge(dfxy, y_counts, on=['t', 'y'], how='left')
        
        # Calculate x/y differences per timestamp
        x_diffs = dfxy.groupby('t')['x'].agg(['min', 'max']).reset_index()
        x_diffs['xdiff'] = x_diffs['max'] - x_diffs['min']
        x_diffs = x_diffs[['t', 'xdiff']]
        
        y_diffs = dfxy.groupby('t')['y'].agg(['min', 'max']).reset_index()
        y_diffs['ydiff'] = y_diffs['max'] - y_diffs['min']
        y_diffs = y_diffs[['t', 'ydiff']]
        
        # Join differences back
        dfxy = pd.merge(dfxy, x_diffs, on='t', how='left')
        dfxy = pd.merge(dfxy, y_diffs, on='t', how='left')
        
        # Summed energies
        xE_sums = dfxy.groupby(['t', 'nX'])['xstripE'].sum().reset_index(name='xE_sum')
        dfxy = pd.merge(dfxy, xE_sums, on=['t', 'nX'], how='left')
        dfxy['xE'] = dfxy['xE_sum'] / dfxy['nX']
        dfxy = dfxy.drop('xE_sum', axis=1)
        
        yE_sums = dfxy.groupby(['t', 'nY'])['ystripE'].sum().reset_index(name='yE_sum')
        dfxy = pd.merge(dfxy, yE_sums, on=['t', 'nY'], how='left')
        dfxy['yE'] = dfxy['yE_sum'] / dfxy['nY']
        dfxy = dfxy.drop('yE_sum', axis=1)
        
        # Filter on x/y differences
        temp3 = dfxy[(dfxy['xdiff'] < 2) & (dfxy['ydiff'] < 2)].copy()
        temp3 = temp3.sort_values(by=['t', 'xstripE', 'ystripE'], ascending=[True, False, False])
        
        # Final cleanup
        result = temp3.drop(['xstripE', 'ystripE', 'xdiff', 'ydiff'], axis=1, errors='ignore')
        result = result.reset_index(drop=True)
        
        # Add event_type column
        result['event_type'] = event_type_name
        
        return result
        
    except Exception as e:
        print(f"Error in process_time_chunk: {str(e)}")
        return None

def extract_ppac_data(raw_df):
    """Extract PPAC detector data from raw dataframe."""
    try:
        # Define PPAC channels
        cathode = raw_df[(raw_df['board'] == 9) & (raw_df['channel'] == 10)].copy()
        anodev = raw_df[(raw_df['board'] == 9) & (raw_df['channel'] == 12)].copy()
        anodeh = raw_df[(raw_df['board'] == 9) & (raw_df['channel'] == 8)].copy()
        
        # Add detector identifier
        if len(cathode) > 0:
            cathode['detector'] = 'cathode'
        if len(anodev) > 0:
            anodev['detector'] = 'anodeV'
        if len(anodeh) > 0:
            anodeh['detector'] = 'anodeH'
        
        # Combine all PPAC data
        ppac_data = pd.concat([cathode, anodev, anodeh], ignore_index=True)
        
        # Convert timetag to seconds
        if len(ppac_data) > 0:
            ppac_data['t'] = np.round(ppac_data['timetag'] * 1e-12, 6)
            
            # Sort by time
            ppac_data = ppac_data.sort_values(by='t').reset_index(drop=True)
            
            # Select columns for output
            ppac_data = ppac_data[['t', 'energy', 'board', 'channel', 'detector', 'timetag', 'nfile']]
            
        return ppac_data
        
    except Exception as e:
        print(f"Error extracting PPAC data: {str(e)}")
        return pd.DataFrame()

def extract_rutherford_data(raw_df):
    """Extract Rutherford detector data from raw dataframe."""
    try:
        # Define Rutherford detector channels
        ruthE = raw_df[(raw_df['board'] == 9) & (raw_df['channel'] == 14)].copy()
        ruthW = raw_df[(raw_df['board'] == 9) & (raw_df['channel'] == 15)].copy()
        
        # Add detector identifier
        if len(ruthE) > 0:
            ruthE['detector'] = 'ruthE'
        if len(ruthW) > 0:
            ruthW['detector'] = 'ruthW'
        
        # Combine all Rutherford data
        ruth_data = pd.concat([ruthE, ruthW], ignore_index=True)
        
        # Convert timetag to seconds
        if len(ruth_data) > 0:
            ruth_data['t'] = np.round(ruth_data['timetag'] * 1e-12, 6)
            
            # Sort by time
            ruth_data = ruth_data.sort_values(by='t').reset_index(drop=True)
            
            # Select columns for output
            ruth_data = ruth_data[['t', 'energy', 'board', 'channel', 'detector', 'timetag', 'nfile']]
            
        return ruth_data
        
    except Exception as e:
        print(f"Error extracting Rutherford data: {str(e)}")
        return pd.DataFrame()

def detmerge(dssd_events, veto_events, time_window=400000e-12):
    """
    Find coincidences between DSSD and veto events.
    
    Parameters:
    -----------
    dssd_events : pd.DataFrame
        DSSD events dataframe
    veto_events : pd.DataFrame
        Veto detector events dataframe
    time_window : float
        Time window for coincidence in seconds
        
    Returns:
    --------
    pd.DataFrame
        DSSD events with veto flag added
    """
    try:
        # If either dataframe is empty, return dssd_events with is_vetoed=False
        if len(dssd_events) == 0:
            return dssd_events
        if len(veto_events) == 0:
            dssd_events['is_vetoed'] = False
            return dssd_events
            
        # Convert timetags to time if not already done
        if 't' not in dssd_events.columns:
            dssd_events['t'] = np.round(dssd_events['tagx'] * 1e-12, 6)
        if 't' not in veto_events.columns:
            veto_events['t'] = np.round(veto_events['tagx'] * 1e-12, 6)
        
        # Create rounded time columns for merging
        dssd_events['t_round'] = np.round(dssd_events['t'], 5)
        veto_events['t_round'] = np.round(veto_events['t'], 5)
        
        # Find exact time matches
        merge_exact = pd.merge(
            dssd_events[['t', 'tagx', 't_round']], 
            veto_events[['t', 'tagx', 't_round']].rename(columns={'t': 't_veto', 'tagx': 'tagx_veto'}),
            on='t_round',
            how='left',
            suffixes=('', '_veto')
        )
        
        # Calculate time differences
        merge_exact['tdiff'] = np.abs(merge_exact['t'] - merge_exact['t_veto'])
        
        # Flag events within time window
        merge_exact['within_window'] = merge_exact['tdiff'] <= time_window
        
        # Group by dssd event and check if any veto event is within window
        vetoed_tags = merge_exact.groupby('tagx')['within_window'].any()
        
        # Add veto flag to original dataframe
        dssd_events['is_vetoed'] = dssd_events['tagx'].map(vetoed_tags).fillna(False)
        
        # Clean up
        dssd_events = dssd_events.drop('t_round', axis=1, errors='ignore')
        
        return dssd_events
        
    except Exception as e:
        print(f"Error in detmerge: {str(e)}")
        # If error occurs, assume no vetoed events
        dssd_events['is_vetoed'] = False
        return dssd_events

def process_file(csv_file, output_paths, shrec_map_path, calibration_path, save_all_events=False, ecut=50):
    """
    Process a single CSV file and extract all detector data.
    
    Parameters:
    -----------
    csv_file : str
        Path to CSV file to process
    output_paths : dict
        Dictionary of output file paths
    shrec_map_path : str
        Path to SHREC mapping file
    calibration_path : str
        Path to calibration file
    save_all_events : bool
        Whether to save all events (including vetoed)
    ecut : int
        Energy cut threshold
        
    Returns:
    --------
    dict
        Event counts by detector type
    """
    from shrec_utils import mapimp, mapboxE, mapboxW, mapboxT, mapboxB, mapveto
    
    try:
        log_message(f"Processing {csv_file}...", output_paths['log'])
        t_start = time.time()
        
        # Read the data in chunks to control memory usage
        chunk_size = 250000
        chunks = []
        
        log_message(f"Reading CSV in chunks of {chunk_size}...", output_paths['log'])
        
        for chunk in pd.read_csv(csv_file, chunksize=chunk_size):
            chunks.append(chunk)
            check_memory()
        
        # Combine chunks
        raw_df = pd.concat(chunks, ignore_index=True)
        log_message(f"Read {len(raw_df)} raw events", output_paths['log'])
        
        # Extract PPAC and Rutherford data before filtering
        ppac_data = extract_ppac_data(raw_df)
        ruth_data = extract_rutherford_data(raw_df)
        
        # Read SHREC map
        shrec_map = pd.read_excel(shrec_map_path, sheet_name=None)
        
        # Set up region processing functions
        region_funcs = {
            'imp': mapimp,
            'boxE': mapboxE,
            'boxW': mapboxW,
            'boxT': mapboxT,
            'boxB': mapboxB,
            'veto': mapveto
        }
        
        # Dictionary to store results
        shrec_results = {}
        
        # Process each detector region
        for region_name, region_func in region_funcs.items():
            log_message(f"Processing {region_name} region...", output_paths['log'])
            
            # Process this region
            processed = process_region_chunk(
                raw_df, region_func, shrec_map, calibration_path, 
                region_name, ecut=ecut
            )
            
            # Store if valid
            if processed is not None and len(processed) > 0:
                shrec_results[region_name] = processed
                log_message(f"Found {len(processed)} {region_name} events", output_paths['log'])
            else:
                shrec_results[region_name] = pd.DataFrame()
                log_message(f"No {region_name} events found", output_paths['log'])
        
        # Free memory
        del raw_df, shrec_map
        gc.collect()
        
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
        for region in region_funcs.keys():
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
    # Configuration variables (modify these as needed)
    file_list_path = 'long_run_4mbar_500V.txt'  # Path to file containing list of CSV files
    data_folder = '../ppac_data/'
    shrec_map_path = os.path.join(data_folder, 'r238_shrec_map.xlsx')
    calibration_path = os.path.join(data_folder, 'r238_calibration_v0_copy-from-r237.txt')
    output_folder = 'processed_data/'
    energy_cut = 50
    save_all_events = False  # Set to True if you want to save all events including vetoed ones
    clear_outputs = True     # Set to True to clear existing output files before processing
    
    # Memory settings
    max_memory_mb = 5120     # 5 GB
    memory_threshold_mb = max_memory_mb * 0.9
    
    # Optional processing range (for partial processing)
    start_from = 0           # Start from this file index
    end_at = None            # End at this file index (None = process all files)
    
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
    log_message(f"Maximum memory: {max_memory_mb} MB", output_paths['log'])
    
    # Load list of files to process
    try:
        csv_files = load_file_list(file_list_path)
        log_message(f"Found {len(csv_files)} files to process", output_paths['log'])
    except Exception as e:
        log_message(f"Error loading file list: {str(e)}", output_paths['log'])
        return
    
    # Apply start/end indices if specified
    if end_at is not None:
        csv_files = csv_files[start_from:end_at]
    elif start_from > 0:
        csv_files = csv_files[start_from:]
    
    log_message(f"Will process {len(csv_files)} files", output_paths['log'])
    
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