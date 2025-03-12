import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
import time
import gc
import numba
import scienceplots

plt.style.use('science')

# =============================================================================
# 1. DATA LOADING WITH MEMORY OPTIMIZATIONS
# =============================================================================

# Define memory-efficient dtypes for each CSV file.
dssd_dtypes = {
    'event_type': 'category',  # limited set of types
    'x': 'int32',
    'y': 'int32',
    'tagx': 'int64',
    'tagy': 'int64',
    'nfile': 'int16',
    'tdelta': 'float32',
    'nX': 'int16',
    'nY': 'int16',
    'xE': 'float32',
    'yE': 'float32',
    'xboard': 'int8',
    'yboard': 'int8',
}
ppac_dtypes = {
    'detector': 'category',
    'timetag': 'int64',
    'energy': 'float32',
    'board': 'int8',
    'channel': 'int8',
    'nfile': 'int16',
}
ruth_dtypes = {
    'detector': 'category',
    'timetag': 'int64',
    'energy': 'float32',
    'board': 'int8',
    'channel': 'int8',
    'nfile': 'int16',
}

# Read CSV files. (For very large files consider adding chunksize)
dssd = pd.read_csv('processed_data/dssd_non_vetoed_events.csv', dtype=dssd_dtypes)
ppac = pd.read_csv('processed_data/ppac_events.csv', dtype=ppac_dtypes)
ruth = pd.read_csv('processed_data/rutherford_events.csv', dtype=ruth_dtypes)

# =============================================================================
# 2. DATA SEGREGATION AND SORTING
# =============================================================================

# Split DSSD data into regions
imp = dssd[dssd['event_type'] == 'imp']
boxE = dssd[dssd['event_type'] == 'boxE']
boxW = dssd[dssd['event_type'] == 'boxW']
boxT = dssd[dssd['event_type'] == 'boxT']
boxB = dssd[dssd['event_type'] == 'boxB']

# Split PPAC data by detector type
cathode = ppac[ppac['detector'] == 'cathode']
anodeV = ppac[ppac['detector'] == 'anodeV']
anodeH = ppac[ppac['detector'] == 'anodeH']

# Split Rutherford data
ruth_E = ruth[ruth['detector'] == 'ruthE']
ruth_W = ruth[ruth['detector'] == 'ruthW']

# Define the coincidence window (in ns) and convert to ps
window_before_ns = 1700  # 1.7 µs before
window_after_ns = 0      # adjust if needed
window_before_ps = window_before_ns * 1000  # convert ns to ps
window_after_ps = window_after_ns * 1000

# Sort the relevant DataFrames by time (or tagx for imp)
cathode_sorted = cathode.sort_values('timetag').reset_index(drop=True)
anodeV_sorted = anodeV.sort_values('timetag').reset_index(drop=True)
anodeH_sorted = anodeH.sort_values('timetag').reset_index(drop=True)
imp_sorted = imp.sort_values('tagx').reset_index(drop=True)

# Create a column 't' (time in seconds) for the IMP events (needed for later decay analysis)
imp_sorted['t'] = imp_sorted['tagx'] / 1e12  # converting picoseconds to seconds

# Cache timetag arrays for PPAC detectors for fast binary search
cathode_timetags = cathode_sorted['timetag'].values
anodeV_timetags = anodeV_sorted['timetag'].values
anodeH_timetags = anodeH_sorted['timetag'].values

# =============================================================================
# 3. DEFINE BINARY SEARCH FUNCTION FOR TIME WINDOWS
# =============================================================================

def find_events_in_window(imp_timetag, detector_timetags, window_before_ps, window_after_ps):
    """
    Find detector events within a specified time window around an IMP event using binary search.
    
    Parameters:
      imp_timetag (int): Timestamp of the IMP event (in ps).
      detector_timetags (ndarray): Sorted array of detector timestamps (in ps).
      window_before_ps (int): Time window before the IMP event (in ps).
      window_after_ps (int): Time window after the IMP event (in ps).
    
    Returns:
      list: Indices of events within the time window.
    """
    lower_bound = imp_timetag - window_before_ps
    upper_bound = imp_timetag + window_after_ps
    lower_idx = np.searchsorted(detector_timetags, lower_bound)
    upper_idx = np.searchsorted(detector_timetags, upper_bound)
    if upper_idx > lower_idx:
        return list(range(lower_idx, upper_idx))
    return []

# =============================================================================
# 4. COINCIDENCE SEARCH BETWEEN IMP AND PPAC EVENTS
# =============================================================================

start_time = time.time()
coincident_events = []
non_ppac_coincident_events = []
total_imp_events = len(imp_sorted)
print(f"Processing {total_imp_events} IMP events...")

for idx, imp_row in imp_sorted.iterrows():
    imp_timetag = imp_row['tagx']  # using tagx as the IMP time
    # Find PPAC events within the specified time window
    cathode_indices = find_events_in_window(imp_timetag, cathode_timetags, window_before_ps, window_after_ps)
    anodeV_indices = find_events_in_window(imp_timetag, anodeV_timetags, window_before_ps, window_after_ps)
    anodeH_indices = find_events_in_window(imp_timetag, anodeH_timetags, window_before_ps, window_after_ps)
    
    if cathode_indices and anodeV_indices and anodeH_indices:
        # Find the closest PPAC event for each detector
        cathode_diffs = np.abs(cathode_timetags[cathode_indices] - imp_timetag)
        anodeV_diffs = np.abs(anodeV_timetags[anodeV_indices] - imp_timetag)
        anodeH_diffs = np.abs(anodeH_timetags[anodeH_indices] - imp_timetag)
        closest_cathode_idx = cathode_indices[np.argmin(cathode_diffs)]
        closest_anodeV_idx = anodeV_indices[np.argmin(anodeV_diffs)]
        closest_anodeH_idx = anodeH_indices[np.argmin(anodeH_diffs)]
        
        # Retrieve the corresponding rows
        cathode_data = cathode_sorted.iloc[closest_cathode_idx]
        anodeV_data = anodeV_sorted.iloc[closest_anodeV_idx]
        anodeH_data = anodeH_sorted.iloc[closest_anodeH_idx]
        
        # Compute time differences (in ps and then ns)
        dt_cathode_ps = cathode_data['timetag'] - imp_timetag
        dt_anodeV_ps = anodeV_data['timetag'] - imp_timetag
        dt_anodeH_ps = anodeH_data['timetag'] - imp_timetag
        
        event_data = {
            # IMP data
            'imp_timetag': imp_timetag,
            'imp_x': imp_row['x'],
            'imp_y': imp_row['y'],
            'imp_tagx': imp_row['tagx'],
            'imp_tagy': imp_row['tagy'],
            'imp_nfile': imp_row['nfile'],
            'imp_tdelta': imp_row['tdelta'],
            'imp_nX': imp_row['nX'],
            'imp_nY': imp_row['nY'],
            'imp_xE': imp_row['xE'],
            'imp_yE': imp_row['yE'],
            'xboard': imp_row['xboard'],
            'yboard': imp_row['yboard'],
            # PPAC data
            'cathode_timetag': cathode_data['timetag'],
            'cathode_energy': cathode_data['energy'],
            'cathode_board': cathode_data['board'],
            'cathode_channel': cathode_data['channel'],
            'cathode_nfile': cathode_data['nfile'],
            'anodeV_timetag': anodeV_data['timetag'],
            'anodeV_energy': anodeV_data['energy'],
            'anodeV_board': anodeV_data['board'],
            'anodeV_channel': anodeV_data['channel'],
            'anodeV_nfile': anodeV_data['nfile'],
            'anodeH_timetag': anodeH_data['timetag'],
            'anodeH_energy': anodeH_data['energy'],
            'anodeH_board': anodeH_data['board'],
            'anodeH_channel': anodeH_data['channel'],
            'anodeH_nfile': anodeH_data['nfile'],
            # Time differences
            'dt_cathode_ps': dt_cathode_ps,
            'dt_anodeV_ps': dt_anodeV_ps,
            'dt_anodeH_ps': dt_anodeH_ps,
            'dt_cathode_ns': dt_cathode_ps / 1000,
            'dt_anodeV_ns': dt_anodeV_ps / 1000,
            'dt_anodeH_ns': dt_anodeH_ps / 1000,
        }
        coincident_events.append(event_data)
    else:
        # Record IMP events with no PPAC coincidences
        non_coincident_data = {
            'timetag': imp_timetag,
            't': imp_timetag / 1e12,
            'x': imp_row['x'],
            'y': imp_row['y'],
            'tagx': imp_row['tagx'],
            'tagy': imp_row['tagy'],
            'nfile': imp_row['nfile'],
            'tdelta': imp_row['tdelta'],
            'nX': imp_row['nX'],
            'nY': imp_row['nY'],
            'xE': imp_row['xE'],
            'yE': imp_row['yE'],
            'xboard': imp_row['xboard'],
            'yboard': imp_row['yboard'],
        }
        non_ppac_coincident_events.append(non_coincident_data)
    
    # Print progress every 10,000 events
    if idx % 10000 == 0 and idx > 0:
        elapsed = time.time() - start_time
        events_per_sec = idx / elapsed
        remaining_time = (total_imp_events - idx) / events_per_sec 
        print(f"Processed {idx}/{total_imp_events} events ({idx/total_imp_events:.1%}) - ETA: {remaining_time:.1f} sec")

# Create DataFrames from the coincidence lists
coincident_imp_df = pd.DataFrame(coincident_events)
non_coincident_imp_df = pd.DataFrame(non_ppac_coincident_events)
print(f"Found {len(coincident_imp_df)} coincidences within the window")
elapsed_time = time.time() - start_time
print(f"Total processing time: {elapsed_time:.2f} seconds")
print(f"Processing rate: {total_imp_events/elapsed_time:.1f} events/sec")

# Free large objects that are no longer needed
del dssd, ppac, ruth, cathode, anodeV, anodeH, imp
gc.collect()

# =============================================================================
# 5. PLOTTING RAW E-ToF AND TIME CORRECTIONS
# =============================================================================

# Convert time differences from ps to µs for plotting convenience
coincident_imp_df['dt_cathode_us'] = coincident_imp_df['dt_cathode_ps'] / 1000
coincident_imp_df['dt_anodeV_us'] = coincident_imp_df['dt_anodeV_ps'] / 1000
coincident_imp_df['dt_anodeH_us'] = coincident_imp_df['dt_anodeH_ps'] / 1000

# Plot raw E-ToF
plt.figure(figsize=(8, 4))
fs = 18
plt.scatter(coincident_imp_df['imp_xE'], coincident_imp_df['dt_anodeH_us'],
            alpha=0.2, s=1, c='blue')
plt.xlabel("SHREC implant energy (keV)", fontsize=fs)
plt.ylabel(r"AnodeH ToF ($\mu$s)", fontsize=fs)
plt.title("Raw E-ToF", fontsize=fs+2)
plt.xlim(0, 14000)
plt.ylim(-1.7, -1.35)
plt.grid(True, alpha=0.3)
plt.savefig("plots/raw_etof.pdf", dpi=1000)

# Apply manual board offsets using vectorized mapping
manual_offsets = {
    0: 0,
    1: -0.045e-6,
    2: -0.065e-6,
    3: -0.085e-6,
    4: -0.105e-6,
    5: -0.125e-6,
}
coincident_imp_df['dt_anodeH_us_corr'] = (coincident_imp_df['dt_anodeH_us'] +
                                          coincident_imp_df['xboard'].map(manual_offsets))
coincident_imp_df['dt_anodeV_us_corr'] = (coincident_imp_df['dt_anodeV_us'] +
                                          coincident_imp_df['xboard'].map(manual_offsets))
coincident_imp_df['dt_cathode_us_corr'] = (coincident_imp_df['dt_cathode_us'] +
                                           coincident_imp_df['xboard'].map(manual_offsets))

# (Additional plots by board can be inserted here following similar practices.)
# For brevity, only the raw E-ToF and correction steps are shown.

# =============================================================================
# 6. DECAY EVENT CANDIDATE IDENTIFICATION
# =============================================================================

# Build pixel history from the IMP events (using imp_sorted that we prepared earlier)
# Group by pixel (x, y) and ensure each group is sorted by time 't'
pixel_groups = imp_sorted.groupby(['x', 'y'])
pixel_history = {pixel: group.sort_values('t') for pixel, group in pixel_groups}

# Define decay time window (in seconds)
min_corr_time = 1e-8  # Minimum time after recoil to consider (10 ns)
max_corr_time = 20    # Maximum time after recoil (20 s)

decay_candidates = []  # List to store candidate decay events

# Loop through each recoil (coincident IMP event)
for recoil_idx, recoil in coincident_imp_df.iterrows():
    pixel = (recoil['imp_x'], recoil['imp_y'])
    recoil_time_sec = recoil['imp_timetag'] / 1e12  # convert to seconds
    if pixel not in pixel_history:
        continue  # No events for this pixel in history
    pixel_df = pixel_history[pixel]
    time_array = pixel_df['t'].values  # Already in seconds
    lower_bound = recoil_time_sec + min_corr_time
    upper_bound = recoil_time_sec + max_corr_time
    start_idx = np.searchsorted(time_array, lower_bound, side='left')
    end_idx = np.searchsorted(time_array, upper_bound, side='right')
    if start_idx < end_idx:
        candidate_events = pixel_df.iloc[start_idx:end_idx].copy()
        candidate_events['recoil_index'] = recoil_idx
        candidate_events['recoil_time_sec'] = recoil_time_sec
        decay_candidates.append(candidate_events)

if decay_candidates:
    decay_candidates_df = pd.concat(decay_candidates, ignore_index=True)
else:
    decay_candidates_df = pd.DataFrame()

print("First few decay candidates:")
print(decay_candidates_df.head())

# =============================================================================
# 7. PPAC ANTICOINCIDENCE CHECK FOR DECAYS
# =============================================================================

if not decay_candidates_df.empty and not non_coincident_imp_df.empty:
    non_coincident_clean = non_coincident_imp_df[['x', 'y']].drop_duplicates()
    decay_candidates_df = decay_candidates_df.merge(
        non_coincident_clean,
        on=['x', 'y'],
        how='left',
        indicator='ppac_flag'
    )
    decay_candidates_df['is_clean'] = decay_candidates_df['ppac_flag'] == 'left_only'
    print("PPAC anticoincidence check counts:")
    print(decay_candidates_df['is_clean'].value_counts())
    print(decay_candidates_df.head())
else:
    print("No decay candidates or non-coincident events available for PPAC anticoincidence check.")

# Calculate log time difference between decay candidate and recoil event
if not decay_candidates_df.empty:
    decay_candidates_df['log_dt'] = np.log(np.abs(decay_candidates_df['t'] - decay_candidates_df['recoil_time_sec']))
    # Plotting 2D histogram for decay events (e.g., decay energy vs. log time difference)
    plt.figure(figsize=(8, 4))
    plt.hist2d(decay_candidates_df['yE'], decay_candidates_df['log_dt'],
               bins=(500, 50), range=((0, 10000), (-3, 3)), cmin=1)
    plt.xlabel('Decay energy (keV)', fontsize=fs)
    plt.ylabel(r'Ln($\Delta$t/ s)/ 10 keV', fontsize=fs)
    plt.title('Decay events: KHS vs energy', fontsize=fs+2)
    plt.savefig('plots/decay_khs.pdf', dpi=1000)

# =============================================================================
# 8. ALPHA ENERGY, TIME GATES, AND CORRELATION WITH RECOILS
# =============================================================================

# Set energy and time gates
alpha_energy_min = 8100    # Minimum alpha energy (keV)
alpha_energy_max = 8400    # Maximum alpha energy (keV)
recoil_energy_min = 2000   # Minimum recoil energy (keV)
recoil_energy_max = 8099   # Maximum recoil energy (keV)
alpha_corr_min = 0.08      # Minimum time difference in seconds
alpha_corr_max = 10        # Maximum time difference in seconds

# Filter alpha candidates by energy on the 'xE' column
filtered_alpha_candidates = decay_candidates_df[
    (decay_candidates_df['xE'] >= alpha_energy_min) & 
    (decay_candidates_df['xE'] <= alpha_energy_max)
].copy()

# Ensure there is a time column; it should already be there from imp_sorted grouping
if 't' not in filtered_alpha_candidates.columns:
    filtered_alpha_candidates['t'] = filtered_alpha_candidates['tagx'] / 1e12

# Initialize new columns for recoil correlation
filtered_alpha_candidates['closest_recoil_index'] = np.nan
filtered_alpha_candidates['recoil_time'] = np.nan
filtered_alpha_candidates['time_difference'] = np.nan
filtered_alpha_candidates['recoil_energy'] = np.nan

# For each alpha candidate, find the preceding recoil in the same pixel from the coincident events
for idx, alpha in filtered_alpha_candidates.iterrows():
    pixel_x = alpha['x']
    pixel_y = alpha['y']
    alpha_time = alpha['t']
    
    # Retrieve all recoil events in the same pixel from the coincident IMP events
    recoils_in_pixel = coincident_imp_df[
        (coincident_imp_df['imp_x'] == pixel_x) & (coincident_imp_df['imp_y'] == pixel_y)
    ]
    # Apply recoil energy gate on the recoil energy (imp_xE)
    recoils_in_pixel = recoils_in_pixel[
        (recoils_in_pixel['imp_xE'] >= recoil_energy_min) &
        (recoils_in_pixel['imp_xE'] <= recoil_energy_max)
    ]
    # Only consider recoils that occurred before the alpha candidate
    recoils_before = recoils_in_pixel[recoils_in_pixel['t'] < alpha_time]
    
    if not recoils_before.empty:
        recoils_before = recoils_before.copy()
        recoils_before['time_diff'] = alpha_time - recoils_before['t']
        # Ensure the time difference is within the correlation window
        recoils_in_window = recoils_before[
            (recoils_before['time_diff'] >= alpha_corr_min) &
            (recoils_before['time_diff'] <= alpha_corr_max)
        ]
        if not recoils_in_window.empty:
            # Choose the recoil event with the smallest time difference
            closest_recoil = recoils_in_window.loc[recoils_in_window['time_diff'].idxmin()]
            filtered_alpha_candidates.at[idx, 'closest_recoil_index'] = closest_recoil.name
            filtered_alpha_candidates.at[idx, 'recoil_time'] = closest_recoil['t']
            filtered_alpha_candidates.at[idx, 'time_difference'] = closest_recoil['time_diff']
            filtered_alpha_candidates.at[idx, 'recoil_energy'] = closest_recoil['imp_xE']
        else:
            filtered_alpha_candidates.at[idx, 'closest_recoil_index'] = np.nan
            filtered_alpha_candidates.at[idx, 'recoil_time'] = np.nan
            filtered_alpha_candidates.at[idx, 'time_difference'] = np.nan
            filtered_alpha_candidates.at[idx, 'recoil_energy'] = np.nan
    else:
        filtered_alpha_candidates.at[idx, 'closest_recoil_index'] = np.nan
        filtered_alpha_candidates.at[idx, 'recoil_time'] = np.nan
        filtered_alpha_candidates.at[idx, 'time_difference'] = np.nan
        filtered_alpha_candidates.at[idx, 'recoil_energy'] = np.nan

# Build the correlated events DataFrame by dropping rows with no recoil correlation
correlated_events = filtered_alpha_candidates.dropna(subset=['closest_recoil_index'])
print("Number of correlated alpha-recoil events:", len(correlated_events))
print(correlated_events.head())

# =============================================================================
# 9. MERGE RECOIL AND ALPHA INFORMATION FOR FINAL CORRELATION ANALYSIS
# =============================================================================

# Rename columns in coincident_imp_df (recoil info) with a prefix "rec_"
recoil_rename = {
    'imp_timetag': 'rec_timetag',
    'imp_x': 'rec_x',
    'imp_y': 'rec_y',
    'imp_tagx': 'rec_tagx',
    'imp_tagy': 'rec_tagy',
    'imp_nfile': 'rec_nfile',
    'imp_tdelta': 'rec_tdelta',
    'imp_nX': 'rec_nX',
    'imp_nY': 'rec_nY',
    'imp_xE': 'rec_xE',
    'imp_yE': 'rec_yE',
    'xboard': 'rec_xboard',
    'yboard': 'rec_yboard',
    'cathode_timetag': 'rec_cathode_timetag',
    'cathode_energy': 'rec_cathode_energy',
    'cathode_board': 'rec_cathode_board',
    'cathode_channel': 'rec_cathode_channel',
    'cathode_nfile': 'rec_cathode_nfile',
    'anodeV_timetag': 'rec_anodeV_timetag',
    'anodeV_energy': 'rec_anodeV_energy',
    'anodeV_board': 'rec_anodeV_board',
    'anodeV_channel': 'rec_anodeV_channel',
    'anodeV_nfile': 'rec_anodeV_nfile',
    'anodeH_timetag': 'rec_anodeH_timetag',
    'anodeH_energy': 'rec_anodeH_energy',
    'anodeH_board': 'rec_anodeH_board',
    'anodeH_channel': 'rec_anodeH_channel',
    'anodeH_nfile': 'rec_anodeH_nfile',
    't': 'rec_t',
    'dt_anodeH_us_corr': 'rec_dt_anodeH_us_corr',
    'dt_anodeV_us_corr': 'rec_dt_anodeV_us_corr',
    'dt_cathode_us_corr': 'rec_dt_cathode_us_corr'
}
recoil_df_renamed = coincident_imp_df.copy().rename(columns=recoil_rename)

# Rename columns in the alpha (correlated) events with prefix "alpha_"
alpha_rename = {
    't': 'alpha_t',
    'x': 'alpha_x',
    'y': 'alpha_y',
    'tagx': 'alpha_tagx',
    'tagy': 'alpha_tagy',
    'nfile': 'alpha_nfile',
    'xboard': 'alpha_xboard',
    'yboard': 'alpha_yboard',
    'tdelta': 'alpha_tdelta',
    'nX': 'alpha_nX',
    'nY': 'alpha_nY',
    'xE': 'alpha_xE',
    'yE': 'alpha_yE',
    'recoil_index': 'alpha_recoil_index',
    'recoil_time_sec': 'alpha_recoil_time',
    'time_difference': 'alpha_time_difference',
    'recoil_energy': 'alpha_recoil_energy'
}
alpha_df_renamed = correlated_events.copy().rename(columns=alpha_rename)

# Merge the alpha and recoil DataFrames using the recoil index
final_correlated_df = alpha_df_renamed.merge(
    recoil_df_renamed,
    left_on='alpha_recoil_index',
    right_index=True,
    how='left'
)

print("Final correlated events dataframe:")
print(final_correlated_df.head())
print("Pixel match check (alpha vs. recoil):")
print(final_correlated_df[['alpha_x', 'alpha_y', 'rec_x', 'rec_y']].head())

# Compute additional columns if needed
final_correlated_df['log_dt'] = np.log10(np.abs(final_correlated_df['alpha_t'] - final_correlated_df['rec_t']))
final_correlated_df['rec_alpha_time'] = np.abs(final_correlated_df['alpha_t'] - final_correlated_df['rec_t'])

# =============================================================================
# 10. PLOTTING CORRELATED RESULTS
# =============================================================================

fs = 16
plt.figure(figsize=(13, 7))
plt.subplot(221)
plt.scatter(final_correlated_df['alpha_xE'], final_correlated_df['rec_alpha_time'],
            s=10, color='red', alpha=0.7, label=r'Correlated $\alpha$')
plt.xlabel('SHREC alpha X-Energy (keV)', fontsize=fs)
plt.ylabel('Correlation time (s)', fontsize=fs)
plt.xlim(8100, 8400)
plt.yscale('log')
plt.legend(fontsize=fs-4, loc='lower left', frameon=True)
plt.ylim(0.001, 20)
plt.title(r'Correlation time vs $\alpha$ energy', fontsize=fs+2)

plt.subplot(222)
plt.hist2d(decay_candidates_df['xE'], decay_candidates_df['log_dt'],
           bins=(500, 50), range=((5000, 10000), (-3, 3)))
plt.fill_betweenx(y=[np.log(alpha_corr_min), np.log(alpha_corr_max)], x1=5000, x2=10000,
                  color='g', alpha=0.2, label=r'$^{246}$Fm gate')
plt.xlabel('Decay energy (keV)', fontsize=fs)
plt.ylabel(r'Ln($\Delta$t/ s)/ 10 keV', fontsize=fs)
plt.title('Decay events: KHS vs energy', fontsize=fs+2)
plt.legend(fontsize=fs-4, loc='upper left', frameon=True)
plt.savefig('plots/log_time_corr_alphas.pdf', dpi=300)

# Correlated E-ToF plot
plt.figure(figsize=(8, 4))
plt.hexbin(coincident_imp_df['imp_xE'], coincident_imp_df['dt_anodeH_us'],
           gridsize=200, extent=(0, 10000, -1.7, -1.5), mincnt=1)
plt.scatter(final_correlated_df['rec_xE'], final_correlated_df['rec_dt_anodeH_us_corr'],
            color='red', alpha=0.4, s=20, label=r'$\alpha$-tagged')
plt.ylim(-1.625, -1.49)
plt.xlim(0, 10000)
plt.xlabel('SHREC implant energy (keV)', fontsize=fs)
plt.ylabel(r'ToF ($\mu$s/ 50 keV)', fontsize=fs)
plt.title(r'$\alpha$-correlated E-ToF', fontsize=fs+2)
plt.legend(loc='lower right', fontsize=fs-4)
plt.savefig('plots/correlated_etof.pdf', dpi=300)

# Correlated beam spot
plt.figure(figsize=(8, 3))
plt.hist2d(final_correlated_df['rec_x'], final_correlated_df['rec_y'],
           bins=(175, 61), range=((-1, 174), (-1, 60)), cmin=1)
plt.xlabel('x-strip', fontsize=fs)
plt.ylabel('y-strip', fontsize=fs)
plt.title(r'$\alpha$ correlated, recoil beam spot', fontsize=fs+2)
plt.colorbar()
plt.savefig('plots/correlated_stripX_stripY.pdf', dpi=300)

# Beam spot projections
plt.figure(figsize=(12, 6))
plt.subplot(221)
plt.hist(final_correlated_df['rec_x'], histtype='step', bins=175, range=(-1, 174))
plt.xlabel('x-strip', fontsize=fs)
plt.ylabel('Counts', fontsize=fs)
plt.subplot(222)
plt.hist(final_correlated_df['rec_y'], histtype='step', bins=60, range=(-1, 60))
plt.xlabel('y-strip', fontsize=fs)
plt.ylabel('Counts', fontsize=fs)
plt.savefig('plots/correlated_beam_spot_projections.pdf', dpi=300)

# Recoil and alpha energy histograms
plt.figure(figsize=(12, 6))
plt.subplot(221)
plt.hist(final_correlated_df['alpha_xE'], histtype='step', bins=60)
plt.xlabel('Alpha energy (keV)', fontsize=fs)
plt.ylabel('Counts / 10 keV', fontsize=fs)
plt.subplot(222)
plt.hist(final_correlated_df['rec_xE'], histtype='step', bins=175, range=(0, 10000))
plt.xlabel('Recoil energy (keV)', fontsize=fs)
plt.ylabel('Counts / 40 keV', fontsize=fs)
plt.savefig('plots/rec_alpha_energy_projections.pdf', dpi=300)

plt.show()
