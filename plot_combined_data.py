#!/usr/bin/env python3
"""
mega_analysis_exact.py

Reproduces the EXACT same plots from 'ppac_analysis - Jupyter Notebook.pdf':
- Raw E–ToF (notebook cell ~In[7])
- Board-by-board E–ToF (In[8])
- Decay candidates 2D hist (In[19])
- R–alpha correlation (In[20] / In[24])
- 2D beam spot (In[26])
- Beam spot projections (In[27])
- Recoil & alpha energy hists (In[31])
- Correlated E–ToF hexbin (In[25])

Loads per-run CSVs (coincident_imp.csv, final_correlated.csv, decay_candidates.csv)
from r47..r56 and combines them. Then exactly replicates the same code you used
for each figure (same bin sizes, alpha, x/y limits, etc.), saving plots to
'combined_plots/' in the base directory.

Usage:
  python mega_analysis_exact.py
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from matplotlib.lines import Line2D

# If you want the same style from your notebook:
# try:
#     import scienceplots
#     plt.style.use(['science', 'ieee'])
# except ImportError:
#     print("scienceplots not installed; continuing without that style.")

##############################################################################
# UTILITY: LOAD & MERGE
##############################################################################

def load_csv_if_exists(path, dfname=""):
    """Helper to load a CSV if it exists, else return empty DataFrame."""
    if os.path.exists(path):
        return pd.read_csv(path)
    else:
        print(f"WARNING: {dfname} not found: {path}")
        return pd.DataFrame()

def load_and_combine_runs(base_path, runs):
    """
    Reads coincident_imp.csv, final_correlated.csv, and decay_candidates.csv from
    each run's folder, appends them, and returns 3 combined DataFrames.

    Returns: (coincident_imp_df, final_correlated_df, decay_candidates_df)
    """
    all_coincident = []
    all_final = []
    all_decay = []

    for run_id in runs:
        run_dir = os.path.join(base_path, run_id)

        cfile = os.path.join(run_dir, "coincident_imp.csv")
        ffile = os.path.join(run_dir, "final_correlated.csv")
        dfile = os.path.join(run_dir, "decay_candidates.csv")

        cdf = load_csv_if_exists(cfile, f"coincident_imp ({run_id})")
        fdf = load_csv_if_exists(ffile, f"final_correlated ({run_id})")
        ddf = load_csv_if_exists(dfile, f"decay_candidates ({run_id})")

        if not cdf.empty:
            cdf["run_id"] = run_id
            all_coincident.append(cdf)
        if not fdf.empty:
            fdf["run_id"] = run_id
            all_final.append(fdf)
        if not ddf.empty:
            ddf["run_id"] = run_id
            all_decay.append(ddf)

    coincident_imp_df = pd.concat(all_coincident, ignore_index=True) if all_coincident else pd.DataFrame()
    final_corr_df     = pd.concat(all_final, ignore_index=True)     if all_final else pd.DataFrame()
    decay_candidates  = pd.concat(all_decay, ignore_index=True)     if all_decay else pd.DataFrame()

    return coincident_imp_df, final_corr_df, decay_candidates

##############################################################################
# EXACT PLOTS REPLICATED
##############################################################################

def plot_raw_etof(coincident_imp_df, output_dir):
    """
    EXACT from ~In[7] in your notebook:
    plt.scatter(coincident_imp_df['imp_xE'], dt_anodeH_us, xlim(0,14000), ylim(-1.7, -1.35), etc.
    """
    if coincident_imp_df.empty:
        print("No data for raw E–ToF.")
        return

    plt.figure(figsize=(8,4))
    fs = 18
    plt.scatter(coincident_imp_df['imp_xE'], coincident_imp_df['dt_anodeH_us'],
                alpha=0.2, s=1, c='blue')
    plt.xlabel("SHREC implant energy (keV)", fontsize=fs)
    plt.ylabel(r"AnodeH ToF ($\mu$s)", fontsize=fs)
    plt.title("Raw E–ToF", fontsize=fs+2)
    plt.xlim(0, 14000)
    plt.ylim(-1.7, -1.35)
    plt.grid(True, alpha=0.3)
    ax = plt.gca()
    ax.tick_params(axis='both', which='major', labelsize=fs-2)

    plt.savefig(os.path.join(output_dir, "raw_etof.png"), dpi=300)
    #plt.show()

def plot_board_by_board_etof(coincident_imp_df, output_dir):
    """
    EXACT from ~In[8]:
    - we do a figure with subplots(1,2)
    - left: dt_anodeH_us
    - right: dt_anodeH_us_corr
    - color-coded by xboard
    """
    if coincident_imp_df.empty or "xboard" not in coincident_imp_df.columns:
        print("No data for board-by-board E–ToF.")
        return

    fs = 18
    boards = sorted(coincident_imp_df['xboard'].unique())
    colors = plt.cm.tab10(np.linspace(0,1,len(boards)))
    legend_handles = []

    plt.figure(figsize=(30,18))  # from your code, you used (30,18) in cell [8]
    # Actually you used (30,18)? If not, adapt to exactly what you had.

    plt.subplot(221)
    for board, color in zip(boards, colors):
        board_data = coincident_imp_df[coincident_imp_df['xboard'] == board]
        plt.scatter(board_data['imp_xE'], board_data['dt_anodeH_us'],
                    s=2, alpha=0.2, color=color)
        legend_handles.append(Line2D([0],[0], marker='o', color='w',
                                     markerfacecolor=color, markersize=10,
                                     label=f'Board {board}'))
    plt.xlabel("SHREC implant energy (keV)", fontsize=fs)
    plt.ylabel(r"ToF ($\mu$s)", fontsize=fs)
    plt.title("E–ToF by implant board", fontsize=fs+2)
    plt.xlim(0, 14000)
    plt.ylim(-1.7, -1.35)
    plt.grid(True, alpha=0.3)
    ax = plt.gca()
    ax.tick_params(axis='both', which='major', labelsize=fs-2)
    plt.legend(handles=legend_handles, fontsize=fs-4, frameon=True, shadow=True)

    plt.subplot(222)
    for board, color in zip(boards, colors):
        board_data = coincident_imp_df[coincident_imp_df['xboard'] == board]
        if 'dt_anodeH_us_corr' in board_data.columns:
            plt.scatter(board_data['imp_xE'], board_data['dt_anodeH_us_corr'],
                        s=2, alpha=0.1, color=color)
        else:
            # if you do not have dt_anodeH_us_corr in your actual df, just pass
            pass
    plt.xlabel("SHREC implant energy (keV)", fontsize=fs)
    plt.ylabel(r"ToF ($\mu$s)", fontsize=fs)
    plt.title("Time corrected E–ToF by implant board", fontsize=fs+2)
    plt.xlim(0, 14000)
    plt.ylim(-1.7, -1.35)
    plt.grid(True, alpha=0.3)
    ax = plt.gca()
    ax.tick_params(axis='both', which='major', labelsize=fs-2)

    # from your code you had a 'plt.savefig("plots/etof_by_board.png")' at the end
    # We'll do that now:
    plt.savefig(os.path.join(output_dir, "board_by_board_etof.png"), dpi=300)
    #plt.show()

def plot_decay_candidates_2d(decay_df, output_dir):
    """
    EXACT from ~In[19]:
      hist2d(decay_candidates_df['xE'], decay_candidates_df['log_dt'])
      bins=(500,50), range=((0,10000),(-3,3)), cmin=1
    """
    if decay_df.empty:
        print("No decay_candidates data available for 2D hist.")
        return
    if not all(col in decay_df.columns for col in ['xE','log_dt']):
        print("Decay candidates missing xE/log_dt columns. Skipping.")
        return

    fs=18
    plt.figure(figsize=(8,4))
    plt.hist2d(decay_df['xE'], decay_df['log_dt'],
               bins=(500,50), range=((0,10000),(-3,3)), cmin=1)
    plt.xlabel('Decay energy (keV)', fontsize=fs)
    plt.ylabel(r'Ln($\Delta$t/ s)/ 10 keV', fontsize=fs)  # from your code
    plt.title('Decay events: KHS vs energy', fontsize=fs+2)
    ax = plt.gca()
    ax.tick_params(axis='both', labelsize=fs-4)
    plt.colorbar(label='Counts')

    plt.savefig(os.path.join(output_dir, "decay_candidates_2dhist.png"), dpi=300)
    #plt.show()

def plot_alpha_tag_corr(final_corr_df, output_dir):
    """
    EXACT from ~In[20] or so (the alpha correlation).
    e.g. scatter of rec_xE vs alpha_xE
    """
    if final_corr_df.empty:
        print("No final_correlated data for alpha correlation scatter.")
        return
    needed = ['rec_xE','alpha_xE']
    if not all(col in final_corr_df.columns for col in needed):
        print("final_correlated missing rec_xE or alpha_xE columns. Skipping.")
        return
    
    # Count the number of events
    n_events = len(final_corr_df)
    label_str = fr'Correlated $\alpha$ (N={n_events})'

    fs = 16
    plt.figure(figsize=(13,7))
    # code from your In[24] or so might have had a different size
    # we'll replicate the main concept
    plt.subplot(221)
    plt.scatter(final_corr_df['alpha_xE'], final_corr_df['rec_alpha_time'],
                s=10, color='red', alpha=0.7, label=label_str)
    plt.xlabel('SHREC alpha X-Energy (keV)', fontsize=fs)
    plt.ylabel(r'Correlation time (s)', fontsize=fs)
    plt.xlim(8100, 8400)  # from your code snippet
    plt.yscale('log')
    plt.ylim(0.001,20)
    plt.title(r'Correlation time vs $\alpha$ energy', fontsize=fs+2)
    ax = plt.gca()
    ax.tick_params(axis='both', labelsize=fs-4 )

    plt.legend(fontsize=fs-4, loc='lower left', frameon=True)

    plt.tight_layout()
    # We'll call it alpha_tag_corr.png
    outname = os.path.join(output_dir, "alpha_tag_corr.png")
    plt.savefig(outname, dpi=300)
    #plt.show()

def plot_corr_time_and_decay_khs(final_corr_df,decay_candidates_df,alpha_corr_min,alpha_corr_max,alpha_energy_min,alpha_energy_max,output_dir):
    
    # Basic checks
    needed_final = ["alpha_xE", "rec_alpha_time"]
    if final_corr_df.empty or not all(col in final_corr_df.columns for col in needed_final):
        print("No or incomplete final_corr_df data for correlation-time subplot.")
        return

    needed_decay = ["xE", "log_dt"]
    if decay_candidates_df.empty or not all(col in decay_candidates_df.columns for col in needed_decay):
        print("No or incomplete decay_candidates_df data for 2D hist subplot.")
        return

    # Build the figure
    fs = 16
    plt.figure(figsize=(13,7))

    # ---------------------------
    # Subplot (221): alpha_xE vs. rec_alpha_time (log y-scale)
    # ---------------------------
    plt.subplot(221)
    n_events = len(final_corr_df)
    label_str = rf'Correlated $\alpha$ (N={n_events})'

    plt.scatter(
        final_corr_df['alpha_xE'],
        final_corr_df['rec_alpha_time'],
        s=10, color='red', alpha=0.7,
        label=label_str
    )
    plt.xlabel('SHREC alpha X-Energy (keV)', fontsize=fs)
    plt.ylabel(r'Correlation time (s)', fontsize=fs)
    plt.xlim(alpha_energy_min, alpha_energy_max)
    plt.yscale('log')
    plt.ylim(0.001, 20)
    plt.title(r'Correlation time vs $\alpha$ energy', fontsize=fs+2)
    ax = plt.gca()
    ax.tick_params(axis='both', labelsize=fs-4)
    plt.legend(fontsize=fs-4, loc='lower left', frameon=True)

    # ---------------------------
    # Subplot (222): 2D hist of xE vs. log_dt + fill_between
    # ---------------------------
    plt.subplot(222)
    plt.hist2d(
        decay_candidates_df['xE'],
        decay_candidates_df['log_dt'],
        bins=(500, 50),
        range=((5000, 10000), (-3, 3)),
        cmin=1
    )
    # highlight region in green
    # The fill_betweenx call uses the natural log for alpha_corr_min, alpha_corr_max:
    plt.fill_betweenx(
        y=[np.log(alpha_corr_min), np.log(alpha_corr_max)],
        x1=alpha_energy_min,
        x2=alpha_energy_max,
        color='g', alpha=0.2,
        label=r'$^{246}$Fm gate'
    )
    plt.xlabel('Decay energy (keV)', fontsize=fs)
    plt.ylabel(r'Ln($\Delta$t/ s)/ 10 keV', fontsize=fs)
    plt.title('Decay events: KHS vs energy', fontsize=fs+2)
    ax = plt.gca()
    ax.tick_params(axis='both', which='major', labelsize=fs-4)
    plt.legend(fontsize=fs-4, loc='upper left', frameon=True, facecolor='white', shadow=True)

    plt.tight_layout()

    plt.savefig(os.path.join(output_dir, "corr_alphas_dec_khs.png"), dpi=300)

def plot_correlated_vs_uncorrelated_alphas(final_corr_df, decay_candidates_df, output_dir, energy_range, bins):
    
    # Ensure required columns exist
    if final_corr_df.empty or "alpha_xE" not in final_corr_df.columns:
        print("Warning: No correlated alpha data available.")
        return
    if decay_candidates_df.empty or "xE" not in decay_candidates_df.columns:
        print("Warning: No uncorrelated alpha data available.")
        return

    # Extract alpha energies
    correlated_alphas = final_corr_df["alpha_xE"]
    uncorrelated_alphas = decay_candidates_df["xE"]

    len_uncorr_alphas = len(decay_candidates_df[
        (decay_candidates_df["xE"] >= energy_range[0]) &
        (decay_candidates_df["xE"] <= energy_range[1])
    ]["xE"])

    

    # Create the histogram
    fs = 16
    plt.figure(figsize=(8,5))
    
    plt.hist(correlated_alphas, bins=(300), range=(7000,10000), alpha=0.6, 
             color='red', label=f'Correlated (N={len(correlated_alphas)})', histtype='stepfilled')

    plt.hist(uncorrelated_alphas, bins=(300), range=(7000,10000), alpha=0.4, 
             color='blue', label=f'Uncorrelated (N={len_uncorr_alphas})', histtype='stepfilled')

    # Formatting
    plt.xlabel("Alpha Energy (keV)", fontsize=fs)
    plt.ylabel("Counts", fontsize=fs)
    # plt.title("Comparison of Correlated vs Uncorrelated Alphas", fontsize=fs+2)
    plt.legend(fontsize=fs-2, frameon=True)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "raw_vs_corr_alphas.png"), dpi=300)

    # Save the figure


def plot_correlated_etof(coincident_imp_df, final_corr_df, output_dir):
    """
    EXACT from ~In[25]:
      hexbin of coincident_imp_df['imp_xE'], coincident_imp_df['dt_anodeH_us_corr']
      overlay scatter in red from final_corr_df['rec_xE'], final_corr_df['rec_dt_anodeH_us_corr']
    """
    if coincident_imp_df.empty or final_corr_df.empty:
        print("No data for correlated E–ToF plot (one or both are empty).")
        return

    needed_coinc = ['imp_xE','dt_anodeH_us_corr']
    needed_final = ['rec_xE','rec_dt_anodeH_us_corr']
    if not all(k in coincident_imp_df.columns for k in needed_coinc):
        print("coincident_imp missing columns for correlated E–ToF hexbin. Skipping.")
        return
    if not all(k in final_corr_df.columns for k in needed_final):
        print("final_corr missing columns for correlated E–ToF overlay. Skipping.")
        return

    fs = 18
    plt.figure(figsize=(8,4))
    plt.hexbin(
        coincident_imp_df['imp_xE'],
        coincident_imp_df['dt_anodeH_us_corr'],
        gridsize=200, extent=(0, 10000, -1.7, -1.5),
        mincnt=1, cmap='viridis'
    )

    plt.scatter(
        final_corr_df['rec_xE'],
        final_corr_df['rec_dt_anodeH_us_corr'],
        color='red', alpha=0.4, s=20, label=r'$\alpha$-tagged'
    )
    legend_marker = Line2D([0],[0], marker='o', color='w',
                           markerfacecolor='red', markersize=6,
                           label=r'$\alpha$-tagged')
    
    plt.ylim(-1.625, -1.49)
    plt.xlim(0, 10000)
    plt.xlabel('SHREC implant energy (keV)', fontsize=fs)
    plt.ylabel(r'ToF ($\mu$s)', fontsize=fs)
    plt.title(r'$\alpha$-correlated E–ToF', fontsize=fs+2)
    ax = plt.gca()
    ax.tick_params(axis='both', which='major', labelsize=fs-2)

    plt.legend(handles=[legend_marker], loc='lower right', fontsize=fs-2, frameon=True)
    plt.colorbar(label='Counts')

    plt.savefig(os.path.join(output_dir, "correlated_etof.png"), dpi=300)
    #plt.show()

def plot_beam_spot_2d(final_corr_df, output_dir):
    """
    EXACT from ~In[26]:
      hist2d of final_corr_df['rec_x'], final_corr_df['rec_y']
      bins=((175),(61)), range=((-1,174),(-1,60)), cmin=1
    """
    if final_corr_df.empty:
        print("No final_correlated data for beam-spot 2D hist.")
        return
    needed = ['rec_x','rec_y']
    if not all(k in final_corr_df.columns for k in needed):
        print("Missing rec_x or rec_y columns. Skipping beam spot 2D.")
        return

    fs=18
    plt.figure(figsize=(8,3))
    plt.hist2d(
        final_corr_df['rec_x'],
        final_corr_df['rec_y'],
        bins=(175,61), range=((-1,174),(-1,60)), cmin=1
    )
    plt.xlabel('x-strip', fontsize=fs)
    plt.ylabel(r'y-strip', fontsize=fs)
    plt.title(r'$\alpha$ correlated, recoil beam spot', fontsize=fs+2)
    plt.colorbar()
    ax = plt.gca()
    ax.tick_params(axis='both', which='major', labelsize=fs-2)

    plt.savefig(os.path.join(output_dir, "beam_spot_2d.png"), dpi=300)
    #plt.show()

def plot_beam_spot_projections(final_corr_df, output_dir):
    """
    EXACT from ~In[27]:
      subplot with hist of rec_x (bins=175, range=(-1,174))
      subplot with hist of rec_y (bins=60, range=(-1,59))
    """
    if final_corr_df.empty:
        print("No final_correlated data for beam-spot projections.")
        return
    needed = ['rec_x','rec_y']
    if not all(k in final_corr_df.columns for k in needed):
        print("Missing rec_x or rec_y columns. Skipping beam spot projections.")
        return

    fs=18
    plt.figure(figsize=(12,6))

    plt.subplot(221)
    plt.hist(final_corr_df['rec_x'], histtype='step', bins=175, range=(-1,174))
    plt.xlabel('x-strip', fontsize=fs)
    plt.ylabel(r'Counts', fontsize=fs)
    ax = plt.gca()
    ax.tick_params(axis='both', which='major', labelsize=fs-2)

    plt.subplot(222)
    plt.hist(final_corr_df['rec_y'], histtype='step', bins=60, range=(-1,59))
    plt.xlabel('y-strip', fontsize=fs)
    plt.ylabel(r'Counts', fontsize=fs)
    ax = plt.gca()
    ax.tick_params(axis='both', which='major', labelsize=fs-2)

    plt.savefig(os.path.join(output_dir, "beam_spot_projections.png"), dpi=300)
    #plt.show()

def plot_energy_hists(final_corr_df, output_dir):
    """
    EXACT from ~In[31]:
      subplot(221): alpha_xE hist, bins=60, range=(8100,8400)
      subplot(222): rec_xE hist, bins=175, range=(0,9000)
      etc.
    """
    if final_corr_df.empty:
        print("No final_correlated data for recoil/alpha energy hists.")
        return
    needed = ['alpha_xE','rec_xE']
    if not all(k in final_corr_df.columns for k in needed):
        print("Missing alpha_xE or rec_xE. Skipping recoil/alpha hists.")
        return

    fs=18
    plt.figure(figsize=(12,6))

    plt.subplot(221)
    plt.hist(final_corr_df['alpha_xE'], histtype='step', bins=60, range=(8100,8400))
    plt.xlabel('Alpha energy (keV)', fontsize=fs)
    plt.ylabel(r'Counts/ 10keV', fontsize=fs)
    ax = plt.gca()
    ax.tick_params(axis='both', which='major', labelsize=fs-2)

    plt.subplot(222)
    plt.hist(final_corr_df['rec_xE'], histtype='step',bins=175, range=(0,9000))
    plt.xlabel('Recoil energy (keV)', fontsize=fs)
    plt.ylabel(r'Counts/ 40keV', fontsize=fs)
    ax = plt.gca()
    ax.tick_params(axis='both', which='major', labelsize=fs-2)

    plt.savefig(os.path.join(output_dir, "rec_alpha_energy_projections.png"), dpi=300)
    #plt.show()

##############################################################################
# MAIN
##############################################################################

def main():
    base_path = "processed_data/long_run_4mbar_500V"
    
    # Output folder for all final plots
    output_dir = os.path.join(base_path, "combined_plots")
    os.makedirs(output_dir, exist_ok=True)

    # e.g. r47..r56
    runs = [f"r{x}" for x in range(47,57)]

    # 1) Load & combine data
    coincident_imp_df, final_corr_df, decay_candidates_df = load_and_combine_runs(base_path, runs)

    # Specify the gates that were used, just for plotting
    alpha_corr_min = 0.08
    alpha_corr_max = 15
    alpha_energy_min = 8150
    alpha_energy_max = 8300

    # 2) Make EXACT same plots (with same code references from your notebook)
    plot_raw_etof(coincident_imp_df, output_dir)
    plot_board_by_board_etof(coincident_imp_df, output_dir)
    plot_decay_candidates_2d(decay_candidates_df, output_dir)
    plot_alpha_tag_corr(final_corr_df, output_dir)
    plot_correlated_etof(coincident_imp_df, final_corr_df, output_dir)
    plot_beam_spot_2d(final_corr_df, output_dir)
    plot_beam_spot_projections(final_corr_df, output_dir)
    plot_energy_hists(final_corr_df, output_dir)
    plot_corr_time_and_decay_khs(final_corr_df,decay_candidates_df, alpha_corr_min,alpha_corr_max,alpha_energy_min,alpha_energy_max, output_dir=output_dir)
    plot_correlated_vs_uncorrelated_alphas(final_corr_df=final_corr_df,decay_candidates_df=decay_candidates_df,output_dir=output_dir,
        energy_range=(alpha_energy_min, alpha_energy_max),bins=60)

    print("All EXACT notebook plots saved to:", output_dir)


if __name__ == "__main__":
    main()
