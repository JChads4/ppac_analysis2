"""
shrec_utils.py

Utility functions for mapping and calibrating SHREC (DSSD) data.
You can add additional utility functions (e.g., mapboxE, mapboxW, mapboxT, etc.) here.
"""

import pandas as pd
import os
def mapimp(dataframe, shrec_map):
    """
    Maps 'IMP' events in the DSSD (front vs back strips).
    
    Parameters:
    -----------
    dataframe : pd.DataFrame
        Raw data containing columns ['board', 'channel', ...]
    shrec_map : dict of pd.DataFrames
        Dictionary of DataFrames (as loaded by pd.read_excel(..., sheet_name=None))

    Returns:
    --------
    impx : pd.DataFrame
        Front side events mapped to X strips.
    impy : pd.DataFrame
        Back side events mapped to Y strips.
    """
    impx = dataframe[((dataframe['board'] <=5) & (dataframe['channel'] <=31))]
    impy = dataframe[((dataframe['board'] == 6) | (dataframe['board'] == 7)) & (dataframe['channel'] <=31)]
    
    xmap = shrec_map['IMP X']
    ymap = shrec_map['IMP Y']
    
    cols = ['board', 'channel']
    impx = impx.join(xmap.set_index(cols), on=cols)
    impy = impy.join(ymap.set_index(cols), on=cols)
    
    return impx, impy


def mapboxE(dataframe, shrec_map):
    """
    Example function for mapping Box E. 
    This can remain unused if you don't need it yet, but it's stored here.
    """
    boxEx = dataframe[(dataframe['board'] == 6) & (dataframe['channel'] <=47) & (dataframe['channel'] >= 32)]
    boxEy = dataframe[(dataframe['board'] == 4) & (dataframe['channel'] <=47) & (dataframe['channel'] >= 32)]
    
    xmap = shrec_map['BOXE X']
    ymap = shrec_map['BOXE Y']
    
    cols = ['board', 'channel']
    boxEx = boxEx.join(xmap.set_index(cols), on=cols)
    boxEy = boxEy.join(ymap.set_index(cols), on=cols)
    
    return boxEx, boxEy


def mapboxW(dataframe, shrec_map):
    """
    Example function for mapping Box W.
    """
    boxWx = dataframe[(dataframe['board'] == 7) & (dataframe['channel'] <=47) & (dataframe['channel'] >= 32)]
    boxWy = dataframe[(dataframe['board'] == 5) & (dataframe['channel'] <=47) & (dataframe['channel'] >= 32)]
    
    xmap = shrec_map['BOXW X']
    ymap = shrec_map['BOXW Y']
    
    cols = ['board', 'channel']
    boxWx = boxWx.join(xmap.set_index(cols), on=cols)
    boxWy = boxWy.join(ymap.set_index(cols), on=cols)
    
    return boxWx, boxWy


def mapboxT(dataframe, shrec_map):
    """
    Example function for mapping Box T.
    """
    boxTx = dataframe[((dataframe['board'] == 8) & (dataframe['channel'] >= 32)) |
                      ((dataframe['board'] == 5) & (dataframe['channel'] >= 48))]
    boxTy = dataframe[(dataframe['board'] == 7) & (dataframe['channel'] >= 48)]
    
    xmap = shrec_map['BOXT X']
    ymap = shrec_map['BOXT Y']
    
    cols = ['board', 'channel']
    boxTx = boxTx.join(xmap.set_index(cols), on=cols)
    boxTy = boxTy.join(ymap.set_index(cols), on=cols)
    
    return boxTx, boxTy


def mapboxB(dataframe, shrec_map):
    """
    Example function for mapping Box B.
    """
    boxBx = dataframe[((dataframe['board'] == 8) & (dataframe['channel'] <=31)) |
                      ((dataframe['board'] == 4) & (dataframe['channel'] >=48))]
    boxBy = dataframe[(dataframe['board'] == 6) & (dataframe['channel'] >=48)]
    
    xmap = shrec_map['BOXB X']
    ymap = shrec_map['BOXB Y']
    
    cols = ['board', 'channel']
    boxBx = boxBx.join(xmap.set_index(cols), on=cols)
    boxBy = boxBy.join(ymap.set_index(cols), on=cols)
    
    return boxBx, boxBy


def mapveto(dataframe, shrec_map):
    """
    Example function for mapping VETO.
    """
    vetox = dataframe[(dataframe['board'] <= 2) & (dataframe['channel'] >= 32)]
    vetoy = dataframe[(dataframe['board'] == 3) & (dataframe['channel'] >= 32)]
    
    xmap = shrec_map['VETO X']
    ymap = shrec_map['VETO Y']
    
    cols = ['board', 'channel']
    vetox = vetox.join(xmap.set_index(cols), on=cols)
    vetoy = vetoy.join(ymap.set_index(cols), on=cols)
    
    return vetox, vetoy

# def sortcalSHREC(xdata, ydata, calibration_path, ecut=50):
#     """
#     Finds coincidences between the front and back strips within a detector,
#     applies the energy calibration, and returns a cleaned DataFrame.

#     Parameters:
#     -----------
#     xdata, ydata : pd.DataFrame
#         The front and back data subsets for a particular detector region (e.g., IMP X / IMP Y).
#     calibration_path : str
#         Path to the SHREC calibration file (e.g., 'r238_calibration_v0_copy-from-r237.txt')
#     ecut : float
#         Energy cut to remove low-energy / noisy events.

#     Returns:
#     --------
#     pd.DataFrame
#         The cleaned and time-sorted DataFrame for this region.
#     """
#     import numpy as np  # local import for clarity, or put at top of file

#     # Convert timetags to time
#     xdata['t'] = np.round(xdata['timetag'] * 1e-12, 6)
#     ydata['t'] = np.round(ydata['timetag'] * 1e-12, 6)

#     # Also prepare the t2 columns used in your merging logic
#     xdata['t2'] = np.round(xdata['timetag'] * 1e-12, 5)
#     ydata['t2'] = np.round(ydata['timetag'] * 1e-12, 5)

#     # Load calibration
#     calfile = pd.read_csv(calibration_path, sep='\t')

#     # Apply the energy cut
#     xdata = xdata.query('energy >= @ecut')
#     ydata = ydata.query('energy >= @ecut')

#     # Join calibration columns
#     xdata = xdata.join(calfile.set_index(['board', 'channel']), on=['board', 'channel'])
#     ydata = ydata.join(calfile.set_index(['board', 'channel']), on=['board', 'channel'])

#     # Compute calibrated energy
#     xdata['calE'] = xdata['m'] * (xdata['energy'] - xdata['b'])
#     ydata['calE'] = ydata['m'] * (ydata['energy'] - ydata['b'])

#     # Drop unneeded columns
#     xdata.drop(['energy', 'm', 'b'], axis=1, inplace=True)
#     ydata.drop(['energy', 'm', 'b'], axis=1, inplace=True)

#     # Merge on (t) and (t2) to find coincidences
#     dfxy1 = xdata.merge(ydata, on='t')
#     dfxy2 = xdata.merge(ydata, on='t2')

#     # Fix column collisions
#     dfxy1 = dfxy1.drop(['t2_y'], axis=1).rename(columns={'t2_x': 't2'})
#     dfxy2 = dfxy2.drop(['t_y'], axis=1).rename(columns={'t_x': 't'})

#     # Combine, drop duplicates
#     dfxy = pd.concat([dfxy1, dfxy2], ignore_index=True).drop_duplicates().drop(['t2'], axis=1)

#     # Rename columns as needed
#     dfxy.columns = [
#         'xboard','xchan','tagx','xflag','nfile','xid','x','t','xstripE',
#         'yboard','ychan','tagy','yflag','nfiley','yid','y','ystripE'
#     ]

#     # Keep only relevant columns
#     dfxy = dfxy[['t','x','y','xstripE','ystripE','tagx','tagy','nfile']]

#     # Time difference cut (optional)
#     dfxy['tdelta'] = dfxy['tagx'] - dfxy['tagy']
#     dfxy = dfxy.loc[dfxy['tdelta'].abs() < 400000]

#     # Multiplicities
#     dfxy['nX'] = dfxy.groupby(['t','x'])['t'].transform('count')
#     dfxy['nY'] = dfxy.groupby(['t','y'])['t'].transform('count')

#     # x / y differences
#     dfxy['xdiff'] = dfxy.groupby('t')['x'].transform('max') - dfxy.groupby('t')['x'].transform('min')
#     dfxy['ydiff'] = dfxy.groupby('t')['y'].transform('max') - dfxy.groupby('t')['y'].transform('min')

#     # Summed energies
#     dfxy['xE'] = dfxy.groupby(['t','nX'])['xstripE'].transform('sum') / dfxy['nX']
#     dfxy['yE'] = dfxy.groupby(['t','nY'])['ystripE'].transform('sum') / dfxy['nY']

#     # Filter on x/y differences
#     temp3 = dfxy.loc[(dfxy['xdiff'] < 2) & (dfxy['ydiff'] < 2)]
#     # temp3.sort_values(by=['t','xstripE','ystripE'], ascending=[True,False,False], inplace=True)
#     # Filter on x/y differences
#     temp3 = dfxy.loc[(dfxy['xdiff'] < 2) & (dfxy['ydiff'] < 2)].copy()  # Add .copy() to avoid warning
#     # Sort values (without inplace)
#     temp3 = temp3.sort_values(by=['t','xstripE','ystripE'], ascending=[True,False,False])

#     temp4 = temp3.drop(['xstripE','ystripE','xdiff','ydiff'], axis=1).reset_index(drop=True)

#     temp4 = temp3.drop(['xstripE','ystripE','xdiff','ydiff'], axis=1).reset_index(drop=True)

#     # Print duration info
#     duration = temp4['t'].max() - temp4['t'].min()
#     print(f"duration = {duration:.1f} s = {duration/3600:.2f} hr")

#     return temp4

def sortcalSHREC(xdata, ydata, calibration_path, ecut=50):
    """
    Finds coincidences between the front and back strips within a detector,
    applies the energy calibration, and returns a cleaned DataFrame.

    Parameters:
    -----------
    xdata, ydata : pd.DataFrame
        The front and back data subsets for a particular detector region (e.g., IMP X / IMP Y).
    calibration_path : str, optional
        Path to the SHREC calibration file. Defaults to global SHREC_CALIBRATION.
    ecut : float, optional
        Energy cut to remove low-energy / noisy events. Default is 50.

    Returns:
    --------
    pd.DataFrame
        The cleaned and time-sorted DataFrame for this region.
    """
    import numpy as np

    # Time conversion using numpy array
    xdata['t'] = np.array(xdata['timetag']*1e-12).round(6)
    ydata['t'] = np.array(ydata['timetag']*1e-12).round(6)
    
    xdata['t2'] = np.array(xdata['timetag']*1e-12).round(5)
    ydata['t2'] = np.array(ydata['timetag']*1e-12).round(5)
    
    # Load calibration file
    calfile = pd.read_csv(calibration_path, sep='\t')
    
    # Apply energy cut
    xdata = xdata.query('energy >= @ecut')
    ydata = ydata.query('energy >= @ecut')
    
    # Join calibration columns
    xdata = xdata.join(calfile.set_index(['board', 'channel']), on=['board', 'channel'])
    ydata = ydata.join(calfile.set_index(['board', 'channel']), on=['board', 'channel'])
    
    # Compute calibrated energy
    xdata['calE'] = xdata['m'] * (xdata['energy'] - xdata['b'])
    ydata['calE'] = ydata['m'] * (ydata['energy'] - ydata['b'])
    
    # Drop unneeded columns
    xdata = xdata.drop(['energy', 'm', 'b'], axis=1)
    ydata = ydata.drop(['energy', 'm', 'b'], axis=1)
    
    # Merge on time and t2
    dfxy1 = xdata.merge(ydata, on=['t'])
    dfxy2 = xdata.merge(ydata, on=['t2'])
    
    # Fix column collisions
    dfxy1 = dfxy1.drop(['t2_y'], axis=1).rename(columns={'t2_x': 't2'})
    dfxy2 = dfxy2.drop(['t_y'], axis=1).rename(columns={'t_x': 't'})
    
    # Combine, drop duplicates
    dfxy = pd.concat([dfxy1, dfxy2]).drop_duplicates(ignore_index=True).drop(['t2'], axis=1).reset_index(drop=True)
    
    # Rename columns as needed
    dfxy.columns = [
        'xboard', 'xchan', 'tagx', 'xflag', 'nfile', 'xid', 'x', 't', 'xstripE', 
        'yboard', 'ychan', 'tagy', 'yflag', 'nfiley', 'yid', 'y', 'ystripE'
    ]
    
    # Keep only relevant columns
    dfxy = dfxy[['t', 'x', 'y', 'xstripE', 'ystripE', 'tagx', 'tagy', 'nfile']]
    
    # Time difference cut
    dfxy['tdelta'] = dfxy['tagx'] - dfxy['tagy']
    dfxy = dfxy.loc[abs(dfxy['tdelta']) < 400000]
    
    # Multiplicities
    dfxy['nX'] = dfxy.groupby(['t', 'x'])['t'].transform('count')
    dfxy['nY'] = dfxy.groupby(['t', 'y'])['t'].transform('count')
    
    # x / y differences
    dfxy['xdiff'] = dfxy.groupby(['t'])['x'].transform('max') - dfxy.groupby(['t'])['x'].transform('min')
    dfxy['ydiff'] = dfxy.groupby(['t'])['y'].transform('max') - dfxy.groupby(['t'])['y'].transform('min')
    
    # Summed energies
    dfxy['xE'] = dfxy.groupby(['t', 'nX'])['xstripE'].transform('sum') / dfxy['nX']
    dfxy['yE'] = dfxy.groupby(['t', 'nY'])['ystripE'].transform('sum') / dfxy['nY']
    
    # Filter on x/y differences
    temp3 = dfxy.loc[(dfxy['xdiff'].values < 2) & (dfxy['ydiff'].values < 2)]
    temp3 = temp3.sort_values(by=['t', 'xstripE', 'ystripE'], ascending=[True, False, False])
    
    # Drop unnecessary columns and reset index
    temp4 = temp3.drop(['xstripE', 'ystripE', 'xdiff', 'ydiff'], axis=1).reset_index(drop=True)
    
    # Calculate and print duration
    duration = max(temp4['t'].values) - min(temp4['t'].values)
    print("duration = %.1f s = %.2f hr" % (duration, duration/3600))
    
    return temp4


def process_shrec_data(csv_file, shrec_map_path, calibration_path, ecut=50):
    """
    Reads data, applies the SHREC mapping to all regions (IMP, Box E, etc.), then sorts
    and calibrates each region. Returns a dictionary of cleaned DataFrames, e.g.:
    {
      'imp': <DataFrame>,
      'boxE': <DataFrame>,
      ...
    }
    """
    import numpy as np

    # Read the raw data
    df = pd.read_csv(csv_file)

    # Read the SHREC map (dictionary of DataFrames)
    shrec_map = pd.read_excel(shrec_map_path, sheet_name=None)
    print(f"Processing {csv_file}... found {len(df)} raw events.")

    # 1) MAP all regions
    impx, impy   = mapimp(df, shrec_map)
    bex, bey     = mapboxE(df, shrec_map)
    bwx, bwy     = mapboxW(df, shrec_map)
    btx, bty     = mapboxT(df, shrec_map)
    bbx, bby     = mapboxB(df, shrec_map)
    vetox, vetoy = mapveto(df, shrec_map)

    # 2) Sort + calibrate each region using sortcalSHREC
    imp_clean   = sortcalSHREC(impx, impy,  calibration_path, ecut=ecut)
    boxE_clean  = sortcalSHREC(bex, bey,   calibration_path, ecut=ecut)
    boxW_clean  = sortcalSHREC(bwx, bwy,   calibration_path, ecut=ecut)
    boxT_clean  = sortcalSHREC(btx, bty,   calibration_path, ecut=ecut)
    boxB_clean  = sortcalSHREC(bbx, bby,   calibration_path, ecut=ecut)
    veto_clean  = sortcalSHREC(vetox, vetoy, calibration_path, ecut=ecut)

    # 3) Return them in a dict
    data = {
        'imp':   imp_clean,
        'boxE':  boxE_clean,
        'boxW':  boxW_clean,
        'boxT':  boxT_clean,
        'boxB':  boxB_clean,
        'veto':  veto_clean
    }

    plt.hist2d(imp_clean['xE'], imp_clean['x'], range = ((0, 11000),(0,174)), bins = (300, 174), cmin = 1)
    plt.show()

    return data



