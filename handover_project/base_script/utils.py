from skyfield.api import load
from datetime import datetime
import numpy as np
import random

from channel_parameters import ChannelParameters

import math
import pandas as pd

"""
Table 6.1.3.3-1 — Link Budget Results (TR 38.821)
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------
| Case | Orbit Alt. | Rx Mode | Freq    | EIRP [dBm] (dBW) | G/T [dB/K] | BW [MHz] | FSPL   | Atm. loss | Shadow loss | Scin. loss | Pol. loss  | Add. loss  | CNR   |
|      | [km]       |         | [GHz]   |                  |            |          | [dB]   | [dB]      | [dB]        | [dB]       | [dB]       | [dB]       | [dB]  |
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------
| SC6  | 600        | DL      | 20.0    | 60.0 (30.0)      |  15.9      | 400.0    | 179.1  | 0.5       | 0.0         | 0.3        | 0.0        | 0.0        | 8.5   |
|      |            | UL      | 30.0    | 76.2 (46.2)      |  13.0      | 400.0    | 182.6  | 0.5       | 0.0         | 0.3        | 0.0        | 0.0        | 18.4  |
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------
| SC9  | 600        | DL      | 2.0     | 78.8 (48.8)      | -31.6      | 30.0     | 159.1  | 0.1       | 3.0         | 2.2        | 0.0        | 0.0        | 6.6   |
|      |            | UL      | 3.0     | 23.0 (-7.0)      |  1.1       | 0.4      | 159.1  | 0.1       | 3.0         | 2.2        | 0.0        | 0.0        | 2.8   |
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------

Table 6.1.1.1-1 - Satellite antenna parameters (TR 38.821)
-------------------------------------------------------------------------------------
| Case | Antenna Aper. | EIRP Density  | Max Gain | 3dB Beamwidth  | Beam Diameter |
|      | [m]           | [dBW/MHz]     | [dBi]    | [deg]          | [Km]          |                     
-------------------------------------------------------------------------------------
| SC6  | 0.5           | 4.0           | 38.5     | 1.7647         | 20            |
-------------------------------------------------------------------------------------
| SC9  | 2.0           | 34.0          | 30.0     | 4.4127         | 50            |
-------------------------------------------------------------------------------------

Table 6.1.1.1-3 - UE antenna parameters (TR 38.821)
---------------------------------------------------------------------------
| Case | Rx Gain  | Antenna Temp.  | Noise Fig. | Tx Power    | Tx Gain  |
|      | [dBi]    | [K]            | [dB]       | [W] (dBm)   | [dBi]    |                     
---------------------------------------------------------------------------
| SC6  | 39.7     | 150            | 1.2        | 2 (33)      | 43.2     |
---------------------------------------------------------------------------
| SC9  | 0 (1x1)  | 290            | 7.0        | 0.2 (23)    | 0 (1x1)  |
|      | 3 (2x1)  |                |            |             | 3 (2x1)  |
---------------------------------------------------------------------------
"""

sc6_parameters = {
    'eirp_ue' : 76.2,   # dBm
    'gt_sat' : 13,      # dBi
    'eirp_sat' : 60,    # dBm
    'gt_ue' : 15.9,     # dBi
    'bw_dl' : 400e6,     # Hz
    'bw_ul' : 400e6,    # Hz
    'freq_dl' : 2e9,    # Hz
    'freq_ul' : 3e9,    # Hz
    'atm_loss' : 5.3,   # dB
    'dl_db_headroom' : 0, # dB
    'ul_db_headroom' : 0,  # dB
    'dlul_snr_variance' : 1, # variance of the noise added to the UE snr measurement to simulate real-world measurement imperfections
    '3gpp_overhead_dl': 0.18, # additional percentage of overhead as specified by 3GPP TS 38.306
    '3gpp_overhead_ul': 0.1,
    'frequency_reuse_factor': 3 # divide the total available bandwidth by this factor
}

sc9_parameters = {
    'eirp_ue' : 23,     # dBm
    'gt_sat' : 1.1,     # dBi
    'eirp_sat' : 78.8,  # dBm
    'gt_ue' : -31.6,    # dBi
    'bw_dl' : 30e6,     # Hz
    'bw_ul' : 0.4e6,    # Hz
    'freq_dl' : 2e9,    # Hz
    'freq_ul' : 3e9,    # Hz
    'atm_loss' : 0.8,   # dB
    'dl_db_headroom' : 2, # dB
    'ul_db_headroom' : 2,  # dB
    'dlul_snr_variance' : 1, # variance of the noise added to the UE snr measurement to simulate real-world measurement imperfections
    '3gpp_overhead_dl': 0.14, # additional percentage of overhead as specified by 3GPP TS 38.306
    '3gpp_overhead_ul': 0.08,
    'frequency_reuse_factor': 3 # divide the total available bandwidth by this factor
}

def get_satellites_at_time(df, target_time):
    """
    Filters a pandas DataFrame and returns a list of tuples containing satellite 
    information for a specific datetime.
    """
    satellites = []
    
    # Check if target_time is a datetime object and format it to match the DataFrame
    if isinstance(target_time, datetime):
        # Formats to "YYYY-MM-DD HH:MM:SS" (e.g., "2025-06-08 00:00:00")
        target_time_str = target_time.strftime("%Y-%m-%d %H:%M:%S")
    else:
        # Fallback just in case you pass a string directly
        target_time_str = str(target_time)
    
    try:
        # 1. Filter the DataFrame
        # .astype(str) ensures it safely compares strings, even if pandas 
        # auto-converted your time column to datetime objects when loading the CSV.
        matched_data = df[df['time'].astype(str) == target_time_str]
        
        # 2. Iterate through the matching rows
        for index, row in matched_data.iterrows():
            sat_info = (
                row['sat_name'],
                float(row['sat_lat']),
                float(row['sat_lon']),
                float(row['sat_height']),
                float(row['occurrence_countdown'])
            )
            satellites.append(sat_info)
            
    except KeyError as e:
        print(f"Error: Missing expected column in DataFrame - {e}")
    except ValueError as e:
        print(f"Error: Data format issue (e.g., empty or non-numeric values) - {e}")
        
    return satellites

def lla_to_ecef(lat, lon, alt):
    # seting the parameters for the WGS84
    a = 6378137.0 # the semi-major axis in meters
    f = 1/298.257223563 # flattening factor
    e2 = 6.69437999014e-3 # square of the eccentricity
    b = 6356752.3142 # the semi-minor axis in meters

    n_lat = a / math.sqrt(1-e2*math.sin(math.radians(lat))**2)

    x = (n_lat + alt) * math.cos(math.radians(lat)) * math.cos(math.radians(lon))
    y = (n_lat + alt) * math.cos(math.radians(lat)) * math.sin(math.radians(lon))
    z = ((1-e2)*n_lat+alt) * math.sin(math.radians(lat))

    return x, y, z

def direct_geodetic_problem(latitude, longitude, distance, bearing):
    """
    Computes the destination point coordinates given a starting point coordinates (latitude and longitude),
    the distance to travel, and the initial bearing in degrees. The Earth is assumed to be a perfect sphere
    (haversine formula).
    
    Args:
        latitude (float): latitude of the starting point in decimal degrees
        longitude (float): longitude of the starting point in decimal degrees
        distance (float): distance to travel in meters
        bearing (float): initial bearing in decimal degrees, clockwise from north (0 = north, 90 = east, ...)
    Returns:
        (float, float): latitude and longitude of the destination point in decimal degrees.
    """
    radius = 6371000 # Earth radius in meters

    # converting to radians for computaitons
    phi1 = math.radians(latitude)
    lambda1 = math.radians(longitude)
    theta = math.radians(bearing)
    delta = distance / radius

    # conversion formulae 
    phi2 = math.asin(math.sin(phi1) * math.cos(delta) + 
                     math.cos(phi1) * math.sin(delta) * math.cos(theta)) # destination latitude
    
    lambda2 = lambda1 + math.atan2(math.sin(theta) * math.sin(delta) * math.cos(phi1), 
                                   math.cos(delta) - math.sin(phi1) * math.sin(phi2)) # destination longitude
    
    return math.degrees(phi2), math.degrees(lambda2) # returning to degrees


def compute_cell_boundaries_lla(center_lat, center_lon, beam_size_m, beams_per_cell):
    """
    Computes the coordinates of the corners of a squared cell on the earth's surface centered on the 
    given point (center_lat, center_lon). The cell dimensions are computed from the given beam size
    in meters and the number of beams in the cell, assuming a square grid of beams.
    Args:
        center_lat (float): latitude of the cell centre in decimal degrees
        center_lon (float): longitude of the cell centre in decimal degrees
        beam_size_m (float): size of the beam footprint in meters
        beams_per_cell (int): number of beams in the cell in one direction 
            (e.g., if cell_width_in_beams=3, the cell is 3 beams wide and 3 beams tall)
    Returns:
        ((float, float), (float, float)): (nw_lat, nw_lon) coordinates of the north-west corner of the cell,
            (se_lat, se_lon) coordinates of the south-east corner of the cell, all in decimal degrees.
    """
    # bearings
    north = 0
    east = 90
    south = 180
    west = 270

    half_cell_size = (beam_size_m * beams_per_cell) / 2

    # starting from the center of the cell, we move west and north to find the north-west corner
    nw_lat, nw_lon = direct_geodetic_problem(center_lat, center_lon, half_cell_size, west)
    nw_lat, nw_lon = direct_geodetic_problem(nw_lat, nw_lon, half_cell_size, north)
    # starting from the center of the cell, we move east and south to find the south-east corner
    se_lat, se_lon = direct_geodetic_problem(center_lat, center_lon, half_cell_size, east)
    se_lat, se_lon = direct_geodetic_problem(se_lat, se_lon, half_cell_size, south)

    return (nw_lat, nw_lon), (se_lat, se_lon)

def check_clusters_visibility(cluster_centers_positions, cell_boundaries, cell_dim_beams, enable_elevation = False, elevation_threshold = 0, sat_lat = 0, sat_lon = 0, sat_alt = 0):
    """
    checks which of the cluster centers are within the satellite cell boundaries defined by the north-west and
    south-east corners latitude and longitude in decimal degrees.
    Example of centers grid numbering for a 5x5 cluster:
        0   1   2   3   4
        5   6   7   8   9
        10  11  12  13  14
        15  16  17  18  19
        20  21  22  23  24
    If the enableElevation flag is set, the function also checks if the satellite elevation angle is above a certain
    threshold for each cluster center, and only considers visible those clusters that are above the elevation threshold.

    Args:
        cluster_centers_positions (list of tuples): list of (lat, lon, alt) coordinates of the cluster centers
            in decimal degrees and meters.
        cell_boundaries ((float, float), (float, float)): (nw_lat, nw_lon) coordinates of the north-west 
            corner of the cell, (se_lat, se_lon) coordinates of the south-east corner of the cell,
            all in decimal degrees.
        cell_dim_beams (int): number of beams in the cell in one direction.
            e.g., if cell_dim_beams = 3, the cell is 3 beams wide and 3 beams tall, totalling 9 beams.
        enable_elevation (bool): if True, the function also checks the satellite elevation angle for each cluster center,
            and only considers visible those clusters that are above the elevation threshold.
        elevation_threshold (float): elevation angle threshold in degrees, only clusters with satellite elevation above
            this threshold are considered visible
        sat_lat (float): latitude of the satellite in decimal degrees
        sat_lon (float): longitude of the satellite in decimal degrees
    Returns:
        numpy.ndarray: matrix with indices of the cluster centers that are within the cell boundaries.
    """
    nw_lat, nw_lon = cell_boundaries[0]
    se_lat, se_lon = cell_boundaries[1]
    
    visible_clusters_indices = []
    cluster_row = []
    rind = 0
    for index, (lat, lon, alt) in enumerate(cluster_centers_positions):
        if nw_lat >= lat >= se_lat and nw_lon <= lon <= se_lon:
            if enable_elevation:
                sat_elevation = ChannelParameters.elevation_angle_deg(lat, lon, sat_lat, sat_lon, sat_alt)
                if sat_elevation >= elevation_threshold:
                    cluster_row.append(index)
                else:
                    cluster_row.append(-1)
            else:
                cluster_row.append(index)
        
        rind += 1
        if rind == cell_dim_beams:
            visible_clusters_indices.append(cluster_row)
            cluster_row = []
            rind = 0
    visible_clusters_indices = list(filter(None, visible_clusters_indices))
    
    visible_clusters_indices_matrix = np.array(visible_clusters_indices)
    out = np.array(strip_none_matrix(visible_clusters_indices_matrix))
    # if out.__contains__(-1):
    #     print(f"visible clusters indices after stripping none: \n{out}")
    return out

def strip_none_matrix(matrix):
    # Handle empty matrix edge case
    if matrix.size == 0:
        return []

    # 1. Keep rows that have at least one non-None value
    valid_rows = [row for row in matrix if any(val != -1 for val in row)]

    # If the matrix is now empty, return early
    if not valid_rows:
        return []

    # 2. Find the indices of columns that have at least one non-None value
    num_cols = len(valid_rows[0])
    valid_col_indices = [
        col_idx for col_idx in range(num_cols) 
        if any(row[col_idx] != -1 for row in valid_rows)
    ]

    # 3. Build the final matrix keeping only the valid columns
    cleaned_matrix = [
        [row[col_idx] for col_idx in valid_col_indices] 
        for row in valid_rows
    ]

    return cleaned_matrix

def get_coverage_beam_indices_matrix(visible_clusters_indices_matrix, cell_dim_beams):
    """
    given the matrix containing the indices of miniclusters that do have visibility with the satellite,
    this function returns a matrix of the same size containing the indices of the satellite beams that cover
    each of the mini clusters.
    Args:
        visible_clusters_indices_matrix (numpy.ndarray): matrix with indices of the cluster centers that are within the cell boundaries
        cell_dim_beams (int): number of beams in the cell in one direction
    Returns:
        numpy.ndarray: matrix with indices of the satellite beams that cover each of the mini clusters. The indices are arranged
            in the same way as the input matrix.
    """
    rows, cols = visible_clusters_indices_matrix.shape
    satellite_beams_matrix = np.arange(0, cell_dim_beams * cell_dim_beams).reshape(cell_dim_beams, cell_dim_beams)
    satellite_beams_matrix = np.roll(satellite_beams_matrix, shift=(-rows, cols), axis=(0, 1))
    coverage_beams_matrix = np.take(satellite_beams_matrix, visible_clusters_indices_matrix)
    return coverage_beams_matrix

def calculate_beams_grid(center_lat, center_lon, beam_size_km, num_beams):
    """
    This function calculates the positions of the beams' centers in a grid pattern around the center position.
    """
    grid_size = int(np.sqrt(num_beams))
    # Ensure inputs are treated as float64 (doubles)
    center_lat = np.float64(center_lat)
    center_lon = np.float64(center_lon)

    KM_PER_DEG_LAT = np.float64(111.32)
    km_per_deg_lon = KM_PER_DEG_LAT * np.cos(np.radians(center_lat))

    indices = np.arange(grid_size)
    center_idx = grid_size // 2 

    col_grid, row_grid = np.meshgrid(indices, indices)

    # Calculate offsets using float64 math
    delta_y_km = (center_idx - row_grid).astype(np.float64) * beam_size_km
    delta_x_km = (col_grid - center_idx).astype(np.float64) * beam_size_km

    # Final Lats and Lons
    lats = (center_lat + (delta_y_km / KM_PER_DEG_LAT)).flatten()
    lons = (center_lon + (delta_x_km / km_per_deg_lon)).flatten()

    # Create altitude as float64 zeros
    alts = np.zeros_like(lats, dtype=np.float64)

    return list(zip(lats, lons, alts))

def compute_distance_m(satellite_lat, satellite_lon, satellite_alt_m, ue_lat, ue_lon, ue_alt_m):
    sat_x, sat_y, sat_z = lla_to_ecef(satellite_lat, satellite_lon, satellite_alt_m)
    ue_x, ue_y, ue_z = lla_to_ecef(ue_lat, ue_lon, ue_alt_m)
    distance_m = round(math.sqrt((sat_x - ue_x)**2 + (sat_y - ue_y)**2 + (sat_z - ue_z)**2), 2)
    return distance_m

def compute_snr(distance_m, parameters):
    # Unpack parameters
    eirp_ue = parameters['eirp_ue']
    gt_sat = parameters['gt_sat']
    eirp_sat = parameters['eirp_sat']
    gt_ue = parameters['gt_ue']
    bw_dl = parameters['bw_dl']
    bw_ul = parameters['bw_ul']
    freq_dl = parameters['freq_dl']
    freq_ul = parameters['freq_ul']

    c = 299792458 
    path_loss_dl_db = 20 * math.log10(distance_m) + 20 * math.log10(freq_dl) + 20 * math.log10(4 * math.pi / c)
    path_loss_ul_db = 20 * math.log10(distance_m) + 20 * math.log10(freq_ul) + 20 * math.log10(4 * math.pi / c)

    # Calculate received power in dBm
    received_power_dl_dbm = eirp_sat + gt_ue - path_loss_dl_db
    received_power_ul_dbm = eirp_ue + gt_sat - path_loss_ul_db

    snr_dl_db = received_power_dl_dbm + 198.6 - 10 * math.log10(bw_dl)
    snr_ul_db = received_power_ul_dbm + 198.6 - 10 * math.log10(bw_ul)

    return snr_dl_db, snr_ul_db

def measure_snr_with_noise(distance_m, parameters):
    snr_dl_db, snr_ul_db = compute_snr(distance_m, parameters)
    noise_variance = parameters['dlul_snr_variance']
    snr_dl_db = round(snr_dl_db + random.gauss(0, math.sqrt(noise_variance)), 4)
    snr_ul_db = round(snr_ul_db + random.gauss(0, math.sqrt(noise_variance)), 4)
    return snr_dl_db, snr_ul_db
    
def compute_shannon(distance_m, parameters):
    snr_dl_db, snr_ul_db = compute_snr(distance_m, parameters)

    snr_dl_linear = 10 ** (snr_dl_db / 10)
    snr_ul_linear = 10 ** (snr_ul_db / 10)

    per_user_bandwidth_dl = parameters['bw_dl'] / parameters['frequency_reuse_factor']
    per_user_bandwidth_ul = parameters['bw_ul'] / parameters['frequency_reuse_factor']

    dl_thr_mbps = round(per_user_bandwidth_dl * math.log2(1 + snr_dl_linear) / 1e6, 4)
    ul_thr_mbps = round(per_user_bandwidth_ul * math.log2(1 + snr_ul_linear) / 1e6, 4)

    return dl_thr_mbps, ul_thr_mbps

def compute_shannon_from_snr(snr_dl_db, snr_ul_db, parameters):
    snr_dl_linear = 10 ** (snr_dl_db / 10)
    snr_ul_linear = 10 ** (snr_ul_db / 10)

    per_user_bandwidth_dl = parameters['bw_dl'] / parameters['frequency_reuse_factor']
    per_user_bandwidth_ul = parameters['bw_ul'] / parameters['frequency_reuse_factor']

    dl_thr_mbps = round(per_user_bandwidth_dl * math.log2(1 + snr_dl_linear) / 1e6, 4)
    ul_thr_mbps = round(per_user_bandwidth_ul * math.log2(1 + snr_ul_linear) / 1e6, 4)

    return dl_thr_mbps, ul_thr_mbps

def reverse_snr_from_thr(dl_ue_thr, ul_ue_thr, parameters):
    snr_dl_linear = (2 ** (dl_ue_thr * 1e6 / parameters['bw_dl'] * parameters['frequency_reuse_factor']) - 1)
    snr_ul_linear = (2 ** (ul_ue_thr * 1e6 / parameters['bw_ul'] * parameters['frequency_reuse_factor']) - 1)

    snr_dl_db = round(10 * math.log10(snr_dl_linear), 4)
    snr_ul_db = round(10 * math.log10(snr_ul_linear), 4)

    return snr_dl_db, snr_ul_db

def get_max_beam_throughput(frame, target_time, satellite_name, mini_cluster_position, scenario):
    mini_cluster_lat, mini_cluster_lon, _ = mini_cluster_position
    dl_total_throughput, ul_total_throughput = 0, 0
    if isinstance(target_time, datetime):
        target_time_str = target_time.strftime("%Y-%m-%d %H:%M:%S")
    else:
        target_time_str = str(target_time)
    try:
        matched_satellite = frame[frame['time'].astype(str) == target_time_str]
        matched_satellite = matched_satellite[matched_satellite['sat_name'].astype(str) == satellite_name]
        sat_lat = float(matched_satellite['sat_lat'].iloc[0])
        sat_lon = float(matched_satellite['sat_lon'].iloc[0])
        sat_alt_m = float(matched_satellite['sat_height'].iloc[0])

        distance_m = compute_distance_m(sat_lat, sat_lon, sat_alt_m, mini_cluster_lat, mini_cluster_lon, 0)
        dl_total_throughput, ul_total_throughput = compute_shannon(distance_m, scenario)
    except KeyError as e:
        print(f"Error: Missing expected column in DataFrame - {e}")
    except ValueError as e:
        print(f"Error: Data format issue (e.g., empty or non-numeric values) - {e}")
    return dl_total_throughput, ul_total_throughput


def get_elevation(frame, target_time, satellite_name, mini_cluster_position):
    """
    Compute the satellite elevation given the minicluster and sat positions.
    Args:
        frame: the dataframe containing the constellation information over time
        satellite_name: name of the satellite we want to compute the snr
        mini_cluster_position: location of the ue we want to compute the snr
    Returns:
        sat_elev: satellite elevation angle in degrees
    """
    mini_cluster_lat, mini_cluster_lon, _ = mini_cluster_position
    sat_elev = 0
    if isinstance(target_time, datetime):
        target_time_str = target_time.strftime("%Y-%m-%d %H:%M:%S")
    else:
        target_time_str = str(target_time)
    
    try:
        matched_satellite = frame[frame['time'].astype(str) == target_time_str]
        matched_satellite = matched_satellite[matched_satellite['sat_name'].astype(str) == satellite_name]
        sat_lat = float(matched_satellite['sat_lat'].iloc[0])
        sat_long = float(matched_satellite['sat_lon'].iloc[0])
        sat_alt_m = float(matched_satellite['sat_height'].iloc[0])

        sat_elev = ChannelParameters.elevation_angle_deg(
                mini_cluster_lat, mini_cluster_lon,
                sat_lat, sat_long,
                sat_alt_m
            )
        
    except KeyError as e:
        print(f"Error: Missing expected column in DataFrame - {e}")
    except ValueError as e:
        print(f"Error: Data format issue (e.g., empty or non-numeric values) - {e}")
        
    return sat_elev

def get_noisy_snr(frame, target_time, satellite_name, mini_cluster_position, parameters):
    """
    Compute the dl and ul snr given the minicluster and sat positions.
    Args:
        frame: the dataframe containing the constellation information over time
        satellite_name: name of the satellite we want to compute the snr
        mini_cluster_position: location of the ue we want to compute the snr
        parameters: specify the scenario (ex. sc6_parameters or sc9_parameters)
    Returns:
        snr_dl_db: the actual dl snr in dB
        snr_ul_db: the actual ul snr in dB
    """
    mini_cluster_lat, mini_cluster_lon, _ = mini_cluster_position
    if isinstance(target_time, datetime):
        target_time_str = target_time.strftime("%Y-%m-%d %H:%M:%S")
    else:
        target_time_str = str(target_time)
    try:
        matched_satellite = frame[frame['time'].astype(str) == target_time_str]
        matched_satellite = matched_satellite[matched_satellite['sat_name'].astype(str) == satellite_name]
        sat_lat = float(matched_satellite['sat_lat'].iloc[0])
        sat_lon = float(matched_satellite['sat_lon'].iloc[0])
        sat_alt_m = float(matched_satellite['sat_height'].iloc[0])
        distance_m = compute_distance_m(sat_lat, sat_lon, sat_alt_m, mini_cluster_lat, mini_cluster_lon, 0)
        
        snr_dl_db, snr_ul_db = compute_snr(distance_m, parameters)
    except KeyError as e:
        print(f"Error: Missing expected column in DataFrame - {e}")
    except ValueError as e:
        print(f"Error: Data format issue (e.g., empty or non-numeric values) - {e}")
    return snr_dl_db, snr_ul_db
   
def get_visibility_time(sat_name, target_time, df):
    """
    Returns the remaining visibility time of a given satellite starting from a given time instant.
    Args:
        frame: the dataframe containing the constellation information over time
        time: current simulation time
        satellite_name: name of the satellite we want to compute the snr
    Returns:
        visibility_time: remaining visibility time of the satellite
    """
    vis_time = 0

    # Convert the entire 'time' column from strings to datetime objects
    df['time'] = pd.to_datetime(df['time'])
    vis_time = df[(df['sat_name'] == sat_name) & (df['time'] == target_time)]['occurrence_countdown'].iloc[0]

    return vis_time

def get_sat_positions(frame, sat_name, time):
    """
    Returns the old, current,and future positions of a given satellite at a given time instant.
    Args:
        frame: the dataframe containing the constellation information over time
        time: current simulation time
        satellite_name: name of the satellite we want to compute the snr
    Returns:
        old_pos:  (sat_lat, sat_lon, sat_alt_m)
        curr_pos: (sat_lat, sat_lon, sat_alt_m)
        fut_pos:  (sat_lat, sat_lon, sat_alt_m)
    """
    old_lat, old_lon, old_alt = None, None, None
    curr_lat, curr_lon, curr_alt = None, None, None
    fut_lat, fut_lon, fut_alt = None, None, None

    if isinstance(time, datetime):
        time_curr = time.strftime("%Y-%m-%d %H:%M:%S")
        time_old = (time - pd.Timedelta(seconds=1)).strftime("%Y-%m-%d %H:%M:%S")
        time_fut = (time + pd.Timedelta(seconds=1)).strftime("%Y-%m-%d %H:%M:%S")
    else:
        time_curr = str(time)
        time_old = str(time - pd.Timedelta(seconds=1))
        time_fut = str(time + pd.Timedelta(seconds=1))

    try:
        # current position
        matched_satellite = frame[frame['time'].astype(str) == time_curr]
        matched_satellite = matched_satellite[matched_satellite['sat_name'].astype(str) == sat_name]
        if not matched_satellite.empty:
            curr_lat = float(matched_satellite['sat_lat'].iloc[0])
            curr_lon = float(matched_satellite['sat_lon'].iloc[0])
            curr_alt = float(matched_satellite['sat_height'].iloc[0])

        # old position
        matched_satellite = frame[frame['time'].astype(str) == time_old]
        matched_satellite = matched_satellite[matched_satellite['sat_name'].astype(str) == sat_name]
        if not matched_satellite.empty:
            old_lat = float(matched_satellite['sat_lat'].iloc[0])
            old_lon = float(matched_satellite['sat_lon'].iloc[0])
            old_alt = float(matched_satellite['sat_height'].iloc[0])

        # future position
        matched_satellite = frame[frame['time'].astype(str) == time_fut]
        matched_satellite = matched_satellite[matched_satellite['sat_name'].astype(str) == sat_name]
        if not matched_satellite.empty:
            fut_lat = float(matched_satellite['sat_lat'].iloc[0])
            fut_lon = float(matched_satellite['sat_lon'].iloc[0])
            fut_alt = float(matched_satellite['sat_height'].iloc[0])

    except KeyError as e:
        print(f"Error: Missing expected column in DataFrame - {e}")
    except ValueError as e:
        print(f"Error: Data format issue (e.g., empty or non-numeric values) - {e}")

    return (old_lat, old_lon, old_alt), (curr_lat, curr_lon, curr_alt), (fut_lat, fut_lon, fut_alt)

def compute_doppler_shift(ue_pos, old_pos, curr_pos, fut_pos, scenario):
    """
    Compute the Doppler shift given the old, current and future positions of the satellite and the user.
    Args:
        ue_pos: (ue_lat, ue_lon, ue_alt_m) position of the user
        old_pos: (sat_lat, sat_lon, sat_alt_m) old position of the satellite
        curr_pos: (sat_lat, sat_lon, sat_alt_m) current position of the satellite
        fut_pos: (sat_lat, sat_lon, sat_alt_m) future position of the satellite
        scenario: current scenario parameters (ex. sc6_parameters or sc9_parameters)
    Returns:
        doppler_shift_dl: Doppler shift in Hz
        doppler_shift_ul: Doppler shift in Hz
    """ 

    freq_ul = scenario['freq_ul']
    freq_dl = scenario['freq_dl']
    dt = 1.0 # time interval in seconds between the old, current and future positions

    # Compute the relative velocity between the user and the satellite
    rel_velocity = compute_relative_velocity(ue_pos, old_pos, curr_pos, fut_pos, dt)

    doppler_shift_dl = freq_dl * rel_velocity / 299792458  
    doppler_shift_ul = freq_ul * rel_velocity / 299792458  

    return doppler_shift_dl, doppler_shift_ul


def compute_relative_velocity(ue_pos, old_pos, curr_pos, fut_pos, dt=1.0):
    """
    Compute the relative velocity between the user and the satellite along the line of sight (LOS) direction.
    Args:
        ue_pos: (ue_lat, ue_lon, ue_alt_m) position of the user
        old_pos: (sat_lat, sat_lon, sat_alt_m) old position of the satellite
        curr_pos: (sat_lat, sat_lon, sat_alt_m) current position of the satellite
        fut_pos: (sat_lat, sat_lon, sat_alt_m) future position of the satellite
        dt: time interval in seconds between the old, current and future positions (default: 1 second)
    Returns:
        relative_velocity: relative velocity in m/s 
                           (positive if the satellite is moving away from the user, 
                            negative if it is moving towards the user)
    """
    if dt <= 0.0:
        raise ValueError(f"dt must be positive, got {dt!r}")
 
    # Guard: need curr_pos and at least one neighbour to estimate velocity
    if curr_pos is None:
        return 0.0
    if old_pos is None and fut_pos is None:
        return 0.0
 
    # Convert available LLA positions to ECEF 
    ue_ecef = lla_to_ecef(ue_pos[0], ue_pos[1], ue_pos[2])
    curr_ecef = lla_to_ecef(curr_pos[0], curr_pos[1], curr_pos[2])
    old_ecef = lla_to_ecef(old_pos[0], old_pos[1], old_pos[2]) if old_pos is not None else None
    fut_ecef = lla_to_ecef(fut_pos[0], fut_pos[1], fut_pos[2]) if fut_pos is not None else None

    # Estimate satellite velocity vector (m/s) in ECEF
    if old_ecef is not None and fut_ecef is not None:
        # Central difference over 2·dt
        sat_vel = tuple(
            (fut_ecef[i] - old_ecef[i]) / (2.0 * dt) for i in range(3)
        )
    elif fut_ecef is not None:
        # Forward difference over dt
        sat_vel = tuple(
            (fut_ecef[i] - curr_ecef[i]) / dt for i in range(3)
        )
    else:
        # Backward difference over dt
        sat_vel = tuple(
            (curr_ecef[i] - old_ecef[i]) / dt for i in range(3)
        )
 
    # Line-of-sight (LOS) unit vector: UE → satellite
    los = tuple(curr_ecef[i] - ue_ecef[i] for i in range(3))
    los_norm = math.sqrt(sum(v ** 2 for v in los))
 
    if los_norm == 0.0:
        # Degenerate case: UE and satellite at the identical point
        return 0.0
    los_unit = tuple(v / los_norm for v in los)
 
    # Radial velocity = projection of sat_vel onto the LOS unit vector 
    radial_velocity = sum(sat_vel[i] * los_unit[i] for i in range(3))
 
    return radial_velocity
