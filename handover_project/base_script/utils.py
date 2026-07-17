from datetime import datetime
import numpy as np
import math
import pandas as pd

from channel_parameters import ChannelParameters
from channel import Channel, sc6_parameters, sc9_parameters


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
    the distance to travel, and the initial bearing in degrees. The Earth is assumed to be a perfect sphere.
    """
    radius = 6371000 # Earth radius in meters

    phi1 = math.radians(latitude)
    lambda1 = math.radians(longitude)
    theta = math.radians(bearing)
    delta = distance / radius

    phi2 = math.asin(math.sin(phi1) * math.cos(delta) + 
                     math.cos(phi1) * math.sin(delta) * math.cos(theta))
    
    lambda2 = lambda1 + math.atan2(math.sin(theta) * math.sin(delta) * math.cos(phi1), 
                                   math.cos(delta) - math.sin(phi1) * math.sin(phi2))
    
    return math.degrees(phi2), math.degrees(lambda2)

def compute_cell_boundaries_lla(center_lat, center_lon, beam_size_m, beams_per_cell):
    north, east, south, west = 0, 90, 180, 270
    half_cell_size = (beam_size_m * beams_per_cell) / 2

    nw_lat, nw_lon = direct_geodetic_problem(center_lat, center_lon, half_cell_size, west)
    nw_lat, nw_lon = direct_geodetic_problem(nw_lat, nw_lon, half_cell_size, north)
    
    se_lat, se_lon = direct_geodetic_problem(center_lat, center_lon, half_cell_size, east)
    se_lat, se_lon = direct_geodetic_problem(se_lat, se_lon, half_cell_size, south)

    return (nw_lat, nw_lon), (se_lat, se_lon)

def check_clusters_visibility(cluster_centers_positions, cell_boundaries, cell_dim_beams, enable_elevation = False, elevation_threshold = 0, sat_lat = 0, sat_lon = 0, sat_alt = 0):
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
    return out

def strip_none_matrix(matrix):
    if matrix.size == 0: return []
    valid_rows = [row for row in matrix if any(val != -1 for val in row)]
    if not valid_rows: return []

    num_cols = len(valid_rows[0])
    valid_col_indices = [col_idx for col_idx in range(num_cols) if any(row[col_idx] != -1 for row in valid_rows)]
    cleaned_matrix = [[row[col_idx] for col_idx in valid_col_indices] for row in valid_rows]
    return cleaned_matrix

def get_coverage_beam_indices_matrix(visible_clusters_indices_matrix, cell_dim_beams):
    rows, cols = visible_clusters_indices_matrix.shape
    satellite_beams_matrix = np.arange(0, cell_dim_beams * cell_dim_beams).reshape(cell_dim_beams, cell_dim_beams)
    satellite_beams_matrix = np.roll(satellite_beams_matrix, shift=(-rows, cols), axis=(0, 1))
    coverage_beams_matrix = np.take(satellite_beams_matrix, visible_clusters_indices_matrix)
    return coverage_beams_matrix

def calculate_beams_grid(center_lat, center_lon, beam_size_km, num_beams):
    grid_size = int(np.sqrt(num_beams))
    center_lat = np.float64(center_lat)
    center_lon = np.float64(center_lon)

    KM_PER_DEG_LAT = np.float64(111.32)
    km_per_deg_lon = KM_PER_DEG_LAT * np.cos(np.radians(center_lat))

    indices = np.arange(grid_size)
    center_idx = grid_size // 2 

    col_grid, row_grid = np.meshgrid(indices, indices)

    delta_y_km = (center_idx - row_grid).astype(np.float64) * beam_size_km
    delta_x_km = (col_grid - center_idx).astype(np.float64) * beam_size_km

    lats = (center_lat + (delta_y_km / KM_PER_DEG_LAT)).flatten()
    lons = (center_lon + (delta_x_km / km_per_deg_lon)).flatten()
    alts = np.zeros_like(lats, dtype=np.float64)

    return list(zip(lats, lons, alts))

def compute_distance_m(satellite_lat, satellite_lon, satellite_alt_m, ue_lat, ue_lon, ue_alt_m):
    sat_x, sat_y, sat_z = lla_to_ecef(satellite_lat, satellite_lon, satellite_alt_m)
    ue_x, ue_y, ue_z = lla_to_ecef(ue_lat, ue_lon, ue_alt_m)
    distance_m = round(math.sqrt((sat_x - ue_x)**2 + (sat_y - ue_y)**2 + (sat_z - ue_z)**2), 2)
    return distance_m

def compute_doppler_shift(ue_pos, old_pos, curr_pos, fut_pos, scenario):
    if curr_pos is None or curr_pos[0] is None: return 0.0, 0.0
    
    if old_pos is None or old_pos[0] is None: old_pos = curr_pos
    if fut_pos is None or fut_pos[0] is None: fut_pos = curr_pos

    freq_ul = scenario['freq_ul']
    freq_dl = scenario['freq_dl']
    dt = 1.0 

    rel_velocity = compute_relative_velocity(ue_pos, old_pos, curr_pos, fut_pos, dt)

    doppler_shift_dl = freq_dl * rel_velocity / 299792458  
    doppler_shift_ul = freq_ul * rel_velocity / 299792458  

    return doppler_shift_dl, doppler_shift_ul

def compute_relative_velocity(ue_pos, old_pos, curr_pos, fut_pos, dt=1.0):
    if dt <= 0.0: return 0.0
    if curr_pos is None or (old_pos is None and fut_pos is None): return 0.0
 
    ue_ecef = lla_to_ecef(ue_pos[0], ue_pos[1], ue_pos[2])
    curr_ecef = lla_to_ecef(curr_pos[0], curr_pos[1], curr_pos[2])
    old_ecef = lla_to_ecef(old_pos[0], old_pos[1], old_pos[2]) if old_pos is not None else None
    fut_ecef = lla_to_ecef(fut_pos[0], fut_pos[1], fut_pos[2]) if fut_pos is not None else None

    if old_ecef is not None and fut_ecef is not None:
        sat_vel = tuple((fut_ecef[i] - old_ecef[i]) / (2.0 * dt) for i in range(3))
    elif fut_ecef is not None:
        sat_vel = tuple((fut_ecef[i] - curr_ecef[i]) / dt for i in range(3))
    else:
        sat_vel = tuple((curr_ecef[i] - old_ecef[i]) / dt for i in range(3))
 
    los = tuple(curr_ecef[i] - ue_ecef[i] for i in range(3))
    los_norm = math.sqrt(sum(v ** 2 for v in los))
 
    if los_norm == 0.0: return 0.0
    los_unit = tuple(v / los_norm for v in los)
    radial_velocity = sum(sat_vel[i] * los_unit[i] for i in range(3))
 
    return radial_velocity


def get_satellites_at_time(df, target_time):
    satellites = []
    target_time_str = target_time.strftime("%Y-%m-%d %H:%M:%S") if isinstance(target_time, datetime) else str(target_time)
    
    try:
        matched_data = df[df['time'].astype(str) == target_time_str]
        for index, row in matched_data.iterrows():
            sat_info = (
                row['sat_name'], float(row['sat_lat']), float(row['sat_lon']),
                float(row['sat_height']), float(row['occurrence_countdown'])
            )
            satellites.append(sat_info)
    except (KeyError, ValueError):
        pass
    return satellites

def get_elevation(frame, target_time, satellite_name, mini_cluster_position):
    mini_cluster_lat, mini_cluster_lon, _ = mini_cluster_position
    target_time_str = target_time.strftime("%Y-%m-%d %H:%M:%S") if isinstance(target_time, datetime) else str(target_time)
    
    try:
        matched_satellite = frame[(frame['time'].astype(str) == target_time_str) & (frame['sat_name'].astype(str) == satellite_name)]
        if matched_satellite.empty: return -1.0

        sat_lat = float(matched_satellite['sat_lat'].iloc[0])
        sat_long = float(matched_satellite['sat_lon'].iloc[0])
        sat_alt_m = float(matched_satellite['sat_height'].iloc[0])

        return ChannelParameters.elevation_angle_deg(mini_cluster_lat, mini_cluster_lon, sat_lat, sat_long, sat_alt_m)
    except (KeyError, ValueError, IndexError):
        return -1.0

def get_visibility_time(sat_name, target_time, df):
    df['time'] = pd.to_datetime(df['time'])
    try:
        return df[(df['sat_name'] == sat_name) & (df['time'] == target_time)]['occurrence_countdown'].iloc[0]
    except IndexError:
        return 0

def get_sat_positions(frame, sat_name, time):
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
        matched_curr = frame[(frame['time'].astype(str) == time_curr) & (frame['sat_name'].astype(str) == sat_name)]
        if not matched_curr.empty:
            curr_lat, curr_lon, curr_alt = float(matched_curr['sat_lat'].iloc[0]), float(matched_curr['sat_lon'].iloc[0]), float(matched_curr['sat_height'].iloc[0])

        matched_old = frame[(frame['time'].astype(str) == time_old) & (frame['sat_name'].astype(str) == sat_name)]
        if not matched_old.empty:
            old_lat, old_lon, old_alt = float(matched_old['sat_lat'].iloc[0]), float(matched_old['sat_lon'].iloc[0]), float(matched_old['sat_height'].iloc[0])

        matched_fut = frame[(frame['time'].astype(str) == time_fut) & (frame['sat_name'].astype(str) == sat_name)]
        if not matched_fut.empty:
            fut_lat, fut_lon, fut_alt = float(matched_fut['sat_lat'].iloc[0]), float(matched_fut['sat_lon'].iloc[0]), float(matched_fut['sat_height'].iloc[0])

    except (KeyError, ValueError):
        pass

    return (old_lat, old_lon, old_alt), (curr_lat, curr_lon, curr_alt), (fut_lat, fut_lon, fut_alt)

def compute_snr(distance_m, parameters):
    return Channel.compute_snr(distance_m, parameters)

def compute_shannon(distance_m, parameters):
    return Channel.compute_shannon(distance_m, parameters)

def compute_shannon_from_snr(snr_dl_db, snr_ul_db, parameters):
    return Channel.compute_shannon_from_snr(snr_dl_db, snr_ul_db, parameters)

def reverse_snr_from_thr(dl_ue_thr, ul_ue_thr, parameters):
    return Channel.reverse_snr_from_thr(dl_ue_thr, ul_ue_thr, parameters)

def get_max_beam_throughput(frame, target_time, satellite_name, mini_cluster_position, scenario):
    mini_cluster_lat, mini_cluster_lon, _ = mini_cluster_position
    target_time_str = target_time.strftime("%Y-%m-%d %H:%M:%S") if isinstance(target_time, datetime) else str(target_time)
    
    try:
        matched_satellite = frame[(frame['time'].astype(str) == target_time_str) & (frame['sat_name'].astype(str) == satellite_name)]
        if matched_satellite.empty: return 0.0, 0.0

        sat_lat = float(matched_satellite['sat_lat'].iloc[0])
        sat_lon = float(matched_satellite['sat_lon'].iloc[0])
        sat_alt_m = float(matched_satellite['sat_height'].iloc[0])

        distance_m = compute_distance_m(sat_lat, sat_lon, sat_alt_m, mini_cluster_lat, mini_cluster_lon, 0)
        return Channel.compute_shannon(distance_m, scenario)
    except (KeyError, ValueError, IndexError):
        return 0.0, 0.0

def get_noisy_snr(frame, target_time, satellite_name, mini_cluster_position, parameters):
    # La logica del rumore era in utils, la manteniamo ma chiamiamo Channel per la fisica base
    import random
    mini_cluster_lat, mini_cluster_lon, _ = mini_cluster_position
    target_time_str = target_time.strftime("%Y-%m-%d %H:%M:%S") if isinstance(target_time, datetime) else str(target_time)
    
    try:
        matched_satellite = frame[(frame['time'].astype(str) == target_time_str) & (frame['sat_name'].astype(str) == satellite_name)]
        if matched_satellite.empty: return 0.0, 0.0
            
        sat_lat = float(matched_satellite['sat_lat'].iloc[0])
        sat_lon = float(matched_satellite['sat_lon'].iloc[0])
        sat_alt_m = float(matched_satellite['sat_height'].iloc[0])
        distance_m = compute_distance_m(sat_lat, sat_lon, sat_alt_m, mini_cluster_lat, mini_cluster_lon, 0)
        
        snr_dl_db, snr_ul_db = Channel.compute_snr(distance_m, parameters)
        noise_variance = parameters.get('dlul_snr_variance', 0)
        if noise_variance > 0:
            snr_dl_db = round(snr_dl_db + random.gauss(0, math.sqrt(noise_variance)), 4)
            snr_ul_db = round(snr_ul_db + random.gauss(0, math.sqrt(noise_variance)), 4)
        return snr_dl_db, snr_ul_db
    except (KeyError, ValueError, IndexError):
        return 0.0, 0.0
