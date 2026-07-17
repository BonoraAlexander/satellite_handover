import random
import utils
import math

"""
This module contains the implementation of different legacy handover strategies (State of the Art). 
Each strategy is implemented as a standalone function representing a distributed, UE-centric MADM approach.
"""

def get_random_visible_satellite(visible_satellites):
    """
    Implementation of a random selection amongst the visible satellites.
    Returns:
        tuple: (next_sat, next_beam_index) or None if no visible satellites.
    """
    number_of_satellites = len(visible_satellites)
    if number_of_satellites == 0:
        return None
        
    random_index = random.randint(0, number_of_satellites - 1) 
    next_sat = visible_satellites[random_index][0] 
    next_beam_index = visible_satellites[random_index][1] 
    return next_sat, next_beam_index

def get_max_elevation_satellite(visible_satellites, current_positions_df, round_time, mini_cluster):
    """
    Selecting the satellite with the maximum elevation angle from the ones in visibility.
    Includes a 30% random pool selection to avoid catastrophic greedy collisions.
    """
    satellites_with_elevation = []
    
    for satellite, beam in visible_satellites:
        satellite_name = satellite[0]
        elevation = utils.get_elevation(current_positions_df, round_time, satellite_name, mini_cluster.position)
        satellites_with_elevation.append((satellite, beam, elevation))
        
    # Sort and pick randomly from the top 30%
    satellites_with_elevation = sorted(satellites_with_elevation, key=lambda x: x[2], reverse=True)
    choices = max(int(0.3 * len(satellites_with_elevation)), 1)
    random_index = random.randint(0, choices - 1)
    
    next_sat = satellites_with_elevation[random_index][0]
    next_beam_index = satellites_with_elevation[random_index][1]
    return next_sat, next_beam_index

def get_max_visibility_satellite(visible_satellites, current_positions_df, round_time):
    """
    Selecting the satellite with the maximum remaining visibility time.
    """
    satellites_with_visibility = []
    
    for satellite, beam in visible_satellites:
        satellite_name = satellite[0]
        vis_time = utils.get_visibility_time(satellite_name, round_time, current_positions_df)
        satellites_with_visibility.append((satellite, beam, vis_time))
        
    # Sort and pick randomly from the top 30%
    satellites_with_visibility = sorted(satellites_with_visibility, key=lambda x: x[2], reverse=True)
    choices = max(int(0.3 * len(satellites_with_visibility)), 1)
    random_index = random.randint(0, choices - 1)
    
    next_sat = satellites_with_visibility[random_index][0]
    next_beam_index = satellites_with_visibility[random_index][1]
    return next_sat, next_beam_index

def get_max_available_throughput_satellite(visible_satellites, round_time, mini_cluster, service_satellites, mainframe, scenario):
    """
    Selecting the satellite capable of providing the highest throughput value considering current load (Shannon capacity).
    """
    satellites_with_throughput = []
    
    for satellite, beam in visible_satellites:
        satellite_name = satellite[0]
        num_connected_ues = 1
        
        if satellite_name in service_satellites:
            num_connected_ues += service_satellites[satellite_name].connected_ues[beam]
            
        max_dl_thr, max_ul_thr = utils.get_max_beam_throughput(mainframe, round_time, satellite_name, mini_cluster.position, scenario)
        thr_score = max_dl_thr / num_connected_ues
        satellites_with_throughput.append((satellite, beam, thr_score))
        
    satellites_with_throughput = sorted(satellites_with_throughput, key=lambda x: x[2], reverse=True)
    best_index = 0
    next_sat = satellites_with_throughput[best_index][0]
    next_beam_index = satellites_with_throughput[best_index][1]
    return next_sat, next_beam_index

def compute_mauf_weight(satellite_info, beam_index, current_positions_df, round_time, mini_cluster, service_satellites, mainframe, scenario):
    """
    Computes the Doppler-Aware Multi-Attribute Utility Function (MAUF) weight W_{i,j} for a specific satellite.
    """
    satellite_name = satellite_info[0]
    
    # 1. Normalized Elevation (E_norm)
    elevation = utils.get_elevation(current_positions_df, round_time, satellite_name, mini_cluster.position)
    e_threshold = 30.0
    if elevation < e_threshold:
        e_norm = 0.0
    else:
        e_norm = (elevation - e_threshold) / (90.0 - e_threshold)
        
    # 2. Normalized Remaining Visibility (V_norm)
    vis_time = utils.get_visibility_time(satellite_name, round_time, current_positions_df)
    v_max = 600.0  
    v_norm = min(vis_time / v_max, 1.0)
    
    # 3. Normalized Load Penalty (L_norm)
    max_capacity_per_beam = 50.0 
    current_load = 0
    if satellite_name in service_satellites:
        current_load = service_satellites[satellite_name].connected_ues[beam_index]
    l_norm = min(current_load / max_capacity_per_beam, 1.0)
    
    # 4. Normalized Doppler Penalty (D_norm)
    # Temporary fallback: approximate Doppler penalty inversely proportional to elevation 
    d_norm = 1.0 - e_norm 
    
    # Weights for the MAUF
    alpha_1 = 0.4  # Elevation Importance
    alpha_2 = 0.3  # Visibility Importance
    alpha_3 = 0.2  # Doppler Penalty Importance
    alpha_4 = 0.1  # Load Penalty Importance
    
    W = (alpha_1 * e_norm) + (alpha_2 * v_norm) - (alpha_3 * d_norm) - (alpha_4 * l_norm)
    return W


def get_dap_bm_satellite(visible_satellites, current_positions_df, round_time, mini_cluster, service_satellites, mainframe, scenario, ue_id, cluster_pending_ues, curr_sat_name=None, curr_beam_index=None):
    """
    Implementation of the Doppler-Aware Predictive Handover (DAP-BM) localized heuristic.
    """
    number_of_satellites = len(visible_satellites)
    if number_of_satellites == 0:
        return None, None
        
    satellites_with_mauf = []
    w_curr = -1.0
    
    MAX_CAPACITY = 15.0 
    
    for satellite, beam in visible_satellites:
        w_score = compute_mauf_weight(satellite, beam, current_positions_df, round_time, mini_cluster, service_satellites, mainframe, scenario)
        
        if satellite[0] in service_satellites:
            try:
                current_load = service_satellites[satellite[0]].connected_ues[beam]
                load_ratio = min(current_load / MAX_CAPACITY, 0.99)
                w_score *= (1.0 - load_ratio) ** 4 
            except Exception:
                pass
                
        satellites_with_mauf.append((satellite, beam, w_score))
        
        if curr_sat_name is not None and satellite[0] == curr_sat_name and beam == curr_beam_index:
            w_curr = w_score

    if curr_sat_name is None or w_curr == -1.0:
        satellites_with_mauf = sorted(satellites_with_mauf, key=lambda x: x[2], reverse=True)
        return satellites_with_mauf[0][0], satellites_with_mauf[0][1]

    # Satisfaction Domain
    if w_curr > 0.60:
        return "STAY", None

    intra_candidates = [x for x in satellites_with_mauf if x[0][0] == curr_sat_name]
    best_intra = sorted(intra_candidates, key=lambda x: x[2], reverse=True)[0] if intra_candidates else None

    inter_candidates = [x for x in satellites_with_mauf if x[0][0] != curr_sat_name]
    best_inter = sorted(inter_candidates, key=lambda x: x[2], reverse=True)[0] if inter_candidates else None

    # Radio Link Failure Protection
    if w_curr < 0.05:
        all_sorted = sorted(satellites_with_mauf, key=lambda x: x[2], reverse=True)
        return all_sorted[0][0], all_sorted[0][1]

    # Inter-HO Evaluation (30% Margin)
    inter_approved = best_inter is not None and best_inter[2] > (w_curr * 1.30)

    # Intra-HO Evaluation (2% Margin)
    intra_approved = best_intra is not None and best_intra[1] != curr_beam_index and best_intra[2] > (w_curr * 1.02)

    if inter_approved and intra_approved:
        if best_inter[2] > best_intra[2]:
            return best_inter[0], best_inter[1]
        else:
            return best_intra[0], best_intra[1]
            
    elif inter_approved:
        return best_inter[0], best_inter[1]
        
    elif intra_approved:
        return best_intra[0], best_intra[1]
        
    else:
        return "STAY", None