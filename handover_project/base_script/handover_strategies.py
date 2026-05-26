import random
import utils

"""
This module contains the implementation of different handover strategies. Each strategy is implemented as a standalone function. 
Each function can have its own paraemters, but the return value of each function should be a tuple (next satellite, next beam index)
specifying the satellite to connect to next and the beam index the UE should connect to in that satellite.

"""

def get_random_visible_satellite(visible_satellites):
    """
    implementation of a random selection amongst the visible satellites. The next satellite is selected randomly
    from the list of the visible satellites.
    Args:
        visible_satellites (list): list of tuples (satellite, beam_index) representing the satellites in visibility and the 
            corresponding beam index that the UE can connect to.
    Returns:
        tuple: (next_sat, next_beam_index) where next_sat is the tuple representing thesatellite to connect to, along with
            the respective beam index. returns None if there are no visible satellties.
    """
    number_of_satellites = len(visible_satellites)
    if(number_of_satellites == 0):
        return None
    random_index = random.randint(0, number_of_satellites-1) # pick a random index
    next_sat = visible_satellites[random_index][0] # get the satellite corresponding to the random index
    next_beam_index = visible_satellites[random_index][1]  # get the beam index that is covering the ue
    return next_sat, next_beam_index

def get_max_elevation_satellite(visible_satellites, current_positions_df, round_time, mini_cluster):
    """
    selecting the satellite with the maximum elevation angle from the ones in visibility for the UE.
    Args:
        visible_satellites (list): list of tuples (satellite, beam_index) representing the satellites in visibility and the 
            corresponding beam index that the UE can connect to.
        current_positions_df (pd.DataFrame): dataframe containing the positions of the visible satellites.
        round_time (int): current simulation time.
        mini_cluster (Beam): the Beam object containing the information on the mini cluster for which we are selecting the next satellite.
    Returns:
        tuple: (next_sat, next_beam_index) where next_sat is the satellite to connect to, along with the respective beam index. 
            returns None if there are no visible satellties.
    """

    satellites_with_elevation = []
    # loop on the visible satellites and get the elevation for each
    for satellite, beam in visible_satellites:
        satellite_name = satellite[0]
        elevation = utils.get_elevation(current_positions_df, round_time, satellite_name, mini_cluster.position)
        satellites_with_elevation.append((satellite, beam, elevation))
    # sort the satellites based on the elevation and pick a random one amongst the top 30%.
    # this to avoid all the UEs to converge to the same satellite
    satellites_with_elevation = sorted(satellites_with_elevation, key=lambda x: x[2], reverse=True)
    choices = max(int(0.3*len(satellites_with_elevation)), 1)
    random_index = random.randint(0, choices-1)
    next_sat = satellites_with_elevation[random_index][0]
    next_beam_index = satellites_with_elevation[random_index][1]
    return next_sat, next_beam_index

def get_max_visibility_satellite(visible_satellites, current_positions_df, round_time):
    """
    selecting the satellite with the maximum remaining visibility time from the ones in visibility for the UE.
    Args:
        visible_satellites (list): list of tuples (satellite, beam_index) representing the satellites in visibility and the 
            corresponding beam index that the UE can connect to.
        current_positions_df (pd.DataFrame): dataframe containing the positions of the visible satellites.
        round_time (int): current simulation time.
    Returns:
        tuple: (next_sat, next_beam_index) where next_sat is the satellite to connect to, along with the respective beam index. 
            returns None if there are no visible satellties.
    """
    satellites_with_visibility = []
    # loop on the visible satellites and get the remaining visibility time for each
    for satellite, beam in visible_satellites:
        satellite_name = satellite[0]
        vis_time = utils.get_visibility_time(satellite_name, round_time, current_positions_df)
        satellites_with_visibility.append((satellite, beam, vis_time))
    # sort the satellites based on the remaining visibility time and pick a random one amongst the top 30%.
    # this to avoid all the UEs to converge to the same satellite
    satellites_with_visibility = sorted(satellites_with_visibility, key=lambda x: x[2], reverse=True)
    choices = max(int(0.3*len(satellites_with_visibility)), 1)
    random_index = random.randint(0, choices-1)
    next_sat = satellites_with_visibility[random_index][0]
    next_beam_index = satellites_with_visibility[random_index][1]
    return next_sat, next_beam_index

def get_max_available_throughput_satellite(visible_satellites, round_time, mini_cluster, service_satellites, mainframe, scenario):
    """
    selecting the satellite capable of providing the highest throughput value also considering the current load from the ones in visibility for the UE.
    Args:
        visible_satellites (list): list of tuples (satellite, beam_index) representing the satellites in visibility and the 
            corresponding beam index that the UE can connect to.
        round_time (int): current simulation time.
        mini_cluster (Beam): the Beam object containing the information on the mini cluster for which we are selecting the next satellite.
        service_satellites (list): list of satellites that are currently providing service.
        mainframe (pd.DataFrame): the main dataframe containing the information on satellite positions over time.
        scenario (dict): dictionary containing the scenario parameters.
    Returns:
        tuple: (next_sat, next_beam_index) where next_sat is the satellite to connect to, along with the respective beam index. 
            returns None if there are no visible satellties.
    """
    satellites_with_throughput = []
    # loop on the visible satellites, compute and save the throughput for each 
    for satellite, beam in visible_satellites:
        satellite_name = satellite[0]
        num_connected_ues = 1
        if satellite_name in service_satellites:
            num_connected_ues += service_satellites[satellite_name].connected_ues[beam]
        # compute the max throughput based on shannon formula, snr, overhead
        max_dl_thr, max_ul_thr = utils.get_max_beam_throughput(mainframe, round_time, satellite_name, mini_cluster.position, scenario)
        # split resources amongst the number of connected users
        thr_score = max_dl_thr / num_connected_ues
        satellites_with_throughput.append((satellite, beam, thr_score))
    # sort the throughput
    satellites_with_throughput = sorted(satellites_with_throughput, key=lambda x: x[2], reverse=True)
    best_index = 0
    next_sat = satellites_with_throughput[best_index][0]
    next_beam_index = satellites_with_throughput[best_index][1]
    return next_sat, next_beam_index

def get_best_neighbor_snr(visible_satellites, curr_sat_name, round_time, mini_cluster, df_satellites_positions, scenario):
    """
    implementation of 3GPP Event-A3 handover condition, that is, the handover is triggered when the UE detects that
    the SNR of a neighboring satellite becomes higher than the SNR of the serving cell by a certain threshold, usually set to 2 dB.
    """
    best_satellite = None
    best_snr_dl = None
    best_beam_index = None
    for satellite_tuple, beam_index in visible_satellites:
        satellite_name = satellite_tuple[0]
        if(curr_sat_name != satellite_name):
            current_snr_dl, _ = utils.get_noisy_snr(df_satellites_positions, round_time, satellite_name, mini_cluster.position, scenario)
            if(best_snr_dl is None or current_snr_dl > best_snr_dl):
                best_snr_dl = current_snr_dl
                best_satellite = satellite_tuple
                best_beam_index = beam_index
    return best_satellite, best_beam_index, best_snr_dl

