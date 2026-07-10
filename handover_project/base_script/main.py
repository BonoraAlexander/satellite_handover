from datetime import datetime, timedelta
import pandas as pd
from tqdm import tqdm
import argparse
import os 
import shutil
import re

from cluster import Cluster
import utils


# initial configuration
df_name_1 = "100km_25beams_sc9_padova_2026_02_18_00_24h.csv"#"100km_25beams_sc9_padova.csv"
########################################
# retrive parameters
data_frame_1 = pd.read_csv(df_name_1)
numbers = re.findall(r'\d+', df_name_1)
beam_size_km = int(numbers[0])
num_beams = int(numbers[1])
year = int(numbers[3])
month = int(numbers[4])
day = int(numbers[5])
minute = int(numbers[6])
duration_h = int(numbers[7])
########################################
enable_elevation_threshold = True
elevation_threshold = 30
enable_doppler_computation = False

simTime = timedelta(hours=duration_h)
num_ues = 100
mu_inter = 30 * 1e-3
mu_intra = 1 * 1e-3 
servers = 1
scenario = utils.sc9_parameters
handover_timer = 40
a3_event_snr_threshold = 2 # dB
enhanced_flag = False # don't take the best satellite, but a random one among the ones that are better than the current one

ho_condition_1 = ("A3", a3_event_snr_threshold, enhanced_flag)
sat_selection_condition_1 = "A3"

# RL parameters
w1 = 1 # capacity weight
w2 = 1 # load weight
w3 = 0 # delay weight
enable_rl_algorithm =  False # if True, use RL algorithm for target satellite selection for HO
eneable_rl_learning = True # if True, the RL agent will learn during the simulation, otherwise it will use a pre-trained model
agent_type = "PPO" # RL agent type: "PPO" or "DQL"
rl_parameters = (w1, w2, w3, enable_rl_algorithm, eneable_rl_learning, agent_type)

####################################
########### ho_condition ###########
####################################
# ("SNR", dl_threshold, ul_threshold): if the SNR goes under certain thresholds then handover to a new satellite
# ("ELEVATION", elev_threshold): if the elevation angle goes under certain thresholds then handover to a new satellite
# ("TIMER", handover_timer): if not already triggered, handover to a new satellite after handover_timer seconds
# ("VISIBILITY"): standard approach, no input needed, handover when satellite goes out of visibility
# ("A3", a3_event_snr_threshold, enhanced_flag): 3GPP Event-A3: monitor neighbouring cells, handover if one becomes better by a3_event_snr_threshold dB. if enhanced flag is true, select a random sat among the ons with higher SNR than the current one by a3_event_snr_threshold dB, otherwise select the best one.

###############################################
########### sat_selection_condition ###########
###############################################
# "RANDOM": a random satellite within the ones in visibility
# "MAX_ELEVATION": the satellite with the highest elevation angle from the current time instant
# "MAX_VISIBILITY": the satellite with the longer visibility window from the current time instant
# "AVL_THR": the satellite with the highest available throughput is selected as target satellite
# "A3": the only possible sat selection when using A3 as ho condition.



# parsing input parameters 
parser = argparse.ArgumentParser(description="Satellite Simulation Script")
parser.add_argument('--servers', type=int, default=servers, help='Number of servers')
parser.add_argument('--num_ues', type=int, default=num_ues, help='Number of User Equipments')
args= parser.parse_args()
servers = args.servers
num_ues = args.num_ues

# (name, position, num_ues, satellites_frame, threshold_snr, satellite servers, satellite mu)
cluster1 = Cluster("Cluster1", (45.40996, 11.89261, 0), num_ues, beam_size_km, num_beams, data_frame_1, servers, mu_inter, mu_intra, scenario, enable_elevation_threshold, elevation_threshold, rl_parameters)
clusters = [cluster1] 

# (# year, month, day, hour, minute, second)
time = datetime(year, month, day, minute, 0, 0) 
end_sim_time = time + simTime

# Initial connection phase: each ue connects to a random satellite
service_sats = {}
for cluster in clusters:    
    cluster.initial_connection_phase(time, service_sats, handover_timer)


# increment the time by 100 ms
time += timedelta(seconds=1)

total_iterations = int((end_sim_time - time).total_seconds()) # *1000 (sec --> ms) & /100 (every 100)

print("\nStarting Simulation...")

with tqdm(total=total_iterations, desc="Simulating") as pbar:
    
    # Monitor the SNR of the current connections and apply conditional handover if needed
    while time < end_sim_time:

        cluster1.monitor(time, service_sats, ho_condition_1, sat_selection_condition_1)
        
        # Display the current time on the right side of the progress bar instead of printing it
        pbar.set_postfix(time=time.strftime("%H:%M:%S"))
        
        # increment the time by 1 sec
        time += timedelta(seconds=1)
        
        # Manually advance the progress bar by 1 tick
        pbar.update(1)


print("Simulation Complete!\n")

if(enable_doppler_computation):
    print("Computing Doppler Shifts for all the UEs ...")

    total_iterations = num_ues
    with tqdm(total=total_iterations, desc="Simulating") as pbar:
        for cluster in clusters: 
            frame = cluster.df_satellites_positions

            # in order to increase lookup speed, convert DataFrame to an O(1) lookup dictionary once per cluster
            # we use zip() as it is very fast for iterating through pandas columns
            sat_positions_map = {
                (str(sat), str(t)): (lat, lon, alt)
                for sat, t, lat, lon, alt in zip(
                    frame['sat_name'], 
                    frame['time'], 
                    frame['sat_lat'], 
                    frame['sat_lon'], 
                    frame['sat_height']
                )
            }

            # cache timestamp: avoid recalculating time strings for the same time instant
            time_cache = {}

            for mini_cluster in cluster.list_beams:
                ue_pos = mini_cluster.position
                
                for ue in mini_cluster.list_ues:
                    handover_info = ue.thr_tracker
                    
                    for line in handover_info:
                        time = line["time"]
                        sat_name = str(line["sat.id"])

                        # fetch or compute time strings (Past, Present, Future)
                        if time not in time_cache:
                            if isinstance(time, datetime):
                                t_curr = time.strftime("%Y-%m-%d %H:%M:%S")
                                t_old = (time - pd.Timedelta(seconds=1)).strftime("%Y-%m-%d %H:%M:%S")
                                t_fut = (time + pd.Timedelta(seconds=1)).strftime("%Y-%m-%d %H:%M:%S")
                            else:
                                t_curr = str(time)
                                t_old = str(time - pd.Timedelta(seconds=1))
                                t_fut = str(time + pd.Timedelta(seconds=1))
                            time_cache[time] = (t_old, t_curr, t_fut)
                        
                        t_old, t_curr, t_fut = time_cache[time]

                        # retrieve positions instantly from the dictionary
                        # .get() safely returns (None, None, None) if the key isn't found
                        default_pos = (None, None, None)
                        old_pos = sat_positions_map.get((sat_name, t_old), default_pos)
                        curr_pos = sat_positions_map.get((sat_name, t_curr), default_pos)
                        fut_pos = sat_positions_map.get((sat_name, t_fut), default_pos)

                        # compute doppler shifts
                        doppler_dl, doppler_ul = utils.compute_doppler_shift(
                            ue_pos, old_pos, curr_pos, fut_pos, scenario
                        )
                        
                        line['doppler_shift_dl_KHz'] = doppler_dl / 1000
                        line['doppler_shift_ul_KHz'] = doppler_ul / 1000
                    pbar.set_postfix(time=time.strftime("%H:%M:%S"))
                    pbar.update(1)
                
    print("Computation completed!\n")

if enable_rl_algorithm:
    cluster1.rl_agent.save_model()

print("Creating the folder with the ue dataframes ...")


output_folders = (
    [f"{cluster.name} dataframes" for cluster in clusters] +
    [f"{cluster.name} throughput" for cluster in clusters] +
    ["Satellite dataframes"]
)

# if there are old results, delete them
for folder in output_folders:
    if os.path.exists(folder):
        shutil.rmtree(folder)

total_iterations = len(service_sats)
with tqdm(total=total_iterations, desc="Simulating") as pbar:
    for name, sat in service_sats.items():
        sat.deactivate()
        pbar.set_postfix(time=time.strftime("%H:%M:%S"))
        pbar.update(1)

print("Folder created!\n")

print("Creating the folder with the sat dataframes ...")

total_iterations = num_ues
with tqdm(total=total_iterations, desc="Simulating") as pbar:
    for cluster in clusters:
        for mini_cluster in cluster.list_beams:
            for ue in mini_cluster.list_ues:
                ue.deactivate(cluster.name)
                pbar.set_postfix(time=time.strftime("%H:%M:%S"))
                pbar.update(1)

print("Folder created!\n")
