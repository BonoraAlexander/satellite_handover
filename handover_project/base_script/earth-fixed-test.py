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
df_name_1 = "200km_25beams_sc9_padova_2026_07_05_12_1h.csv"
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
num_ues = 25
max_serving_sats = 10
mu_inter = 30 * 1e-3
mu_intra = 1 * 1e-3 
servers = 1
scenario = utils.sc9_parameters
pointing = True # True if the satellite is pointing (Earth-Fixed), False otherwise (Earth-Moving). This flag doesn't affect the UE antenna orientation anyway. So, if the satellite is moving, the UE antenna will always point to the satellite, but the satellite will depending on "pointing".
handover_timer = 40
a3_event_snr_threshold = 2 # dB
enhanced_flag = False # don't take the best satellite, but a random one among the ones that are better than the current one

ho_condition_1 = ("VISIBILITY")
sat_selection_condition_1 = "RANDOM"

# RL parameters
w1 = 1 # capacity weight
w2 = 1 # load weight
w3 = 0 # delay weight
enable_rl_algorithm =  False # if True, use RL algorithm for target satellite selection for HO
eneable_rl_learning = False # if True, the RL agent will learn during the simulation, otherwise it will use a pre-trained model
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
cluster1 = Cluster("Cluster1", (45.40996, 11.89261, 0), num_ues, beam_size_km, num_beams, data_frame_1, servers, mu_inter, mu_intra, scenario, pointing, enable_elevation_threshold, elevation_threshold, rl_parameters, max_serving_sats)
clusters = [cluster1] 

# (# year, month, day, hour, minute, second)
time = datetime(year, month, day, minute, 0, 0) 
end_sim_time = time + simTime

total_iterations = int((end_sim_time - time).total_seconds()) # *1000 (sec --> ms) & /100 (every 100)

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