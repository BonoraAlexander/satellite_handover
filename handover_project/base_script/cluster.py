from ue import Ue
from beam import Beam
from satellite import Satellite
from datetime import datetime, timedelta
import utils
import handover_strategies as strategies
import random
from satellite import Satellite
import numpy as np
import math
import rl_agent

class Cluster:
    def __init__(self, name, position, num_ues, beam_size_km, num_beams, satellites_frame, servers, mu_inter, mu_intra, scenario, enable_elevation = False, elevation_threshold = 0, rl_parameters = (0, 0, 0, False, False, "PPO")):
        self.name = name
        self.position = position
        self.num_ues = num_ues
        self.beam_size_km = beam_size_km
        self.num_beams = num_beams
        self.enable_elevation = enable_elevation
        self.elevation_threshold = elevation_threshold
        self.df_satellites_positions = satellites_frame
        self.sat_servers = servers
        self.sat_mu_inter = mu_inter
        self.sat_mu_intra = mu_intra
        self.scenario = scenario # stores the struct containing the parameters characterizing the scenario (e.g.: frequency, EIRP, etc) from the utils class.
        self.w1 = rl_parameters[0]
        self.w2 = rl_parameters[1]
        self.w3 = rl_parameters[2]
        self.enable_rl_algorithm = rl_parameters[3]
        self.enable_rl_learning = rl_parameters[4]
        self.agent_type = rl_parameters[5]

        # beams computation
        self.positions = self.calculate_beams_grid(self.position[0], self.position[1], self.beam_size_km, self.num_beams)
        self.list_beams = [Beam(self.name + "-Beam" + str(ii+1), ii, self.positions[ii], int(num_ues/num_beams), self.beam_size_km, int(np.sqrt(num_beams)), servers, mu_inter, mu_intra) for ii in range(self.num_beams)]

        if(self.enable_rl_algorithm):
            if(self.agent_type == "PPO"):
                self.rl_agent = rl_agent.PPOAgent() if self.enable_rl_algorithm else None
            elif(self.agent_type == "DQL"):
                self.rl_agent = rl_agent.DDQLAgent() if self.enable_rl_algorithm else None
            else:
                raise ValueError("Invalid agent type. Choose either 'PPO' or 'DQL'.")
            self.rl_agent = self.rl_agent.load_model() if self.rl_agent.load_model() is not None else self.rl_agent
            self.rl_agent.set_learning(self.enable_rl_learning)

    # in order to compute the position of the beams, we assume that they are arranged in a grid centered on the cluster position, 
    # and that the distance between adjacent beams is equal to the beam size. We then compute the latitude and longitude of each 
    # beam based on the center position and the beam size, taking into account the curvature of the Earth. 
    def calculate_beams_grid(self, center_lat, center_lon, beam_size_km, num_beams):
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



    # only at the beginning of the simulation, we attech every UEs of all the mini-cluster to a beam 
    # of a random satellite within the visibility.
    def initial_connection_phase(self, time, service_sats, handover_timer = 0):
        """
        This function handles the initial connection phase for the UEs in the mini-cluster.
        It should connect each UE to a random visible satellite. If no available satellite is visible, the UE will be 
        considered out of service until a satellite becomes visible.
        Inputs:
            - time: the current time of the simulation, used to determine the visible satellites at this time.
            - service_sats: a dictionary that will be updated with the satellites that are serving the UEs of the cluster.
        
        """
        # Find all visible satellites at the given time
        # round to the closest sec
        round_time = (time + timedelta(microseconds=500000)).replace(microsecond=0)
        visible_sats = utils.get_satellites_at_time(self.df_satellites_positions, round_time)

        # visible_sats identifies all the satellites visibled by at least one mini-cluster of the cluster.
        # we need to find which mini-clusters can see each satellite
        # "visible_sats_for_each_minicluster" is a list of lists, where the i-th element is the list of 
        # satellites visible from the i-th mini-cluster
        visible_sats_for_each_minicluster = [[] for _ in range(self.num_beams)]
        for sat in visible_sats:
            sat_lat, sat_lon = sat[1], sat[2]
            sat_cell_boundaries = utils.compute_cell_boundaries_lla(sat_lat, sat_lon, self.beam_size_km*1000, int(np.sqrt(self.num_beams)))
            visible_clusters_indices = utils.check_clusters_visibility(self.positions, sat_cell_boundaries, int(np.sqrt(self.num_beams)))

            # case when the current examinated satellite illimunate only a portion of a specific mini-cluster but not its center, i.e., it
            # illuminates less then 50%, so we conclude that mini-cluster cannot be served by that satellite beam.
            if(visible_clusters_indices.size == 0):
                continue

            satellite_beam_indices = utils.get_coverage_beam_indices_matrix(visible_clusters_indices, int(np.sqrt(self.num_beams)))
            
            rows, cols = visible_clusters_indices.shape
            for ii in range(rows):
                for jj in range(cols):
                    idx_cluster = visible_clusters_indices[ii][jj]
                    idx_sat_beam = satellite_beam_indices[ii][jj]
                    if(idx_sat_beam != -1):
                        visible_sats_for_each_minicluster[idx_cluster].append((sat, idx_sat_beam))
        
        for index, mini_cluster in enumerate(self.list_beams):
            mini_cluster.initial_connection_phase(visible_sats_for_each_minicluster[index], time, service_sats, handover_timer)
                
        return service_sats
    


    def monitor(self, time, service_sats, ho_condition, sat_selection_condition):
        """
            This function handles the monitoring of the current connections of the UEs and the handover process if needed. 
            It should be called at each time step of the simulation. For each mini-cluster, it checks the visibility of the satellites
            and determines if a handover is needed for each UE. If a handover is needed, it selects the target satellite randomly. 
            For the moment the only ho condition is the visibility. 

            visible_sats_for_each_minicluster is a list of lists, where each element is a list containing the visible 
            satellites for the corresponding mini-cluster as follows: [(sat1, idx_sat_beam1), (sat2, idx_sat_beam2), ...] 
            Specifically, satX is a tuple with the satellite info (name, lat, lon, alt) and idx_sat_beamX is the index of the 
            beam of the satellite that covers the mini-cluster.

            For example, if we have 3 mini-clusters, the structure of visible_sats_for_each_minicluster will be as follows:

            __                                                                                                              __
            |                                                                                                                |
            | [(sat1, idx_sat_beam1), ...]   ,   [(sat1, idx_sat_beam1), ...]   ,   [(sat1, idx_sat_beam1), ...]   ,   ...   |
            |_                                                                                                              _|
            
                mini-cluster 1 (index 0)            mini-cluster 2 (index 1)           mini-cluster 3 (index 2)        ...


            Handle the handover process for the UE.
            Possible scenarios:
            1) The UE is currently connected to a satellite and needs to handover to a new beam of the same satellite (intra-satellite handover).
            2) The UE is currently connected to a satellite and needs to handover to a new beam of a different satellite (inter-satellite handover).
            3) The UE is not currently connected to any satellite and needs to connect to a new beam of a satellite (inter-satellite handover).
            4) The Ue is not currently connected to any satellite and no satellite is visible, so it remains out of service (inter-satellite handover)).
        """
        # round to the closest sec
        round_time = (time + timedelta(microseconds=500000)).replace(microsecond=0)

        # check the visibility of all satellites respect to the whole cluster
        visible_sats = utils.get_satellites_at_time(self.df_satellites_positions, round_time)
        visible_sats_for_each_minicluster = [[] for _ in range(self.num_beams)]
        for sat in visible_sats:
            sat_lat, sat_lon, sat_alt = sat[1], sat[2], sat[3]
            sat_cell_boundaries = utils.compute_cell_boundaries_lla(sat_lat, sat_lon, self.beam_size_km*1000, int(np.sqrt(self.num_beams)))
            visible_clusters_indices = utils.check_clusters_visibility(self.positions, sat_cell_boundaries, int(np.sqrt(self.num_beams)), self.enable_elevation, self.elevation_threshold, sat_lat, sat_lon, sat_alt)

            # case when the current examinated satellite illimunate only a portion of a specific mini-cluster but not its center, i.e., it
            # illuminates less then 50%, so we conclude that mini-cluster cannot be served by that satellite beam.
            if(visible_clusters_indices.size == 0):
                continue

            satellite_beam_indices = utils.get_coverage_beam_indices_matrix(visible_clusters_indices, int(np.sqrt(self.num_beams)))
            rows, cols = visible_clusters_indices.shape
            for ii in range(rows):
                for jj in range(cols):
                    idx_cluster = visible_clusters_indices[ii][jj]
                    idx_sat_beam = satellite_beam_indices[ii][jj]
                    if(idx_sat_beam != -1):
                        visible_sats_for_each_minicluster[idx_cluster].append((sat, idx_sat_beam))

        # extract the rows of the dataframe related to the current time instant
        if isinstance(round_time, datetime):
            target_time_str = round_time.strftime("%Y-%m-%d %H:%M:%S")
        else:
            target_time_str = str(round_time)
        curr_time_df =self.df_satellites_positions[self.df_satellites_positions['time'].astype(str) == target_time_str].copy()


        # all the UEs who want to perform handover
        ho_ues = []

        # # make a screenshot of the current load for all the serving satellite
        for satellite in service_sats:
            service_sats[satellite].connected_ues_screenshot = service_sats[satellite].connected_ues.copy()

        # check if each UE needs to perform an intra or an inter handover based on the handover condition.
        for mini_cluster in self.list_beams:
            for ue in mini_cluster.list_ues:
                curr_sat, curr_beam_index = ue.get_connection_info()
                ue.intra_handover_flag = False
                ue.inter_handover_flag = False
                next_sat = None
                next_beam_index = None
                event = ho_condition[0]

                # ============== Detect the type of required handover (if needed) ==============

                # check if the current satellite is still visible from the mini-cluster of the UE
                if curr_sat is None: # UE is not connected to any satellite
                    index = -1
                else:
                    index = next((i for i, (obj, *_) in enumerate(visible_sats_for_each_minicluster[mini_cluster.index]) if obj[0] == curr_sat.name), -1)
                    
                if (index == -1): # the current satellite is not visible anymore --> inter handover
                    ue.inter_handover_flag = True
                elif (curr_beam_index != visible_sats_for_each_minicluster[mini_cluster.index][index][1]): # the current satellite is still visible --> check if intra handover is needed
                    ue.intra_handover_flag = True

                if(event == "SNR"):
                    dl_threshold, ul_threshold = ho_condition[1], ho_condition[2]
                    snr_dl, snr_ul = utils.get_snr(self.df_satellites_positions, round_time, curr_sat.name, mini_cluster.position, self.scenario)
                    dl_measurement_noise = random.gauss(0, self.scenario['dlul_snr_variance'])
                    ul_measurement_noise = random.gauss(0, self.scenario['dlul_snr_variance'])
                    snr_dl += dl_measurement_noise
                    snr_ul += ul_measurement_noise
                    if(snr_dl < dl_threshold or snr_ul < ul_threshold):
                        ue.inter_handover_flag = True
                elif(event == "TIMER"):
                    if(ue.time_to_next_handover <= 0):
                        ue.inter_handover_flag = True
                    else:
                        ue.time_to_next_handover -= 1 # decrease the time to next handover by 1 second
                elif(event == "ELEVATION"):
                    elev_threshold = ho_condition[1]
                    sat_elev = utils.get_elevation(curr_time_df, round_time, curr_sat.name, mini_cluster.position)
                    if(sat_elev < elev_threshold):
                        ue.inter_handover_flag = True
                elif(event == "A3"):
                    snr_difference_threshold = ho_condition[1]
                    enhanced_flag = ho_condition[2]
                    snr_dl = -100
                    best_snr_dl = -100
                    if curr_sat is not None:
                        if index != -1:
                            snr_dl, _ = utils.get_noisy_snr(self.df_satellites_positions, round_time, curr_sat.name, mini_cluster.position, self.scenario)
                        choices = len(visible_sats_for_each_minicluster[mini_cluster.index])
                        if(not enhanced_flag):
                            best_satellite, best_beam_index, best_snr_dl = strategies.get_best_neighbor_snr(visible_sats_for_each_minicluster[mini_cluster.index], curr_sat.name, round_time, mini_cluster, self.df_satellites_positions, self.scenario)
                        else:
                            best_satellite, best_beam_index, best_snr_dl = strategies.get_a_better_neighbor_snr(visible_sats_for_each_minicluster[mini_cluster.index], curr_sat.name, round_time, mini_cluster, self.df_satellites_positions, self.scenario, snr_dl, snr_difference_threshold)

                    else:
                        choices = len(visible_sats_for_each_minicluster[mini_cluster.index])
                        if(not enhanced_flag):
                            best_satellite, best_beam_index, best_snr_dl = strategies.get_best_neighbor_snr(visible_sats_for_each_minicluster[mini_cluster.index], "", round_time, mini_cluster, self.df_satellites_positions, self.scenario)
                        else:
                            best_satellite, best_beam_index, best_snr_dl = strategies.get_a_better_neighbor_snr(visible_sats_for_each_minicluster[mini_cluster.index], "", round_time, mini_cluster, self.df_satellites_positions, self.scenario, snr_dl, snr_difference_threshold)

                    satellite_out_visibility = index == -1
                    if((best_satellite is not None) and (best_snr_dl - snr_dl > snr_difference_threshold or satellite_out_visibility)):
                        # print(f"we perform inter-satellite handover since the neighboring snr is {best_snr_dl}, the current snr is {snr_dl}, and the difference is {best_snr_dl - snr_dl}.")
                        ue.inter_handover_flag = True
                        next_sat = best_satellite
                        next_beam_index = best_beam_index
                        # print()
                # print(f"=== UE {ue.id} CONNECTION INFORMAITON ===")
                # print(f"\tue.inter_handover_flag: {ue.inter_handover_flag}")
                # print(f"\tue.intra_handover_flag: {ue.intra_handover_flag}")
                # print(f"\tcurr_sat: {curr_sat}")
                # print(f"\tcurr_beam_index: {curr_beam_index}")
                # print(f"\tnext_sat: {next_sat}")
                # print(f"\tnext_beam_index: {next_beam_index}")



                ###################### RL HO ######################

                if(self.enable_rl_algorithm):
                    # if an intra or inter handover is needed, this is the time to train our RL algorithm
                    if(ue.inter_handover_flag):
                        # visible_sats_for_each_minicluster[mini_cluster.index] is a list containing (sat, idx_sat_beam) for each visible satellite of the mini-cluster
                        # sat is a tuple containing (sat_name, sat_lat, sat_lon, sat_alt, occurence_count_down)
                        states = []
                        sat_infos = []
                        for iii in visible_sats_for_each_minicluster[mini_cluster.index]:
                            sat_name, sat_lat, sat_lon, sat_alt, occurence_countdown = iii[0]
                            beam_index = iii[1]
                            distance_m = utils.compute_distance_m(sat_lat, sat_lon, sat_alt, mini_cluster.position[0], mini_cluster.position[1], 0)
                            snr_dl_db, _ = utils.compute_snr(distance_m, self.scenario)
                            # load is the number of UEs already connected to that beam of that satellite
                            load = 0
                            if(sat_name in service_sats):
                                load = service_sats[sat_name].connected_ues[beam_index] 
                            # if the current satellite is the same as the one we are considering (intra handover)
                            intra_flag = 0
                            current_sat_name = "ULISSE"
                            if(ue.connected_to is not None):
                                current_sat_name = ue.connected_to.name
                            if(current_sat_name == sat_name):
                                intra_flag = 1
                            candidate_sat = iii[0]
                            # sat_info is composed as follow ( (sat_name, sat_lat, sat_lon, sat_alt, occurence_countdown) , beam_index, snr_dl_db )
                            sat_infos.append((candidate_sat, beam_index, snr_dl_db))
                            connected_to = ue.connected_to
                            connected_to_beam = ue.connected_to_beam
                            curr_sat_load = 0
                            if connected_to is not None:
                                curr_sat_load = service_sats[connected_to.name].connected_ues_screenshot[connected_to_beam]
                            states.append((snr_dl_db, load, curr_sat_load))

                        # RL target satellite selection
                        if sat_infos:
                            sat_info = self.rl_agent.rl_algorithm_selection(ue.id, sat_infos, states)
                            next_sat, next_beam_index = sat_info[0], sat_info[1]
                            # this UE wants to be a sonar 
                            ho_ues.append((ue, sat_info))

                        else:
                            next_sat, next_beam_index = None, None
                        
                        selected_sat_name = "xxx"
                        if(next_sat is not None):
                            selected_sat_name = next_sat[0]

                        # if the next satellite is new then save it in the service_sats dictionary
                        # after that, perform the handover to the selected satellite and beam index
                        if(next_sat is not None):
                            selected_sat_name = next_sat[0]
                            if selected_sat_name not in service_sats:
                                sat = Satellite(selected_sat_name, self.sat_servers, self.sat_mu_inter, self.sat_mu_intra, self.num_beams)
                                service_sats[selected_sat_name] = sat
                            next_sat = service_sats[selected_sat_name]
                            if(ho_condition[0] == "TIMER"):
                                ue.time_to_next_handover = ho_condition[1] -1 # reset the time to next handover in case of fixed timer handover condition
                        if(ue.connected_to is not None and (ue.connected_to.name == selected_sat_name)):
                            ue.intra_handover(time, next_sat, next_beam_index)
                        else:
                            ue.inter_handover(time, next_sat, next_beam_index)
                    elif ue.intra_handover_flag:
                        next_sat = curr_sat
                        next_beam_index = visible_sats_for_each_minicluster[mini_cluster.index][index][1]
                        ue.intra_handover(time, next_sat, next_beam_index)

                ################################################

                ###################### CLASSIC HO ######################
                else:
                    # ============== Performe the handover (if selected) ==============
                    if(ue.inter_handover_flag): # handover to a new beam of a new satellite

                        # possible satellites beams towards which the ue could handover
                        choices = len(visible_sats_for_each_minicluster[mini_cluster.index])
                        # if no one, the ue goes out of service
                        if(choices == 0):
                            ue.time_to_next_handover = 0 # reset the time to next handover in case of fixed timer handover condition
                        # if there is at least one, select a random one among them and handover
                        elif(sat_selection_condition == "RANDOM"):
                            next_sat, next_beam_index = strategies.get_random_visible_satellite(visible_sats_for_each_minicluster[mini_cluster.index])
                        elif(sat_selection_condition == "MAX_ELEVATION"):
                            next_sat, next_beam_index = strategies.get_max_elevation_satellite(visible_sats_for_each_minicluster[mini_cluster.index], curr_time_df, round_time, mini_cluster)
                        elif(sat_selection_condition == "MAX_VISIBILITY"):
                            next_sat, next_beam_index = strategies.get_max_visibility_satellite(visible_sats_for_each_minicluster[mini_cluster.index], curr_time_df, round_time)
                        elif(sat_selection_condition == "AVL_THR"):
                            next_sat, next_beam_index = strategies.get_max_available_throughput_satellite(visible_sats_for_each_minicluster[mini_cluster.index], round_time, mini_cluster, service_sats, self.df_satellites_positions, self.scenario)
                        elif(sat_selection_condition == "A3"):
                            pass # since the next satellite informations are filled by the trigger event function, there is no need to do anything here.
                        if(next_sat is not None):
                            selected_sat_name = next_sat[0]
                            if selected_sat_name not in service_sats:
                                sat = Satellite(selected_sat_name, self.sat_servers, self.sat_mu_inter, self.sat_mu_intra, self.num_beams)
                                service_sats[selected_sat_name] = sat
                            next_sat = service_sats[selected_sat_name]
                            if(ho_condition[0] == "TIMER"):
                                ue.time_to_next_handover = ho_condition[1] -1 # reset the time to next handover in case of fixed timer handover condition
                        # if there is at least one, select the less busy satellite which could guarantee the higer thorughput
                        # handover to the selected sat (if no one, go out of service)
                        ue.inter_handover(time, next_sat, next_beam_index)
                    
                    elif(ue.intra_handover_flag): # handover to a new visible beam of the same satellite
                        next_sat = curr_sat
                        next_beam_index = visible_sats_for_each_minicluster[mini_cluster.index][index][1]
                        ue.intra_handover(time, next_sat, next_beam_index)

                ################################################

        # RL rewards computation
        if(self.enable_rl_algorithm):
            rewards = []
            for user, sat_info in ho_ues:
                ue_id = user.id
                if sat_info[0] is None:
                    rewards.append((ue_id, 0))
                    continue
                current_sat_name = sat_info[0][0]
                current_beam_index = sat_info[1]
                snr_dl_db = sat_info[2]
                # check to be sure everything is correct
                if(current_sat_name != user.connected_to.name or current_beam_index != user.connected_to_beam):
                    print(f"ERROR: {current_sat_name} != {user.connected_to.name} or {current_beam_index} != {user.connected_to_beam}")
                current_sat = service_sats[current_sat_name]

                # REWARD FUNCTION V1
                # load = current_sat.connected_ues[current_beam_index]
                # delay = user.remaining_handover_execution_time

                # snr_max = 24 # according to the scenario
                # load_max = 100 # number of UEs connected to the same beam of the same satellite
                # max_delay = 1000 # ms
                # capacity_factor = min(1, math.log2(1 + 10**(snr_dl_db/10)) / math.log2(1 + 10**(snr_max/10))) 
                # load_factor = min(1, (load-1) / (load_max))
                # delay_factor = min(1, delay / max_delay)
                # reward = self.w1 * capacity_factor - self.w2 * load_factor - self.w3 * delay_factor

                # REWARD FUNCTION V2
                max_dl_thr, max_ul_thr = utils.get_max_beam_throughput(self.df_satellites_positions, time, current_sat_name, (ue.lat, ue.lon, ue.alt), self.scenario)
                load = current_sat.connected_ues[current_beam_index]
                dl_ue_throughput = max_dl_thr / load
                ul_ue_throughput = max_ul_thr / load
                equivalent_snr_dl_db, equivalent_snr_ul_db = utils.reverse_snr_from_thr(dl_ue_throughput, ul_ue_throughput, self.scenario)
                equivalent_snr_dl_db -= self.scenario['dl_db_headroom']
                equivalent_snr_ul_db -= self.scenario['ul_db_headroom']
                dl_ue_throughput, ul_ue_throughput = utils.compute_shannon_from_snr(equivalent_snr_dl_db, equivalent_snr_ul_db, self.scenario)
                if(user.remaining_handover_execution_time >= 1000):
                    dl_ue_throughput = 0
                    ul_ue_throughput = 0
                elif(user.remaining_handover_execution_time > 0):
                    dl_ue_throughput = dl_ue_throughput * (1 - ue.remaining_handover_execution_time/1000)
                    ul_ue_throughput = ul_ue_throughput * (1 - ue.remaining_handover_execution_time/1000)
                dl_ue_throughput *= (1 - self.scenario['3gpp_overhead_dl']) 
                ul_ue_throughput *= (1 - self.scenario['3gpp_overhead_ul'])

                reward = min(1,(self.w1 * dl_ue_throughput)/40)

                rewards.append((ue_id, reward))
            if rewards:
                self.rl_agent.rl_rewards(rewards)

        self.save_instant_throughput(time)

    def save_instant_throughput(self, target_time):
        for mini_cluster in self.list_beams:
            for ue in mini_cluster.list_ues:
                serving_satellite, serving_beam_index = ue.get_connection_info()
                if(serving_satellite is None or serving_beam_index is None):
                    thr_info = {
                        "time": target_time,
                        "ue.id": ue.id,
                        "sat.id": None,
                        "max_dl_thr": 0,
                        "max_ul_thr": 0,
                        "connected_users": 1,
                        "ho_duration": 0,
                        "dl_thr": 0,
                        "ul_thr": 0
                    }
                    ue.thr_tracker.append(thr_info)
                    continue
                max_dl_thr, max_ul_thr = utils.get_max_beam_throughput(self.df_satellites_positions, target_time, serving_satellite.name, mini_cluster.position, self.scenario)
                connected_users = serving_satellite.connected_ues[serving_beam_index]
                # instant throughput computation
                dl_ue_throughput = max_dl_thr / connected_users
                ul_ue_throughput = max_ul_thr / connected_users
                # print(f"Instant throughput for UE {ue.id}: DL={dl_ue_throughput}, UL={ul_ue_throughput}")
                # reverse the shannon formula to compute the equivalent snr from the throughput
                equivalent_snr_dl_db, equivalent_snr_ul_db = utils.reverse_snr_from_thr(dl_ue_throughput, ul_ue_throughput, self.scenario)
                # print(f"Equivalent SNR for UE {ue.id}: DL={equivalent_snr_dl_db} dB, UL={equivalent_snr_ul_db} dB")
                equivalent_snr_dl_db -= self.scenario['dl_db_headroom']
                equivalent_snr_ul_db -= self.scenario['ul_db_headroom']
                # print(f"Equivalent SNR after accounting for headroom for UE {ue.id}: DL={equivalent_snr_dl_db} dB, UL={equivalent_snr_ul_db} dB")
                # now compute the throughput of the UE after accounting for the headroom
                dl_ue_throughput, ul_ue_throughput = utils.compute_shannon_from_snr(equivalent_snr_dl_db, equivalent_snr_ul_db, self.scenario)
                # print(f"Instant throughput for UE {ue.id} after accounting for headroom: DL={dl_ue_throughput}, UL={ul_ue_throughput}")
                ho_duration_ms = ue.remaining_handover_execution_time
                if(ue.remaining_handover_execution_time >= 1000):
                    dl_ue_throughput = 0
                    ul_ue_throughput = 0
                    ue.remaining_handover_execution_time -= 1000
                elif(ue.remaining_handover_execution_time > 0):
                    dl_ue_throughput = dl_ue_throughput * (1 - ue.remaining_handover_execution_time/1000)
                    ul_ue_throughput = ul_ue_throughput * (1 - ue.remaining_handover_execution_time/1000)
                    ue.remaining_handover_execution_time = 0

                dl_ue_throughput *= (1 - self.scenario['3gpp_overhead_dl']) # apply the 3GPP overhead to the throughput
                ul_ue_throughput *= (1 - self.scenario['3gpp_overhead_ul'])

                thr_info = {
                        "time": target_time,
                        "ue.id": ue.id,
                        "sat.id": serving_satellite.name,
                        "max_dl_thr": max_dl_thr,
                        "max_ul_thr": max_ul_thr,
                        "connected_users": connected_users,
                        "ho_duration": ho_duration_ms,
                        "dl_thr": dl_ue_throughput,
                        "ul_thr": ul_ue_throughput
                    }
                ue.thr_tracker.append(thr_info)