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
import random

class Cluster:
    def __init__(self, name, position, num_ues, beam_size_km, num_beams, satellites_frame, servers, mu_inter, mu_intra, scenario, pointing=True, enable_elevation = False, elevation_threshold = 0, rl_parameters = (0, 0, 0, False, False, "PPO"), max_serving_sats = 10):
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
        self.pointing = pointing
        self.w1 = rl_parameters[0]
        self.w2 = rl_parameters[1]
        self.w3 = rl_parameters[2]
        self.enable_rl_algorithm = rl_parameters[3]
        self.enable_rl_learning = rl_parameters[4]
        self.agent_type = rl_parameters[5]
        self.max_serving_sats = max_serving_sats

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

        # control the number of possible visible satellites to the maximum number of serving satellites
        visible_sats = random.sample(visible_sats, min(len(visible_sats), self.max_serving_sats))
        # already also configure the list of the serving satellites, even if we cannot assign any UE to them, since they are visible and could be used in the future. this allows also the mantain constant the number of serving satellites during the simulation.
        for sat in visible_sats:
            sat_name = sat[0]
            if sat_name not in service_sats:
                sat = Satellite(sat_name, self.sat_servers, self.sat_mu_inter, self.sat_mu_intra, self.num_beams)
                service_sats[sat_name] = sat
        
        for index, mini_cluster in enumerate(self.list_beams):
            mini_cluster.initial_connection_phase(visible_sats, time, service_sats, handover_timer)
                
        return service_sats
    


    def monitor(self, time, service_sats, ho_condition, sat_selection_condition):
        """
            This function handles the monitoring of the current connections of the UEs and the handover process if needed. 
            It should be called at each time step of the simulation.

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
        # if a current serving satellite is not visible anymore, we increment this variable, so as to know how many satellites add to the serving list
        count_of_sats_went_out_of_visibility = 0
        # count how many satellites went out of visibility and pop them from service_sats list
        for sat in list(service_sats.values()):
            exists = any(item[0] == sat.name for item in visible_sats)
            if not exists:
                service_sats.pop(sat.name)
                count_of_sats_went_out_of_visibility += 1
        # print(f"\n{count_of_sats_went_out_of_visibility} went out of visibility!")
        # selects the next serving satellites to include
        for iii in range(count_of_sats_went_out_of_visibility):
            # select the sat with the longest remaing visibility time and not already in service_sats
            next_sat = max((x for x in visible_sats if x[0] not in service_sats),key=lambda x: x[3],default=None)
            next_sat_name = next_sat[0]
            if next_sat_name in service_sats:
                print("Something wrong, I can feel it ... (in the next service sat selection)")
            else:
                sat = Satellite(next_sat_name, self.sat_servers, self.sat_mu_inter, self.sat_mu_intra, self.num_beams)
                service_sats[next_sat_name] = sat
                # print(f"Include {sat.name}")
        #extract the actual list of visible satellite according to service_sats, so only those are actually pointing toward the terrestrial cluster
        actual_visible_sats = [sat for sat in visible_sats if sat[0] in service_sats]
        # print(f"Now there are {len(service_sats)} serving")
                
        # extract the rows of the dataframe related to the current time instant
        if isinstance(round_time, datetime):
            target_time_str = round_time.strftime("%Y-%m-%d %H:%M:%S")
        else:
            target_time_str = str(round_time)
        curr_time_df =self.df_satellites_positions[self.df_satellites_positions['time'].astype(str) == target_time_str].copy()


        # all the UEs who want to perform handover
        ho_ues = []

        # make a screenshot of the current load for all the serving satellite
        for satellite in service_sats:
            service_sats[satellite].connected_ues_screenshot = service_sats[satellite].connected_ues.copy()

        # check if each UE needs to perform an intra or an inter handover based on the handover condition.
        for mini_cluster in self.list_beams:
            for ue in mini_cluster.list_ues:
                curr_sat, curr_beam_index = ue.get_connection_info()
                curr_sat_name = curr_sat.name
                ue.intra_handover_flag = False
                ue.inter_handover_flag = False
                next_sat = None
                next_beam_index = None
                event = ho_condition[0]

                # ============== Detect the type of required handover (if needed) ==============
                # check if the current satellite is still visible
                # UE is not connected to any satellite or the previous sat went out of visibility
                if curr_sat is None or curr_sat.name not in service_sats: 
                    ue.inter_handover_flag = True

                if(event == "SNR"):
                    dl_threshold, ul_threshold = ho_condition[1], ho_condition[2]
                    # handle the case in which the curr sat is None, i.e., the UE is not connected to any satellite, so it is out of service.
                    if(curr_sat is not None):
                        elevation_angle_deg = utils.get_elevation(curr_time_df, round_time, curr_sat.name, mini_cluster.position)
                        snr_dl, snr_ul = utils.get_noisy_snr(self.df_satellites_positions, round_time, curr_sat.name, mini_cluster.position, self.scenario, elevation_angle_deg, self.pointing)
                        dl_measurement_noise = random.gauss(0, self.scenario['dlul_snr_variance'])
                        ul_measurement_noise = random.gauss(0, self.scenario['dlul_snr_variance'])
                        snr_dl += dl_measurement_noise
                        snr_ul += ul_measurement_noise
                        if(snr_dl < dl_threshold or snr_ul < ul_threshold):
                            ue.inter_handover_flag = True
                    else:
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
                        snr_dl, _ = utils.get_noisy_snr(self.df_satellites_positions, round_time, curr_sat.name, mini_cluster.position, self.scenario, self.pointing)
                        choices = len(actual_visible_sats)
                        if(not enhanced_flag):
                            best_satellite, best_beam_index, best_snr_dl = strategies.get_best_neighbor_snr(actual_visible_sats, curr_sat.name, round_time, mini_cluster, self.df_satellites_positions, self.scenario)
                        else:
                            best_satellite, best_beam_index, best_snr_dl = strategies.get_a_better_neighbor_snr(actual_visible_sats, curr_sat.name, round_time, mini_cluster, self.df_satellites_positions, self.scenario, snr_dl, snr_difference_threshold)

                    else:
                        choices = len(actual_visible_sats)
                        if(not enhanced_flag):
                            best_satellite, best_beam_index, best_snr_dl = strategies.get_best_neighbor_snr(actual_visible_sats, "", round_time, mini_cluster, self.df_satellites_positions, self.scenario)
                        else:
                            best_satellite, best_beam_index, best_snr_dl = strategies.get_a_better_neighbor_snr(actual_visible_sats, "", round_time, mini_cluster, self.df_satellites_positions, self.scenario, snr_dl, snr_difference_threshold)

                    satellite_out_visibility = ue.inter_handover_flag
                    if((best_satellite is not None) and (best_snr_dl - snr_dl > snr_difference_threshold or satellite_out_visibility)):
                        ue.inter_handover_flag = True
                        next_sat = best_satellite
                        next_beam_index = best_beam_index

                ###################### CLASSIC HO ######################
                # ============== Performe the handover (if selected) ==============
                if(ue.inter_handover_flag): # handover to a new beam of a new satellite

                    # possible satellites beams towards which the ue could handover
                    choices = len(actual_visible_sats)
                    # if no one, the ue goes out of service
                    if(choices == 0):
                        ue.time_to_next_handover = 0 # reset the time to next handover in case of fixed timer handover condition
                    # if there is at least one, select a random one among them and handover
                    elif(sat_selection_condition == "RANDOM"):
                        next_sat = strategies.get_random_visible_satellite(actual_visible_sats)
                    elif(sat_selection_condition == "MAX_ELEVATION"):
                        next_sat = strategies.get_max_elevation_satellite(actual_visible_sats, curr_time_df, round_time, mini_cluster)
                    elif(sat_selection_condition == "MAX_VISIBILITY"):
                        next_sat = strategies.get_max_visibility_satellite(actual_visible_sats, curr_time_df, round_time)
                    elif(sat_selection_condition == "AVL_THR"):
                        next_sat, = strategies.get_max_available_throughput_satellite(actual_visible_sats, round_time, mini_cluster, service_sats, self.df_satellites_positions, self.scenario, self.pointing)
                    elif(sat_selection_condition == "A3"):
                        pass # since the next satellite informations are filled by the trigger event function, there is no need to do anything here.
                    if(next_sat is not None):
                        selected_sat_name = next_sat[0]
                        if selected_sat_name not in service_sats:
                            print("Something wrong, I can feel it ... (It should already be within service_sats list!)")
                        next_sat = service_sats[selected_sat_name]
                        if(ho_condition[0] == "TIMER"):
                            ue.time_to_next_handover = ho_condition[1] -1 # reset the time to next handover in case of fixed timer handover condition
                    # if there is at least one, select the less busy satellite which could guarantee the higer thorughput
                    # handover to the selected sat (if no one, go out of service)
                    ue.inter_handover(time, next_sat, mini_cluster.index)

                ################################################

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
                elevation_angle_deg = utils.get_elevation(self.df_satellites_positions, target_time, serving_satellite.name, mini_cluster.position)
                max_dl_thr, max_ul_thr = utils.get_max_beam_throughput(self.df_satellites_positions, target_time, serving_satellite.name, mini_cluster.position, self.scenario, elevation_angle_deg, self.pointing)
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