"""
This module defines the Cluster class, which represents a specific geographic area containing multiple beams (mini-clusters) and User Equipments (UEs).
It serves as the interface between the physical layer and the control plane, managing physical visibility, simulating the Slotted ALOHA RACH protocol with collision handling, generating stochastic Markov traffic, and periodically invoking the centralized SDN Controller for global handover orchestration.
"""

from beam import Beam
from datetime import datetime, timedelta
import utils
import random
import numpy as np
from channel import Channel
from sdn_controller import SDN_Controller

class Cluster:
    def __init__(self, name, position, num_ues, beam_size_km, num_beams, satellites_frame, servers, mu_inter, mu_intra, scenario, enable_elevation=False, elevation_threshold=0):
        """
        Initializes the geographic cluster, sets up the physical boundaries, and instantiates the local beam grid.
        It also links the centralized SDN controller to this cluster and configures the stochastic ON/OFF Markov traffic model to simulate bursty user behaviors.
        """
        self.name = name
        self.position = position
        self.num_ues = num_ues
        self.beam_size_km = beam_size_km
        self.num_beams = num_beams
        self.enable_elevation = enable_elevation
        self.elevation_threshold = elevation_threshold
        self.frame = satellites_frame
        self.sat_servers = servers
        self.sat_mu_inter = mu_inter
        self.sat_mu_intra = mu_intra
        self.scenario = scenario

        self.sdn_controller = SDN_Controller(self, scenario, satellites_frame)

        self.positions = self.calculate_beams_grid(self.position[0], self.position[1], self.beam_size_km, self.num_beams)
        self.list_beams = [Beam(f"{self.name}-Beam{ii+1}", ii, self.positions[ii], int(num_ues/num_beams), self.beam_size_km, int(np.sqrt(num_beams)), servers, mu_inter, mu_intra) for ii in range(self.num_beams)]

        self.t_on_mean = 3.0   
        self.t_off_mean = 12.0 
        
        self.p_on_to_off = 1.0 / self.t_on_mean
        self.p_off_to_on = 1.0 / self.t_off_mean
        
        for mini_cluster in self.list_beams:
            for ue in mini_cluster.list_ues:
                ue.traffic_state = 'ON' if random.random() < (self.t_on_mean / (self.t_on_mean + self.t_off_mean)) else 'OFF'

    def calculate_beams_grid(self, center_lat, center_lon, beam_size_km, num_beams):
        """
        Computes the precise geocentric coordinates (latitude, longitude, altitude) for the center of each individual beam within the cluster's overarching footprint based on the specified beam size and grid layout.
        """
        grid_size = int(np.sqrt(num_beams))
        center_lat, center_lon = np.float64(center_lat), np.float64(center_lon)
        
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

    def initial_connection_phase(self, time, service_sats, handover_timer=0):
        """
        Bootstraps the simulation environment by evaluating the initial line-of-sight visibility of the satellite constellation and triggering the first routing optimization phase via the SDN controller.
        """
        round_time = (time + timedelta(microseconds=500000)).replace(microsecond=0)
        visible_sats = utils.get_satellites_at_time(self.frame, round_time)
        visible_sats_for_each_minicluster = [[] for _ in range(self.num_beams)]
        
        for sat in visible_sats:
            sat_lat, sat_lon = sat[1], sat[2]
            sat_cell_boundaries = utils.compute_cell_boundaries_lla(sat_lat, sat_lon, self.beam_size_km * 1000, int(np.sqrt(self.num_beams)))
            visible_clusters_indices = utils.check_clusters_visibility(self.positions, sat_cell_boundaries, int(np.sqrt(self.num_beams)))

            if visible_clusters_indices.size == 0:
                continue

            satellite_beam_indices = utils.get_coverage_beam_indices_matrix(visible_clusters_indices, int(np.sqrt(self.num_beams)))
            rows, cols = visible_clusters_indices.shape
            for ii in range(rows):
                for jj in range(cols):
                    idx_cluster = visible_clusters_indices[ii][jj]
                    idx_sat_beam = satellite_beam_indices[ii][jj]
                    if idx_sat_beam != -1:
                        visible_sats_for_each_minicluster[idx_cluster].append((sat, idx_sat_beam))
        
        for index, mini_cluster in enumerate(self.list_beams):
            mini_cluster.initial_connection_phase(visible_sats_for_each_minicluster[index], time, service_sats, handover_timer)
                
        self.sdn_controller.run_optimization(time, service_sats, visible_sats_for_each_minicluster)
        return service_sats

    def monitor(self, time, service_sats, ho_condition, sat_selection_condition):
        """
        Represents the core operational loop executed at each time step. 
        It evaluates the physical geometry of the constellation, updates UE Markov traffic states, applies EMA-filtered Gaussian noise to the physical layer metrics, and models the Slotted ALOHA RACH protocol including collision detection and exponential backoff mechanisms. 
        It periodically (or critically) invokes the SDN controller to re-optimize network routing.
        """
        round_time = (time + timedelta(microseconds=500000)).replace(microsecond=0)

        visible_sats = utils.get_satellites_at_time(self.frame, round_time)
        visible_sats_for_each_minicluster = [[] for _ in range(self.num_beams)]
        
        for sat in visible_sats:
            sat_lat, sat_lon, sat_alt = sat[1], sat[2], sat[3]
            sat_cell_boundaries = utils.compute_cell_boundaries_lla(sat_lat, sat_lon, self.beam_size_km * 1000, int(np.sqrt(self.num_beams)))
            visible_clusters_indices = utils.check_clusters_visibility(self.positions, sat_cell_boundaries, int(np.sqrt(self.num_beams)), self.enable_elevation, self.elevation_threshold, sat_lat, sat_lon, sat_alt)

            if visible_clusters_indices.size == 0:
                continue

            satellite_beam_indices = utils.get_coverage_beam_indices_matrix(visible_clusters_indices, int(np.sqrt(self.num_beams)))
            rows, cols = visible_clusters_indices.shape
            for ii in range(rows):
                for jj in range(cols):
                    idx_cluster = visible_clusters_indices[ii][jj]
                    idx_sat_beam = satellite_beam_indices[ii][jj]
                    if idx_sat_beam != -1:
                        visible_sats_for_each_minicluster[idx_cluster].append((sat, idx_sat_beam))

        for mini_cluster in self.list_beams:
            for ue in mini_cluster.list_ues:
                curr_sat, _ = ue.get_connection_info()
                
                if ue.traffic_state == 'ON':
                    if random.random() < self.p_on_to_off: 
                        ue.traffic_state = 'OFF'
                else:
                    if random.random() < self.p_off_to_on: 
                        ue.traffic_state = 'ON'
                
                if curr_sat is not None:
                    try:
                        raw_snr_dl, raw_snr_ul = utils.get_noisy_snr(self.frame, round_time, curr_sat.name, mini_cluster.position, self.scenario)
                        raw_elev = utils.get_elevation(self.frame, round_time, curr_sat.name, mini_cluster.position)
                    except:
                        raw_snr_dl, raw_snr_ul, raw_elev = 0, 0, 0
                else:
                    raw_snr_dl, raw_snr_ul, raw_elev = 0, 0, 0
                
                ue.update_ema_filters(raw_snr_dl, raw_snr_ul, raw_elev)

        rach_attempts = {}
        for mini_cluster in self.list_beams:
            for ue in mini_cluster.list_ues:
                if getattr(ue, 'scheduled_handover', None) is not None:
                    sched_time = ue.scheduled_handover['time']
                    
                    if sched_time <= time and ue.remaining_handover_execution_time == 0:
                        target_sat_obj = ue.scheduled_handover['sat']
                        if target_sat_obj is not None: 
                            rach_attempts.setdefault(target_sat_obj.name, []).append(ue)
        
        collided_ues = set()
        NUM_PREAMBLES = 64 
        
        for sat, ues_in_rach in rach_attempts.items():
            if len(ues_in_rach) > 1:
                
                preamble_map = {}
                for ue in ues_in_rach:
                    p = random.randint(1, NUM_PREAMBLES)
                    if p in preamble_map:
                        preamble_map[p].append(ue)
                    else:
                        preamble_map[p] = [ue]
                
                for p, users_with_same_preamble in preamble_map.items():
                    if len(users_with_same_preamble) > 1:
                        for ue in users_with_same_preamble:
                            collided_ues.add(ue.id) 

        for mini_cluster in self.list_beams:
            for ue in mini_cluster.list_ues:
                was_in_ho = ue.remaining_handover_execution_time > 0
                
                ue.execute_scheduled_handover(time)
                
                is_in_ho = ue.remaining_handover_execution_time > 0
                
                if (not was_in_ho) and is_in_ho and (ue.id in collided_ues):
                    backoff_delay_ms = random.randint(500, 1500) 
                    ue.remaining_handover_execution_time += backoff_delay_ms

        trigger_sdn = False
        if time.second % 5 == 0:
            trigger_sdn = True
        else:
            for mini_cluster in self.list_beams:
                for ue in mini_cluster.list_ues:
                    curr_sat, _ = ue.get_connection_info()
                    if curr_sat is not None and ue.ema_elevation is not None:
                        try:
                            elev_old = utils.get_elevation(self.frame, round_time - timedelta(seconds=1), curr_sat.name, mini_cluster.position)
                            derivata_elev = ue.ema_elevation - elev_old
                            
                            if derivata_elev < 0:
                                t_crit = (ue.ema_elevation - self.elevation_threshold) / abs(derivata_elev)
                                if t_crit <= 1.0:
                                    trigger_sdn = True
                                    break
                        except:
                            pass
                if trigger_sdn: break

        if trigger_sdn and sat_selection_condition == "PREHO":
            self.sdn_controller.run_optimization(time, service_sats, visible_sats_for_each_minicluster)

        self.save_instant_throughput(time)

    def save_instant_throughput(self, target_time):
        """
        Computes and records the instantaneous effective throughput for each UE in the cluster. 
        It evaluates the Shannon capacity based on the current channel state and the dynamic user load sharing the same beam resource. 
        It also scales down the available throughput to account for structural 3GPP protocol overheads and active handover execution downtimes.
        """
        active_users_map = {}
        for mc in self.list_beams:
            for u in mc.list_ues:
                sat_obj, b_idx = u.get_connection_info()
                if sat_obj is not None and getattr(u, 'traffic_state', 'OFF') == 'ON':
                    key = (sat_obj.name, b_idx)
                    active_users_map[key] = active_users_map.get(key, 0) + 1

        for mini_cluster in self.list_beams:
            for ue in mini_cluster.list_ues:
                serving_satellite, serving_beam_index = ue.get_connection_info()
                
                if serving_satellite is None or serving_beam_index is None:
                    ue.thr_tracker.append({
                        "time": target_time, "ue.id": ue.id, "sat.id": None,
                        "max_dl_thr": 0, "max_ul_thr": 0, "connected_users": 1,
                        "ho_duration": 0, "dl_thr": 0, "ul_thr": 0
                    })
                    continue
                
                max_dl_thr, max_ul_thr = utils.get_max_beam_throughput(self.frame, target_time, serving_satellite.name, mini_cluster.position, self.scenario)
                
                key = (serving_satellite.name, serving_beam_index)
                active_users_in_beam = max(1, active_users_map.get(key, 0))
                
                dl_ue_throughput = max_dl_thr / active_users_in_beam
                ul_ue_throughput = max_ul_thr / active_users_in_beam
                
                eq_snr_dl, eq_snr_ul = Channel.reverse_snr_from_thr(dl_ue_throughput, ul_ue_throughput, self.scenario)
                eq_snr_dl -= self.scenario['dl_db_headroom']
                eq_snr_ul -= self.scenario['ul_db_headroom']
                
                dl_ue_throughput, ul_ue_throughput = Channel.compute_shannon_from_snr(eq_snr_dl, eq_snr_ul, self.scenario)
                
                ho_duration_ms = ue.remaining_handover_execution_time
                if ue.remaining_handover_execution_time >= 1000:
                    dl_ue_throughput, ul_ue_throughput = 0, 0
                    ue.remaining_handover_execution_time -= 1000
                elif ue.remaining_handover_execution_time > 0:
                    loss_factor = 1 - (ue.remaining_handover_execution_time / 1000)
                    dl_ue_throughput *= loss_factor
                    ul_ue_throughput *= loss_factor
                    ue.remaining_handover_execution_time = 0
                
                dl_ue_throughput *= (1 - self.scenario['3gpp_overhead_dl'])
                ul_ue_throughput *= (1 - self.scenario['3gpp_overhead_ul'])

                ue.thr_tracker.append({
                        "time": target_time, "ue.id": ue.id, "sat.id": serving_satellite.name,
                        "max_dl_thr": max_dl_thr, "max_ul_thr": max_ul_thr,
                        "connected_users": active_users_in_beam, "ho_duration": ho_duration_ms,
                        "dl_thr": dl_ue_throughput, "ul_thr": ul_ue_throughput
                    })