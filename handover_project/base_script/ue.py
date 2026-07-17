"""
This file defines the User Equipment (UE) class for the Non-Terrestrial Network (NTN) simulation. 
Under the centralized SDN architecture, the UE operates as a passive terminal. 
It maintains its geographical position, applies Exponential Moving Average (EMA) filters to smooth physical channel noise, tracks performance metrics, and executes scheduled handover directives strictly dictated by the central controller.
"""

import pandas as pd
import os

class Ue:
    def __init__(self, ue_id, position):
        """
        Initializes the User Equipment (UE) with its unique identifier and geographical coordinates. 
        It sets up the initial disconnected state, internal variables for SDN scheduled handovers, 
        EMA filters for signal smoothing, and metric trackers for post-simulation analysis.
        """
        self.id = ue_id
        self.lat = position[0]
        self.lon = position[1]
        self.alt = position[2]
        
        self.connected_to = None
        self.connected_to_beam = None
        
        self.scheduled_handover = None 
        
        self.ema_snr_dl = None
        self.ema_snr_ul = None
        self.ema_elevation = None
        self.EMA_ALPHA = 0.3 

        self.handover_tracker = []
        self.thr_tracker = []
        self.remaining_handover_execution_time = 0 

    def update_ema_filters(self, raw_snr_dl, raw_snr_ul, raw_elev):
        """
        Updates the Exponential Moving Average (EMA) filters for downlink SNR, uplink SNR, and perceived elevation. 
        This mitigates the Gaussian noise from the physical layer readings, ensuring metric stability for the SDN controller's decision-making process.
        """
        if self.ema_snr_dl is None:
            self.ema_snr_dl = raw_snr_dl
            self.ema_snr_ul = raw_snr_ul
            self.ema_elevation = raw_elev
        else:
            self.ema_snr_dl = (self.EMA_ALPHA * raw_snr_dl) + ((1 - self.EMA_ALPHA) * self.ema_snr_dl)
            self.ema_snr_ul = (self.EMA_ALPHA * raw_snr_ul) + ((1 - self.EMA_ALPHA) * self.ema_snr_ul)
            self.ema_elevation = (self.EMA_ALPHA * raw_elev) + ((1 - self.EMA_ALPHA) * self.ema_elevation)
        
        return self.ema_snr_dl, self.ema_snr_ul, self.ema_elevation

    def execute_scheduled_handover(self, current_time):
        """
        Acts as the passive execution engine for the UE. 
        It continuously checks if the current simulation time matches or exceeds the execution timestamp dictated by the SDN's Temporal Trigger Spreading (TTS) mechanism. 
        If the trigger condition is met, it routes the execution to either an intra-satellite or inter-satellite handover.
        """
        if self.scheduled_handover is not None:
            exec_time = self.scheduled_handover['time']
            
            if current_time >= exec_time:
                dest_sat = self.scheduled_handover['sat']
                dest_beam = self.scheduled_handover['beam']
                
                curr_sat, curr_beam_index = self.get_connection_info()
                
                if curr_sat is not None and dest_sat is not None and curr_sat.name == dest_sat.name:
                    self.intra_handover(current_time, curr_sat, dest_beam)
                else:
                    self.inter_handover(current_time, dest_sat, dest_beam)
                
                self.scheduled_handover = None
                return True 
                
        return False

    def get_connection_info(self):
        """
        Returns the current routing pointers, specifically the active satellite object and the active beam index to which the UE is currently connected.
        """
        return self.connected_to, self.connected_to_beam

    def connect_to_satellite(self, satellite, beam_index):
        """
        Physically establishes the connection to a new satellite node by updating the internal pointers to the target satellite and beam.
        """
        self.connected_to = satellite
        self.connected_to_beam = beam_index

    def disconnect(self):
        """
        Severs the current connection, effectively putting the UE in a link failure or Out of Service (OOS) state by clearing the active routing pointers.
        """
        self.connected_to = None
        self.connected_to_beam = None

    def deactivate(self, cluster_name, label):
        """
        Terminates the UE's lifecycle at the end of the simulation. 
        It exports the logged handover events and throughput telemetry into structured CSV files for post-simulation analytics.
        """
        if self.handover_tracker:
            df_ho = pd.DataFrame(self.handover_tracker)
            output_folder_ho = f"{cluster_name} dataframes_{label}"
            os.makedirs(output_folder_ho, exist_ok=True)
            df_ho.to_csv(os.path.join(output_folder_ho, f"{self.id}_handover_events.csv"), index=False)

        if self.thr_tracker:
            df_thr = pd.DataFrame(self.thr_tracker)
            output_folder_thr = f"{cluster_name} throughput_{label}"
            os.makedirs(output_folder_thr, exist_ok=True)
            df_thr.to_csv(os.path.join(output_folder_thr, f"{self.id}_thr_over_time.csv"), index=False)

    def inter_handover(self, time, dest_sat, dest_beam_index):
        """
        Manages the physical transition to a different satellite node (Inter-HO). 
        It handles varying connection states, including complete loss of coverage, reconnection from an offline state, and standard node-to-node transitions. 
        It invokes the target satellite's handover manager and logs the transition metrics.
        """
        curr_sat, curr_beam_index = self.get_connection_info()

        if dest_sat is None:
            event_type = "lost_conn" if curr_sat is not None else "out_serv"
            curr_sat_name = curr_sat.name if curr_sat is not None else None

            handover_info = {
                    "arrival_time": time, "event_type": event_type, "ue_id": self.id,
                    "from_satellite": curr_sat_name, "from_beam_index": curr_beam_index,
                    "dest_satellite": None, "dest_beam_index": None,
                    "start_time": time, "departure_time": 0, "duration": 1, "dest_number_ues": None
                }

            if curr_sat is not None:
                curr_sat.disconnect_ue(curr_beam_index)
                curr_sat.handover_manager.handover_tracker.append(handover_info)

            self.disconnect()
            self.remaining_handover_execution_time = handover_info["duration"]
            self.handover_tracker.append(handover_info)
            return 
        
        if curr_sat is None: 
            handover_info = {
                    "arrival_time": time, "event_type": "rest_conn", "ue_id": self.id,
                    "from_satellite": None, "from_beam_index": None,
                    "dest_satellite": dest_sat.name, "dest_beam_index": dest_beam_index,
                    "start_time": time, "departure_time": 0, "duration": 1,
                    "dest_number_ues": dest_sat.connected_ues.copy()
                }
            dest_sat.handover_manager.handover_tracker.append(handover_info)
            dest_sat.connect_ue(dest_beam_index)

            self.remaining_handover_execution_time = handover_info["duration"] * 1000
            self.handover_tracker.append(handover_info)
            self.connect_to_satellite(dest_sat, dest_beam_index)

        else: 
            handover_info = curr_sat.handover_manager.process_handover_inter(time, self, curr_beam_index, dest_sat, dest_beam_index)

            dest_sat.handover_manager.handover_tracker.append(handover_info)
            dest_sat.connect_ue(dest_beam_index)
            curr_sat.handover_manager.handover_tracker.append(handover_info)
            curr_sat.disconnect_ue(curr_beam_index)

            self.remaining_handover_execution_time = handover_info["duration"] * 1000
            self.handover_tracker.append(handover_info)
            self.connect_to_satellite(dest_sat, dest_beam_index)

    def intra_handover(self, time, dest_sat, dest_beam_index):
        """
        Manages the logical transition between two different beams within the same serving satellite (Intra-HO). 
        It processes the structural latency and updates the internal counters of the current satellite node.
        """
        curr_sat, curr_beam_index = self.get_connection_info()
        handover_info = curr_sat.handover_manager.process_handover_intra(time, self, curr_beam_index, dest_sat, dest_beam_index)
        
        curr_sat.handover_manager.handover_tracker.append(handover_info)
        curr_sat.disconnect_ue(curr_beam_index)
        dest_sat.connect_ue(dest_beam_index)

        self.connect_to_satellite(dest_sat, dest_beam_index) 
        self.remaining_handover_execution_time = handover_info["duration"] * 1000
        self.handover_tracker.append(handover_info)