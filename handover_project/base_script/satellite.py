"""
This file defines the Satellite class and its internal HandoverManager. 
It represents the data plane nodes of the Non-Terrestrial Network, tracking connected users, 
propagating orbital positions via TLEs, and physically executing the handover directives issued by the central SDN controller.
"""

from skyfield.api import load, EarthSatellite
import pandas as pd
import os
from datetime import datetime

class Satellite:
    """
    Represents a single LEO satellite node within the network constellation. 
    It stores physical properties, manages the user load across its active beams, 
    and contains an internal HandoverManager to process network transitions.
    """
    def __init__(self, name, servers=1, mu_inter=0.03, mu_intra=0.001, num_beams=0, tle_data=None):
        """
        Initializes the satellite instance with its specific name, orbital parameters, and beam configuration. 
        It sets up the user counter array for load tracking and instantiates the internal HandoverManager.
        """
        self.name = name
        self.tle_data = tle_data
        self.num_beams = num_beams
        
        self.connected_ues = [0] * num_beams 
        
        self.handover_manager = self.HandoverManager(
            self, servers=servers, mu_inter=mu_inter, mu_intra=mu_intra
        )

    class HandoverManager:
        """
        Acts as the Data Plane engine of the satellite. 
        In this SDN architecture, congestion is resolved centrally via Temporal Trigger Spreading (TTS). 
        Therefore, this manager acts as a passive node that deterministically executes pure physical propagation delays without stochastic queueing.
        """
        def __init__(self, satellite, servers, mu_inter, mu_intra):
            """
            Initializes the handover manager with processing latencies. 
            It prepares an empty tracker list to log events, relying entirely on the central SDN for execution scheduling.
            """
            self.satellite = satellite
            self.servers = servers
            self.mu_inter = mu_inter
            self.mu_intra = mu_intra
            self.handover_tracker = []

        def process_handover_intra(self, arrival_time, ue, curr_beam_index, dest_satellite, dest_beam_index):
            """
            Processes a logical intra-satellite handover (beam switching). 
            It calculates the purely physical execution duration without queueing and returns a dictionary containing the detailed handover event metrics.
            """
            event_type = "intra_ho"
            duration = self.mu_intra 
            
            end_time_dt = arrival_time + pd.Timedelta(seconds=duration)

            handover_info = {
                "arrival_time": arrival_time,
                "event_type": event_type,
                "ue_id": ue.id,
                "from_satellite": self.satellite.name,
                "from_beam_index": curr_beam_index,
                "dest_satellite": dest_satellite.name,
                "dest_beam_index": dest_beam_index,
                "start_time": arrival_time,
                "departure_time": end_time_dt,
                "duration": duration,
                "dest_number_ues": dest_satellite.connected_ues.copy()
            }
                
            return handover_info

        def process_handover_inter(self, arrival_time, ue, curr_beam_index, dest_satellite, dest_beam_index):
            """
            Processes a physical inter-satellite handover (node switching). 
            It calculates the deterministic execution duration based on physical latency parameters and returns the structured handover event dictionary.
            """
            event_type = "inter_ho"
            duration = self.mu_inter 
            
            end_time_dt = arrival_time + pd.Timedelta(seconds=duration)

            handover_info = {
                "arrival_time": arrival_time,
                "event_type": event_type,
                "ue_id": ue.id,
                "from_satellite": self.satellite.name,
                "from_beam_index": curr_beam_index,
                "dest_satellite": dest_satellite.name,
                "dest_beam_index": dest_beam_index,
                "start_time": arrival_time,
                "departure_time": end_time_dt,
                "duration": duration,
                "dest_number_ues": dest_satellite.connected_ues.copy()
            }

            return handover_info

        def deactivate(self, label):
            """
            Terminates the manager's operations and exports the tracked physical transition logs for this specific satellite node into a structured CSV file for performance evaluation.
            """
            if self.handover_tracker:
                df = pd.DataFrame(self.handover_tracker)
                output_folder = f"Satellite dataframes_{label}"
                os.makedirs(output_folder, exist_ok=True)
                
                filename = f"{self.satellite.name}_handover_events.csv"
                full_path = os.path.join(output_folder, filename)
                df.to_csv(full_path, index=False)


    def connect_ue(self, beam_index):
        """
        Handles the connection of a single User Equipment (UE) by incrementing the active connection counter for the specified beam index.
        """
        self.connected_ues[beam_index] += 1
        return self.connected_ues[beam_index]
    
    def disconnect_ue(self, beam_index):
        """
        Handles the disconnection of a User Equipment (UE) by safely decrementing the connection counter for the specified beam index, preventing negative values.
        """
        if self.connected_ues[beam_index] > 0:
            self.connected_ues[beam_index] -= 1
        return self.connected_ues

    def deactivate(self, label):
        """
        Deactivates the satellite node at the end of the simulation. 
        It triggers the internal HandoverManager to save all tracked data and then cleanly removes the manager instance.
        """
        if self.handover_manager:
            self.handover_manager.deactivate(label)
            self.handover_manager = None

    def get_position(self, time):
        """
        Propagates the satellite's Two-Line Elements (TLEs) using the Skyfield library to compute and return the exact geocentric coordinates (latitude, longitude, altitude) at a requested simulation timestamp.
        """
        ts = load.timescale()
        t = ts.utc(time.year, time.month, time.day, time.hour, time.minute, time.second)

        satellite_obj = EarthSatellite(self.tle_data[1], self.tle_data[2])
        geocentric = satellite_obj.at(t) 
        subpoint = geocentric.subpoint()

        latitude = subpoint.latitude.degrees
        longitude = subpoint.longitude.degrees
        height = subpoint.elevation.m

        return latitude, longitude, height