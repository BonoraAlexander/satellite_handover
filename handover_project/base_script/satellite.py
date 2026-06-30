from pyclbr import Class
from skyfield.api import load, EarthSatellite
import heapq
import random
import pandas as pd
import os
from datetime import datetime

class Satellite:
    def __init__(self, name, servers = 1, mu_inter = 0.03, mu_intra = 0.001, num_beams = 0, tle_data = None):
        self.name = name
        self.tle_data = tle_data
        self.handover_manager = self.HandoverManager(self, servers = servers, mu_inter = mu_inter , mu_intra = mu_intra)
        self.num_beams = num_beams
        self.connected_ues = [0] * num_beams # I create a satellite only when a Ue wants to connect to him
        self.connected_ues_screenshot = [0] * num_beams


    class HandoverManager:
        # typical values for number of servers n and mean service time 1/mu could be the following:
        # n = 8-16-32, mu = 1/30ms. These values can be adjusted to simulate different scenarios and 
        # evaluate the associated performance impact. 
        # note that the UE shall not be unable to communicate for the whole duration of the handover,
        # only for the time it takes for the disconnection from the serving satellite and RACH to the target.
        def __init__(self, satellite, servers, mu_inter, mu_intra):
            # print(f"\n=== initializing hom for {satellite.name} ===")
            self.satellite = satellite
            self.servers = servers
            self.mu_inter = mu_inter
            self.mu_intra = mu_intra
            self.events_queue_inter = [0.0] * servers
            self.events_queue_intra = [0.0] * servers
            heapq.heapify(self.events_queue_inter)
            heapq.heapify(self.events_queue_intra)
            self.handover_tracker = []

        def process_handover_intra(self, arrival_time, ue, curr_beam_index, dest_satellite, dest_beam_index):
            """
            Generate the handover_info struct, and specifically the queue delay, for intra-handover.
            """
            event_type = "intra_ho"
            # duration = random.expovariate(self.mu_intra)    
            duration = self.mu_intra # 1 ms
            earliest_time = heapq.heappop(self.events_queue_intra)
            arrival_time_rel = arrival_time.timestamp()
            start_time = max(arrival_time_rel, earliest_time)
            end_time = start_time + duration
            heapq.heappush(self.events_queue_intra, end_time)
            start_time_dt = datetime.fromtimestamp(start_time)
            end_time_dt = datetime.fromtimestamp(end_time)

            handover_info = {
                "arrival_time": arrival_time,
                "event_type": event_type,
                "ue_id": ue.id,
                "from_satellite": self.satellite.name,
                "from_beam_index": curr_beam_index,
                "dest_satellite": dest_satellite.name,
                "dest_beam_index": dest_beam_index,
                "start_time": start_time_dt,
                "departure_time": end_time_dt,
                "duration": duration,
                "dest_number_ues": dest_satellite.connected_ues.copy()
            }
                
            return handover_info

        def process_handover_inter(self, arrival_time, ue, curr_beam_index, dest_satellite, dest_beam_index):
            """
            Generate the handover_info struct, and specifically the queue delay, for inter-handover.
            """
            
            event_type = "inter_ho"
            # duration = random.expovariate(self.mu_inter)
            duration = self.mu_inter  # 30 ms
            earliest_time = heapq.heappop(self.events_queue_inter)
            arrival_time_rel = arrival_time.timestamp()
            start_time = max(arrival_time_rel, earliest_time)
            end_time = start_time + duration
            heapq.heappush(self.events_queue_inter, end_time)
            start_time_dt = datetime.fromtimestamp(start_time)
            end_time_dt = datetime.fromtimestamp(end_time)

            handover_info = {
                "arrival_time": arrival_time,
                "event_type": event_type,
                "ue_id": ue.id,
                "from_satellite": self.satellite.name,
                "from_beam_index": curr_beam_index,
                "dest_satellite": dest_satellite.name,
                "dest_beam_index": dest_beam_index,
                "start_time": start_time_dt,
                "departure_time": end_time_dt,
                "duration": duration,
                "dest_number_ues": dest_satellite.connected_ues.copy()
            }

            return handover_info


        def deactivate(self):
            """
            This function should be called once the satellite is no longer needed. It will save the output dataframe to a csv file.
            """
            df = pd.DataFrame(self.handover_tracker)
            # print(f"\n=== saving output file for {self.satellite.name} ===")
            filename = f"{self.satellite.name}_handover_events.csv"

            output_folder = "Satellite dataframes"
            full_path = os.path.join(output_folder, filename)
            os.makedirs(output_folder, exist_ok=True)

            df.to_csv(full_path, index=False)


    def connect_ue(self, beam_index):
        """
        Handles the connection of a single UE.
        """
        self.connected_ues[beam_index] += 1
        return self.connected_ues[beam_index]
    
    def deactivate(self):
        """
        Deactivate the satellite. This should be called once the satellite is no longer needed.
        It will call the respective deactivate method of the handover_manager class to save the output dataframe to a csv file.
        """
        self.handover_manager.deactivate()
        self.handover_manager = None

    def disconnect_ue(self, index):
        """
        Handles the disconnection of a single UE.
        """
        if self.connected_ues[index] > 0:
            self.connected_ues[index] -= 1
        return self.connected_ues

    def get_position(self, time):
        ts = load.timescale()
        t = ts.utc(time.year, time.month, time.day, time.hour, time.minute, time.second)

        satellite = EarthSatellite(self.tle_data[1], self.tle_data[2])  # Create EarthSatellite object
        geocentric = satellite.at(t)  # Use the created EarthSatellite object here
        subpoint = geocentric.subpoint()

        latitude = subpoint.latitude.degrees
        longitude = subpoint.longitude.degrees
        height = subpoint.elevation.m

        return latitude, longitude, height


# ========================= TO FIX =========================
    
"""

    

    def get_rate(self, frame, time):

        return self.get_max_rate(frame, time) / self.load if self.load > 0 else 0
"""