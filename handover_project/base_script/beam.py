from ue import Ue
from satellite import Satellite
from datetime import timedelta
import numpy as np
import random

class Beam:
    def __init__(self, name, index, position, num_ues, beam_size_km, cell_dim_beams, sat_servers, sat_mu_inter, sat_mu_intra):
        self.name = name
        self.index = index
        self.num_ues = num_ues
        self.position = position
        self.beam_size_km = beam_size_km
        self.cell_dim_beams = cell_dim_beams
        self.list_ues = [Ue(self.name + "-Ue" + str(ii+1), position) for ii in range(num_ues)]
        self.sat_servers = sat_servers
        self.sat_mu_inter = sat_mu_inter
        self.sat_mu_intra = sat_mu_intra

    
    def initial_connection_phase(self, visible_sats, time, service_sats, handover_timer = 0):
        """
        Initial connection phase: each UE connects to a random satellite among the visible ones.
        """
        if(len(visible_sats) > 0):
            # For each UE, connect to a random satellite
            for ue in self.list_ues:
                random_index = random.randint(0, len(visible_sats)-1)
                ue.time_to_next_handover = random.randint(1, handover_timer) # set a random condition for the first handover timer
                selected_sat_name = visible_sats[random_index][0]
                selected_sat_beam = self.index # should be the same of the terrestrial mini-cluster
                # is this satellite already configured?
                if selected_sat_name not in service_sats:
                    sat = Satellite(selected_sat_name, self.sat_servers, self.sat_mu_inter, self.sat_mu_intra, int(pow(self.cell_dim_beams, 2)))
                    service_sats[selected_sat_name] = sat

                ue.connect_to_satellite(service_sats[selected_sat_name], selected_sat_beam)
                service_sats[selected_sat_name].connect_ue(selected_sat_beam)

                handover_info = {
                    "arrival_time": time,
                    "event_type": "init_con",
                    "ue_id": ue.id,
                    "from_satellite": None,
                    "from_beam_index": None,
                    "dest_satellite": service_sats[selected_sat_name].name,
                    "dest_beam_index": selected_sat_beam,
                    "start_time": time,
                    "departure_time": 0,
                    "duration": 0,
                    "dest_number_ues": service_sats[selected_sat_name].connected_ues.copy()
                }
                service_sats[selected_sat_name].handover_manager.handover_tracker.append(handover_info)
                ue.handover_tracker.append(handover_info)
        else: # all the ues are out of service
            for ue in self.list_ues:
                handover_info = {
                    "arrival_time": time,
                    "event_type": "out_serv",
                    "ue_id": ue.id,
                    "from_satellite": None,
                    "from_beam_index": None,
                    "dest_satellite": None,
                    "dest_beam_index": None,
                    "start_time": time,
                    "departure_time": 0,
                    "duration": 1.0,
                    "dest_number_ues": 0
                }
                ue.handover_tracker.append(handover_info)

    
