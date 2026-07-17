import utils
import cluster
import numpy as np
import pandas as pd

# Importiamo lo scenario dal nuovo Livello Fisico per istanziare correttamente il Cluster
from channel import sc9_parameters

beam_size_m = 50000 # 50 km beam
cell_dim_beams = 5

# Definizione del centro del cluster (Es. Padova)
center_lat = 45.40969451353066
center_lon = 11.89266205327162

# Aggiunto '0.0' come altitudine per coerenza con l'unpacking (lat, lon, alt)
cluster_position = (center_lat, center_lon, 0.0) 
cluster_name = "Test Cluster"
num_ues = 0
beam_size_km = beam_size_m / 1000  # Convert meters to kilometers
num_beams = 25

# Parametri mock per soddisfare la nuova architettura SDN/MAC del Cluster
dummy_frame = pd.DataFrame() 
sat_servers = 1
mu_inter = 0.030
mu_intra = 0.001
scenario = sc9_parameters

# Calcolo dei confini di cella
cell_boundaries = utils.compute_cell_boundaries_lla(center_lat, center_lon, beam_size_m, cell_dim_beams)
print("\n=== CELL BOUNDARIES ===")
print(cell_boundaries)

# Istanziazione del Cluster aggiornata con la nuova firma
test_cluster = cluster.Cluster(
    name=cluster_name, 
    position=cluster_position, 
    num_ues=num_ues, 
    beam_size_km=beam_size_km, 
    num_beams=num_beams, 
    satellites_frame=dummy_frame, 
    servers=sat_servers, 
    mu_inter=mu_inter, 
    mu_intra=mu_intra, 
    scenario=scenario
)

print("\n=== POSIZIONI DEI CENTRI DEI MINI-CLUSTER ===")
for positions in test_cluster.positions:
    print(positions)

visible_cluster_indices = utils.check_clusters_visibility(test_cluster.positions, cell_boundaries, cell_dim_beams)
print("\n=== INDICI DEI MINI-CLUSTER IN VISIBILITA' ===")
for row in visible_cluster_indices:
    print(row)


print("\n\n" + "="*50)
print("TEST: NUOVO CLUSTER TRASLATO AD EST")
print("="*50)

new_center_lat = center_lat
new_center_lon = center_lon + 1.0  # Move 1 degrees east
new_cluster_position = (new_center_lat, new_center_lon, 0.0)

test_cluster_new = cluster.Cluster(
    name=cluster_name, 
    position=new_cluster_position, 
    num_ues=num_ues, 
    beam_size_km=beam_size_km, 
    num_beams=num_beams, 
    satellites_frame=dummy_frame, 
    servers=sat_servers, 
    mu_inter=mu_inter, 
    mu_intra=mu_intra, 
    scenario=scenario
)

print("\n=== POSIZIONI DEI CENTRI DEI MINI-CLUSTER (TRASLATO) ===")
for positions in test_cluster_new.positions:
    print(positions)
    
visible_cluster_indices = utils.check_clusters_visibility(test_cluster_new.positions, cell_boundaries, cell_dim_beams)
print("\n=== INDICI DEI MINI-CLUSTER IN VISIBILITA' ===")
print(visible_cluster_indices)


# =================================================================================
# Test the transposition to get the correct satellite side beam index.
# This is a different example, not related to the previous cluster, 
# just to test the get_coverage_beam_indices_matrix function.
# =================================================================================
print("\n\n=== TEST MATRICE DI COPERTURA BEAM ===")
visible_clusters_indices_matrix = np.array([[0], [3], [6]])
cell_dim_beams_test = 3

print("Coverage beam indices matrix:")
print(utils.get_coverage_beam_indices_matrix(visible_clusters_indices_matrix, cell_dim_beams_test))
