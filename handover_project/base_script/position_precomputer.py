import numpy as np
import pandas as pd
import itertools
import time
from functools import partial
from tqdm.contrib.concurrent import process_map

# Moduli custom del simulatore
from satellite import Satellite
from channel_parameters import ChannelParameters

def generate_time_vector(start_time, offset_seconds, step_seconds):
    """
    Genera un vettore di istanti temporali a partire da `start_time`,
    con durata totale `offset_seconds` e intervallo `step_seconds`.
    """
    return start_time + np.arange(0, offset_seconds + step_seconds, step_seconds).astype('timedelta64[s]')

def evaluate_time_instant(time_instant, cluster, tle_list, lat_ue, lon_ue, beam_footprint_m, cell_dim):
    """
    Valuta quali satelliti rientrano nel raggio d'azione dell'utente per un dato istante temporale.
    Calcola l'altezza e le coordinate geocentriche propagando i TLE.
    """
    results = [] 
    
    for satellite_name, line1, line2 in tle_list: 
        # Istanziazione temporanea del satellite per il calcolo orbitale
        sat = Satellite(satellite_name, tle_data=(satellite_name, line1, line2)) 
        
        # Estrazione posizione convertendo il formato numpy in datetime standard
        sat_lat, sat_lon, sat_height_m = sat.get_position(time_instant.astype('datetime64[us]').item())
        
        # Filtro anomalie o propagazioni TLE non valide (sotto 100km o sopra 5000km)
        if sat_height_m < 100 or sat_height_m > 5e6:
            continue

        # Filtro Bounding Box: verifica la presenza nel raggio d'azione geometrico
        within = ChannelParameters.within_range(
            lat_ue,
            lon_ue,
            sat_lat,
            sat_lon,
            beam_footprint_m * cell_dim
        )

        if within is not None:
            results.append({
                "cluster_id": cluster,
                "time": time_instant,
                "sat_name": satellite_name,
                "sat_lat": sat_lat,
                "sat_lon": sat_lon,
                "sat_height": sat_height_m,
            })

    return results

def main():
    print("=== Avvio Pre-computazione Posizioni Orbitali (Digital Twin) ===")
    start_time = time.time() 

    # ====================================================================================== #
    # PARAMETRI DI SIMULAZIONE E TOPOLOGIA
    # ====================================================================================== #
    epoch_time = np.datetime64('2026-02-19T00:00:00')   
    simulation_duration_seconds = 6000                  
    simulation_step_seconds = 1                         
    
    # Coordinate UE (Target)
    lat_ue, lon_ue = 45.40996, 11.89261  # Porta Portello, Padova, IT
    
    filename = "250km_sc9_padova"
    cluster_id = 1
    max_workers = None      # Usa tutti i core della CPU disponibili
    cell_dim = 5
    
    # Selezione Scenario 3GPP
    sc9 = True
    sc6 = False

    if sc9:
        beam_footprint_m = 50_000  # Diametro fascio SC9 [m]
    elif sc6:
        beam_footprint_m = 20_000  # Diametro fascio SC6 [m]
    else:
        print("\nErrore: Nessuno scenario (SC6/SC9) selezionato.")
        return
    # ====================================================================================== #

    # Caricamento parametri TLE
    with open('Starlink_TLE.txt', 'r') as f:
        lines = [l.strip() for l in f.readlines()]
    tle_list = [(lines[i], lines[i+1], lines[i+2]) for i in range(0, len(lines)-2, 3)]
    print(f"[{len(tle_list)}] Satelliti caricati con successo dal file TLE.")

    # Impacchettamento funzione con parametri parziali per la parallelizzazione
    worker_evaluate_time_instant = partial(
        evaluate_time_instant,
        cluster=cluster_id,
        tle_list=tle_list,
        lat_ue=lat_ue,
        lon_ue=lon_ue,
        beam_footprint_m=beam_footprint_m,
        cell_dim=cell_dim
    )

    print(f"Generazione vettore temporale ({simulation_duration_seconds}s, step {simulation_step_seconds}s)...")
    times = generate_time_vector(epoch_time, simulation_duration_seconds, simulation_step_seconds)
    print(f"Anteprima istanti temporali: {times[:3]}")

    print("\n=== Avvio Multiprocessing (Calcolo Parallelo) ===")
    results_list = process_map(
        worker_evaluate_time_instant, 
        times, 
        max_workers=max_workers,  
        chunksize=1
    )
    
    # Appiattimento della lista e generazione Dataframe
    flat_results = list(itertools.chain.from_iterable(results_list))
    df = pd.DataFrame(flat_results)
    df['time'] = pd.to_datetime(df['time'])

    print("\nCiclo di computazione completato. Anteprima risultati:")
    print(df.head())
    print(f"Totale Record Generati: {len(df)}")

    end_time = time.time()
    print(f"Elaborazione conclusa in {end_time - start_time:.2f} secondi.")

    print("\n=== Calcolo del Time-to-Loss (Occurrence Countdown) ===")
    df['occurrence_countdown'] = df.groupby('sat_name').cumcount(ascending=False) + 1

    print(f"\n=== Salvataggio File: {filename}.csv ===")
    df.to_csv(f'{filename}.csv', index=False)
    print("Salvataggio completato con successo.")

if __name__ == '__main__':
    main()