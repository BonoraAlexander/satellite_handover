from datetime import datetime
import numpy as np
from satellite import Satellite
from channel_parameters import ChannelParameters
import pandas as pd
import itertools

# for parallel processing
import time
from tqdm.contrib.concurrent import process_map
from functools import partial

def generate_time_vector(start_time, offset_seconds, step_seconds):
    """
        Generates a vector of time instants starting from `start_time`, with a total duration of `offset_seconds` and a step of `step_seconds`.

        Args:
            start_time (numpy.datetime64): The starting time instant
            offset_seconds (int): Total duration in seconds for which to generate time instants
            step_seconds (int): Time step in seconds between consecutive time instants
        Returns:
            numpy.ndarray: An array of time instants
    """
    return start_time + np.arange(0, offset_seconds + step_seconds, step_seconds).astype('timedelta64[s]')

def evaluate_time_instant(time_instant, cluster, tle_list, lat_ue, lon_ue, beam_footprint_m, cell_dim):
    """
    Given a time instant, a list of satellite TLEs, and some transmission parameters, this function 
    evaluates which satellites are within range of the UE and computes the relevant transmission parameters
    for those satellites.
        Args:
            time_instant (numpy.datetime64): The time instant to evaluate
            cluster (int): The cluster ID for which to evaluate the satellite positions (just copied to the dataframe)
            tle_list (list of tuples): List of TLE data for the satellites, where each tuple contains (satellite_name, line1, line2)
            lat_ue (float): Latitude of the user equipment (UE) in decimal degrees
            lon_ue (float): Longitude of the user equipment (UE) in decimal degrees
            beam_footprint_m (float): The diameter of the satellite beam footprint in meters
        Returns:
            list of dicts: A list of dictionaries containing the computed parameters for each satellite within range in the following form:
                {
                    "cluster_id": cluster,
                    "time": time_instant,
                    "sat_name": satellite_name,
                    "sat_lat": sat_lat,
                    "sat_lon": sat_lon,
                    "sat_height": sat_height_m
                }
    """

    results = [] # list to store results for all satellites within range at this time instant
    for satellite_name, line1, line2 in tle_list: 
        sat = Satellite(satellite_name, tle_data=(satellite_name, line1, line2)) # create satellite object from TLE data
        # get satellite position at the given time instant. Note that we need to convert the numpy.datetime64 to a standard Python datetime object
        sat_lat, sat_lon, sat_height_m = sat.get_position(time_instant.astype('datetime64[us]').item())
        # skip if the satellite height is negative or above 5000 km to filter out erroneous position calculations
        if(sat_height_m < 100 or sat_height_m > 5e6):
            continue

        # check if the satellite is within range of the UE based on the beam footprint
        within = ChannelParameters.within_range(
            lat_ue,
            lon_ue,
            sat_lat,
            sat_lon,
            beam_footprint_m * cell_dim
        )

        # if the satellite is within range, compute the channel parameters and save the results.
        if within is not None:
           
            # save the results in the list as a dictionary
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
    print("=== starting position pre-computation ===")
    start_time = time.time() # just to act cool and print the time taken for the computations at the end

    # simulation parameters
    # epoch_time = np.datetime64('2025-06-08T00:00:00')   # epoch time for position computation [y, m, d, h, m, s]
    epoch_time = np.datetime64('2026-07-05T12:00:00Z')   # epoch time for position computation [y, m, d, h, m, s]
    simulation_duration_seconds = 3600                  # total simulation duration [s]
    simulation_step_seconds = 1                         # time step for position computation [s]
    # lat_ue, lon_ue = 18.29817, -64.82818                # ue location [decimal degrees]
    # cluster locations:
    lat_ue, lon_ue = 45.40996, 11.89261 # Porta Portello, Padova, IT
    # lat_ue, lon_ue = 45.43903, 10.99435 # Arena di Verona, Verona, IT
    # lat_ue, lon_ue = 46.06250, 11.11497 # MUSE, Trento, IT
    # lat_ue, lon_ue = 48.14295, 11.57997 # hofgarten, moanco di baviera
    # lat_ue, lon_ue = 47.04240, 8.328983 #richard wagner museum, lucerna
    filename = "100km_25beams_sc9_padova_2026_07_05_12_00_1h"
    cluster_id = 1
    max_workers = None      # none to use all availabe cpu cores, or set to a specific number
    sc9 = True
    sc6 = False
    cell_dim = 5

 # ====================================================================================== #

    if sc9:
        beam_footprint_m = 100_000  # beam diameter [m]
    elif sc6:
        beam_footprint_m = 20_000  # beam diameter [m]
    else:
        print("\nNo scenario selected. Please set sc9 or sc6 to True.")

    # loading the TLE parameters and create the list of tuples
    with open('starlink_2026_07_05T11_30_40Z.tle', 'r') as f:
        lines = [l.strip() for l in f.readlines()]
    tle_list = [(lines[i], lines[i+1], lines[i+2]) for i in range(0, len(lines)-2, 3)]
    print(f"Loaded {len(tle_list)} satellites from TLE file")

    # since the function we want to parallelize (evaluate_time_instant) takes multiple arguments,
    # we use functools.partial to create a new function with the other arguments fixed,
    # so that it only takes the time instant as input. This allows us to easily use it with process_map for parallel processing.
    worker_evaluate_time_instant = partial(
        evaluate_time_instant,
        cluster = cluster_id,
        tle_list=tle_list,
        lat_ue=lat_ue,
        lon_ue=lon_ue,
        beam_footprint_m=beam_footprint_m,
        cell_dim=cell_dim
    )

    # generate the vector of time instants for which to compute the satellite positions
    print(f"Generating time vector for {simulation_duration_seconds} s with step of {simulation_step_seconds} s...")
    times = generate_time_vector(epoch_time, simulation_duration_seconds, simulation_step_seconds)
    print(times[:3]) # Show first three

    print("\n=== starting main computation cycle ===")
    results_list = process_map(
        worker_evaluate_time_instant, 
        times, 
        max_workers=max_workers,  # Specify your core num_uescount
        chunksize=1
    )
    
    # flatten the list of lists of dictionaries into a single list of dictionaries, and convert to a dataframe
    flat_results = list(itertools.chain.from_iterable(results_list))
    df = pd.DataFrame(flat_results)

    # set the correct time format for the dataframe
    df['time'] = pd.to_datetime(df['time'])

    print("finished main computation cycle, results preview: \n")
    print(df.head())
    print(f"total records: {len(df)}")

    end_time = time.time()
    print(f"processed in {end_time - start_time:.2f} seconds.")

    print("\n=== counting the number of occurences of each satellite for visibility condition ===")
    df['occurrence_countdown'] = df.groupby('sat_name').cumcount(ascending=False) + 1

    print("\n=== saving output file ===")
    df.to_csv(f'{filename}.csv', index=False)
    print(f"saved as {filename}.csv")

# whatever
if __name__ == '__main__':
    main()