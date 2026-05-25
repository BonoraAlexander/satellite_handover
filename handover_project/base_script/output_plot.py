import pandas as pd
from datetime import datetime, timedelta
import utils
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
from pathlib import Path
from scipy.stats import gaussian_kde
import seaborn as sns
import numpy as np
from pathlib import Path
import re
import random

pd.options.mode.chained_assignment = None  # default='warn'

# ========================================================================================================= # 

# 1. How many satellites in visibility over time
visible_sats_over_time = True
# 2. Average handover rate 
average_handover_rate = True
# 3. Average handover duration
average_handover_duration = True
# 4. Average service time before the next handover event
average_service_time = True
# 5. Number of handover processes handled by each satellite
ho_handled = True
# 6. Average number and duration of out of services
out_of_service = True
# 7  Throughput considering HO outage time 
get_throuthput_ho_v2 = True
# 8. Number of ping-pong handovers
ping_pong_handovers = True
# 9. Doppler Shifts
doppler_shifts = True

# Save the results into a csv
save_plot_values = False


# ================================================================================================

# dataframes parameters
df_name = "100km_25beams_sc9_padova.csv"
padova_lat, padova_lon = 45.40996, 11.89261
dfnames = [df_name] 
fnames = ["padova"]
enable_elevation_threshold = True
elevation_threshold = 30

# simulation parameters
output_folder = "plots"
period = '20 min'
num_ues_label = 100
simTimeStart = datetime(2026, 2, 19, 0, 0, 0) 
simTimeEnd = datetime(2026, 2, 19, 0, 20, 0) 
time_step = timedelta(seconds=1)
num_ues_to_plot = 1


# retrive parameterts
numbers = re.findall(r'\d+', df_name)
beam_size_km = int(numbers[0])
num_beams = int(numbers[1])
padova_positions = utils.calculate_beams_grid(padova_lat, padova_lon, beam_size_km, num_beams)

# ================================================================================================


colors1 = ['skyblue', 'lightcoral', 'palegreen', 'mocassin', 'plum', 'tan', 'lightpink', 'lightgray', 'darkkhaki', 'paleturquoise']
colors2 = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']



# ========================================================================================================= # 

# 1. How many satellites in visibility over time
if(visible_sats_over_time):
    print("1. Printing the number of visible satellites ...")
    plt.figure(figsize=(10, 5))
    index = 0
    for df_name, fname in zip(dfnames, fnames):
        data_frame = pd.read_csv(df_name)
        time = simTimeStart
        end_sim_time = simTimeEnd

        visible_sats = []
        timestamps = []
        while time < end_sim_time:
            visible_satellites = utils.get_satellites_at_time(data_frame, time)
            visible_sats.append(len(visible_satellites))
            timestamps.append(time)
            time += timedelta(seconds=1)

        plt.plot(timestamps, visible_sats, color=colors1[index], linestyle='-', label = fname)
        index += 1
        plt.title('Visible Satellites Over Time')
        plt.legend()
        plt.xlabel('Time (HH:MM:SS)')
        plt.ylabel('Number of Visible Satellites')
        plt.grid(True)
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        plt.xticks(rotation=45)
        plt.tight_layout()
        os.makedirs(output_folder, exist_ok=True)
        file_name = "1-satellite_visibility.png"
        file_path = os.path.join(output_folder, file_name)
        #os.makedirs(os.path.join(output_folder, fname), exist_ok=True)
        plt.savefig(file_path, dpi=300, bbox_inches='tight')

        if(save_plot_values):
            values_df = pd.DataFrame({'timestamp': timestamps, 'elapsed_seconds': np.arange(len(timestamps)), 'visible_satellites': visible_sats})
            csv_file_name = "1-satellite_visibility_values.csv"
            target_dir = os.path.join(output_folder, fname)
            csv_file_path = os.path.join(target_dir, csv_file_name)
            os.makedirs(target_dir, exist_ok=True)
            values_df.to_csv(csv_file_path, index=False)
    plt.close()

    print("   Completed!")

    print("1.1 Printing the number of visible beams for each ue ...")
    for df_name, fname in zip(dfnames, fnames):
        plt.figure(figsize=(10, 5))
        index = 0
        data_frame = pd.read_csv(df_name)
        beams_names = ["Beam NO", "Beam Center", "Beam SE"]
        visible_sat_beam_NO = []
        visible_sat_beam_center = []
        visible_sat_beam_SE = []


        time = simTimeStart
        end_sim_time = simTimeEnd

        # compute the visibile beams for each minicluster
        while time < end_sim_time:
            visible_sats = utils.get_satellites_at_time(data_frame, time)
            visible_sats_for_each_minicluster = [[] for _ in range(num_beams)]
            for sat in visible_sats:
                sat_lat, sat_lon, sat_alt = sat[1], sat[2], sat[3]
                sat_cell_boundaries = utils.compute_cell_boundaries_lla(sat_lat, sat_lon, beam_size_km*1000, int(np.sqrt(num_beams)))
                visible_clusters_indices = utils.check_clusters_visibility(padova_positions, sat_cell_boundaries, int(np.sqrt(num_beams)), enable_elevation_threshold, elevation_threshold, sat_lat, sat_lon, sat_alt)
                if(len(visible_clusters_indices) == 0):
                    continue
                satellite_beam_indices = utils.get_coverage_beam_indices_matrix(visible_clusters_indices, int(np.sqrt(num_beams)))
                
                rows, cols = visible_clusters_indices.shape
                for ii in range(rows):
                    for jj in range(cols):
                        idx_cluster = visible_clusters_indices[ii][jj]
                        idx_sat_beam = satellite_beam_indices[ii][jj]
                        if (idx_sat_beam != -1):
                            visible_sats_for_each_minicluster[idx_cluster].append((sat, idx_sat_beam))

            visible_sat_beam_NO.append(len(visible_sats_for_each_minicluster[0]))
            visible_sat_beam_center.append(len(visible_sats_for_each_minicluster[int(num_beams/2)]))
            visible_sat_beam_SE.append(len(visible_sats_for_each_minicluster[-1]))

            time += timedelta(seconds=1)

        total_seconds = int((simTimeEnd - simTimeStart).total_seconds())
        timestamps = [simTimeStart + timedelta(seconds=i) for i in range(total_seconds + 1)][:-1]

        plt.plot(timestamps, visible_sat_beam_NO, color=colors1[index], linestyle='-', label = beams_names[index])
        index += 1
        plt.plot(timestamps, visible_sat_beam_center, color=colors1[index], linestyle='-', label = beams_names[index])
        index += 1
        plt.plot(timestamps, visible_sat_beam_SE, color=colors1[index], linestyle='-', label = beams_names[index])
        index += 1
            
        plt.title('Visible Beams Over Time')
        plt.legend()
        plt.xlabel('Time (HH:MM:SS)')
        plt.ylabel('Number of Visible Beams')
        plt.grid(True)
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        plt.xticks(rotation=45)
        plt.tight_layout()
        os.makedirs(output_folder, exist_ok=True)
        file_name = f"1.1-beam_visibility_{fname}.png"
        file_path = os.path.join(output_folder, file_name)
        #os.makedirs(os.path.join(output_folder, fname), exist_ok=True)
        plt.savefig(file_path, dpi=300, bbox_inches='tight')

        if(save_plot_values):
            values_df = pd.DataFrame({'timestamp': timestamps, 'elapsed_seconds': np.arange(len(timestamps)), 'visible_sat_beam_NO': visible_sat_beam_NO, 'visible_sat_beam_center': visible_sat_beam_center, 'visible_sat_beam_SE': visible_sat_beam_SE})
            csv_file_name = "1.1-satellite_visibility_values.csv"
            target_dir = os.path.join(output_folder, fname)
            csv_file_path = os.path.join(target_dir, csv_file_name)
            os.makedirs(target_dir, exist_ok=True)
            values_df.to_csv(csv_file_path, index=False)
        plt.close()

    print("   Completed!\n")

# ========================================================================================================= # 


# 2. Average handover rate
if(average_handover_rate):

    print("2. Printing the average handover rate ...")

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, (df_name, fname) in enumerate(zip(dfnames, fnames)):
        folder_path = Path("Cluster" + str(i+1) + " dataframes")
        intra_ho_count = []
        inter_ho_count = []
        
        for file_path in folder_path.glob('*.csv'):
            df = pd.read_csv(file_path)
            count_intra = len(df[df['event_type'] == 'intra_ho'])
            count_inter = len(df[df['event_type'] == 'inter_ho'])
            intra_ho_count.append(count_intra)
            inter_ho_count.append(count_inter)

        # Base color for this cluster
        color = plt.cm.tab10(i)

        # Dictionary to hold data for CSV saving later
        csv_data = {}
        
        # Check for enough data and variance to do a KDE
        if len(intra_ho_count) > 1 and min(intra_ho_count) != max(intra_ho_count):
            kde_intra = gaussian_kde(intra_ho_count)
            x_min_intra, x_max_intra = min(intra_ho_count), max(intra_ho_count)
            margin_intra = (x_max_intra - x_min_intra) * 0.2
            kde_x_intra = np.linspace(x_min_intra - margin_intra, x_max_intra + margin_intra, 500)
            kde_y_intra = kde_intra(kde_x_intra)

            # Plot solid line for Intra
            ax.plot(kde_x_intra, kde_y_intra, color=color, linestyle='-', linewidth=1.5)
            ax.fill_between(kde_x_intra, kde_y_intra, alpha=0.2, color=color,
                            label=f"Cluster {i+1}: {fname}, intra")
            
            idx_max = np.argmax(kde_y_intra)
            x_peak, y_peak = kde_x_intra[idx_max], kde_y_intra[idx_max]
            ax.plot(x_peak, y_peak, marker='*', color=color, markersize=14, markeredgecolor='black', zorder=5)
            ax.annotate(f'  {x_peak:.1f}', xy=(x_peak, y_peak), fontsize=8, color=color, va='bottom')

            # Store for CSV
            csv_data['intra_ho_x'] = kde_x_intra
            csv_data['intra_ho_density'] = kde_y_intra
        else:
            print(f"Skipping KDE for Cluster {i+1} Intra HO due to lack of variance.")

        if len(inter_ho_count) > 1 and min(inter_ho_count) != max(inter_ho_count):
            kde_inter = gaussian_kde(inter_ho_count)
            x_min_inter, x_max_inter = min(inter_ho_count), max(inter_ho_count)
            margin_inter = (x_max_inter - x_min_inter) * 0.2
            kde_x_inter = np.linspace(x_min_inter - margin_inter, x_max_inter + margin_inter, 500)
            kde_y_inter = kde_inter(kde_x_inter)

            # Plot dashed line for Inter to distinguish it
            ax.plot(kde_x_inter, kde_y_inter, color=color, linestyle='--', linewidth=1.5)
            ax.fill_between(kde_x_inter, kde_y_inter, alpha=0.1, color=color, # Lighter alpha
                            label=f"Cluster {i+1}: {fname}, inter")
            
            idx_max = np.argmax(kde_y_inter)
            x_peak, y_peak = kde_x_inter[idx_max], kde_y_inter[idx_max]
            ax.plot(x_peak, y_peak, marker='o', color=color, markersize=8, markeredgecolor='black', zorder=5)
            ax.annotate(f'  {x_peak:.1f}', xy=(x_peak, y_peak), fontsize=8, color=color, va='bottom')

            # Store for CSV
            csv_data['inter_ho_x'] = kde_x_inter
            csv_data['inter_ho_density'] = kde_y_inter
        else:
            print(f"Skipping KDE for Cluster {i+1} Inter HO due to lack of variance.")

        # --- SAVE PLOT VALUES ---
        # We now save both intra and inter curves if they exist
        if save_plot_values and csv_data:
            os.makedirs(os.path.join(output_folder, fname), exist_ok=True)
            kde_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in csv_data.items()])) 
            kde_df.to_csv(os.path.join(output_folder, fname, "2-handover.csv"), index=False)

    ax.set_title(f'Probability Density of Handovers - All Clusters ({period} Period) - {num_ues_label} UEs')
    ax.set_xlabel('Number of Handovers')
    ax.set_ylabel('Probability Density')
    ax.grid(axis='y', alpha=0.3)
    ax.legend(title="Clusters", bbox_to_anchor=(1.05, 1), loc='upper left')

    os.makedirs(output_folder, exist_ok=True)
    combined_file_path = os.path.join(output_folder, "2-handover_pdf.png")
    fig.savefig(combined_file_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("   Completed!\n")

        
# ========================================================================================================= # 

# 3. Average handover duration
if(average_handover_duration):
    print("3. Printing the average handover duration ...")

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, (df_name, fname) in enumerate(zip(dfnames, fnames)):
        folder_path = Path("Cluster" + str(i+1) + " dataframes")
        intra_ho_duration = []
        inter_ho_duration = []
        
        for file_path in folder_path.glob('*.csv'):
            df = pd.read_csv(file_path)

            df_1 = df[df['event_type'] == 'intra_ho']
            df_2 = df[df['event_type'] == 'inter_ho']

            arr_naive_1 = pd.to_datetime(df_1['arrival_time'], errors='coerce')
            arr_naive_2 = pd.to_datetime(df_2['arrival_time'], errors='coerce')
            arr_1 = pd.to_datetime(df_1['arrival_time'], errors='coerce', utc=True)
            arr_2 = pd.to_datetime(df_2['arrival_time'], errors='coerce', utc=True)
            dep_1 = pd.to_datetime(df_1['departure_time'], errors='coerce', utc=True)
            dep_2 = pd.to_datetime(df_2['departure_time'], errors='coerce', utc=True)
            duration_series_1 = dep_1 - arr_1
            duration_series_2 = dep_2 - arr_2
            duration_ms_1 = duration_series_1.dt.total_seconds() * 1000
            duration_ms_2 = duration_series_2.dt.total_seconds() * 1000
            mean_1 = duration_ms_1.mean()
            mean_2 = duration_ms_2.mean()
            
            if pd.notna(mean_1):
                intra_ho_duration.append(mean_1)
            if pd.notna(mean_2):
                inter_ho_duration.append(mean_2)

        # Base color for this cluster
        color = plt.cm.tab10(i)

        # Dictionary to hold data for CSV saving later
        csv_data = {}
        
        # Check for enough data and variance to do a KDE
        if len(intra_ho_duration) > 1 and min(intra_ho_duration) != max(intra_ho_duration):
            kde_intra = gaussian_kde(intra_ho_duration)
            x_min_intra, x_max_intra = min(intra_ho_duration), max(intra_ho_duration)
            margin_intra = (x_max_intra - x_min_intra) * 0.2
            kde_x_intra = np.linspace(x_min_intra - margin_intra, x_max_intra + margin_intra, 500)
            kde_y_intra = kde_intra(kde_x_intra)

            # Plot solid line for Intra
            ax.plot(kde_x_intra, kde_y_intra, color=color, linestyle='-', linewidth=1.5)
            ax.fill_between(kde_x_intra, kde_y_intra, alpha=0.2, color=color,
                            label=f"Cluster {i+1}: {fname}, intra")
            
            idx_max = np.argmax(kde_y_intra)
            x_peak, y_peak = kde_x_intra[idx_max], kde_y_intra[idx_max]
            ax.plot(x_peak, y_peak, marker='*', color=color, markersize=14, markeredgecolor='black', zorder=5)
            ax.annotate(f'  {x_peak:.1f}', xy=(x_peak, y_peak), fontsize=8, color=color, va='bottom')

            # Store for CSV
            csv_data['intra_ho_x'] = kde_x_intra
            csv_data['intra_ho_density'] = kde_y_intra
        else:
            print(f"Skipping KDE for Cluster {i+1} Intra HO due to lack of variance.")

        if len(inter_ho_duration) > 1 and min(inter_ho_duration) != max(inter_ho_duration):
            kde_inter = gaussian_kde(inter_ho_duration)
            x_min_inter, x_max_inter = min(inter_ho_duration), max(inter_ho_duration)
            margin_inter = (x_max_inter - x_min_inter) * 0.2
            kde_x_inter = np.linspace(x_min_inter - margin_inter, x_max_inter + margin_inter, 500)
            kde_y_inter = kde_inter(kde_x_inter)

            # Plot dashed line for Inter to distinguish it
            ax.plot(kde_x_inter, kde_y_inter, color=color, linestyle='--', linewidth=1.5)
            ax.fill_between(kde_x_inter, kde_y_inter, alpha=0.1, color=color, # Lighter alpha
                            label=f"Cluster {i+1}: {fname}, inter")
            
            idx_max = np.argmax(kde_y_inter)
            x_peak, y_peak = kde_x_inter[idx_max], kde_y_inter[idx_max]
            ax.plot(x_peak, y_peak, marker='o', color=color, markersize=8, markeredgecolor='black', zorder=5)
            ax.annotate(f'  {x_peak:.1f}', xy=(x_peak, y_peak), fontsize=8, color=color, va='bottom')

            # Store for CSV
            csv_data['inter_ho_x'] = kde_x_inter
            csv_data['inter_ho_density'] = kde_y_inter
        else:
            print(f"Skipping KDE for Cluster {i+1} Inter HO due to lack of variance.")

        # --- SAVE PLOT VALUES ---
        # We now save both intra and inter curves if they exist
        if save_plot_values and csv_data:
            os.makedirs(os.path.join(output_folder, fname), exist_ok=True)
            kde_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in csv_data.items()])) 
            kde_df.to_csv(os.path.join(output_folder, fname, "3-handover_duration.csv"), index=False)

    ax.set_title(f'Probability Density of Handovers Duration - All Clusters ({period} Period) - {num_ues_label} UEs')
    ax.set_xlabel('Duration [ms]')
    ax.set_ylabel('Probability Density')
    ax.grid(axis='y', alpha=0.3)
    ax.legend(title="Clusters", bbox_to_anchor=(1.05, 1), loc='upper left')

    os.makedirs(output_folder, exist_ok=True)
    combined_file_path = os.path.join(output_folder, "3-handover_duration.png")
    fig.savefig(combined_file_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("   Completed!\n")


# ========================================================================================================= # 

# 4. Average service time before the next handover event
if(average_service_time):
    print("4. Printing the average time before next handover ...")

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, (df_name, fname) in enumerate(zip(dfnames, fnames)):
        folder_path = Path("Cluster" + str(i+1) + " dataframes")
        
        # Master lists for this specific cluster
        cluster_beam_durations = []
        cluster_sat_durations = []
        
        for file_path in folder_path.glob('*.csv'):
            df = pd.read_csv(file_path)
            if df.empty:
                continue
            
            # parse time safely and ensure it is chronological
            df['arrival_time'] = pd.to_datetime(df['arrival_time'], errors='coerce', utc=True)
            df = df.sort_values('arrival_time')
            
            # set up our state trackers for this specific UE
            curr_sat, curr_beam = None, None
            sat_start_time, beam_start_time = None, None
            
            for row in df.itertuples():
                t = row.arrival_time
                
                # safely extract destinations (handling strings of 'None' or NaNs from CSVs)
                dest_sat = row.dest_satellite if pd.notna(row.dest_satellite) and str(row.dest_satellite) != 'None' else None
                dest_beam = row.dest_beam_index if pd.notna(row.dest_beam_index) and str(row.dest_beam_index) != 'None' else None

                # case A: the UE disconnected entirely (out_serv, lost_conn)
                if dest_sat is None:
                    if curr_sat is not None:
                        cluster_sat_durations.append((t - sat_start_time).total_seconds())
                        curr_sat = None # Reset state
                    if curr_beam is not None:
                        cluster_beam_durations.append((t - beam_start_time).total_seconds())
                        curr_beam = None # Reset state
                        
                # case B: the UE is connected to a satellite
                else:
                    # did the satellite change? (initial connection or inter_ho)
                    if (row.event_type == 'inter_ho' or row.event_type == 'init_con'):
                        # Close out the old tracking periods if they exist
                        if curr_sat is not None:
                            cluster_sat_durations.append((t - sat_start_time).total_seconds())
                        if curr_beam is not None:
                            cluster_beam_durations.append((t - beam_start_time).total_seconds())
                        
                        # start tracking the new satellite and beam
                        curr_sat, curr_beam = dest_sat, dest_beam
                        sat_start_time, beam_start_time = t, t
                        
                    # the Satellite is the same. Did the Beam change? (intra_ho)
                    elif (row.event_type == 'intra_ho'):
                        # Close out the old beam tracking period
                        if curr_beam is not None:
                            cluster_beam_durations.append((t - beam_start_time).total_seconds())
                        
                        # Start tracking the new beam (Satellite tracking continues uninterrupted)
                        curr_beam = dest_beam
                        beam_start_time = t

        # Base color for this cluster
        color = plt.cm.tab10(i)

        # Dictionary to hold data for CSV saving later
        csv_data = {}
        
        # Check for enough data and variance to do a KDE
        if len(cluster_beam_durations) > 1 and min(cluster_beam_durations) != max(cluster_beam_durations):
            kde_intra = gaussian_kde(cluster_beam_durations)
            x_min_intra, x_max_intra = min(cluster_beam_durations), max(cluster_beam_durations)
            margin_intra = (x_max_intra - x_min_intra) * 0.2
            kde_x_intra = np.linspace(x_min_intra - margin_intra, x_max_intra + margin_intra, 500)
            kde_y_intra = kde_intra(kde_x_intra)

            # Plot solid line for Intra
            ax.plot(kde_x_intra, kde_y_intra, color=color, linestyle='-', linewidth=1.5)
            ax.fill_between(kde_x_intra, kde_y_intra, alpha=0.2, color=color,
                            label=f"Cluster {i+1}: {fname}, intra")
            
            idx_max = np.argmax(kde_y_intra)
            x_peak, y_peak = kde_x_intra[idx_max], kde_y_intra[idx_max]
            ax.plot(x_peak, y_peak, marker='*', color=color, markersize=14, markeredgecolor='black', zorder=5)
            ax.annotate(f'  {x_peak:.1f}', xy=(x_peak, y_peak), fontsize=8, color=color, va='bottom')

            # Store for CSV
            csv_data['intra_ho_x'] = kde_x_intra
            csv_data['intra_ho_density'] = kde_y_intra
        else:
            print(f"Skipping KDE for Cluster {i+1} Intra HO due to lack of variance.")

        if len(cluster_sat_durations) > 1 and min(cluster_sat_durations) != max(cluster_sat_durations):
            kde_inter = gaussian_kde(cluster_sat_durations)
            x_min_inter, x_max_inter = min(cluster_sat_durations), max(cluster_sat_durations)
            margin_inter = (x_max_inter - x_min_inter) * 0.2
            kde_x_inter = np.linspace(x_min_inter - margin_inter, x_max_inter + margin_inter, 500)
            kde_y_inter = kde_inter(kde_x_inter)

            # Plot dashed line for Inter to distinguish it
            ax.plot(kde_x_inter, kde_y_inter, color=color, linestyle='--', linewidth=1.5)
            ax.fill_between(kde_x_inter, kde_y_inter, alpha=0.1, color=color, # Lighter alpha
                            label=f"Cluster {i+1}: {fname}, inter")
            
            idx_max = np.argmax(kde_y_inter)
            x_peak, y_peak = kde_x_inter[idx_max], kde_y_inter[idx_max]
            ax.plot(x_peak, y_peak, marker='o', color=color, markersize=8, markeredgecolor='black', zorder=5)
            ax.annotate(f'  {x_peak:.1f}', xy=(x_peak, y_peak), fontsize=8, color=color, va='bottom')

            # Store for CSV
            csv_data['inter_ho_x'] = kde_x_inter
            csv_data['inter_ho_density'] = kde_y_inter
        else:
            print(f"Skipping KDE for Cluster {i+1} Inter HO due to lack of variance.")

        # --- SAVE PLOT VALUES ---
        # We now save both intra and inter curves if they exist
        if save_plot_values and csv_data:
            os.makedirs(os.path.join(output_folder, fname), exist_ok=True)
            kde_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in csv_data.items()])) 
            kde_df.to_csv(os.path.join(output_folder, fname, "4-service_time.csv"), index=False)

    ax.set_title(f'Probability Density of Service Time - All Clusters ({period} Period) - {num_ues_label} UEs')
    ax.set_xlabel('Service Time [s]')
    ax.set_ylabel('Probability Density')
    ax.grid(axis='y', alpha=0.3)
    ax.legend(title="Clusters", bbox_to_anchor=(1.05, 1), loc='upper left')

    os.makedirs(output_folder, exist_ok=True)
    combined_file_path = os.path.join(output_folder, "4-service_time.png")
    fig.savefig(combined_file_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("   Completed!\n")

# ========================================================================================================= # 

# 5. Number of handover processes handled by each satellite
if(ho_handled):
    print("5. Priting the average number of handover handled for each satellite ...")
    folder_path = Path('Satellite dataframes')
    intra_ho_count = []
    inter_ho_count = []
    num_sats = 0
    fname = 'Satellites'

    fig, ax = plt.subplots(figsize=(12, 6))
    

    for file_path in folder_path.glob('*.csv'):
        # needed beacuse there are some saved satellites which have empty df
        try:
            df = pd.read_csv(file_path)
            count_intra = len(df[df['event_type'] == 'intra_ho'])
            count_inter = len(df[df['event_type'] == 'inter_ho'])
            intra_ho_count.append(count_intra)
            inter_ho_count.append(count_inter)
            num_sats += 1
        except Exception as e:
            print("Empty satellite dataframe!")
            continue

    # Dictionary to hold data for CSV saving later
    csv_data = {}

    # Check for enough data and variance to do a KDE
    if len(intra_ho_count) > 1 and min(intra_ho_count) != max(intra_ho_count):
        kde_intra = gaussian_kde(intra_ho_count)
        x_min_intra, x_max_intra = min(intra_ho_count), max(intra_ho_count)
        margin_intra = (x_max_intra - x_min_intra) * 0.2
        kde_x_intra = np.linspace(x_min_intra - margin_intra, x_max_intra + margin_intra, 500)
        kde_y_intra = kde_intra(kde_x_intra)

        # Plot solid line for Intra
        ax.plot(kde_x_intra, kde_y_intra, color=colors1[0], linestyle='-', linewidth=1.5)
        ax.fill_between(kde_x_intra, kde_y_intra, alpha=0.2, color=colors1[0],
                        label=f"Intra-handovers")
        
        idx_max = np.argmax(kde_y_intra)
        x_peak, y_peak = kde_x_intra[idx_max], kde_y_intra[idx_max]
        ax.plot(x_peak, y_peak, marker='*', color=colors1[0], markersize=14, markeredgecolor='black', zorder=5)
        ax.annotate(f'  {x_peak:.1f}', xy=(x_peak, y_peak), fontsize=8, color=colors1[0], va='bottom')

        # Store for CSV
        csv_data['intra_ho_x'] = kde_x_intra
        csv_data['intra_ho_density'] = kde_y_intra
    else:
        print(f"Skipping KDE for Intra HO due to lack of variance.")

    if len(inter_ho_count) > 1 and min(inter_ho_count) != max(inter_ho_count):
        kde_inter = gaussian_kde(inter_ho_count)
        x_min_inter, x_max_inter = min(inter_ho_count), max(inter_ho_count)
        margin_inter = (x_max_inter - x_min_inter) * 0.2
        kde_x_inter = np.linspace(x_min_inter - margin_inter, x_max_inter + margin_inter, 500)
        kde_y_inter = kde_inter(kde_x_inter)

        # Plot dashed line for Inter to distinguish it
        ax.plot(kde_x_inter, kde_y_inter, color=colors1[1], linestyle='--', linewidth=1.5)
        ax.fill_between(kde_x_inter, kde_y_inter, alpha=0.1, color=colors1[1], # Lighter alpha
                        label=f"Inter-handovers")
        
        idx_max = np.argmax(kde_y_inter)
        x_peak, y_peak = kde_x_inter[idx_max], kde_y_inter[idx_max]
        ax.plot(x_peak, y_peak, marker='o', color=colors1[1], markersize=8, markeredgecolor='black', zorder=5)
        ax.annotate(f'  {x_peak:.1f}', xy=(x_peak, y_peak), fontsize=8, color=colors1[1], va='bottom')

        # Store for CSV
        csv_data['inter_ho_x'] = kde_x_inter
        csv_data['inter_ho_density'] = kde_y_inter
    else:
        print(f"Skipping KDE for Inter HO due to lack of variance.")

    # --- SAVE PLOT VALUES ---
    # We now save both intra and inter curves if they exist
    if save_plot_values and csv_data:
        os.makedirs(os.path.join(output_folder, fname), exist_ok=True)
        kde_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in csv_data.items()])) 
        kde_df.to_csv(os.path.join(output_folder, fname, "5-num_of_hos_per_sat.csv"), index=False)

    ax.set_title(f'Cumulative Distribution of Out and In Handovers per {num_sats} serving satellites ({period} Period) - {num_ues_label} UEs')
    ax.set_xlabel('Number of Handovers')
    ax.set_ylabel('Probability Density')
    ax.grid(axis='y', alpha=0.3)
    ax.legend(title="Clusters", bbox_to_anchor=(1.05, 1), loc='upper left')

    os.makedirs(output_folder, exist_ok=True)
    combined_file_path = os.path.join(output_folder, "5-ho_per_sat.png")
    fig.savefig(combined_file_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("   Completed!\n")


# ========================================================================================================= #


# 6. Average number and duration of out of services
if(out_of_service):
    print("6. Printing the average out of service time (lost_conn to rest_conn) ...")

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, (df_name, fname) in enumerate(zip(dfnames, fnames)):
        folder_path = Path("Cluster" + str(i+1) + " dataframes")
        
        # Master list for out-of-service durations for this cluster
        cluster_out_serv_durations = []
        
        for file_path in folder_path.glob('*.csv'):
            df = pd.read_csv(file_path)
            if df.empty:
                continue
            
            # parse time safely and ensure it is chronological
            df['arrival_time'] = pd.to_datetime(df['arrival_time'], errors='coerce', utc=True)
            df = df.sort_values('arrival_time')
            
            # state tracker for out of service periods
            out_serv_start_time = None
            
            for row in df.itertuples():
                t = row.arrival_time
                
                # safely extract destinations
                dest_sat = row.dest_satellite if pd.notna(row.dest_satellite) and str(row.dest_satellite) != 'None' else None

                # case A: the UE is disconnected entirely (out_serv / lost_conn event)
                if dest_sat is None:
                    # If we aren't already tracking a disconnection, start tracking now
                    if out_serv_start_time is None:
                        out_serv_start_time = t
                        
                # case B: the UE is connected to a satellite (rest_conn event)
                else:
                    # If we were tracking a disconnection, close it out and record the duration
                    if out_serv_start_time is not None:
                        cluster_out_serv_durations.append((t - out_serv_start_time).total_seconds())
                        out_serv_start_time = None # Reset state for the next potential lost_conn

        # Base color for this cluster
        color = plt.cm.tab10(i)

        # Dictionary to hold data for CSV saving later
        csv_data = {}
        
        # Check for enough data and variance to do a KDE
        if len(cluster_out_serv_durations) > 1 and min(cluster_out_serv_durations) != max(cluster_out_serv_durations):
            kde_out = gaussian_kde(cluster_out_serv_durations)
            x_min_out, x_max_out = min(cluster_out_serv_durations), max(cluster_out_serv_durations)
            margin_out = (x_max_out - x_min_out) * 0.2
            kde_x_out = np.linspace(x_min_out - margin_out, x_max_out + margin_out, 500)
            kde_y_out = kde_out(kde_x_out)

            # Plot solid line for Out of Service times
            ax.plot(kde_x_out, kde_y_out, color=color, linestyle='-', linewidth=1.5)
            ax.fill_between(kde_x_out, kde_y_out, alpha=0.3, color=color,
                            label=f"Cluster {i+1}: {fname}")
            
            # Mark the peak (highest probability density)
            idx_max = np.argmax(kde_y_out)
            x_peak, y_peak = kde_x_out[idx_max], kde_y_out[idx_max]
            ax.plot(x_peak, y_peak, marker='s', color=color, markersize=8, markeredgecolor='black', zorder=5)
            ax.annotate(f'  {x_peak:.1f}', xy=(x_peak, y_peak), fontsize=8, color=color, va='bottom')

            # Store for CSV
            csv_data['out_serv_x'] = kde_x_out
            csv_data['out_serv_density'] = kde_y_out
        else:
            print(f"Skipping KDE for Cluster {i+1} Out of Service Time due to lack of variance or data.")

        if save_plot_values and csv_data:
            os.makedirs(os.path.join(output_folder, fname), exist_ok=True)
            kde_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in csv_data.items()])) 
            kde_df.to_csv(os.path.join(output_folder, fname, "6-out_of_service_time.csv"), index=False)

    # Finalize chart formatting
    ax.set_title(f'Probability Density of Out of Service Time - All Clusters ({period} Period) - {num_ues_label} UEs')
    ax.set_xlabel('Out of Service Duration [s]')
    ax.set_ylabel('Probability Density')
    ax.grid(axis='y', alpha=0.3)
    ax.legend(title="Clusters", bbox_to_anchor=(1.05, 1), loc='upper left')

    os.makedirs(output_folder, exist_ok=True)
    combined_file_path = os.path.join(output_folder, "6-out_of_service_time.png")
    fig.savefig(combined_file_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("   Completed!\n")
    
    

# ========================================================================================================= #

# 7 Get the throughput considering the handover outage time (v2)
if(get_throuthput_ho_v2):
    print("7. Plotting the average throughput considering the handover outage time ...")

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, (df_name, fname) in enumerate(zip(dfnames, fnames)):
        folder_path = Path("Cluster" + str(i+1) + " throughput")
        ues_thr = []

        for file_path in folder_path.glob('*.csv'):
            df = pd.read_csv(file_path)
            thr = df['dl_thr'].tolist()
            ues_thr.append(thr)

        min_len = min(len(t) for t in ues_thr)
        ues_thr = [t[:min_len] for t in ues_thr]

        avg_thr = np.mean(ues_thr, axis=0).tolist()
        time_vector = pd.date_range(start=simTimeStart, periods=len(avg_thr), freq='1s')
        color = plt.cm.tab10(i)

        ax.plot(time_vector, avg_thr, label=f"Cluster {i+1}: {fname}", color=color)
        print(f"{fname} avg thr: ", np.mean(avg_thr))
        if(save_plot_values):
            os.makedirs(os.path.join(output_folder, fname), exist_ok=True)
            df_export = pd.DataFrame({
                'Seconds': range(len(avg_thr)),
                'Timestamp': time_vector,
                'Cluster_Thr': avg_thr
            })
            csv_file_path = os.path.join(output_folder, fname, "7-DL_throughput_ho_values.csv")
            df_export.to_csv(csv_file_path, index=False)

    ax.set_title('Average DL Throughputs over Time - All Clusters')
    ax.set_xlabel('Time')
    ax.set_ylabel('DL Throughput [Mbit/s]')
    ax.grid(True)
    ax.legend(title="Clusters", bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%M:%S'))

    os.makedirs(output_folder, exist_ok=True)
    combined_file_path = os.path.join(output_folder, "7-DL_throughput_ho.png")
    fig.savefig(combined_file_path, dpi=300, bbox_inches='tight')
    plt.close()
    print("   Completed!\n")

# ========================================================================================================= #


# 8. Number of ping-pong handovers
if(ping_pong_handovers):
    print("8. Printing the average number of ping-pong handovers ...")
    folder_path = Path('Cluster1 dataframes')
    ping_pong_count = []
    num_ues = 0
    for file_path in folder_path.glob('*.csv'):
        df = pd.read_csv(file_path)
        count = 0
        for r1, r2 in zip(df.itertuples(), df.iloc[1:].itertuples()):
            ev1 = r1.from_satellite
            ev2 = r2.dest_satellite
            
            if ev1 == ev2:
                count += 1
        ping_pong_count.append(count)
        num_ues += 1

    print(f"   Average number of ping-pong handovers: {np.mean(ping_pong_count)}")
    print("   Completed!\n")

# ========================================================================================================= #


# 9. UL/DL Doppler Shifts over time
if(doppler_shifts):

    print("9. Plotting the UL/DL Doppler Shifts over time ...")

    plt.figure(figsize=(14, 7))
    for i, (df_name, fname) in enumerate(zip(dfnames, fnames)):
        folder_path = Path("Cluster" + str(i+1) + " throughput")
        
        for file_path in folder_path.glob('*.csv'):
            df = pd.read_csv(file_path)
            
            # Convert time to datetime
            df['time'] = pd.to_datetime(df['time'])
            # Check where 'sat.id' is different from the previous row's 'sat.id'.
            handovers = df[(df['sat.id'].shift(1).notna()) & (df['sat.id'] != df['sat.id'].shift(1))]
            
            dl_label = f'DL Cluster{str(i+1)}' 
            ul_label = f'UL Cluster{str(i+1)}' 
            plt.plot(df['time'], df['doppler_shift_dl_KHz'], label=dl_label, color=colors1[i])
            plt.plot(df['time'], df['doppler_shift_ul_KHz'], label=ul_label, color=colors2[i])
            
            # Mark handover events with a red X
            if not handovers.empty:
                ho_label = f'HO Cluster{str(i+1)}'
                plt.scatter(handovers['time'], handovers['doppler_shift_ul_KHz'], 
                            color='red', marker='X', s=100, zorder=5, label=ho_label)
                
            break # Only plot the first file for this cluster

    # Format the plot
    plt.xlabel('Time')
    plt.ylabel('Doppler Shift (KHz)')
    plt.title('DL and UL Doppler Shifts over Time with Handover Events')
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    combined_file_path = os.path.join(output_folder, "9-Doppler Shifts.png")
    plt.savefig(combined_file_path)

    print("   Completed!")

# 9.1 UL/DL Doppler Shifts for a single random satellite connection (focus plot)
if(doppler_shifts):
    print("9.1 Plotting Doppler Shifts for a single random satellite connection ...")

    folder_path = Path("Cluster1 throughput")
    # take the first CSV file
    file_path = list(folder_path.glob('*.csv'))[0] 
    
    df = pd.read_csv(file_path)
    df['time'] = pd.to_datetime(df['time'])
    unique_sats = df['sat.id'].dropna().unique()
    chosen_sat = random.choice(unique_sats)
    print(f"    -> Selected satellite: {chosen_sat}")
    df_sat = df[df['sat.id'] == chosen_sat]

    plt.figure(figsize=(12, 6))
    plt.plot(df_sat['time'], df_sat['doppler_shift_dl_KHz'], label='DL', color='#1f77b4', linewidth=2)
    plt.plot(df_sat['time'], df_sat['doppler_shift_ul_KHz'], label='UL', color='#ff7f0e', linewidth=2)

    plt.xlabel('Time')
    plt.ylabel('Doppler Shi        # --- SAVE PLOT VALUES ---ft (KHz)')
    plt.title(f'DL and UL Doppler Shifts (Connection to {chosen_sat})')
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    combined_file_path = os.path.join(output_folder, "9.1-Doppler Shift Focus.png")
    plt.savefig(combined_file_path)

    print("   Completed!\n")