import math

"""
Table 6.1.3.3-1 — Link Budget Results (TR 38.821)
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------
| Case | Orbit Alt. | Rx Mode | Freq    | EIRP [dBm] (dBW) | G/T [dB/K] | BW [MHz] | FSPL   | Atm. loss | Shadow loss | Scin. loss | Pol. loss  | Add. loss  | CNR   |
|      | [km]       |         | [GHz]   |                  |            |          | [dB]   | [dB]      | [dB]        | [dB]       | [dB]       | [dB]       | [dB]  |
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------
| SC6  | 600        | DL      | 20.0    | 60.0 (30.0)      |  15.9      | 400.0    | 179.1  | 0.5       | 0.0         | 0.3        | 0.0        | 0.0        | 8.5   |
|      |            | UL      | 30.0    | 76.2 (46.2)      |  13.0      | 400.0    | 182.6  | 0.5       | 0.0         | 0.3        | 0.0        | 0.0        | 18.4  |
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------
| SC9  | 600        | DL      | 2.0     | 78.8 (48.8)      | -31.6      | 30.0     | 159.1  | 0.1       | 3.0         | 2.2        | 0.0        | 0.0        | 6.6   |
|      |            | UL      | 3.0     | 23.0 (-7.0)      |  1.1       | 0.4      | 159.1  | 0.1       | 3.0         | 2.2        | 0.0        | 0.0        | 2.8   |
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------

Table 6.1.1.1-1 - Satellite antenna parameters (TR 38.821)
-------------------------------------------------------------------------------------
| Case | Antenna Aper. | EIRP Density  | Max Gain | 3dB Beamwidth  | Beam Diameter |
|      | [m]           | [dBW/MHz]     | [dBi]    | [deg]          | [Km]          |                    
-------------------------------------------------------------------------------------
| SC6  | 0.5           | 4.0           | 38.5     | 1.7647         | 20            |
-------------------------------------------------------------------------------------
| SC9  | 2.0           | 34.0          | 30.0     | 4.4127         | 50            |
-------------------------------------------------------------------------------------

Table 6.1.1.1-3 - UE antenna parameters (TR 38.821)
---------------------------------------------------------------------------
| Case | Rx Gain  | Antenna Temp.  | Noise Fig. | Tx Power    | Tx Gain  |
|      | [dBi]    | [K]            | [dB]       | [W] (dBm)   | [dBi]    |                    
---------------------------------------------------------------------------
| SC6  | 39.7     | 150            | 1.2        | 2 (33)      | 43.2     |
---------------------------------------------------------------------------
| SC9  | 0 (1x1)  | 290            | 7.0        | 0.2 (23)    | 0 (1x1)  |
|      | 3 (2x1)  |                |            |             | 3 (2x1)  |
---------------------------------------------------------------------------
"""

sc6_parameters = {
    'eirp_ue' : 76.2,   # dBm
    'gt_sat' : 13,      # dBi
    'eirp_sat' : 60,    # dBm
    'gt_ue' : 15.9,     # dBi
    'bw_dl' : 400e6,    # Hz
    'bw_ul' : 400e6,    # Hz
    'freq_dl' : 2e9,    # Hz
    'freq_ul' : 3e9,    # Hz
    'atm_loss' : 5.3,   # dB
    'dl_db_headroom' : 0, # dB
    'ul_db_headroom' : 0,  # dB
    'dlul_snr_variance' : 1, # Varianza rumore gaussiano (Scenario Stocastico)
    '3gpp_overhead_dl': 0.18, # Overhead 3GPP TS 38.306
    '3gpp_overhead_ul': 0.1
}

sc9_parameters = {
    'eirp_ue' : 23,     # dBm
    'gt_sat' : 1.1,     # dBi
    'eirp_sat' : 78.8,  # dBm
    'gt_ue' : -31.6,    # dBi
    'bw_dl' : 30e6,     # Hz
    'bw_ul' : 0.4e6,    # Hz
    'freq_dl' : 2e9,    # Hz
    'freq_ul' : 3e9,    # Hz
    'atm_loss' : 0.8,   # dB
    'dl_db_headroom' : 2, # dB
    'ul_db_headroom' : 2,  # dB
    'dlul_snr_variance' : 1, # Varianza rumore gaussiano (Scenario Stocastico)
    '3gpp_overhead_dl': 0.14, # Overhead 3GPP TS 38.306
    '3gpp_overhead_ul': 0.08
}

class Channel:
    """
    Classe statica che modella il livello fisico (PHY) della rete NTN.
    Implementa il calcolo del Link Budget tramite FSPL e la stima della 
    capacità teorica tramite il teorema di Shannon-Hartley.
    """
    
    @staticmethod
    def compute_snr(distance_m, parameters):
        eirp_ue = parameters['eirp_ue']
        gt_sat = parameters['gt_sat']
        eirp_sat = parameters['eirp_sat']
        gt_ue = parameters['gt_ue']
        bw_dl = parameters['bw_dl']
        bw_ul = parameters['bw_ul']
        freq_dl = parameters['freq_dl']
        freq_ul = parameters['freq_ul']

        c = 299792458 
        path_loss_dl_db = 20 * math.log10(distance_m) + 20 * math.log10(freq_dl) + 20 * math.log10(4 * math.pi / c)
        path_loss_ul_db = 20 * math.log10(distance_m) + 20 * math.log10(freq_ul) + 20 * math.log10(4 * math.pi / c)

        received_power_dl_dbm = eirp_sat + gt_ue - path_loss_dl_db
        received_power_ul_dbm = eirp_ue + gt_sat - path_loss_ul_db

        # Il valore 198.6 incapsula la costante di Boltzmann (-10*log10(kT))
        snr_dl_db = received_power_dl_dbm + 198.6 - 10 * math.log10(bw_dl)
        snr_ul_db = received_power_ul_dbm + 198.6 - 10 * math.log10(bw_ul)

        return snr_dl_db, snr_ul_db

    @staticmethod
    def compute_shannon(distance_m, parameters):
        snr_dl_db, snr_ul_db = Channel.compute_snr(distance_m, parameters)
        return Channel.compute_shannon_from_snr(snr_dl_db, snr_ul_db, parameters)

    @staticmethod
    def compute_shannon_from_snr(snr_dl_db, snr_ul_db, parameters):
        snr_dl_linear = 10 ** (snr_dl_db / 10)
        snr_ul_linear = 10 ** (snr_ul_db / 10)

        dl_thr_mbps = round(parameters['bw_dl'] * math.log2(1 + snr_dl_linear) / 1e6, 4)
        ul_thr_mbps = round(parameters['bw_ul'] * math.log2(1 + snr_ul_linear) / 1e6, 4)

        return dl_thr_mbps, ul_thr_mbps

    @staticmethod
    def reverse_snr_from_thr(dl_ue_thr, ul_ue_thr, parameters):
        if dl_ue_thr <= 0.0:
            snr_dl_db = -100.0
        else:
            snr_dl_linear = (2 ** (dl_ue_thr * 1e6 / parameters['bw_dl']) - 1)
            snr_dl_db = -100.0 if snr_dl_linear <= 0 else round(10 * math.log10(snr_dl_linear), 4)

        if ul_ue_thr <= 0.0:
            snr_ul_db = -100.0
        else:
            snr_ul_linear = (2 ** (ul_ue_thr * 1e6 / parameters['bw_ul']) - 1)
            snr_ul_db = -100.0 if snr_ul_linear <= 0 else round(10 * math.log10(snr_ul_linear), 4)

        return snr_dl_db, snr_ul_db
