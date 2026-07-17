import math

class ChannelParameters:
    """
    Costanti globali fisiche e metodi statici per la geometria orbitale di base.
    """
    EARTH_RADIUS_KM = 6371.0    # Raggio della Terra in chilometri
    DATA_SIZE_BITS = 80000      # Costante generica di trasmissione (Bits)
    
    @staticmethod
    def within_range(observer_lat, observer_lon, satellite_lat, satellite_lon, range_m):
        """
        Determina tramite approssimazione planare (bounding box) se un satellite
        si trova all'interno di un raggio specifico rispetto all'osservatore.
        Utile per un primo scrematore computazionalmente leggero prima dei calcoli sferici.
        """
        range_km = range_m / 1000.0  
        delta_x = math.degrees(range_km / ChannelParameters.EARTH_RADIUS_KM)
        delta_y = delta_x / abs(math.cos(math.radians(observer_lat)))

        if observer_lat - delta_x < satellite_lat < observer_lat + delta_x:
            if observer_lon - delta_y < satellite_lon < observer_lon + delta_y:
                return satellite_lat, satellite_lon
        return None

    @staticmethod
    def elevation_angle_deg(device_lat, device_lon, satellite_lat, satellite_lon, satellite_height):
        """
        Calcola l'angolo di elevazione esatto (in gradi) tra un osservatore terrestre 
        e un satellite usando la trigonometria sferica e il teorema dei coseni.
        Questa funzione è la base cinematica per il calcolo della Utilità Base dell'SDN.
        """
        satellite_lat_rad = math.radians(satellite_lat)
        satellite_lon_rad = math.radians(satellite_lon)
        
        # Le altezze devono includere il raggio terrestre per riferirsi al centro del pianeta
        satellite_height_m = satellite_height + (ChannelParameters.EARTH_RADIUS_KM * 1000)
        
        observer_lat_rad = math.radians(device_lat)
        observer_lon_rad = math.radians(device_lon)
        observer_height_m = ChannelParameters.EARTH_RADIUS_KM * 1000 

        # Calcolo dell'angolo al centro della terra (gamma)
        gamma = math.acos(
            math.cos(observer_lat_rad) * math.cos(satellite_lat_rad) * math.cos(satellite_lon_rad - observer_lon_rad)
            + math.sin(observer_lat_rad) * math.sin(satellite_lat_rad)
        )

        radius_ratio = observer_height_m / satellite_height_m
        
        # Calcolo dell'angolo di elevazione (proiezione tangente)
        cos_elevation_angle = math.sin(gamma) / math.sqrt(1 + radius_ratio**2 - 2 * radius_ratio * math.cos(gamma))

        elevation_angle = math.acos(cos_elevation_angle)
        
        return math.degrees(elevation_angle)