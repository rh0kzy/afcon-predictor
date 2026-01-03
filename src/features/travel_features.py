import pandas as pd
import numpy as np
from src.utils.geo_data import CAF_CAPITALS, MOROCCO_CITIES

def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians 
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    # haversine formula 
    dlat = lat2 - lat1 
    dlon = lon2 - lon1 
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a)) 
    r = 6371 # Radius of earth in kilometers. Use 3956 for miles
    return c * r

def calculate_travel_distance(df):
    """
    Calculates the travel distance for each team to the match city.
    If the city is not in Morocco host cities, it defaults to Rabat.
    """
    def get_dist(team, city):
        if team not in CAF_CAPITALS:
            return 5000 # Default for non-CAF or unknown
        
        team_coords = CAF_CAPITALS[team]
        
        if city in MOROCCO_CITIES:
            city_coords = MOROCCO_CITIES[city]
        else:
            # Default to Rabat if city unknown or not in Morocco
            city_coords = MOROCCO_CITIES['Rabat']
            
        return haversine(team_coords[0], team_coords[1], city_coords[0], city_coords[1])

    df['home_travel_dist'] = df.apply(lambda x: get_dist(x['home_team'], x['city']), axis=1)
    df['away_travel_dist'] = df.apply(lambda x: get_dist(x['away_team'], x['city']), axis=1)
    
    return df
