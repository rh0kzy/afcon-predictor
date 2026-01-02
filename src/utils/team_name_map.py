TEAM_NAME_MAP = {
    "Congo DR": "DR Congo",
    "Democratic Republic of the Congo": "DR Congo",
    "Ivory Coast": "Ivory Coast",
    "Côte d'Ivoire": "Ivory Coast",
    "Swaziland": "Eswatini",
    "Cape Verde Islands": "Cape Verde",
    "Cabo Verde": "Cape Verde",
    # Add more mappings as discovered during data cleaning
}

def normalize_team_name(name):
    return TEAM_NAME_MAP.get(name, name)
