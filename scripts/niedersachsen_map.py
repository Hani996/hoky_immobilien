import os
import sqlite3
import requests
import pandas as pd
import folium
import copy
import re

# -----------------------------
# 1. LOAD DATA
# -----------------------------
db_filename = "property_prices.db"
table_name = "prices"

if os.path.exists(db_filename):
    conn = sqlite3.connect(db_filename)
    df = pd.read_sql_query(f"SELECT geo_krs, obj_purchasePrice FROM {table_name}", conn)
    conn.close()
else:
    df = pd.read_csv("data/price_clean.csv")[["geo_krs", "obj_purchasePrice"]]

# -----------------------------
# 2. CLEAN FUNCTION
# -----------------------------
def normalize(name):
    name = name.lower()
    name = name.replace("_kreis", "")
    name = name.replace("_", " ")
    name = name.replace("-", " ")
    name = re.sub(r"\s+", " ", name).strip()
    return name

# Disambiguate Osnabrück BEFORE normalizing: the raw CSV has both
# "Osnabrück_Kreis" (district) and bare "Osnabrück" (city). Stripping
# "_kreis" makes them collapse to the identical string "osnabrück",
# so groupby silently averages district + city together before any
# manual_map can separate them again. Give the city a distinct raw
# label first so normalize() keeps them apart.
df["geo_krs"] = df["geo_krs"].replace({"Osnabrück": "Osnabrück_Stadt"})

df["geo_krs_clean"] = df["geo_krs"].apply(normalize)

# -----------------------------
# 3. FIXED MANUAL MAPPING (CORRECTED)
# IMPORTANT: must match GEOJSON AFTER normalization
# -----------------------------
manual_map = {
    "hannover": "hanover",
    "nienburg weser": "nienburg",
    "hameln pyrmont": "hamelin pyrmont",          # geojson misspells "Hameln" as "Hamelin"
    "salzgitter": "salzgitter städte",
    "wolfsburg": "wolfsburg städte",
    "braunschweig": "braunschweig städte",
    "wilhelmshaven": "wilhelmshaven städte",
    "emden": "emden städte",
    "oldenburg oldenburg": "oldenburg städte",    # city -> Stadt polygon
    # NOTE: plain "oldenburg" (the Landkreis) is intentionally NOT
    # remapped — it already matches geojson's plain "Oldenburg" district.
    "lüchow dannenberg": "lüchow danneberg",
    "osterode am harz": "osterode",
    "rotenburg wümme": "rotenburg",               # geojson drops "Wümme"
    "heidekreis": "soltau fallingbostel",         # geojson still uses the
                                                   # pre-2011 district name
    "osnabrück stadt": "osnabrück städte",
    # NOTE: plain "osnabrück" (the Landkreis) is intentionally NOT
    # remapped — it already matches geojson's plain "Osnabrück" district.
}

df["geo_krs_clean"] = df["geo_krs_clean"].replace(manual_map)

# NOTE ON DELMENHORST: this GeoJSON source (isellsoap/deutschlandGeoJSON,
# 4_kreise/2_hoch.geo.json) is simply missing a polygon for Delmenhorst
# under any name/property — verified by scanning every feature. It will
# always show as "no data" (or be absent entirely) on this map regardless
# of mapping. To include it, swap in a different/updated GeoJSON source.
# -----------------------------
# 4. AGGREGATE
# -----------------------------
avg_prices = (
    df.groupby("geo_krs_clean")["obj_purchasePrice"]
    .mean()
    .reset_index()
)

# -----------------------------
# 5. LOAD GEOJSON
# -----------------------------
url = "https://raw.githubusercontent.com/isellsoap/deutschlandGeoJSON/master/4_kreise/2_hoch.geo.json"
raw_geojson = requests.get(url).json()

filtered_features = [
    f for f in raw_geojson["features"]
    if f["properties"].get("NAME_1") == "Niedersachsen"
]

# -----------------------------
# 6. NORMALIZE GEOJSON NAMES
# -----------------------------
for f in filtered_features:
    name = f["properties"]["NAME_3"]
    name = normalize(name)

    # align spelling differences manually (geo side only)
    name = name.replace("dannenberg", "danneberg")

    f["properties"]["name"] = name

geojson_data = {
    "type": "FeatureCollection",
    "features": filtered_features
}

# -----------------------------
# 7. MAP
# -----------------------------
m = folium.Map(location=[52.6, 9.8], zoom_start=8, tiles="cartodbpositron")

folium.Choropleth(
    geo_data=geojson_data,
    name="choropleth",
    data=avg_prices,
    columns=["geo_krs_clean", "obj_purchasePrice"],
    key_on="feature.properties.name",
    fill_color="YlOrRd",
    fill_opacity=0.75,
    line_opacity=0.3,
    legend_name="Average Purchase Price (€)",
    nan_fill_color="#e0e0e0"
).add_to(m)

# -----------------------------
# 8. TOOLTIP
# -----------------------------
tooltip_geojson = copy.deepcopy(geojson_data)

price_lookup = avg_prices.set_index("geo_krs_clean")["obj_purchasePrice"].to_dict()

for f in tooltip_geojson["features"]:
    name = f["properties"]["name"]
    val = price_lookup.get(name)

    f["properties"]["display_price"] = (
        f"{val:,.2f} €" if pd.notna(val) else "No data available"
    )

folium.GeoJson(
    tooltip_geojson,
    style_function=lambda x: {
        "fillColor": "#ffffff",
        "color": "black",
        "weight": 0.3,
        "fillOpacity": 0
    },
    tooltip=folium.GeoJsonTooltip(
        fields=["name", "display_price"],
        aliases=["District:", "Avg Price:"],
        sticky=True
    )
).add_to(m)

# -----------------------------
# 9. SAVE
# -----------------------------
output_html = "niedersachsen_price_heatmap.html"
m.save(output_html)

print(f"✓ Map saved as: {output_html}")