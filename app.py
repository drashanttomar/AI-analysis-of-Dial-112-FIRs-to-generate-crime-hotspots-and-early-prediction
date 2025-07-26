import streamlit as st
import pandas as pd
import joblib
import numpy as np
from datetime import date, time, datetime, timedelta
import pytz
import base64
# Assuming ml_logic and forecast are available as local modules
# from ml_logic import predict_crime_proba
# from forecast import prepare_time_series_data, train_forecaster, make_forecast
import plotly.express as px
import pydeck as pdk
import json
import os
from streamlit.components.v1 import html as st_html

# Dummy functions for ml_logic and forecast if not available
# In a real scenario, ensure these modules and their functions are properly implemented and accessible.
def predict_crime_proba(hour, dayofweek, month, latitude, longitude):
    """
    Dummy function to simulate crime probability prediction.
    Replace with actual model prediction logic from ml_logic.py.
    """
    # Simple dummy logic: higher probability for certain times/locations
    # This should be replaced by your actual loaded model's prediction
    if 28.6 < latitude < 28.7 and 77.1 < longitude < 77.3: # Delhi area
        if 18 <= hour <= 23: # Evening/night
            return [("Theft", 80.0), ("Assault", 60.0), ("Vandalism", 40.0)]
        elif 0 <= hour <= 6: # Late night/early morning
            return [("Robbery", 75.0), ("Assault", 55.0), ("Theft", 30.0)]
        else:
            return [("Minor Offense", 20.0), ("Theft", 10.0)]
    else:
        return [("Minor Incident", 5.0)] # Low probability elsewhere


def prepare_time_series_data(df):
    """
    Dummy function to prepare time series data.
    Replace with actual logic from forecast.py.
    """
    # Group by date and count crimes
    if 'Date' in df.columns and 'Crime Head' in df.columns:
        df['ds'] = pd.to_datetime(df['Date']).dt.normalize()
        time_series = df.groupby('ds').size().reset_index(name='y')
        return time_series
    return pd.DataFrame(columns=['ds', 'y'])

class DummyForecaster:
    def __init__(self, time_series_df):
        self.time_series_df = time_series_df

    def fit(self):
        pass # No actual training for dummy

    def make_future_dataframe(self, periods, freq='D'):
        last_date = self.time_series_df['ds'].max()
        return pd.DataFrame({'ds': pd.date_range(start=last_date + pd.Timedelta(days=1), periods=periods, freq=freq)})

    def predict(self, future_df):
        # Simple linear trend for dummy prediction
        # In a real model, this would be actual forecasting
        if not self.time_series_df.empty:
            last_y = self.time_series_df['y'].iloc[-1]
        else:
            last_y = 0 # Default if no historical data

        future_df['yhat'] = np.linspace(last_y, last_y * 1.05, len(future_df))
        future_df['yhat_lower'] = future_df['yhat'] * 0.9
        future_df['yhat_upper'] = future_df['yhat'] * 1.1
        return future_df[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

def train_forecaster(time_series_df):
    """
    Dummy function to simulate training a forecaster.
    Replace with actual logic from forecast.py (e.g., Prophet model).
    """
    return DummyForecaster(time_series_df)

def make_forecast(model, periods):
    """
    Dummy function to make a forecast.
    Replace with actual logic from forecast.py.
    """
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    return forecast


# Cesium configuration - REPLACE WITH YOUR ACTUAL TOKEN
CESIUM_TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJqdGkiOiJhMmRhMDliMy01YmU3LTRiMjctOTdlYy0xMjQ3ZjlmZDcyMzMiLCJpZCI6MzIzNzk5LCJpYXQiOjE3NTMyNTk5MTd9.6DtEv4TPEn9t6mH6GoFtqmAtnu24B8-Ig03h4ctolAY"

# Set full-screen animated GIF background with no dim
def set_gif_background(gif_path):
    with open(gif_path, "rb") as f:
        gif_bytes = f.read()
        b64_gif = base64.b64encode(gif_bytes).decode()

    css_code = f"""
    <style>
    html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {{
        background: url("data:image/gif;base64,{b64_gif}") no-repeat center center fixed;
        background-size: cover;
    }}
    </style>
    """
    st.markdown(css_code, unsafe_allow_html=True)

# Call with your actual file
# Ensure '9.gif' is in the same directory as your Streamlit script or provide a full path
if os.path.exists("9.gif"):
    set_gif_background("9.gif")
else:
    st.warning("Background GIF '9.gif' not found. Please ensure it's in the correct directory.")


# Setup Streamlit layout
st.set_page_config(layout="wide")
st.title("AI analysis of Dial 112, FIRs to generate crime hotspots and early prediction for L&O")

# Load model and data
model = None
label_encoder = None
try:
    # Check if model files exist before loading
    if os.path.exists("crime_predictor.pkl") and os.path.exists("crime_label_encoder.pkl"):
        model = joblib.load("crime_predictor.pkl")
        label_encoder = joblib.load("crime_label_encoder.pkl")
    else:
        st.warning("Model files (crime_predictor.pkl, crime_label_encoder.pkl) not found. Prediction features will be simulated.")
except Exception as e:
    st.error(f"Failed to load model files: {str(e)}. Prediction features will be simulated.")

fir_data = pd.DataFrame()
call_data = pd.DataFrame()
try:
    # Check if data files exist before loading
    if os.path.exists("FIR_Dataset_10K_India_Locations.csv"):
        fir_data = pd.read_csv("FIR_Dataset_10K_India_Locations.csv", parse_dates=["Timestamp"])
        fir_data.rename(columns={"Timestamp": "Date", "Crime_Type": "Crime Head", "State": "State/UT"}, inplace=True)
    else:
        st.error("FIR_Dataset_10K_India_Locations.csv not found. Please ensure it's in the correct directory.")

    if os.path.exists("Dial_112_10K_India_Locations.csv"):
        call_data = pd.read_csv("Dial_112_10K_India_Locations.csv")
    else:
        st.error("Dial_112_10K_India_Locations.csv not found. Please ensure it's in the correct directory.")

except Exception as e:
    st.error(f"Failed to load data files: {str(e)}")
    st.stop()

# Ensure fir_data has required columns for initial filtering, provide dummy if empty
if fir_data.empty or not all(col in fir_data.columns for col in ["State/UT", "Crime Head", "Latitude", "Longitude", "Date", "City"]):
    st.warning("FIR_Dataset_10K_India_Locations.csv is missing required columns or is empty. Using dummy data for demonstration.")
    fir_data = pd.DataFrame({
        "Date": pd.to_datetime(["2023-01-01 10:00", "2023-01-05 14:30", "2023-01-10 20:00", "2023-01-15 03:00"]),
        "Crime Head": ["Theft", "Assault", "Burglary", "Vandalism"],
        "State/UT": ["Delhi", "Maharashtra", "Karnataka", "Delhi"],
        "City": ["New Delhi", "Mumbai", "Bengaluru", "New Delhi"],
        "Latitude": [28.6139, 19.0760, 12.9716, 28.5355],
        "Longitude": [77.2090, 72.8777, 77.5946, 77.3910]
    })
    fir_data["Date"] = pd.to_datetime(fir_data["Date"]) # Ensure Date is datetime type

# Sidebar filters
st.sidebar.header("🔍 Filter Crime Data")
selected_state = st.sidebar.selectbox("Select State", ["All"] + sorted(fir_data["State/UT"].unique().tolist()))
selected_crime = st.sidebar.selectbox("Select Crime Type", ["All"] + sorted(fir_data["Crime Head"].unique().tolist()))
selected_date = st.sidebar.date_input("📅 Select Date", min_value=date(2000, 1, 1), value=date.today())
selected_time = st.sidebar.time_input("⏰ Select Time", value=time(12, 0))
selected_tz = st.sidebar.selectbox("🌐 Select Time Zone", options=["Asia/Kolkata", "UTC", "US/Pacific", "Europe/London", "Australia/Sydney"])
show_predictions = st.sidebar.checkbox("Show Predicted Crime Risk", value=True)
map_style = st.sidebar.selectbox("Map Style", ["3D Globe (Cesium)", "3D Hexbin", "3D Point Cloud", "2D Heatmap"])

# Cesium theme options with enhanced controls
cesium_theme_options = {
    "Cesium World Terrain": "terrain_default",
    "Bing Maps Aerial": "bing_aerial",
    "Black Marble (Night)": "black_marble",
    "ESRI World Imagery": "esri_world",
    "OpenStreetMap": "osm",
    "Natural Earth II": "natural_earth"
}
selected_cesium_theme = st.sidebar.selectbox("🌎 Cesium Globe Theme", list(cesium_theme_options.keys()))

# Enhanced Cesium controls
st.sidebar.header("🌦️ Cesium Visualization Controls")
enable_3d_buildings = st.sidebar.checkbox("Show 3D Buildings", value=True)
enable_day_night = st.sidebar.checkbox("Enable Day/Night Cycle", value=True)
enable_shadows = st.sidebar.checkbox("Enable Shadows", value=True)

# Combine date + time + timezone with proper handling
selected_datetime = datetime.combine(selected_date, selected_time)
try:
    selected_datetime_tz = pytz.timezone(selected_tz).localize(selected_datetime)
    utc_datetime = selected_datetime_tz.astimezone(pytz.UTC)
except pytz.UnknownTimeZoneError:
    st.error(f"Unknown time zone: {selected_tz}. Defaulting to UTC.")
    selected_datetime_tz = pytz.utc.localize(selected_datetime)
    utc_datetime = selected_datetime_tz

# Filter historical data based on selected_date
filtered_historical_data = fir_data.copy()
filtered_historical_data = filtered_historical_data[filtered_historical_data["Date"].dt.date <= selected_date]

if selected_state != "All":
    filtered_historical_data = filtered_historical_data[filtered_historical_data["State/UT"] == selected_state]
if selected_crime != "All":
    filtered_historical_data = filtered_historical_data[filtered_historical_data["Crime Head"] == selected_crime]

filtered_historical_data = filtered_historical_data.sort_values("Date", ascending=False).head(500)

# Prepare data for visualization
def prepare_visualization_data(historical_data, include_predictions=False, selected_state_filter="All", selected_crime_filter="All"):
    historical_points = []
    for _, row in historical_data.iterrows():
        historical_points.append({
            "lat": row["Latitude"],
            "lon": row["Longitude"],
            "type": "historical",
            "crime_type": row["Crime Head"],
            "date": str(row["Date"].strftime("%Y-%m-%d")),
            "time": str(row["Date"].strftime("%H:%M")),
            "city": row["City"],
            "state": row["State/UT"],
            "value": 1,
            "latitude": row["Latitude"],
            "longitude": row["Longitude"]
        })

    predicted_points = []
    if include_predictions:
        # Start with all FIR data for prediction source
        pred_source_data = fir_data.copy()

        # Apply state and crime filters to the prediction source data
        if selected_state_filter != "All":
            pred_source_data = pred_source_data[pred_source_data["State/UT"] == selected_state_filter]
        if selected_crime_filter != "All":
            pred_source_data = pred_source_data[pred_source_data["Crime Head"] == selected_crime_filter]

        pred_source_data = pred_source_data[["Latitude", "Longitude"]].drop_duplicates()

        if not pred_source_data.empty:
            # Sample up to 200 unique locations for predictions to balance performance and coverage
            pred_locations = pred_source_data.sample(min(200, len(pred_source_data)), random_state=42).reset_index(drop=True)
        else:
            st.warning("No unique locations found in FIR data to generate predictions for the selected filters. Try broadening your selection.")
            pred_locations = pd.DataFrame()

        if not pred_locations.empty:
            for _, row in pred_locations.iterrows():
                lat, lon = row["Latitude"], row["Longitude"]
                
                # Call the (possibly dummy) predict_crime_proba function
                top_predictions = predict_crime_proba(
                    hour=selected_datetime_tz.hour,
                    dayofweek=selected_datetime_tz.weekday(),
                    month=selected_datetime_tz.month,
                    latitude=lat,
                    longitude=lon
                )
                top_crime, top_prob = top_predictions[0] if top_predictions else ("Unknown", 0.0)

                # Only add if there's a significant probability
                if top_prob > 0:
                    predicted_points.append({
                        "lat": lat,
                        "lon": lon,
                        "type": "predicted",
                        "crime_type": top_crime,
                        "probability": top_prob,
                        "value": top_prob / 100, # Normalize probability for height/size scaling
                        "latitude": lat,
                        "longitude": lon
                    })

    return historical_points + predicted_points

# Load regional data
regional_data_for_cesium = []
try:
    if os.path.exists("world_capitals.csv"):
        world_capitals_df = pd.read_csv("world_capitals.csv")
        if 'Capital' in world_capitals_df.columns and 'Latitude' in world_capitals_df.columns and 'Longitude' in world_capitals_df.columns:
            world_capitals_df = world_capitals_df.rename(columns={
                'Capital': 'name',
                'Latitude': 'latitude',
                'Longitude': 'longitude'
            })
        elif 'name' in world_capitals_df.columns and 'lat' in world_capitals_df.columns and 'lon' in world_capitals_df.columns:
            world_capitals_df = world_capitals_df.rename(columns={
                'lat': 'latitude',
                'lon': 'longitude'
            })
        else:
            st.sidebar.warning("`world_capitals.csv` columns not recognized. Expected 'Capital', 'Latitude', 'Longitude' or 'name', 'lat', 'lon'. Using limited regional labels (Delhi region).")
            raise ValueError("Column mismatch")

        regional_data_for_cesium = world_capitals_df[['name', 'latitude', 'longitude']].to_dict(orient='records')
        st.sidebar.info(f"Loaded {len(regional_data_for_cesium)} global regional labels.")
    else:
        st.sidebar.warning("`world_capitals.csv` not found. Displaying limited regional labels (Delhi region).")
        # Fallback to Delhi-centric locations
        regional_data_for_cesium = [
            {"name": "PAHARGANJ", "latitude": 28.6475, "longitude": 77.2007},
            {"name": "RAMLILA GROUND", "latitude": 28.6366, "longitude": 77.2346},
            {"name": "BASANT LANE RAILWAY COLONY", "latitude": 28.6339, "longitude": 77.1994},
            {"name": "TELEGRAPH PLACE", "latitude": 28.6292, "longitude": 77.1917},
            {"name": "DIZ STAFF QUARTERS", "latitude": 28.6264, "longitude": 77.2069},
            {"name": "BLOCK B", "latitude": 28.6480, "longitude": 77.1850},
            {"name": "KHAN MARKET", "latitude": 28.6015, "longitude": 77.2289},
            {"name": "CONNAUGHT PLACE", "latitude": 28.6310, "longitude": 77.2167},
            {"name": "JANPATH", "latitude": 28.6200, "longitude": 77.2170},
            {"name": "RASHTRAPATI BHAVAN", "latitude": 28.6143, "longitude": 77.1993},
            {"name": "INDIA GATE", "latitude": 28.6129, "longitude": 77.2295}
        ]

except Exception as e:
    st.sidebar.error(f"Error loading `world_capitals.csv`: {e}. Using limited regional labels.")
    # Fallback to Delhi-centric locations
    regional_data_for_cesium = [
        {"name": "PAHARGANJ", "latitude": 28.6475, "longitude": 77.2007},
        {"name": "RAMLILA GROUND", "latitude": 28.6366, "longitude": 77.2346},
        {"name": "BASANT LANE RAILWAY COLONY", "latitude": 28.6339, "longitude": 77.1994},
        {"name": "TELEGRAPH PLACE", "latitude": 28.6292, "longitude": 77.1917},
        {"name": "DIZ STAFF QUARTERS", "latitude": 28.6264, "longitude": 77.2069},
        {"name": "BLOCK B", "latitude": 28.6480, "longitude": 77.1850},
        {"name": "KHAN MARKET", "latitude": 28.6015, "longitude": 77.2289},
        {"name": "CONNAUGHT PLACE", "latitude": 28.6310, "longitude": 77.2167},
        {"name": "JANPATH", "latitude": 28.6200, "longitude": 77.2170},
        {"name": "RASHTRAPATI BHAVAN", "latitude": 28.6143, "longitude": 77.1993},
        {"name": "INDIA GATE", "latitude": 28.6129, "longitude": 77.2295}
    ]

# Render Cesium globe with enhanced features
def render_cesium(data_points, selected_theme):
    if not CESIUM_TOKEN:
        st.error("Cesium token is missing! Visualization will not work.")
        return None

    # Theme configuration
    themes = {
        "terrain_default": {
            "js": "viewer.imageryLayers.addImageryProvider(new Cesium.IonImageryProvider({ assetId: 3812 }));",
            "lighting": "true"
        },
        "bing_aerial": {
            "js": """
            viewer.imageryLayers.addImageryProvider(new Cesium.BingMapsImageryProvider({
                url: 'https://dev.virtualearth.net',
                key: '',  // Add your Bing Maps key here if you have one, otherwise it might default to a limited version
                mapStyle: Cesium.BingMapsStyle.AERIAL
            }));""",
            "lighting": "true"
        },
        "black_marble": {
            "js": "viewer.imageryLayers.addImageryProvider(new Cesium.IonImageryProvider({ assetId: 3845 }));",
            "lighting": "false"
        },
        "esri_world": {
            "js": """
            viewer.imageryLayers.addImageryProvider(new Cesium.ArcGisMapServerImageryProvider({
                url: 'https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer'
            }));""",
            "lighting": "true"
        },
        "osm": {
            "js": """
            viewer.imageryLayers.addImageryProvider(new Cesium.OpenStreetMapImageryProvider({
                url: 'https://a.tile.openstreetmap.org/'
            }));""",
            "lighting": "true"
        },
        "natural_earth": {
            "js": """
            viewer.imageryLayers.addImageryProvider(new Cesium.TileMapServiceImageryProvider({
                url: Cesium.buildModuleUrl('Assets/Textures/NaturalEarthII')
            }));""",
            "lighting": "true"
        }
    }

    # Prepare day/night cycle JS with fixed time handling
    day_night_js = ""
    if enable_day_night:
        # Pass the UTC datetime from Streamlit to Cesium's clock
        day_night_js = f"""
        // Set initial time to synchronize with Streamlit's selected time
        const startTime = Cesium.JulianDate.fromDate(new Date('{utc_datetime.isoformat()}'));
        viewer.clock.currentTime = startTime;
        viewer.clock.startTime = startTime;
        // Set stopTime to allow time progression if needed, or keep it fixed
        viewer.clock.stopTime = Cesium.JulianDate.addSeconds(startTime, 24 * 3600, new Cesium.JulianDate()); // Example: allow 24 hours of animation
        viewer.clock.clockRange = Cesium.ClockRange.LOOP_STOP; // Play once then stop, or LOOP_INDIFINITE

        // Enable lighting based on time of day
        viewer.scene.globe.enableLighting = true; // Ensure lighting is enabled for day/night cycle
        viewer.shadowMap.enabled = {'true' if enable_shadows else 'false'}; // Apply shadows setting
        """
    else:
        day_night_js = f"""
        viewer.scene.globe.enableLighting = {themes.get(selected_theme, themes["terrain_default"])["lighting"]};
        viewer.shadowMap.enabled = {'true' if enable_shadows else 'false'};
        """


    # Prepare buildings JS
    buildings_js = "viewer.scene.primitives.add(Cesium.createOsmBuildings());" if enable_3d_buildings else ""

    # Create entities for visualization
    entities = []
    for point in data_points:
        # Define base color based on type
        color_r, color_g, color_b, color_a = (128, 128, 128, 255) # Historical: Grey
        pixel_size = 10
        outline_width = 2
        point_height = 0 # For CLAMP_TO_GROUND, height is 0

        if point["type"] == "predicted":
            color_r, color_g, color_b, color_a = (255, 0, 0, 255) # Predicted: Red
            # Scale pixel size by probability for predicted points
            pixel_size = 20 + (point.get('value', 0) * 30) # More prominent for higher probability
            outline_width = 3

        entities.append({
            "id": f"{point['type']}_{len(entities)}",
            "name": point.get('crime_type', point.get('name', 'N/A')),
            "position": {
                "longitude": point['lon'],
                "latitude": point['lat'],
                "height": point_height # Always 0 for CLAMP_TO_GROUND points
            },
            "point_style": {
                "color": {"r": color_r, "g": color_g, "b": color_b, "a": color_a},
                "pixelSize": pixel_size,
                "outlineColor": {"r": 255, "g": 255, "b": 255, "a": 255}, # White outline for all points
                "outlineWidth": outline_width,
                "heightReference": "CLAMP_TO_GROUND" # Make both historical and predicted points stick to the ground
            },
            "description": f"""
                <h3>{'Historical Crime' if point["type"] == "historical" else 'Future Crime (Predicted)'}:</h3>
                <b>Crime Type:</b> {point['crime_type'] if point["type"] == "historical" else point['crime_type']}<br>
                {'<b>Date:</b> ' + point['date'] + '<br>' if point["type"] == "historical" else ''}
                <b>Time:</b> {point['time'] if point["type"] == "historical" else selected_datetime_tz.strftime('%Y-%m-%d %H:%M %Z')}<br>
                {'<b>Probability:</b> ' + f"{point['probability']:.2f}%<br>" if point["type"] == "predicted" else ''}
                <b>Coordinates:</b> {point['latitude']:.4f}, {point['longitude']:.4f}<br>
                {'<b>Location:</b> ' + point['city'] + ', ' + point['state'] if point["type"] == "historical" else ''}
            """,
            "type": "crime_point" # Unified type for crime points
        })

    # Add regional labels
    for region in regional_data_for_cesium:
        entities.append({
            "id": f"region_{region['name'].replace(' ', '_').replace('.', '')}",
            "name": region['name'],
            "position": {
                "longitude": region['longitude'],
                "latitude": region['latitude'],
                "height": 0 # Labels are on the ground
            },
            "label_style": {
                "text": region['name'],
                "font": '14px sans-serif',
                "fillColor": {"r": 255, "g": 255, "b": 255, "a": 255}, # White text
                "outlineColor": {"r": 0, "g": 0, "b": 0, "a": 255}, # Black outline
                "outlineWidth": 1,
                "style": "FILL_AND_OUTLINE",
                "verticalOrigin": "BOTTOM",
                "pixelOffset": {"x": 0, "y": -5},
                # Make labels fade and scale with distance for better visibility
                "translucencyByDistance": {"near": 1000000, "far": 80000000, "nearValue": 1.0, "farValue": 0.0},
                "scaleByDistance": {"near": 1000000, "far": 80000000, "nearValue": 1.0, "farValue": 0.0},
                "showBackground": False
            },
            "type": "region_label"
        })

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Crime Visualization</title>
        <script src="https://cesium.com/downloads/cesiumjs/releases/1.104/Build/Cesium/Cesium.js"></script>
        <link href="https://cesium.com/downloads/cesiumjs/releases/1.104/Build/Cesium/Widgets/widgets.css" rel="stylesheet">
        <style>
            html, body, #cesiumContainer {{
                width: 100%;
                height: 100%;
                margin: 0;
                padding: 0;
                overflow: hidden;
            }}
        </style>
    </head>
    <body>
        <div id="cesiumContainer"></div>
        <script>
            Cesium.Ion.defaultAccessToken = '{CESIUM_TOKEN}';

            const viewer = new Cesium.Viewer('cesiumContainer', {{
                timeline: true,
                animation: true,
                baseLayerPicker: false,
                shouldAnimate: true,
                terrainProvider: Cesium.createWorldTerrain(),
                shadows: {'true' if enable_shadows else 'false'}
            }});

            // Apply selected theme
            {themes.get(selected_theme, themes["terrain_default"])["js"]}
            
            // Apply day/night cycle and shadows dynamically
            {day_night_js}

            // Add entities
            const entities = {json.dumps(entities)};
            entities.forEach(entity => {{
                const position = Cesium.Cartesian3.fromDegrees(
                    entity.position.longitude,
                    entity.position.latitude,
                    entity.position.height
                );

                if (entity.type === "region_label") {{
                    const ls = entity.label_style || {{}};
                    const lsFillColor = ls.fillColor || {{r: 255, g: 255, b: 255, a: 255}};
                    const lsOutlineColor = ls.outlineColor || {{r: 0, g: 0, b: 0, a: 255}};
                    viewer.entities.add({{
                        id: entity.id,
                        position: position,
                        label: {{
                            text: ls.text,
                            font: ls.font,
                            fillColor: Cesium.Color.fromCssColorString(
                                `rgba(${{lsFillColor.r}}, ${{lsFillColor.g}}, ${{lsFillColor.b}}, ${{lsFillColor.a / 255}})`
                            ),
                            outlineColor: Cesium.Color.fromCssColorString(
                                `rgba(${{lsOutlineColor.r}}, ${{lsOutlineColor.g}}, ${{lsOutlineColor.b}}, ${{lsOutlineColor.a / 255}})`
                            ),
                            outlineWidth: ls.outlineWidth,
                            style: Cesium.LabelStyle[ls.style || 'FILL'],
                            verticalOrigin: Cesium.VerticalOrigin[ls.verticalOrigin || 'CENTER'],
                            pixelOffset: new Cesium.Cartesian2(ls.pixelOffset.x, ls.pixelOffset.y),
                            disableDepthTestDistance: Number.POSITIVE_INFINITY, // Always show label
                            translucencyByDistance: new Cesium.NearFarScalar(ls.translucencyByDistance.near, ls.translucencyByDistance.nearValue, ls.translucencyByDistance.far, ls.translucencyByDistance.farValue),
                            scaleByDistance: new Cesium.NearFarScalar(ls.scaleByDistance.near, ls.scaleByDistance.nearValue, ls.scaleByDistance.far, ls.scaleByDistance.farValue),
                            showBackground: ls.showBackground
                        }}
                    }});
                }} else if (entity.type === "crime_point") {{
                    const ps = entity.point_style || {{}};
                    const psColor = ps.color || {{r: 0, g: 0, b: 0, a: 255}};
                    const psOutlineColor = ps.outlineColor || {{r: 0, g: 0, b: 0, a: 255}};

                    viewer.entities.add({{
                        id: entity.id,
                        name: entity.name,
                        position: position,
                        point: {{
                            color: Cesium.Color.fromCssColorString(
                                `rgba(${{psColor.r}}, ${{psColor.g}}, ${{psColor.b}}, ${{psColor.a / 255}})`
                            ),
                            pixelSize: ps.pixelSize || 10,
                            outlineColor: Cesium.Color.fromCssColorString(
                                `rgba(${{psOutlineColor.r}}, ${{psOutlineColor.g}}, ${{psOutlineColor.b}}, ${{psOutlineColor.a / 255}})`
                            ),
                            outlineWidth: ps.outlineWidth || 0,
                            heightReference: Cesium.HeightReference[ps.heightReference || 'CLAMP_TO_GROUND']
                        }},
                        description: entity.description
                    }});
                }}
            }});

            // Add 3D buildings if enabled
            {buildings_js}

            // Zoom to all entities or a default view if no entities
            if (viewer.entities.values.length > 0) {{
                viewer.zoomTo(viewer.entities);
            }} else {{
                // Default view for India if no entities are present
                viewer.camera.setView({{
                    destination : Cesium.Cartesian3.fromDegrees(78.6569, 22.9734, 10000000.0) // Lon, Lat, Height (zoom out)
                }});
            }}
        </script>
    </body>
    </html>
    """
    
    return st_html(html, height=700)

# Build interactive map
if map_style == "3D Globe (Cesium)":
    st.subheader("3D Crime Visualization on Globe")
    data_points = prepare_visualization_data(filtered_historical_data, show_predictions, selected_state, selected_crime)
    render_cesium(data_points, cesium_theme_options[selected_cesium_theme])
else:
    # Set a default initial view for PyDeck maps (India centered)
    view_state = pdk.ViewState(
        latitude=22.9734,
        longitude=78.6569,
        zoom=4,
        pitch=45 if map_style != "2D Heatmap" else 0,
        bearing=0
    )

    layers = []
    if map_style == "3D Hexbin":
        if not filtered_historical_data.empty:
            hex_layer = pdk.Layer(
                "HexagonLayer",
                data=filtered_historical_data,
                get_position=["Longitude", "Latitude"],
                radius=10000,
                elevation_scale=50,
                pickable=True,
                extruded=True,
                coverage=1,
            )
            layers.append(hex_layer)
        else:
            st.warning("No historical data to display for 3D Hexbin map with current filters.")
    elif map_style == "3D Point Cloud":
        if not filtered_historical_data.empty:
            scatter_layer = pdk.Layer(
                "ScatterplotLayer",
                data=filtered_historical_data,
                get_position=["Longitude", "Latitude"],
                get_radius=500,
                get_fill_color=[255, 0, 0, 160], # Red color for historical points
                get_elevation=100,
                pickable=True,
                extruded=True,
            )
            layers.append(scatter_layer)
        else:
            st.warning("No historical data to display for 3D Point Cloud map with current filters.")
    else: # 2D Heatmap
        if not filtered_historical_data.empty:
            heatmap_layer = pdk.Layer(
                "HeatmapLayer",
                data=filtered_historical_data,
                get_position=["Longitude", "Latitude"],
                aggregation="MEAN", # or 'SUM'
                get_weight=1,
                pickable=True,
            )
            layers.append(heatmap_layer)
        else:
            st.warning("No historical data to display for 2D Heatmap with current filters.")

    if show_predictions:
        st.subheader(f"🔴 Predicted Crime Risk for {selected_datetime_tz.strftime('%Y-%m-%d %H:%M %Z')}")
        # The pred_source_data is now filtered by selected_state and selected_crime
        pred_source_data_for_pydeck = fir_data.copy()
        if selected_state != "All":
            pred_source_data_for_pydeck = pred_source_data_for_pydeck[pred_source_data_for_pydeck["State/UT"] == selected_state]
        if selected_crime != "All":
            pred_source_data_for_pydeck = pred_source_data_for_pydeck[pred_source_data_for_pydeck["Crime Head"] == selected_crime]

        pred_source_data_for_pydeck = pred_source_data_for_pydeck[["Latitude", "Longitude"]].drop_duplicates()

        if not pred_source_data_for_pydeck.empty:
            pred_locations = pred_source_data_for_pydeck.sample(min(200, len(pred_source_data_for_pydeck)), random_state=42).reset_index(drop=True)
        else:
            pred_locations = pd.DataFrame() # Ensure pred_locations is a DataFrame

        predictions = []
        if not pred_locations.empty:
            for _, row in pred_locations.iterrows():
                lat, lon = row["Latitude"], row["Longitude"]
                top_predictions = predict_crime_proba(
                    hour=selected_datetime_tz.hour,
                    dayofweek=selected_datetime_tz.weekday(),
                    month=selected_datetime_tz.month,
                    latitude=lat,
                    longitude=lon
                )
                top_crime, top_prob = top_predictions[0] if top_predictions else ("Unknown", 0.0)
                if top_prob > 0:
                    predictions.append({
                        "latitude": lat,
                        "longitude": lon,
                        "crime_type": top_crime,
                        "probability": top_prob,
                        "elevation": top_prob * 100, # Scale elevation by probability
                        "radius": 500 + (top_prob * 5) # Scale radius by probability
                    })

        pred_df = pd.DataFrame(predictions)
        
        if not pred_df.empty:
            pred_layer = pdk.Layer(
                "ScatterplotLayer",
                data=pred_df,
                get_position=["longitude", "latitude"],
                get_radius="radius",
                get_fill_color=[255, 0, 0, 200], # Bright red for predictions
                get_elevation="elevation",
                pickable=True,
                extruded=True,
            )
            layers.append(pred_layer)
        else:
            st.info("No predictions generated for PyDeck map with current settings (probability might be too low).")

    if layers: # Only render deck if there are layers to display
        # Dynamic tooltip based on whether predictions are shown and what's being hovered
        tooltip_html = """
            function (object) {
                if (object.object && object.object.crime_type) { // This is a predicted point
                    return `<b>Crime Type:</b> ${object.object.crime_type}<br><b>Probability:</b> ${object.object.probability.toFixed(2)}%`;
                } else if (object.object && object.object['Crime Head']) { // This is a historical point
                    return `<b>Crime Type:</b> ${object.object['Crime Head']}<br><b>Date:</b> ${new Date(object.object['Date']).toLocaleDateString()}`;
                }
                return null;
            }
        """

        deck = pdk.Deck(
            map_style="mapbox://styles/mapbox/dark-v10", # Using a dark map style for better contrast
            initial_view_state=view_state,
            layers=layers,
            tooltip={
                "html": tooltip_html,
                "style": {
                    "backgroundColor": "steelblue",
                    "color": "white"
                }
            }
        )
        st.pydeck_chart(deck, use_container_width=True) # Use use_container_width for better responsiveness
    else:
        st.info("No map layers to display. Adjust filters or check data availability.")


# Crime Forecast tab
with st.expander("Crime Forecast by Region or Crime Type"):
    # Ensure options are unique and not NaN, and include "All"
    state_options = ["All"] + sorted(fir_data["State/UT"].dropna().unique().tolist())
    crime_options = ["All"] + sorted(fir_data["Crime Head"].dropna().unique().tolist())

    selected_forecast_state = st.selectbox("Select State/UT for Forecast", state_options)
    selected_forecast_crime = st.selectbox("Select Crime Type (Optional)", crime_options)
    granularity = st.radio("Forecast Granularity", ["Daily", "Weekly", "Monthly"], horizontal=True)

    forecast_data = fir_data.copy()
    if selected_forecast_state != "All":
        forecast_data = forecast_data[forecast_data["State/UT"] == selected_forecast_state]
    if selected_forecast_crime != "All":
        forecast_data = forecast_data[forecast_data["Crime Head"] == selected_forecast_crime]

    time_series = prepare_time_series_data(forecast_data)

    if granularity == "Weekly":
        time_series["ds"] = pd.to_datetime(time_series["ds"])
        time_series = time_series.set_index("ds").resample("W").sum().reset_index()
    elif granularity == "Monthly":
        time_series["ds"] = pd.to_datetime(time_series["ds"])
        time_series = time_series.set_index("ds").resample("M").sum().reset_index()

    if len(time_series) > 10 and not time_series.empty: # Ensure enough data points and not empty after resampling
        model_forecast = train_forecaster(time_series) # Use a different variable name to avoid conflict
        forecast = make_forecast(model_forecast, periods=30) # Make forecast for 30 periods

        today = pd.Timestamp.now().normalize()
        future_forecast = forecast[forecast["ds"] >= today].copy()

        # Combine actuals and forecast for plotting
        combined = pd.merge(
            time_series,
            forecast[['ds', 'yhat']],
            on='ds',
            how='outer'
        ).rename(columns={"y": "Actual", "yhat": "Forecast"})

        fig = px.line(combined, x="ds", y=["Actual", "Forecast"],
                      labels={"ds": "Date", "value": "Crime Count", "variable": "Type"},
                      title=f"{granularity} Actual vs Forecasted Crime Counts")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Download Forecast")
        if not future_forecast.empty:
            csv_data = future_forecast[["ds", "yhat"]].rename(columns={"ds": "Date", "yhat": "Predicted_Crimes"})
            st.download_button(
                label="Download Forecast CSV",
                data=csv_data.to_csv(index=False),
                file_name=f"{selected_forecast_state.replace(' ', '_')}_{selected_forecast_crime.replace(' ', '_')}_{granularity}_forecast.csv",
                mime="text/csv"
            )
        else:
            st.info("No future forecast data available for download with current selections.")
    else:
        st.warning("Not enough data for reliable forecasting. Try a broader state or crime type, or ensure your data covers a sufficient time period.")