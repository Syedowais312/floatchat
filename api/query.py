from flask import Flask, request, jsonify
import psycopg2
import numpy as np
from sentence_transformers import SentenceTransformer
from geopy.geocoders import Nominatim
from geopy.distance import distance
import google.generativeai as genai
import re
import json
import math
import os
from dotenv import load_dotenv
from flask_cors import CORS


load_dotenv()
HOST_NAME=os.getenv("HOST_NAME", default=None)
PASSWORD=os.getenv("PASSWORD", default=None)
USER=os.getenv("USER_NAME", default=None)
DB_NAME=os.getenv("DATABASE_NAME", default=None)
DB_CONFIG = {
    "host": HOST_NAME,
    "database": DB_NAME,
    "user": USER,
    "password": PASSWORD
}

TOP_K = 1
RADIUS_METERS = 50_000
model = SentenceTransformer("all-MiniLM-L6-v2")
dotenv_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path)

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
gemini_model = genai.GenerativeModel("models/gemini-1.5-flash")

print("DB_CONFIG:", DB_CONFIG)
print("HOST_NAME:", HOST_NAME)
print("PASSWORD:", PASSWORD)
print("USER:", USER)
print("DB_NAME:", DB_NAME)
print("GOOGLE_API_KEY:", os.getenv("GOOGLE_API_KEY"))
def get_embedding(text):
    return model.encode(text).tolist()

def llm_explain(user_query, profile_data):
    summary = f"Profile at lat={profile_data['lat']}, lon={profile_data['lon']} on {profile_data['time']}. Depth levels:\n"
    for lvl in profile_data["depth_levels"]:
        summary += f"- Pressure: {lvl['pres']} dbar, Temp: {lvl['temp']}°C, Salinity: {lvl['salinity']}\n"

    prompt = f"""
Hey! You asked: "{user_query}"

Here's a friendly explanation of the ocean data I found:

{summary}

In simple terms, describe what the ocean conditions are like here, relate it to the user's question, and keep it short, clear, and conversational – like you're chatting with the user. You can add emojis if it feels natural.
"""

    response = gemini_model.generate_content(prompt)
    return response.text.strip()


def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def geocode_place(place_name):
    geolocator = Nominatim(user_agent="floatchat")
    location = geolocator.geocode(place_name)
    if location:
        return location.latitude, location.longitude
    return None, None


def extract_lat_lon(user_query):
    lat_match = re.search(r"lat\s*=\s*([-+]?\d*\.?\d+)", user_query, re.IGNORECASE)
    lon_match = re.search(r"long\s*=\s*([-+]?\d*\.?\d+)", user_query, re.IGNORECASE)
    if lat_match and lon_match:
        return float(lat_match.group(1)), float(lon_match.group(1))
    return None, None


def query_profiles(user_query):
    lat, lon = extract_lat_lon(user_query)
    if lat is None or lon is None:
        lat, lon = geocode_place(user_query)

    query_emb = get_embedding(user_query)

    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    cur.execute("SELECT id, latitude, longitude, juld, embedding FROM profiles")
    profiles = cur.fetchall()

    sims = []
    for profile in profiles:
        profile_id, plat, plon, juld, emb = profile
        if isinstance(emb, str):
            emb = json.loads(emb)

        if lat is not None and lon is not None:
            dist = distance((lat, lon), (plat, plon)).meters
            if dist > RADIUS_METERS:
                continue
        else:
            dist = 0

        sim = cosine_similarity(query_emb, emb)
        distance_score = 1 / (1 + dist)
        combined_score = 0.7 * sim + 0.3 * distance_score
        sims.append((combined_score, profile_id, plat, plon, juld))

    top_profiles = sorted(sims, key=lambda x: x[0], reverse=True)[:TOP_K]

    results = []
    for _, profile_id, plat, plon, juld in top_profiles:
        cur.execute("""
            SELECT pres, temp, psal
            FROM profile_levels
            WHERE profile_id = %s
            ORDER BY n_levels ASC
            LIMIT 15
        """, (profile_id,))
        levels = cur.fetchall()

        depth_levels = []
        for l in levels:
            pres, temp, sal = l
            if pres is None or temp is None or sal is None:
                continue
            if math.isnan(pres) or math.isnan(temp) or math.isnan(sal):
                continue
            depth_levels.append({"pres": pres, "temp": temp, "salinity": sal})

        profile_data = {
            "profile_id": profile_id,
            "lat": plat,
            "lon": plon,
            "time": juld.strftime("%Y-%m-%d %H:%M:%S"),
            "depth_levels": depth_levels
        }
        profile_data["query_explain"] = llm_explain(user_query, profile_data)
        results.append(profile_data)

    conn.close()
    return results


app = Flask(__name__)

@app.route("/query", methods=["POST"])
def query():
    data = request.get_json()
    user_query = data.get("query")
    if not user_query:
        return jsonify({"error": "Missing 'query' field"}), 400

    try:
        results = query_profiles(user_query)
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "FloatChat API is running 🚀"})


if __name__ == "__main__":
    CORS(app)
    app.run(debug=True)

