import requests
import csv
import os
import numpy as np
import pandas as pd
import math
from datetime import datetime

class TravelAIEngine:
    def __init__(self, serp_key, aviation_key):
        self.serp_key = serp_key
        self.av_key = aviation_key
        self.dataset_file = "india_comprehensive_dataset_2026.csv"

         # ---------------- LOAD DISTANCE DATA ---------------- #
        self.dist_df = pd.read_csv("indian-cities-dataset.csv")
        self.dist_df.columns = [c.lower().strip() for c in self.dist_df.columns]

        required = {"origin", "destination", "distance"}
        if not required.issubset(self.dist_df.columns):
            raise ValueError(
                f"Distance dataset must contain {required}. "
                f"Found: {list(self.dist_df.columns)}"
            )

        # Normalize city names
        self.dist_df["origin"] = self.dist_df["origin"].astype(str).str.lower().str.strip()
        self.dist_df["destination"] = self.dist_df["destination"].astype(str).str.lower().str.strip()

        # Create fast lookup dictionary
        self.distance_lookup = {
            (row["origin"], row["destination"]): float(row["distance"])
            for _, row in self.dist_df.iterrows()
        }
        
         # ---------------- TRAVEL MODES ---------------- #
        self.TRAVEL_MODE_PROFILE = {
            "Bus": {
                "base": 1200,
                "per_person": True,
                "comfort": 0.9
            },
            "Train": {
                "base": 2200,
                "per_person": True,
                "comfort": 1.0
            },
            "Flight": {
                "base": 6500,
                "per_person": True,
                "comfort": 1.2
            },
            "Car": {
                "base": 14,     # ₹ per km
                "per_km": True,
                "comfort": 1.05
            }
        }

    def fetch_live_weather(self, city):
        try:
            geo_url = "https://geocoding-api.open-meteo.com/v1/search"
            geo_params = {"name": city, "count": 1}
            geo_res = requests.get(geo_url, params=geo_params, timeout=10).json()

            if "results" not in geo_res or not geo_res["results"]:
                return "25°C (Clear)"

            lat = geo_res["results"][0]["latitude"]
            lon = geo_res["results"][0]["longitude"]

            weather_url = "https://api.open-meteo.com/v1/forecast"
            w_params = {
                "latitude": lat,
                "longitude": lon,
                "current_weather": True
            }

            w_res = requests.get(weather_url, params=w_params, timeout=10).json()

            if "current_weather" in w_res:
                temp = w_res["current_weather"]["temperature"]
                code = w_res["current_weather"]["weathercode"]
                desc = {0: "Clear", 1: "Cloudy", 61: "Rain", 95: "Storm"}.get(code, "Clear")
                return f"{temp}°C ({desc})"

        except Exception:
            pass

        return "25°C (Clear)"

    def fetch_hotel_price(self, city):
        try:
            url = "https://serpapi.com/search"
            params = {
                "engine": "google_hotels",
                "q": f"Hotels in {city}, India",
                "currency": "INR",
                "api_key": self.serp_key
            }
            res = requests.get(url, params=params, timeout=10).json()

            if "properties" in res and res["properties"]:
                price = res["properties"][0].get("rate_per_night", {}).get("extracted_lowest")
                return float(price) if price else 5800.0
        except Exception:
            pass

        return 5800.0
    # ---------------- DISTANCE ---------------- #
    def estimate_distance_km(self, source, destination):
        s = source.lower().strip()
        d = destination.lower().strip()

        return self.distance_lookup.get(
            (s, d),
            self.distance_lookup.get((d, s), 800)  # India-safe fallback
        )
     # ---------------- TRANSPORT COST ---------------- #
    def get_transport_cost(self, mode, source, destination, people):
        """
        Real-time ready.
        Currently uses fallback cost (stable & ML-safe).
        """
        profile = self.TRAVEL_MODE_PROFILE.get(mode)
        if not profile:
           return 0.0

        distance = self.estimate_distance_km(source, destination)

        if profile.get("per_km"):
           cost = distance * profile["base"]
        else:
           cost = profile["base"]

        if profile.get("per_person"):
           cost *= people

        return round(cost * profile["comfort"], 2)
    
     # ---------------- MAIN PLANNER ---------------- #
    def plan_trip(self, source, destination, days=3, people=1, travel_mode="Flight"):
        if source.lower() == destination.lower():
            return None

        src_weather = self.fetch_live_weather(source)
        dest_weather = self.fetch_live_weather(destination)
        hotel_rate = self.fetch_hotel_price(destination)
        transport_cost = self.get_transport_cost(
            travel_mode, source, destination, people
        )

        stay_cost = hotel_rate * max(1, days - 1)
        misc_cost = 3200.0 * days * people

        report = {
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Source": source,
            "Destination": destination,
            "Travel_Mode": travel_mode,
            "Transport_Cost": round(transport_cost, 2),
            "Source_Weather": src_weather,
            "Dest_Weather": dest_weather,
            "Days": days,
            "People": people,
            "Grand_Total": round(transport_cost + stay_cost + misc_cost, 2)
        }

        self._save_to_csv(report)
        return report

    def _save_to_csv(self, data):
        file_exists = os.path.isfile(self.dataset_file)
        with open(self.dataset_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=data.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(data)

if __name__ == "__main__":
    bot = TravelAIEngine(
        serp_key=os.getenv("718b1b6b6d16e7565670055ff2d7b6608aaa40c072e4ab0a21937a49311f6171"),
        aviation_key=os.getenv("caaeb0963625efa4e8d9196cf2eda3b3")
    )

    # Extract unique cities from distance dataset
    cities = sorted(
        set(bot.dist_df["origin"]).union(set(bot.dist_df["destination"]))
    )

    modes = list(bot.TRAVEL_MODE_PROFILE.keys())

    for _ in range(50):
        src, dest = np.random.choice(cities, 2, replace=False)
        res = bot.plan_trip(
            src.title(),
            dest.title(),
            days=np.random.randint(2, 8),
            people=np.random.randint(1, 5),
            travel_mode=np.random.choice(modes)
        )

        if res:
            print(f"✅ {res['Source']} → {res['Destination']} ({res['Travel_Mode']}) ₹{res['Grand_Total']}")

    print("📂 Dataset generated successfully")
