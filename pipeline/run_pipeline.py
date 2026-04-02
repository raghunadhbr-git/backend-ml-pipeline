import os
import pandas as pd
from sqlalchemy import create_engine, text
from datetime import datetime

# =========================
# START
# =========================
start_time = datetime.utcnow()
print("🚀 ML PIPELINE STARTED")
print("=" * 60)

# =========================
# ENV VARIABLES
# =========================
EVENTS_DB_URL = os.getenv("EVENTS_DB_URL")
PRODUCT_DB_URL = os.getenv("PRODUCT_DB_URL")
RECO_DB_URL = os.getenv("RECO_DB_URL")

assert EVENTS_DB_URL, "EVENTS_DB_URL missing"
assert PRODUCT_DB_URL, "PRODUCT_DB_URL missing"
assert RECO_DB_URL, "RECO_DB_URL missing"

print("✅ ENV LOADED")

# =========================
# DB CONNECTIONS
# =========================
events_engine = create_engine(EVENTS_DB_URL)
product_engine = create_engine(PRODUCT_DB_URL)
reco_engine = create_engine(RECO_DB_URL)

print("✅ DB CONNECTED")

# =========================
# FETCH DATA
# =========================
print("📥 Fetching events...")

events_df = pd.read_sql("""
SELECT user_id, event_type, object_id
FROM user_events
WHERE user_id IS NOT NULL
AND object_type = 'product'
""", events_engine)

print(f"✅ Events: {len(events_df)}")

print("📥 Fetching products...")

products_df = pd.read_sql("""
SELECT id, name
FROM products
""", product_engine)

print(f"✅ Products: {len(products_df)}")

# =========================
# FEATURE ENGINEERING
# =========================
weights = {
    "view_product": 1,
    "add_to_cart": 3,
    "checkout": 5
}

events_df["score"] = events_df["event_type"].map(weights).fillna(0)

features = (
    events_df.groupby(["user_id", "object_id"])["score"]
    .sum()
    .reset_index()
)

features.rename(columns={"object_id": "product_id"}, inplace=True)

# =========================
# TOP K
# =========================
features["rank"] = features.groupby("user_id")["score"].rank(ascending=False)

top_k = features[features["rank"] <= 10]

print(f"✅ Top K rows: {len(top_k)}")

# =========================
# SAVE
# =========================
print("💾 Saving...")

top_k.to_sql("recommendations", reco_engine, if_exists="replace", index=False)

print("✅ Saved successfully")

# =========================
# END
# =========================
end_time = datetime.utcnow()
print("=" * 60)
print("🏁 PIPELINE COMPLETED")
print(f"⏱ Duration: {end_time - start_time}")
