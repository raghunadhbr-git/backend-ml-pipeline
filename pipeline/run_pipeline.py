import os
import pandas as pd
from sqlalchemy import create_engine, text
from datetime import datetime
import smtplib
from email.message import EmailMessage

# =========================
# START
# =========================
start_time = datetime.utcnow()
print("🚀 ML PIPELINE STARTED")
print("=" * 60)

try:
    # =========================
    # ENV VARIABLES
    # =========================
    EVENTS_DB_URL = os.getenv("EVENTS_DB_URL")
    PRODUCT_DB_URL = os.getenv("PRODUCT_DB_URL")
    RECO_DB_URL = os.getenv("RECO_DB_URL")

    MAIL_USERNAME = os.getenv("MAIL_USERNAME")
    MAIL_PASSWORD = os.getenv("MAIL_PASSWORD")
    MAIL_CC = os.getenv("MAIL_CC")

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
    print("💾 Saving recommendations...")

    top_k.to_sql("recommendations", reco_engine, if_exists="replace", index=False)

    print("✅ Saved successfully")

    # =========================
    # EMAIL NOTIFICATION
    # =========================
    print("📧 Sending email...")

    if MAIL_USERNAME and MAIL_PASSWORD:
        try:
            msg = EmailMessage()
            msg["Subject"] = "ML Pipeline Success 🚀"
            msg["From"] = MAIL_USERNAME
            msg["To"] = MAIL_USERNAME

            if MAIL_CC:
                msg["Cc"] = MAIL_CC

            msg.set_content(f"""
ML Recommendation Pipeline Completed ✅

Events processed: {len(events_df)}
Products processed: {len(products_df)}
Recommendations generated: {len(top_k)}

Status: SUCCESS 🚀
""")

            recipients = [MAIL_USERNAME]
            if MAIL_CC:
                recipients.append(MAIL_CC)

            with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
                server.login(MAIL_USERNAME, MAIL_PASSWORD)
                server.send_message(msg, to_addrs=recipients)

            print("✅ Email sent successfully")

        except Exception as e:
            print("❌ Email failed:", str(e))
    else:
        print("⚠️ Email skipped (missing credentials)")

    # =========================
    # END
    # =========================
    end_time = datetime.utcnow()
    print("=" * 60)
    print("🏁 PIPELINE COMPLETED")
    print(f"⏱ Duration: {end_time - start_time}")

# =========================
# ERROR HANDLING
# =========================
except Exception as e:
    print("❌ PIPELINE FAILED")
    print("Error:", str(e))
    raise
