import streamlit as st
from PIL import Image
import pandas as pd
import requests
import plotly.express as px
import boto3, os
from pathlib import Path

# ============================
# Config
# ============================
API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000").rstrip("/")
PREDICT_ENDPOINT = f"{API_URL}/predict"
S3_BUCKET = os.getenv("S3_BUCKET", "house-price-data-ab")
REGION = os.getenv("AWS_REGION", "ap-southeast-1")


s3 = boto3.client("s3", region_name=REGION)

def load_from_s3(key, local_path):
    """Download from S3 if not already cached locally."""
    local_path = Path(local_path)
    if not local_path.exists():
        os.makedirs(local_path.parent, exist_ok=True)
        try:
            print(f"DEBUG: Attempting to download from Bucket: '{S3_BUCKET}' with Key: '{key}'")
            s3.download_file(S3_BUCKET, key, str(local_path))
            print(f"✅ Successfully downloaded {key}")
        except Exception as e:
            print(f"❌ ERROR downloading {key}: {e}")
            raise e
    return str(local_path)

# Paths (ensure available locally by fetching from S3 if missing)
HOLDOUT_ENGINEERED_PATH = load_from_s3(
    "processed/feature_engineered_holdout.csv",
    "data/processed/feature_engineered_holdout.csv"
)
HOLDOUT_META_PATH = load_from_s3(
    "processed/clean_holdout.csv",
    "data/processed/clean_holdout.csv"
)

# ============================
# Data loading
# ============================
@st.cache_data
def load_data():
    fe = pd.read_csv(HOLDOUT_ENGINEERED_PATH)
    meta = pd.read_csv(HOLDOUT_META_PATH, parse_dates=["date"])[["date", "city_full"]]

    if len(fe) != len(meta):
        st.warning("⚠️ Engineered and meta holdout lengths differ. Aligning by index.")
        min_len = min(len(fe), len(meta))
        fe = fe.iloc[:min_len].copy()
        meta = meta.iloc[:min_len].copy()

    disp = pd.DataFrame(index=fe.index)
    disp["date"] = meta["date"]
    disp["region"] = meta["city_full"]
    disp["year"] = disp["date"].dt.year
    disp["month"] = disp["date"].dt.month
    disp["actual_price"] = fe["price"]

    return fe, disp

fe_df, disp_df = load_data()

# ============================
# UI
# ============================

# ============================
# Sidebar UI (Filters)
# ============================

with st.sidebar:
    st.image("https://www.gstatic.com/lamda/images/gemini_sparkle_v002_d473530c2731a4f9173d1.svg", width=50)
    st.title("Filters")
    st.markdown("Select parameters to analyze model performance on holdout data.")
    
    years = sorted(disp_df["year"].unique())
    months = list(range(1, 13))
    regions = ["All"] + sorted(disp_df["region"].dropna().unique())

    selected_year = st.selectbox("Year", years)
    selected_month = st.selectbox("Month", options=months)
    selected_region = st.selectbox("Region", regions)
    
    st.divider()
    st.caption("Author: Arif Budiman")
    st.caption("Model: XGBoost Regressor")

# ============================
# Main Dashboard Area
# ============================
st.set_page_config(layout="wide")
try:
    image = Image.open("assets/sold-banner.jpg")
    st.image(image, use_column_width=True)
except FileNotFoundError:
    st.warning("Banner image not found. Please check the file path.")

st.title("Housing Price Prediction")
st.subheader(f"Analyzing: {selected_region} | Period: {selected_year}-{selected_month:02d}")

if st.button("Generate Predictions 🚀", use_container_width=True):
    try:
        # --- A. FILTER DATA TAHUNAN ---
        # Kita ambil 1 tahun penuh agar grafik tren tidak menumpuk (landai)
        year_mask = (disp_df["year"] == selected_year)
        if selected_region != "All":
            year_mask &= (disp_df["region"] == selected_region)
        
        yearly_data = disp_df[year_mask].copy()
        idx_yearly = yearly_data.index

        if len(idx_yearly) == 0:
            st.warning("No data found for the selected year and region.")
        else:
            with st.spinner("Fetching predictions from API..."):
                # --- B. API CALL (BATCH) ---
                payload = fe_df.loc[idx_yearly].to_dict(orient="records")
                resp = requests.post(PREDICT_ENDPOINT, json=payload, timeout=60)
                resp.raise_for_status()
                preds = resp.json().get("predictions", [])

                # --- C. POST-PROCESSING ---
                yearly_data["prediction"] = pd.Series(preds, index=yearly_data.index).astype(float)
                yearly_data["abs_error"] = (yearly_data["prediction"] - yearly_data["actual_price"]).abs()
                yearly_data["error_pct"] = (yearly_data["abs_error"] / yearly_data["actual_price"]) * 100

                # Data khusus untuk bulan terpilih (Metrik & Tabel)
                results_view = yearly_data[yearly_data["month"] == selected_month].copy()

            # --- D. TAMPILAN METRIK (Bulan Terpilih) ---
            if not results_view.empty:
                mae = results_view["abs_error"].mean()
                mape = results_view["error_pct"].mean()
                
                c1, c2, c3 = st.columns(3)
                c1.metric("Avg. Actual Price", f"${results_view['actual_price'].mean():,.2f}")
                c2.metric("Mean Absolute Error (MAE)", f"${mae:,.2f}", 
                          delta=f"{mape:.2f}% MAPE", delta_color="inverse")
                c3.metric("Data Points", len(results_view))
            else:
                st.info(f"No holdout data available for {selected_month:02d}/{selected_year}.")

            st.divider()

            # --- E. TABS VISUALISASI ---
            tab_chart, tab_table, tab_dist = st.tabs(["📈 Trend Analysis", "📋 Raw Data", "🎯 Error Distribution"])

            with tab_chart:
                st.write(f"### Yearly Performance Trend — {selected_year}")
                
                # Agregasi bulanan agar garis tren landai dan rapi
                monthly_avg = yearly_data.groupby("month")[["actual_price", "prediction"]].mean().reset_index()

                fig = px.line(
                    monthly_avg, x="month", y=["actual_price", "prediction"],
                    markers=True, template="plotly_white", labels={"value": "Price", "month": "Month"}
                )

                # Styling: Biru Tua (Actual) vs Biru Muda (Pred)
                fig.data[0].update(line_color="#1f77b4", name="Actual Price", line=dict(width=3))
                fig.data[1].update(line_color="#a5d8ff", name="Prediction", line=dict(width=3))

                # Highlight Bulan Terpilih (Shading Merah)
                fig.add_vrect(
                    x0=selected_month - 0.5, x1=selected_month + 0.5,
                    fillcolor="red", opacity=0.1, layer="below", line_width=0
                )

                fig.update_layout(
                    hovermode="x unified",
                    xaxis=dict(showgrid=False, tickmode='linear', dtick=1),
                    yaxis=dict(showgrid=True, gridcolor='whitesmoke'),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                st.plotly_chart(fig, use_container_width=True)

            with tab_table:
                st.write("### Prediction Details")
                st.dataframe(
                    results_view.style.background_gradient(subset=['error_pct'], cmap='YlOrRd'), 
                    use_container_width=True
                )
                
                # Tombol Download CSV
                csv = results_view.to_csv(index=False).encode('utf-8')
                st.download_button("Download as CSV", data=csv, file_name="housing_preds.csv", mime="text/csv")

            with tab_dist:
                st.write("### Accuracy Distribution")
                if not results_view.empty:
                    fig_hist = px.histogram(results_view, x="error_pct", nbins=15, 
                                            title="Error Percentage Frequency",
                                            color_discrete_sequence=['#9467bd'])
                    st.plotly_chart(fig_hist, use_container_width=True)

    except Exception as e:
        st.error(f"System Error: {e}")
        st.exception(e)

else:
    # Tampilan awal sebelum tombol diklik
    st.info("👈 Please select filters and click **Generate Predictions** to begin the analysis.")