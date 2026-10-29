"""
Hotel Booking Behavior Segmentation - Streamlit App
-----------------------------------------------------
A user-friendly web interface around the final clustering pipeline built in
the project's notebooks (07 - preprocessing, 13 - final model, 14 - segment
naming, 15 - new-data prediction).

This app does not retrain or refit anything. It loads the exact fitted
artifacts already saved by the notebooks and applies them to new input,
mirroring the assign_segment() function from Notebook 15.

Run locally with:
    streamlit run scripts/app.py
This works from ANY working directory - all data paths below are resolved
relative to this file's own location (scripts/), not to wherever the
`streamlit run` command happens to be invoked from.
"""

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Hotel Booking Segmentation",
    page_icon="🏨",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Paths - anchored to this script's own location (scripts/../ = project root)
# so the app works no matter what directory it's launched from.
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

PREPROCESSING_PIPELINE_PATH = PROJECT_ROOT / "models" / "preprocessing_pipeline.joblib"
FINAL_MODEL_PATH = PROJECT_ROOT / "models" / "final_kmeans_model.joblib"
FEATURE_LISTS_PATH = PROJECT_ROOT / "config" / "feature_lists.json"
PREPROCESSED_FEATURE_NAMES_PATH = PROJECT_ROOT / "config" / "preprocessed_feature_names.json"
SEGMENT_MAPPING_PATH = PROJECT_ROOT / "reports" / "tables" / "14_segment_name_mapping.csv"
SEGMENT_SUMMARY_PATH = PROJECT_ROOT / "reports" / "tables" / "14_final_segment_summary.csv"
CLUSTER_PROFILE_PATH = PROJECT_ROOT / "reports" / "tables" / "13_cluster_profile_summary.csv"

SEGMENT_COLORS = {0: "#4C72B0", 1: "#DD8452", 2: "#55A868", 3: "#C44E52"}


# ---------------------------------------------------------------------------
# Cached loaders - these run once per session, not on every interaction
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model_artifacts():
    """Load the fitted preprocessing pipeline and final clustering model."""
    pipeline = joblib.load(PREPROCESSING_PIPELINE_PATH)
    model = joblib.load(FINAL_MODEL_PATH)
    return pipeline, model


@st.cache_data
def load_config_and_tables():
    """Load the feature lists, column names, and reference tables."""
    with open(FEATURE_LISTS_PATH, "r") as f:
        feature_lists = json.load(f)
    with open(PREPROCESSED_FEATURE_NAMES_PATH, "r") as f:
        preprocessed_feature_names = json.load(f)

    segment_mapping = pd.read_csv(SEGMENT_MAPPING_PATH)
    segment_summary = pd.read_csv(SEGMENT_SUMMARY_PATH)
    cluster_profile = pd.read_csv(CLUSTER_PROFILE_PATH, index_col=0)

    return feature_lists, preprocessed_feature_names, segment_mapping, segment_summary, cluster_profile


try:
    preprocessing_pipeline, final_model = load_model_artifacts()
    feature_lists, preprocessed_feature_names, segment_mapping, segment_summary, cluster_profile = (
        load_config_and_tables()
    )
    ARTIFACTS_LOADED = True
except FileNotFoundError as error:
    ARTIFACTS_LOADED = False
    MISSING_FILE_ERROR = str(error)

if ARTIFACTS_LOADED:
    CORE_NUMERIC_FEATURES = feature_lists["core_numeric_features"]
    CORE_CATEGORICAL_FEATURES = feature_lists["core_categorical_features"]


# ---------------------------------------------------------------------------
# Prediction logic (mirrors Notebook 15's assign_segment function)
# ---------------------------------------------------------------------------
def assign_segment(raw_booking: dict) -> dict:
    """Preprocess a raw booking dict and return its predicted segment."""
    booking = dict(raw_booking)

    used_default_room_match = False
    if "room_type_mismatch" not in booking:
        booking["room_type_mismatch"] = 0
        used_default_room_match = True

    required_fields = CORE_NUMERIC_FEATURES + CORE_CATEGORICAL_FEATURES
    booking_row = pd.DataFrame([booking])[required_fields]

    preprocessed_row = preprocessing_pipeline.transform(booking_row)
    preprocessed_row_df = pd.DataFrame(preprocessed_row, columns=preprocessed_feature_names)

    cluster_id = int(final_model.predict(preprocessed_row_df)[0])
    segment_name = segment_mapping.loc[segment_mapping["cluster_id"] == cluster_id, "segment_name"].values[0]

    distances_to_centroids = np.linalg.norm(
        final_model.cluster_centers_ - preprocessed_row_df.values, axis=1
    )

    return {
        "cluster_id": cluster_id,
        "segment_name": segment_name,
        "distance_to_assigned_centroid": float(distances_to_centroids[cluster_id]),
        "distance_to_all_centroids": distances_to_centroids,
        "used_default_room_type_mismatch": used_default_room_match,
    }


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("🏨 Hotel Booking Behavior Segmentation")
st.caption(
    "Assigns a new booking to one of four data-driven behavior segments, "
    "using the final K-Means model built in this project's notebooks."
)

if not ARTIFACTS_LOADED:
    st.error(
        "Could not find the saved model files. Make sure you are running this app from the "
        "project's root folder (the one containing `models/`, `config/`, and `reports/`).\n\n"
        f"Details: {MISSING_FILE_ERROR}"
    )
    st.stop()

tab_predict, tab_dashboard = st.tabs(["🔮 Predict a Booking", "📊 Segment Dashboard"])

# ---------------------------------------------------------------------------
# TAB 1: Predict a Booking
# ---------------------------------------------------------------------------
with tab_predict:
    st.subheader("Enter Booking Details")
    st.write(
        "Fill in the fields below the way they would appear on a real booking, then click "
        "**Predict Segment**. Every field mirrors a column in the original dataset."
    )

    with st.form("booking_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Property & Channel**")
            hotel = st.selectbox("Hotel", ["City Hotel", "Resort Hotel"])
            meal = st.selectbox("Meal plan", ["BB", "HB", "FB", "SC", "Undefined"])
            market_segment = st.selectbox(
                "Market segment",
                ["Online TA", "Offline TA/TO", "Direct", "Corporate", "Groups", "Complementary", "Aviation", "Undefined"],
            )
            distribution_channel = st.selectbox(
                "Distribution channel", ["TA/TO", "Direct", "Corporate", "GDS", "Undefined"]
            )
            deposit_type = st.selectbox("Deposit type", ["No Deposit", "Non Refund", "Refundable"])
            customer_type = st.selectbox(
                "Customer type", ["Transient", "Transient-Party", "Contract", "Group"]
            )

        with col2:
            st.markdown("**Stay Details**")
            lead_time = st.number_input("Lead time (days before arrival)", min_value=0, max_value=800, value=45)
            stays_in_weekend_nights = st.number_input("Weekend nights", min_value=0, max_value=20, value=2)
            stays_in_week_nights = st.number_input("Week nights", min_value=0, max_value=40, value=3)
            adults = st.number_input("Adults", min_value=1, max_value=10, value=2)
            children = st.number_input("Children", min_value=0, max_value=10, value=0)
            babies = st.number_input("Babies", min_value=0, max_value=10, value=0)

        with col3:
            st.markdown("**Value & Engagement**")
            adr = st.number_input("Average daily rate (ADR)", min_value=0.0, max_value=1000.0, value=110.0, step=1.0)
            required_car_parking_spaces = st.number_input("Parking spaces requested", min_value=0, max_value=8, value=0)
            total_of_special_requests = st.number_input("Special requests", min_value=0, max_value=5, value=1)
            booking_changes = st.number_input("Booking changes so far", min_value=0, max_value=20, value=0)
            is_repeated_guest = st.checkbox("Repeat guest?")
            previous_cancellations = st.number_input("Previous cancellations", min_value=0, max_value=30, value=0)

        st.markdown("**Room Assignment** *(usually unknown until check-in)*")
        room_known = st.checkbox(
            "I know whether the assigned room differs from the reserved room type",
            value=False,
        )
        room_type_mismatch = None
        if room_known:
            room_type_mismatch = st.radio(
                "Does the assigned room type differ from the reserved room type?",
                ["No", "Yes"],
                horizontal=True,
            )
            room_type_mismatch = 1 if room_type_mismatch == "Yes" else 0

        submitted = st.form_submit_button("Predict Segment", use_container_width=True)

    if submitted:
        raw_booking = {
            "lead_time": lead_time,
            "stays_in_weekend_nights": stays_in_weekend_nights,
            "stays_in_week_nights": stays_in_week_nights,
            "adults": adults,
            "children": children,
            "babies": babies,
            "adr": adr,
            "required_car_parking_spaces": required_car_parking_spaces,
            "total_of_special_requests": total_of_special_requests,
            "booking_changes": booking_changes,
            "is_repeated_guest": int(is_repeated_guest),
            "previous_cancellations": previous_cancellations,
            "hotel": hotel,
            "meal": meal,
            "market_segment": market_segment,
            "distribution_channel": distribution_channel,
            "deposit_type": deposit_type,
            "customer_type": customer_type,
        }
        if room_type_mismatch is not None:
            raw_booking["room_type_mismatch"] = room_type_mismatch

        result = assign_segment(raw_booking)
        cluster_id = result["cluster_id"]
        segment_row = segment_mapping[segment_mapping["cluster_id"] == cluster_id].iloc[0]
        profile_row = cluster_profile.loc[cluster_id]

        st.divider()
        st.markdown(
            f"### Predicted Segment: "
            f"<span style='color:{SEGMENT_COLORS[cluster_id]}'>{result['segment_name']}</span>",
            unsafe_allow_html=True,
        )

        metric_col1, metric_col2, metric_col3 = st.columns(3)
        metric_col1.metric("Cluster ID", cluster_id)
        metric_col2.metric("Distance to Segment Center", f"{result['distance_to_assigned_centroid']:.2f}")
        metric_col3.metric(
            "Segment Size (of all bookings)",
            f"{profile_row['pct_of_total']:.1f}%",
        )

        st.info(f"**Why this segment:** {segment_row['business_interpretation']}")
        st.caption(f"Key characteristics: {segment_row['key_characteristics']}")

        if result["used_default_room_type_mismatch"]:
            st.warning(
                "Room assignment wasn't provided, so it was assumed the guest will get the room type "
                "they reserved. This is a common situation for a genuinely new booking (see Notebook 15 "
                "for why), and it's a documented assumption, not a certainty."
            )

        # Comparison chart: this booking vs. its assigned segment's average
        st.markdown("#### How this booking compares to its segment's average")
        compare_metrics = {
            "Lead time (days)": (lead_time, profile_row["lead_time"]),
            "ADR": (adr, profile_row["adr"]),
            "Stay length (nights)": (
                stays_in_weekend_nights + stays_in_week_nights,
                profile_row["total_stay_nights"],
            ),
            "Special requests": (total_of_special_requests, profile_row["total_of_special_requests"]),
        }

        fig, axes = plt.subplots(1, len(compare_metrics), figsize=(12, 3.5))
        for ax, (label, (this_value, segment_avg)) in zip(axes, compare_metrics.items()):
            ax.bar(["This booking", "Segment avg"], [this_value, segment_avg],
                   color=[SEGMENT_COLORS[cluster_id], "lightgray"])
            ax.set_title(label, fontsize=10)
            for i, v in enumerate([this_value, segment_avg]):
                ax.text(i, v, f"{v:.1f}", ha="center", va="bottom", fontsize=8)
        plt.tight_layout()
        st.pyplot(fig)

        with st.expander("See distance to every segment (lower = closer fit)"):
            distance_df = pd.DataFrame({
                "Segment": segment_mapping.sort_values("cluster_id")["segment_name"].values,
                "Distance": result["distance_to_all_centroids"],
            }).sort_values("Distance")
            st.dataframe(distance_df, hide_index=True, use_container_width=True)

# ---------------------------------------------------------------------------
# TAB 2: Segment Dashboard
# ---------------------------------------------------------------------------
with tab_dashboard:
    st.subheader("The Four Booking Behavior Segments")
    st.write(
        "These segments were discovered with unsupervised clustering (K-Means, K=4) across "
        "notebooks 08-13 of the project, then named from their observed characteristics in "
        "Notebook 14. Every number below comes directly from that analysis."
    )

    display_summary = segment_summary.copy()
    st.dataframe(display_summary, hide_index=True, use_container_width=True)

    st.markdown("#### Segment Comparison")
    kpi_col1, kpi_col2 = st.columns(2)

    with kpi_col1:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(segment_summary["Segment"], segment_summary["Avg ADR"],
               color=[SEGMENT_COLORS[i] for i in range(4)])
        ax.set_title("Average ADR by Segment")
        ax.tick_params(axis="x", rotation=25, labelsize=8)
        st.pyplot(fig)

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(segment_summary["Segment"], segment_summary["Repeat Guest %"],
               color=[SEGMENT_COLORS[i] for i in range(4)])
        ax.set_title("Repeat Guest % by Segment")
        ax.tick_params(axis="x", rotation=25, labelsize=8)
        st.pyplot(fig)

    with kpi_col2:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(segment_summary["Segment"], segment_summary["Avg Stay (nights)"],
               color=[SEGMENT_COLORS[i] for i in range(4)])
        ax.set_title("Average Stay Length by Segment")
        ax.tick_params(axis="x", rotation=25, labelsize=8)
        st.pyplot(fig)

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(segment_summary["Segment"], segment_summary["Observed Cancellation %"],
               color=[SEGMENT_COLORS[i] for i in range(4)])
        ax.set_title("Observed Cancellation Rate by Segment")
        ax.tick_params(axis="x", rotation=25, labelsize=8)
        st.pyplot(fig)

    st.markdown("#### Share of Bookings by Segment")
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(
        segment_summary["% of Bookings"],
        labels=segment_summary["Segment"],
        autopct="%1.1f%%",
        colors=[SEGMENT_COLORS[i] for i in range(4)],
        startangle=90,
    )
    st.pyplot(fig)

    st.markdown("#### Full Characteristics and Naming Rationale")
    for _, row in segment_mapping.iterrows():
        with st.expander(f"Cluster {row['cluster_id']} — {row['segment_name']}"):
            st.write(f"**Key characteristics:** {row['key_characteristics']}")
            st.write(f"**Business interpretation:** {row['business_interpretation']}")

    st.caption(
        "⚠️ Reminder: the observed cancellation rate is a post-clustering descriptive statistic, "
        "not something the model was trained to predict, and differences across segments are "
        "reported as observed associations - not claims of cause and effect."
    )
