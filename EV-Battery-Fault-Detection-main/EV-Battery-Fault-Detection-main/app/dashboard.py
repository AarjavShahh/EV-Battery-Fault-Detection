import streamlit as st
import pandas as pd
from pathlib import Path

# =========================================================
# CONFIG
# =========================================================

st.set_page_config(
    page_title="EV Battery Fault Detection",
    page_icon="🔋",
    layout="wide"
)

ROOT = Path(__file__).resolve().parent.parent

DASHBOARD_FILE = (
    ROOT / "data" / "dashboard_data.csv"
)

CYCLE_FILE = (
    ROOT / "results" / "cycle_anomaly_summary.csv"
)

# =========================================================
# LOAD SMALL DATASETS ONLY
# =========================================================

@st.cache_data
def load_dashboard_data():

    return pd.read_csv(DASHBOARD_FILE)


@st.cache_data
def load_cycle_summary():

    return pd.read_csv(CYCLE_FILE)


try:

    battery = load_dashboard_data()
    anomalies = load_cycle_summary()

except Exception as e:

    st.error("Could not load dashboard data.")
    st.exception(e)
    st.stop()

# =========================================================
# TITLE
# =========================================================

st.title("🔋 EV Battery Fault Detection")

st.caption(
    "LSTM Autoencoder Based Battery Anomaly Monitoring"
)

st.divider()

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.header("Battery Monitoring")

selected_cycle = st.sidebar.slider(
    "Select Battery Cycle",
    int(battery["cycle"].min()),
    int(battery["cycle"].max()),
    100
)

# =========================================================
# SELECTED CYCLE
# =========================================================

selected = battery[
    battery["cycle"] == selected_cycle
].iloc[0]

# =========================================================
# METRICS
# =========================================================

st.subheader(
    f"🔋 Battery Cycle {selected_cycle}"
)

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        "Average Voltage",
        f"{selected['voltage']:.3f} V"
    )

with col2:

    st.metric(
        "Average Current",
        f"{selected['current']:.3f} A"
    )

with col3:

    st.metric(
        "Average Temperature",
        f"{selected['temperature']:.2f} °C"
    )

with col4:

    st.metric(
        "Sensor Readings",
        f"{int(selected['readings']):,}"
    )

# =========================================================
# ANOMALY INFORMATION
# =========================================================

st.divider()

st.subheader("🚨 LSTM Anomaly Detection")

cycle_result = anomalies[
    anomalies["cycle"] == selected_cycle
]

if len(cycle_result) > 0:

    result = cycle_result.iloc[0]

    anomaly_percent = float(
        result["anomaly_percentage"]
    )

    average_error = float(
        result["average_error"]
    )

    maximum_error = float(
        result["maximum_error"]
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Anomaly Percentage",
            f"{anomaly_percent:.2f}%"
        )

    with col2:

        st.metric(
            "Average Error",
            f"{average_error:.6f}"
        )

    with col3:

        st.metric(
            "Maximum Error",
            f"{maximum_error:.6f}"
        )

    if anomaly_percent > 5:

        st.error(
            "🚨 ANOMALY DETECTED"
        )

    else:

        st.success(
            "✅ NORMAL"
        )

else:

    st.info(
        "No anomaly result available."
    )

# =========================================================
# BATTERY SENSOR TRENDS
# =========================================================

st.divider()

st.subheader("📈 Battery Sensor Trends")

trend = battery.sort_values("cycle")

tab1, tab2, tab3 = st.tabs(
    [
        "Voltage",
        "Current",
        "Temperature"
    ]
)

with tab1:

    chart = trend.set_index("cycle")[
        ["voltage"]
    ]

    st.line_chart(
        chart,
        width="stretch"
    )

with tab2:

    chart = trend.set_index("cycle")[
        ["current"]
    ]

    st.line_chart(
        chart,
        width="stretch"
    )

with tab3:

    chart = trend.set_index("cycle")[
        ["temperature"]
    ]

    st.line_chart(
        chart,
        width="stretch"
    )

# =========================================================
# ANOMALY TREND
# =========================================================

st.divider()

st.subheader(
    "📊 Anomaly Percentage Across Battery Life"
)

anomaly_chart = anomalies.sort_values(
    "cycle"
).set_index("cycle")[
    ["anomaly_percentage"]
]

st.line_chart(
    anomaly_chart,
    width="stretch"
)

# =========================================================
# TOP ANOMALOUS CYCLES
# =========================================================

st.subheader(
    "⚠️ Most Anomalous Battery Cycles"
)

top = anomalies.sort_values(
    "anomaly_percentage",
    ascending=False
).head(10)

st.dataframe(
    top,
    width="stretch",
    hide_index=True
)

# =========================================================
# INFORMATION
# =========================================================

st.divider()

st.subheader("🧠 How the System Works")

st.markdown(
    """
    **1. Battery Data**

    Voltage, current and temperature measurements are collected.

    **2. LSTM Autoencoder**

    The model learns normal battery behavior.

    **3. Reconstruction Error**

    The model reconstructs the sensor sequence.

    **4. Anomaly Detection**

    High reconstruction error indicates abnormal behavior.

    **5. Dashboard**

    The detected abnormal behavior is displayed for monitoring.
    """
)

st.warning(
    "ANOMALY indicates behavior different from the learned "
    "normal pattern. It is not a confirmed diagnosis of a "
    "specific physical fault."
)

st.caption(
    "EV Battery Fault Detection Using LSTM Autoencoder"
)