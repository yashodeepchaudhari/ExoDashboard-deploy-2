import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import io

st.set_page_config(page_title="Light Curve Explorer", layout="wide")

st.title("🧠 Stellar Light Curves Visualization")
st.caption("Upload a CSV to analyze stellar transits and dips interactively")

# ================= HELPER FUNCTIONS =================
def parse_uploaded_file(uploaded_file):
    """Parses metadata from comments and reads the CSV data."""
    metadata = {}
    content = uploaded_file.getvalue().decode("utf-8")
    lines = content.splitlines()
    
    # Extract metadata from lines starting with '#'
    data_start_line = 0
    for i, line in enumerate(lines):
        if line.startswith('#'):
            parts = line.lstrip('#').split(':')
            if len(parts) == 2:
                metadata[parts[0].strip()] = parts[1].strip()
        elif line.strip() == "":
            continue
        else:
            data_start_line = i
            break
            
    # Reset pointer and read data
    uploaded_file.seek(0)
    df = pd.read_csv(uploaded_file, comment='#')
    
    # Normalize column names (e.g., 'time' -> 'Time')
    df.columns = [c.strip().capitalize() for c in df.columns]
    return df, metadata

# ================= DATA SELECTION =================
uploaded_file = st.file_uploader("Upload Light Curve CSV (KIC format)", type=["csv"])

if uploaded_file is not None:
    try:
        data, meta = parse_uploaded_file(uploaded_file)
        
        # ================= SIDEBAR INFO =================
        st.sidebar.header("🌟 Star Metadata")
        if meta:
            for key, value in meta.items():
                st.sidebar.write(f"**{key.replace('_', ' ').title()}:** {value}")
        else:
            st.sidebar.info("No metadata found in CSV comments.")

        # ================= MAIN VISUALIZATION =================
        col1, col2 = st.columns([3, 1])

        with col2:
            st.subheader("Interactive Controls")
            show_err = st.checkbox("Show Error Bars", value=False)
            
            # Identify dip range (if folded, it's usually near 0)
            is_folded = meta.get("folded", "False").lower() == "true"
            
            t_min, t_max = float(data["Time"].min()), float(data["Time"].max())
            time_range = st.slider("Select Time Range", t_min, t_max, (t_min, t_max))
            
            if is_folded:
                if st.button("🔍 Center on Transit Dip"):
                    time_range = (-0.5, 0.5)
                    st.rerun()

        with col1:
            # Filter data based on slider
            filtered_data = data[(data["Time"] >= time_range[0]) & (data["Time"] <= time_range[1])]
            
            fig = go.Figure()

            # Plot the flux data
            fig.add_trace(go.Scatter(
                x=filtered_data["Time"], 
                y=filtered_data["Flux"],
                mode='markers',
                marker=dict(size=4, color='rgba(0, 200, 255, 0.7)', line=dict(width=0)),
                name='Flux',
                error_y=dict(type='data', array=filtered_data["Flux_err"], visible=show_err) if "Flux_err" in filtered_data else None
            ))

            # Add vertical line for the dip center
            fig.add_vline(x=0 if is_folded else data["Time"].median(), 
                         line_dash="dash", line_color="red", 
                         annotation_text="Expected Transit Center")

            fig.update_layout(
                title=f"Light Curve Visualization (Points: {len(filtered_data)})",
                xaxis_title="Time (Relative or Days)",
                yaxis_title="Normalized Flux",
                template="plotly_dark",
                height=600,
                hovermode="closest"
            )
            
            st.plotly_chart(fig, use_container_width=True)

        # ================= STATISTICS =================
        st.subheader("Transit Details")
        st.info(f"The deepest point in this view is at Flux = **{filtered_data['Flux'].min():.4f}**")
        
        stat_col1, stat_col2, stat_col3 = st.columns(3)
        stat_col1.metric("Min Flux", f"{data['Flux'].min():.4f}")
        stat_col2.metric("Max Flux", f"{data['Flux'].max():.4f}")
        stat_col3.metric("Std Dev", f"{data['Flux'].std():.4f}")

    except Exception as e:
        st.error(f"Error processing file: {e}")
        st.write("Please ensure the CSV follows the format: `time, flux, flux_err` with optional '#' comments.")
else:
    st.info("👋 Please upload a CSV file to begin. You can use the 'KIC_10004519.csv' file you have.")
    
    # Show example format
    with st.expander("See expected CSV format"):
        st.code("""
# kic_id: 10004519
# period: 9.8032
# folded: True
time,flux,flux_err
-4.89,0.032,0.00006
-4.88,1.666,0.00008
...
        """)
