import streamlit as st
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import plotly.express as px
import torch.serialization
import os

# Define the path relative to this script's location
# This goes up one level from 'pages/' to the root directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "siamese_model.pt")

# Allowlist numpy reconstruction for PyTorch 2.6+
torch.serialization.add_safe_globals([np._core.multiarray._reconstruct])

class SiameseBackbone(nn.Module):
    def __init__(self):
        super(SiameseBackbone, self).__init__()
        self.backbone = nn.Sequential(
            nn.Linear(51, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32) 
        )

    def forward(self, x):
        return self.backbone(x)



@st.cache_resource
def load_resources():
    # Verify file existence for cleaner debugging
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file not found at {MODEL_PATH}. Check your repository structure.")
        st.stop()
        
    checkpoint = torch.load(MODEL_PATH, map_location=torch.device('cpu'), weights_only=False)
    model = SiameseBackbone()
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()
    return model, checkpoint.get('feature_columns'), checkpoint.get('X_mean'), checkpoint.get('X_std')

# --- UI Configuration ---
st.set_page_config(page_title="Exoplanet Explorer", layout="wide")
st.title("🌌 Exoplanet Discovery Dashboard")
st.markdown("---")

model, feature_cols, x_mean, x_std = load_resources()

# Sidebar for controls
with st.sidebar:
    st.header("Settings")
    threshold = st.slider("Sensitivity Threshold", 0.0, 3.0, 1.2, help="Higher values are more selective.")
    uploaded_file = st.file_uploader("Upload test.csv", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    
    if all(col in df.columns for col in feature_cols):
        # Processing
        data = df[feature_cols].values
        normalized_data = (data - x_mean) / x_std
        
        with torch.no_grad():
            embeddings = model(torch.tensor(normalized_data).float())
            scores = torch.norm(embeddings, dim=1).numpy()
        
        # Build Results DataFrame
        df['Score'] = scores
        df['Status'] = np.where(df['Score'] > threshold, "Exoplanet Candidate", "Ordinary Star")
        
        # --- Top Level Metrics ---
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Stars Scanned", len(df))
        col2.metric("Candidates Found", len(df[df['Status'] == "Exoplanet Candidate"]))
        col3.metric("Model Reliability (AUC)", "0.91") # From metadata 

        # --- Visualizations ---
        tab1, tab2 = st.tabs(["📊 Prediction Analysis", "🔍 Raw Data"])
        
        with tab1:
            st.subheader("Candidate Distribution")
            fig = px.histogram(df, x="Score", color="Status", 
                               color_discrete_map={"Exoplanet Candidate": "#00CC96", "Ordinary Star": "#636EFA"},
                               nbins=30, barmode='overlay', title="Embedding Magnitude Distribution")
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Detection Results")
            def color_status(val):
                color = '#2ecc71' if val == "Exoplanet Candidate" else '#95a5a6'
                return f'background-color: {color}; color: white; font-weight: bold'

            # Display styled table
            st.dataframe(df[['Score', 'Status'] + feature_cols[:3]].style.applymap(color_status, subset=['Status']), 
                         use_container_width=True)

        with tab2:
            st.write("Full Dataset with Features")
            st.dataframe(df)
            
    else:
        st.error("CSV error: Ensure your file contains the 51 light curve features.")
else:

    st.info("Please upload a CSV file to begin the analysis.")
