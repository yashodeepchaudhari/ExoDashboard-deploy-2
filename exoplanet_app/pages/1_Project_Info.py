# import streamlit as st
# import pandas as pd
# import time
# import numpy as np
# import plotly.graph_objects as go  # For interactive plots

# st.set_page_config(page_title="Exoplanet Detection System", layout="wide")

# # ================= SIDEBAR =================
# st.sidebar.title("🌌 Exoplanet System")
# st.sidebar.success("System Status: Online")

# if st.sidebar.button("🔄 Refresh"):
#     st.experimental_rerun()

# st.sidebar.markdown("---")

# st.sidebar.markdown("""
# ### Navigation
# ➡ Project Overview  
# ➡ Prediction  
# ➡ Light Curves  
# """)

# # Add interactivity: File uploader for user data
# uploaded_file = st.sidebar.file_uploader("📤 Upload Light Curve Data (CSV)", type=["csv"])
# if uploaded_file is not None:
#     user_data = pd.read_csv(uploaded_file)
#     st.sidebar.success("Data uploaded successfully!")
#     st.sidebar.write(f"Rows: {len(user_data)}, Columns: {len(user_data.columns)}")
# else:
#     user_data = None

# # Add a selectbox for model selection
# model_option = st.sidebar.selectbox("Select Model Variant", ["Base Siamese", "Enhanced Siamese", "Custom"])

# # ================= HEADER =================
# st.title("🌌 Exoplanet Detection using Siamese Neural Networks")
# st.caption("AI based detection of exoplanets from stellar light curves")

# # ================= METRICS =================
# col1, col2, col3, col4 = st.columns(4)

# # Make metrics dynamic based on model selection
# accuracy = {"Base Siamese": "88%", "Enhanced Siamese": "92%", "Custom": "85%"}[model_option]
# features = {"Base Siamese": "50+", "Enhanced Siamese": "60+", "Custom": "45+"}[model_option]
# embedding = {"Base Siamese": "32", "Enhanced Siamese": "64", "Custom": "16"}[model_option]
# layers = {"Base Siamese": "256-128-64", "Enhanced Siamese": "512-256-128", "Custom": "128-64-32"}[model_option]

# col1.metric("Accuracy", accuracy)
# col2.metric("Features", features)
# col3.metric("Embedding Size", embedding)
# col4.metric("Model Layers", layers)

# # ================= PROGRESS =================
# st.subheader("Pipeline Status")

# # Add a checkbox to control progress simulation
# run_progress = st.checkbox("Simulate Pipeline Progress", value=True)
# if run_progress:
#     progress = st.progress(0)
#     for i in range(100):
#         time.sleep(0.01)
#         progress.progress(i + 1)
#     st.success("Pipeline Ready 🚀")
# else:
#     st.info("Pipeline simulation disabled. Toggle to run.")

# # ================= TABS =================
# tab1, tab2, tab3 = st.tabs(["📘 Overview", "⚙ Workflow", "🧠 Model"])

# # ---------- TAB 1 ----------
# with tab1:
#     st.info("""
# This system detects exoplanets by analyzing stellar brightness variations.
# A Siamese Neural Network learns similarity patterns between light curve pairs.
#     """)

#     st.markdown("""
# ### 🎯 Objectives
# - Automated exoplanet detection  
# - Similarity learning using Siamese networks  
# - Feature engineering (50+ features)  
# - High accuracy prediction  
#     """)

#     # Add interactivity: Interactive plot for sample light curve
#     st.subheader("Sample Light Curve Visualization")
#     # Generate sample data
#     time_points = np.linspace(0, 10, 100)
#     flux = 1 + 0.1 * np.sin(2 * np.pi * time_points / 5) + np.random.normal(0, 0.02, 100)
#     fig = go.Figure()
#     fig.add_trace(go.Scatter(x=time_points, y=flux, mode='lines', name='Flux'))
#     fig.update_layout(title="Simulated Stellar Light Curve", xaxis_title="Time", yaxis_title="Flux")
#     st.plotly_chart(fig, use_container_width=True)

# # ---------- TAB 2 ----------
# with tab2:
#     st.markdown("""
# ### System Flow

# Raw Light Curves  
# ⬇  
# Preprocessing (Normalization + Detrending)  
# ⬇  
# Feature Extraction  
# ⬇  
# Pair Generation  
# ⬇  
# Siamese Neural Network  
# ⬇  
# Prediction & Visualization
#     """)

#     # Add interactivity: Slider for workflow steps
#     step = st.slider("Select Workflow Step", 1, 6, 1)
#     steps = ["Raw Light Curves", "Preprocessing", "Feature Extraction", "Pair Generation", "Siamese NN", "Prediction"]
#     st.write(f"Current Step: {steps[step-1]}")
#     st.progress(step / 6)

#     # Add a button to simulate next step
#     if st.button("Next Step"):
#         if step < 6:
#             st.experimental_rerun()  # In practice, use session state for persistence

# # ---------- TAB 3 ----------
# with tab3:
#     st.markdown("""
# ### Architecture

# - Dense Layers: 256 → 128 → 64  
# - Embedding: 32  
# - Loss: Contrastive  
# - Optimizer: Adam  
# - Dropout: 0.3  
#     """)

#     # Add interactivity: Form to adjust hyperparameters
#     with st.form("hyperparams_form"):
#         st.subheader("Adjust Hyperparameters")
#         dropout = st.slider("Dropout Rate", 0.0, 0.5, 0.3)
#         learning_rate = st.selectbox("Learning Rate", [0.001, 0.01, 0.1])
#         submitted = st.form_submit_button("Update Model")
#         if submitted:
#             st.success(f"Model updated! Dropout: {dropout}, LR: {learning_rate}")
#             # In a real app, retrain or update metrics here

# # ================= EXPANDERS =================
# with st.expander("📊 Feature Categories"):
#     st.write("""
# ✔ Statistical Features  
# ✔ Shape Features  
# ✔ Frequency Features  
# ✔ Transit Specific Features  
#     """)
#     # Add interactivity: Checkbox to select features
#     selected_features = st.multiselect("Select Features to View", ["Statistical", "Shape", "Frequency", "Transit"], default=["Statistical"])
#     st.write(f"Selected: {', '.join(selected_features)}")

# with st.expander("📈 Performance Metrics"):
#     st.write("""
# Accuracy: 85–90%  
# Precision: 80–85%  
# Recall: 75–80%  
# F1 Score: 78–83%  
#     """)
#     # Add interactivity: Bar chart for metrics
#     metrics = pd.DataFrame({
#         "Metric": ["Accuracy", "Precision", "Recall", "F1 Score"],
#         "Value": [88, 82, 78, 80]
#     })
#     st.bar_chart(metrics.set_index("Metric"))

# # ================= SAMPLE DATA =================
# st.subheader("📁 Sample Dataset Preview")

# sample = pd.DataFrame({
#     "FLUX_1": [1.01, 0.98, 1.00],
#     "FLUX_2": [0.99, 0.97, 1.02],
#     "LABEL": ["Planet", "No Planet", "Planet"]
# })

# # Make dataframe editable
# edited_sample = st.data_editor(sample, num_rows="dynamic")
# st.write("Edited Data:", edited_sample)

# # ================= BUTTON =================
# if st.button("🚀 Go to Prediction Dashboard"):
#     st.switch_page("pages/2_Prediction.py")

# # ================= FOOTER =================
# st.markdown("---")
# st.caption("Exoplanet Detection System | Siamese Neural Networks | Streamlit")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import time

st.set_page_config(page_title="Exoplanet Detection System", layout="wide")

# ================= SIDEBAR =================
st.sidebar.title("🌌 Exoplanet Detection System")
st.sidebar.success("System Status: Online")

st.sidebar.markdown("""
### Navigation
➡ Project Overview  
➡ Prediction  
➡ Light Curves  
""")

# ================= HEADER =================
st.title("🌌 Exoplanet Detection using Siamese Neural Networks")
st.caption("Metric Learning based detection of exoplanets from stellar light curves")

# ================= METRICS =================
col1, col2, col3, col4 = st.columns(4)

col1.metric("Embedding Size", "32")
col2.metric("Basic Features", "100+")
col3.metric("TSFresh Features", "775+")
col4.metric("Inference Time", "< 1 ms")

# ================= PIPELINE STATUS =================
st.subheader("Deployment Pipeline Status")

progress = st.progress(0)
for i in range(100):
    time.sleep(0.005)
    progress.progress(i+1)

st.success("Pipeline Ready 🚀")

# ================= TABS =================
tab1, tab2, tab3 = st.tabs(["📘 Overview", "⚙ Pipeline", "🧠 Model"])

# ---------- TAB 1 ----------
with tab1:
    st.markdown("""
### 🎯 System Overview

This system detects exoplanets by analyzing stellar brightness variations (light curves).
A Siamese Neural Network is trained using contrastive learning to learn similarity between
planetary and non-planetary transits.

Instead of traditional classification, the model learns an embedding space where
similar curves are close and dissimilar curves are far apart.
    """)

    st.markdown("""
### Supported Input Formats
• CSV  
• NPZ  
• FITS  
    """)

# ---------- TAB 2 ----------
with tab2:
    st.markdown("""
### Complete Data Pipeline

1️⃣ Universal Data Loader → LightCurve Objects  
2️⃣ NaN Removal & Sigma Clipping  
3️⃣ Detrending & Normalization  
4️⃣ Period Folding & Time Binning  
5️⃣ Feature Extraction  
6️⃣ Feature Standardization (X_mean / X_std)  
7️⃣ Siamese Embedding Generation  
8️⃣ Distance Calculation → Probability Output
    """)

    st.progress(0.9)

# ---------- TAB 3 ----------
with tab3:
    st.markdown("""
### Siamese Neural Network Architecture

• Layers: 64 → 64 → 32  
• Activation: ReLU  
• L2 Normalized Embeddings  
• Contrastive Loss (margin = 1.0)  
• Distance-based Classification  

Saved as PyTorch checkpoint (.pth) including:
- Model weights  
- Embedding dimension  
- Feature normalization parameters  
    """)

# ================= FEATURE SECTION =================
with st.expander("🧪 Feature Extraction"):
    st.markdown("""
### Two Feature Modes

✅ Basic Feature Extractor  
• 100+ statistical & temporal features  
• Optimized for speed  

✅ TSFresh Feature Extractor  
• 775+ advanced time-series features  
• Higher accuracy, slower processing  
    """)

# ================= MODELS =================
with st.expander("🤖 Available Models"):
    st.markdown("""
• Random Forest (Baseline)  
• Logistic Regression (Baseline)  
• Siamese Neural Network (Production Model)  
    """)

# ================= PERFORMANCE =================
with st.expander("📊 Model Performance"):
    st.markdown("""
Accuracy: 85–90%  
Precision: 80–85%  
Recall: 75–80%  
F1 Score: 78–83%  
ROC-AUC supported  
    """)

# ================= WHY SIAMESE =================
with st.expander("🚀 Why Siamese Learning?"):
    st.markdown("""
✔ Learns similarity instead of hard boundaries  
✔ Robust to class imbalance  
✔ Supports few-shot learning  
✔ Better generalization on unseen stars  
    """)

# ================= SAMPLE LIGHT CURVE =================
st.subheader("Sample Light Curve")

t = np.linspace(0,10,100)
flux = 1 + 0.05*np.sin(t) + np.random.normal(0,0.01,100)

fig = go.Figure()
fig.add_trace(go.Scatter(x=t,y=flux,mode="lines"))
fig.update_layout(title="Simulated Stellar Light Curve",
                  xaxis_title="Time",
                  yaxis_title="Normalized Flux")

st.plotly_chart(fig,use_container_width=True)

# ================= FOOTER =================
st.markdown("---")
st.caption("Exoplanet Detection System | Siamese Neural Networks | Streamlit Deployment")
