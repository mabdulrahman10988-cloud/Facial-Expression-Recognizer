import streamlit as st
import requests
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Facial Expression Recognition",
    layout="wide"
)

st.markdown("""
    <style>
        .main { background-color: #ffffff; }
        .stApp { background-color: #f8f9fa; }
        .header-box {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
        }
        .header-box h1 { margin: 0; font-size: 32px; font-weight: 600; }
        .emotion-card {
            background: white;
            border-radius: 12px;
            padding: 25px;
            border: 1px solid #e0e0e0;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        }
        .emotion-large {
            font-size: 48px;
            font-weight: 700;
            color: #667eea;
            margin-bottom: 10px;
        }
        .confidence-text {
            font-size: 24px;
            font-weight: 600;
            color: #333;
            margin-bottom: 5px;
        }
        .confidence-value {
            font-size: 32px;
            font-weight: 700;
            color: #667eea;
        }
        .section-title {
            font-size: 18px;
            font-weight: 600;
            color: #333;
            margin-bottom: 15px;
            margin-top: 20px;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown(
    '<div class="header-box"><h1>Facial Expression Recognition</h1></div>',
    unsafe_allow_html=True
)

col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown('<div class="section-title">Image Input</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, use_container_width=True)

        if st.button("Analyze", use_container_width=True, type="primary"):
            with st.spinner("Processing..."):
                try:
                    api_url = "http://localhost:8000"
                    response = requests.post(
                        f"{api_url}/predict",
                        files={"file": ("image.jpg", uploaded_file.getvalue())}
                    )

                    if response.status_code == 200:
                        result = response.json()
                        if result["success"]:
                            st.session_state.prediction_result = result
                            st.rerun()
                        else:
                            st.error(f"Error: {result['error']}")
                    else:
                        st.error("API error")

                except requests.exceptions.ConnectionError:
                    st.error("Cannot connect to backend. Start FastAPI server first.")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
    else:
        st.info("Upload an image to begin analysis")

with col2:
    st.markdown('<div class="section-title">Analysis Result</div>', unsafe_allow_html=True)

    if "prediction_result" in st.session_state:
        result = st.session_state.prediction_result
        emotion = result["predicted_emotion"]
        confidence = result["confidence"]

        st.markdown(
            f"""
            <div class="emotion-card">
                <div class="emotion-large">{emotion}</div>
                <div class="confidence-text">Confidence</div>
                <div class="confidence-value">{confidence*100:.1f}%</div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown('<div class="section-title">Score Distribution</div>', unsafe_allow_html=True)
        emotions_data = result["all_emotions"]

        fig, ax = plt.subplots(figsize=(10, 5))
        emotions = list(emotions_data.keys())
        scores = list(emotions_data.values())
        colors = ['#667eea' if e == emotion else '#d0d0d0' for e in emotions]

        bars = ax.barh(emotions, scores, color=colors, height=0.6)
        ax.set_xlim(0, 1)
        ax.set_xlabel("Confidence", fontsize=11, color="#666")
        ax.set_ylabel("")
        ax.tick_params(axis='y', labelsize=10)
        ax.tick_params(axis='x', labelsize=9)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#e0e0e0')
        ax.spines['bottom'].set_color('#e0e0e0')

        for i, (bar, score) in enumerate(zip(bars, scores)):
            ax.text(score + 0.02, bar.get_y() + bar.get_height()/2,
                   f'{score*100:.1f}%', va='center', fontsize=10, fontweight='600')

        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)

        st.markdown('<div class="section-title">Detailed Breakdown</div>', unsafe_allow_html=True)
        table_data = {
            "Emotion": list(emotions_data.keys()),
            "Score": [f"{score*100:.2f}%" for score in emotions_data.values()]
        }
        st.dataframe(table_data, use_container_width=True, hide_index=True)

    else:
        st.info("Results will appear here after analysis")