import streamlit as st
import torch
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification
import re
import time
from datetime import datetime
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Configure Streamlit page
st.set_page_config(
    page_title="Essay Scoring System",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .score-display {
        font-size: 4rem;
        font-weight: bold;
        text-align: center;
        padding: 2rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .score-1 { background: linear-gradient(45deg, #ff6b6b, #ee5a52); color: white; }
    .score-2 { background: linear-gradient(45deg, #ffa726, #ff9800); color: white; }
    .score-3 { background: linear-gradient(45deg, #ffee58, #fdd835); color: black; }
    .score-4 { background: linear-gradient(45deg, #9ccc65, #8bc34a); color: white; }
    .score-5 { background: linear-gradient(45deg, #66bb6a, #4caf50); color: white; }
    .score-6 { background: linear-gradient(45deg, #42a5f5, #2196f3); color: white; }
    
    .stTextArea textarea {
        font-family: 'Arial', sans-serif;
        font-size: 16px;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model_and_tokenizer():
    """
    Load the trained BERT model and tokenizer
    """
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load tokenizer
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
        # Load model
        model = BertForSequenceClassification.from_pretrained(
            'bert-base-uncased',
            num_labels=1,
            output_attentions=False,
            output_hidden_states=False,
        )
        
        # Load the trained weights
        model.load_state_dict(torch.load('bert_essay_scoring_1.pt', map_location=device))
        model.to(device)
        model.eval()
        
        return model, tokenizer, device
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.info("Make sure 'bert_essay_scoring_1.pt' is in the same directory as this app.")
        return None, None, None

def clean_essay(text):
    """
    Clean the essay text (same preprocessing as training)
    """
    text = str(text).lower()
    
    # Replace inverted/fancy quotes with normal or remove
    text = text.replace("â€œ", "").replace("â€", "")
    text = text.replace('"', "").replace("'", "")
    
    # Remove multiple spaces, tabs, and newlines
    text = re.sub(r"\s+", " ", text)
    
    # Trim leading/trailing spaces
    text = text.strip()
    
    return text

def predict_score(text, model, tokenizer, device, min_score=1, max_score=6, max_len=256):
    """
    Predicts the score of a single essay.
    """
    if model is None or tokenizer is None:
        return None
    
    # Clean the text
    cleaned_text = clean_essay(text)
    
    # Put the model in evaluation mode
    model.eval()

    # Tokenize the input text
    encoded_text = tokenizer.encode_plus(
        cleaned_text,
        add_special_tokens=True,
        max_length=max_len,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

    # Move tensors to the correct device
    input_ids = encoded_text['input_ids'].to(device)
    attention_mask = encoded_text['attention_mask'].to(device)

    # Make the prediction
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits.squeeze().cpu().numpy()

    # Denormalize and round the score
    predicted_score = round(logits * (max_score - min_score) + min_score)
    
    # Ensure score is within valid range
    #predicted_score = max(min_score, min(max_score, predicted_score))

    return predicted_score, logits

def analyze_essay(text):
    """
    Analyze essay characteristics
    """
    words = text.split()
    sentences = re.split(r'[.!?]+', text)
    
    analysis = {
        'word_count': len(words),
        'sentence_count': len([s for s in sentences if s.strip()]),
        'avg_words_per_sentence': len(words) / len([s for s in sentences if s.strip()]) if sentences else 0,
        'unique_words': len(set(words))
    }
    
    return analysis

def get_score_interpretation(score):
    """
    Provide interpretation for each score
    """
    interpretations = {
        1: "Needs Significant Improvement - The essay shows minimal understanding and organization.",
        2: "Below Average - The essay demonstrates some understanding but lacks clarity and structure.",
        3: "Average - The essay shows adequate understanding with some organizational issues.",
        4: "Good - The essay demonstrates good understanding and organization with minor issues.",
        5: "Very Good - The essay shows strong understanding and clear organization.",
        6: "Excellent - The essay demonstrates exceptional understanding, organization, and expression."
    }
    return interpretations.get(score, "Score out of range")

def main():
    # Header
    st.markdown('<h1 class="main-header">📝 Automated Essay Scoring System</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Load model
    with st.spinner("Loading BERT model... This may take a moment."):
        model, tokenizer, device = load_model_and_tokenizer()
    
    if model is None:
        st.error("Failed to load the model. Please check if 'bert_essay_scoring_1.pt' exists in the current directory.")
        st.stop()
    
    
    # Sidebar with information
    with st.sidebar:
        st.header("📋 Scoring Rubric")
        st.markdown("""
        - **6**: Excellent
        - **5**: Very Good
        - **4**: Good
        - **3**: Average
        - **2**: Below Average
        - **1**: Needs Improvement
        """)
        
        if st.button("📚 Sample Essay"):
            sample_essay = """The author suggests that studying Venus is a worthy pursuit despite the dangers it presents because if something happens to Earth, if we study Venus we will know some stuff to do. Like all the needs to stay alive, keep breathing, and is it even possible. Because if a war happens with the USA and they blow up America, we will need somewhere to go. So in order for us to know all of these things, we will have to study about other planets to survive at. Venus also looks the safest out of all the other planets. It's shaped like Earth, close to Earth, so it's in the same atmosphere so it gets the same oxygen. The shape and size is pretty similar so there should be enough room for people. It's close to the sun so we don't know if it's safe. It could be safe if the sun rotates away from it so people can actually live there. Since Venus is so close to the sun, we will have to send someone there to see if it's liveable. If we ever decided to move to Venus, we will have to bring so much stuff and it would take a while. So I don't think everybody will make it before the US is destroyed."""
            st.session_state['sample_essay'] = sample_essay

    # Main content area
    col1, col2 = st.columns([4, 1])
    
    with col1:
        st.header("✍️ Enter Your Essay")
        
        # Text input
        if 'sample_essay' in st.session_state:
            default_text = st.session_state['sample_essay']
            del st.session_state['sample_essay']  # Clear after use
        else:
            default_text = ""
            
        essay_text = st.text_area(
            "Paste or type your essay here:",
            value=default_text,
            height=400,
            placeholder="Enter your essay text here...",
            help="Enter the essay you want to score. The system works best with essays between 150-500 words."
        )
        
        # Predict button
        col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
        with col_btn2:
            predict_button = st.button("🎯 Score Essay", type="primary", use_container_width=True)
    
    with col2:
        st.header("📈 Essay Analysis")
        
        if essay_text.strip():
            analysis = analyze_essay(essay_text)
            
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Word Count", analysis['word_count'])
                st.metric("Unique Words", analysis['unique_words'])
            
            st.metric("Avg Words/Sentence", f"{analysis['avg_words_per_sentence']:.1f}")
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Prediction results
    if predict_button and essay_text.strip():
        if len(essay_text.split()) < 20:
            st.warning("⚠️ Essay seems too short. Please enter a longer essay for better accuracy.")
        else:
            with st.spinner("🤔 Analyzing essay..."):
                start_time = time.time()
                score, confidence = predict_score(essay_text, model, tokenizer, device)
                processing_time = time.time() - start_time
            
            if score is not None:
                st.markdown("---")
                st.header("🎯 Prediction Results")
                
                # Display score with color coding
                score_html = f'<div class="score-display score-{score}">Score: {score}/6</div>'
                st.markdown(score_html, unsafe_allow_html=True)
                
                # Score interpretation
                interpretation = get_score_interpretation(score)
                st.info(f"**Interpretation:** {interpretation}")
                
                # Confidence visualization
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number+delta",
                    value = score,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Essay Score"},
                    gauge = {
                        'axis': {'range': [None, 6]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 2], 'color': "lightgray"},
                            {'range': [2, 4], 'color': "gray"},
                            {'range': [4, 6], 'color': "lightgreen"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 6
                        }
                    }
                ))
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
                
            else:
                st.error("Failed to predict score. Please try again.")
    
    elif predict_button:
        st.warning("Please enter an essay to score.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>🤖 Powered by BERT | Built with Streamlit | Essay Scoring System</p>
        <p><small>Note: This is an automated scoring system. Results should be used as guidance alongside human evaluation.</small></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()