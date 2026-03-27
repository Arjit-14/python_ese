import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# --- INITIALIZATION ---
@st.cache_resource
def download_nltk_resources():
    try:
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('vader_lexicon')
    except:
        st.warning("NLTK download failed. Check internet connection.")

download_nltk_resources()

def ensure_nltk_resources():
    resources = ['punkt', 'stopwords', 'vader_lexicon', 'punkt_tab']
    for res in resources:
        try:
            nltk.download(res, quiet=True)
        except:
            pass

ensure_nltk_resources()

def process_feedback(text_series):
    try:
        # Stopwords and Stemmer setup
        stop_words = set(stopwords.words('english'))
        ps = PorterStemmer()
        all_stems = []
        
        for text in text_series:
            if pd.isna(text): continue
            # Tokenization (Unit 4 requirement) [cite: 14]
            tokens = word_tokenize(str(text).lower())
            # Clean and Stem [cite: 14]
            stems = [ps.stem(w) for w in tokens if w.isalpha() and w not in stop_words]
            all_stems.extend(stems)
        
        if not all_stems:
            return pd.Series()
            
        return pd.Series(all_stems).value_counts().head(10)
    except Exception as e:
        # This will print the actual error to your Streamlit logs
        print(f"NLP Error: {e}")
        return pd.Series()

# --- 1. DATA LOADING ---
def load_data():
    df = pd.read_csv("dataset.csv", engine='python') 
    df['Rating'] = pd.to_numeric(df['Rating'], errors='coerce') 
    return df

df = load_data()

# --- 2. DASHBOARD SETUP & FILTERS ---
st.set_page_config(page_title="GATEWAYS 2025 Dashboard", layout="wide") 

st.sidebar.header("Filter Insights")
selected_event = st.sidebar.multiselect("Select Event", options=df["Event Name"].unique(), default=df["Event Name"].unique())
selected_state = st.sidebar.multiselect("Select State", options=df["State"].unique(), default=df["State"].unique())

# .copy() fixes the SettingWithCopyWarning
filtered_df = df[(df["Event Name"].isin(selected_event)) & (df["State"].isin(selected_state))].copy()

st.title("GATEWAYS 2025 - National Level Fest Insights")
st.markdown("#### Core Organizing Team Analytics Dashboard")

kpi1, kpi2, kpi3 = st.columns(3)
with kpi1:
    st.metric("Total Participants", len(filtered_df))
with kpi2:
    st.metric("Total Revenue", f"₹{filtered_df['Amount Paid'].sum()}") 
with kpi3:
    st.metric("Average Rating", round(filtered_df['Rating'].mean(), 2) if not filtered_df.empty else 0) 

st.divider()

# --- 3. PART 1: PARTICIPATION TRENDS & MAP ---
st.header("1. Analysis of Participation Trends")
# Added 'College-wise Analysis' tab
tab1, tab_college, tab2 = st.tabs(["Event Distribution", "College-wise Analysis", "State-wise India Map"])

with tab1:
    st.subheader("Participation by Event")
    event_counts = filtered_df['Event Name'].value_counts()
    fig1, ax1 = plt.subplots()
    event_counts.plot(kind='bar', color='teal', ax=ax1) 
    ax1.set_ylabel("Count")
    st.pyplot(fig1)

with tab_college:
    st.subheader("Top 10 Participating Colleges")
    # Calculation for college-wise trends 
    college_counts = filtered_df['College'].value_counts().head(10)
    
    if not college_counts.empty:
        fig_coll, ax_coll = plt.subplots()
        # Horizontal bar chart for better readability of college names
        college_counts.plot(kind='barh', color='orchid', ax=ax_coll)
        ax_coll.set_xlabel("Number of Participants")
        ax_coll.invert_yaxis()  # Highest on top
        st.pyplot(fig_coll)
    else:
        st.write("No college data available.")

with tab2:
    st.subheader("State-wise Participant Density")
    state_data = filtered_df.groupby('State').size().reset_index(name='Counts')
    
    if not state_data.empty:
        # Create a horizontal bar chart which acts as a 'ranked' map analysis
        fig_map, ax_map = plt.subplots(figsize=(10, 6))
        
        # Sort data to show the 'Trend' (L4 Analysis requirement)
        state_data = state_data.sort_values(by='Counts', ascending=True)
        
        colors = plt.cm.Reds(np.linspace(0.3, 0.9, len(state_data)))
        ax_map.barh(state_data['State'], state_data['Counts'], color=colors)
        
        ax_map.set_title("Participant Distribution Across India")
        ax_map.set_xlabel("Number of Students")
        
        # Add labels to the bars for "Actionable Insights"
        for i, v in enumerate(state_data['Counts']):
            ax_map.text(v + 0.5, i, str(v), color='black', fontweight='bold')
            
        st.pyplot(fig_map)
        
        # Mandatory ESE Insight
        top_state = state_data.iloc[-1]['State']
        st.success(f"📍 **Geospatial Insight:** {top_state} is the primary hub for GATEWAYS 2025.")
    else:
        st.write("Select a state from the sidebar to view the distribution.")

# --- 4. PART 2: FEEDBACK & SENTIMENT ANALYSIS ---
st.header("2. Participant Feedback & Sentiment Analysis")

tab_rating, tab_sentiment, tab_words = st.tabs(["⭐ Rating Overview", "🎭 Sentiment Analysis", "🔤 Keyword Trends"])

with tab_rating:
    st.subheader("Overall Rating Distribution")
    rating_counts = filtered_df['Rating'].value_counts().sort_index()
    
    if not rating_counts.empty:
        fig_pie, ax_pie = plt.subplots(figsize=(6, 6))
        
        # Custom function to display both Count and Percentage
        def make_autopct(values):
            def my_autopct(pct):
                total = sum(values)
                val = int(round(pct*total/100.0))
                return f'{val}\n({pct:.1f}%)'
            return my_autopct

        # Modern Donut Chart
        wedges, texts, autotexts = ax_pie.pie(
            rating_counts, 
            labels=rating_counts.index, 
            autopct=make_autopct(rating_counts),
            startangle=140, 
            colors=plt.cm.Pastel1.colors, 
            wedgeprops={'edgecolor': 'white', 'linewidth': 2},
            textprops={'fontsize': 10}
        )
        
        # Adding the center circle for the "Donut" look
        centre_circle = plt.Circle((0,0), 0.70, fc='white')
        fig_pie.gca().add_artist(centre_circle)
        
        # Styling the labels inside the pie
        plt.setp(autotexts, size=9, weight="bold")
        
        ax_pie.set_title(f"Total Ratings: {len(filtered_df)}", pad=20)
        st.pyplot(fig_pie)
    else:
        st.write("No rating data available.")

with tab_sentiment:
    st.subheader("Feedback Sentiment Classification")
    sia = SentimentIntensityAnalyzer()
    
    def get_sentiment(text):
        score = sia.polarity_scores(str(text))['compound']
        if score >= 0.05: return 'Positive'
        elif score <= -0.05: return 'Negative'
        else: return 'Neutral'

    if not filtered_df.empty:
        filtered_df['Sentiment'] = filtered_df['Feedback on Fest'].apply(get_sentiment)
        sentiment_counts = filtered_df['Sentiment'].value_counts()
        
        fig_sent, ax_sent = plt.subplots()
        # Ensure colors match the sentiment found
        color_map = {'Positive': '#2ecc71', 'Neutral': '#f1c40f', 'Negative': '#e74c3c'}
        current_colors = [color_map[s] for s in sentiment_counts.index]
        
        sentiment_counts.plot(kind='bar', color=current_colors, ax=ax_sent)
        ax_sent.set_ylabel("Number of Feedbacks")
        st.pyplot(fig_sent)
        
        pos_perc = (sentiment_counts.get('Positive', 0) / len(filtered_df)) * 100
        st.divider()
        st.subheader("Organizing Team Summary")
        if pos_perc > 70:
            st.success(f"🏆 **Excellent Event!** {pos_perc:.1f}% of feedback is positive.")
        elif pos_perc > 40:
            st.warning(f"📈 **Good Effort.** {pos_perc:.1f}% positive feedback.")
        else:
            st.error(f"⚠️ **Action Required.** Only {pos_perc:.1f}% positive feedback.")

with tab_words:
    st.subheader("Top Frequency Keywords (Stemmed)")
    top_stems = process_feedback(filtered_df['Feedback on Fest'])
    
    if not top_stems.empty:
        fig_words, ax_words = plt.subplots()
        top_stems.plot(kind='bar', color='#3498db', ax=ax_words)
        plt.xticks(rotation=45)
        st.pyplot(fig_words)
    else:
        st.write("No text data to analyze.")

st.divider()
st.dataframe(filtered_df)

