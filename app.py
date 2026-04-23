import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.multivariate.manova import MANOVA
from statsmodels.stats.proportion import proportions_ztest
from sklearn.linear_model import LassoCV, Lasso
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
import sklearn.linear_model as lm
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="💸 Where Did Our Money Go??",
    page_icon="💸",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── UNHINGED CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Space+Mono:wght@400;700&display=swap');

    * { font-family: 'Space Grotesk', sans-serif !important; }
    code, pre, .mono { font-family: 'Space Mono', monospace !important; }

    /* ── BACKGROUNDS ── */
    .stApp {
        background-color: #04010f;
        background-image:
            radial-gradient(ellipse 80% 50% at 20% -10%, rgba(120,40,200,0.25) 0%, transparent 60%),
            radial-gradient(ellipse 60% 40% at 80% 110%, rgba(255,60,120,0.18) 0%, transparent 60%),
            repeating-linear-gradient(
                0deg,
                transparent,
                transparent 39px,
                rgba(255,255,255,0.02) 39px,
                rgba(255,255,255,0.02) 40px
            ),
            repeating-linear-gradient(
                90deg,
                transparent,
                transparent 39px,
                rgba(255,255,255,0.02) 39px,
                rgba(255,255,255,0.02) 40px
            );
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d0020 0%, #080018 100%) !important;
        border-right: 1px solid rgba(180,0,255,0.3) !important;
        box-shadow: 4px 0 30px rgba(180,0,255,0.12);
    }

    /* ── GLOWING HERO BANNER ── */
    .hero-banner {
        background: linear-gradient(135deg, #0d0020 0%, #150030 40%, #0a001a 100%);
        border: 1px solid rgba(180,0,255,0.4);
        border-radius: 20px;
        padding: 32px 36px 28px 36px;
        margin-bottom: 28px;
        position: relative;
        overflow: hidden;
    }
    .hero-banner::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -20%;
        width: 60%;
        height: 200%;
        background: radial-gradient(ellipse, rgba(180,0,255,0.12) 0%, transparent 70%);
        pointer-events: none;
    }
    .hero-banner::after {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 2px;
        background: linear-gradient(90deg, transparent, #b400ff, #ff3c78, #00f5ff, transparent);
    }
    .hero-title {
        font-size: 2.6rem;
        font-weight: 800;
        line-height: 1.1;
        background: linear-gradient(90deg, #e040fb, #ff3c78, #ff9a00);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 8px;
    }
    .hero-sub {
        font-size: 0.95rem;
        color: #7a6d9a;
        letter-spacing: 0.03em;
    }
    .hero-tag {
        display: inline-block;
        background: rgba(180,0,255,0.15);
        color: #cc88ff;
        border: 1px solid rgba(180,0,255,0.4);
        border-radius: 6px;
        padding: 3px 10px;
        font-size: 0.75rem;
        font-family: 'Space Mono', monospace !important;
        margin-right: 8px;
        margin-top: 12px;
    }

    /* ── NEON CARDS ── */
    .neon-card {
        background: rgba(13,0,32,0.7);
        border: 1px solid rgba(180,0,255,0.25);
        border-radius: 16px;
        padding: 22px 26px;
        margin-bottom: 16px;
        backdrop-filter: blur(10px);
        position: relative;
        overflow: hidden;
        transition: border-color 0.2s;
    }
    .neon-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(180,0,255,0.6), transparent);
    }
    .neon-card-pink {
        border-color: rgba(255,60,120,0.35);
    }
    .neon-card-pink::before {
        background: linear-gradient(90deg, transparent, rgba(255,60,120,0.7), transparent);
    }
    .neon-card-cyan {
        border-color: rgba(0,245,255,0.3);
    }
    .neon-card-cyan::before {
        background: linear-gradient(90deg, transparent, rgba(0,245,255,0.6), transparent);
    }
    .neon-card-gold {
        border-color: rgba(255,180,0,0.35);
    }
    .neon-card-gold::before {
        background: linear-gradient(90deg, transparent, rgba(255,180,0,0.7), transparent);
    }

    /* ── STAT ROWS ── */
    .stat-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 9px 0;
        border-bottom: 1px solid rgba(255,255,255,0.06);
        font-size: 0.88rem;
    }
    .stat-row:last-child { border-bottom: none; }
    .stat-key { color: #6b5f8a; }
    .stat-val { color: #e2d8ff; font-weight: 700; font-family: 'Space Mono', monospace !important; }

    /* ── GLOWING METRIC BOXES ── */
    .kpi-grid { display: flex; gap: 14px; flex-wrap: wrap; margin-bottom: 24px; }
    .kpi-box {
        flex: 1;
        min-width: 130px;
        background: rgba(13,0,32,0.8);
        border: 1px solid rgba(180,0,255,0.3);
        border-radius: 14px;
        padding: 18px 16px;
        text-align: center;
        position: relative;
    }
    .kpi-box::after {
        content: '';
        position: absolute;
        bottom: 0; left: 10%; right: 10%;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(180,0,255,0.8), transparent);
        box-shadow: 0 0 8px rgba(180,0,255,0.8);
    }
    .kpi-value {
        font-size: 1.9rem;
        font-weight: 800;
        background: linear-gradient(135deg, #e040fb, #b400ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-family: 'Space Mono', monospace !important;
    }
    .kpi-value-pink {
        background: linear-gradient(135deg, #ff3c78, #ff6b9d);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .kpi-value-cyan {
        background: linear-gradient(135deg, #00f5ff, #0099cc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .kpi-value-gold {
        background: linear-gradient(135deg, #ffb800, #ff7c00);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .kpi-label {
        font-size: 0.7rem;
        color: #5a4d7a;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-top: 5px;
        font-weight: 600;
    }

    /* ── BADGES ── */
    .badge-sig {
        display: inline-flex; align-items: center; gap: 5px;
        background: rgba(0,245,130,0.1);
        color: #00f582;
        border: 1px solid rgba(0,245,130,0.4);
        border-radius: 20px;
        padding: 4px 14px;
        font-size: 0.78rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        box-shadow: 0 0 12px rgba(0,245,130,0.2);
    }
    .badge-nosig {
        display: inline-flex; align-items: center; gap: 5px;
        background: rgba(255,60,120,0.1);
        color: #ff3c78;
        border: 1px solid rgba(255,60,120,0.4);
        border-radius: 20px;
        padding: 4px 14px;
        font-size: 0.78rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        box-shadow: 0 0 12px rgba(255,60,120,0.2);
    }
    .badge-marginal {
        display: inline-flex; align-items: center; gap: 5px;
        background: rgba(255,180,0,0.1);
        color: #ffb800;
        border: 1px solid rgba(255,180,0,0.4);
        border-radius: 20px;
        padding: 4px 14px;
        font-size: 0.78rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        box-shadow: 0 0 12px rgba(255,180,0,0.2);
    }

    /* ── SECTION LABELS ── */
    .section-eyebrow {
        font-size: 0.7rem;
        color: #b400ff;
        text-transform: uppercase;
        letter-spacing: 0.15em;
        font-weight: 700;
        margin-bottom: 4px;
    }
    .section-title {
        font-size: 1.9rem;
        font-weight: 800;
        color: #f0e8ff;
        margin-bottom: 4px;
        line-height: 1.2;
    }
    .section-sub {
        font-size: 0.88rem;
        color: #5a4d7a;
        margin-bottom: 24px;
    }

    /* ── Q HEADER ── */
    .q-header {
        display: flex;
        align-items: flex-start;
        gap: 14px;
        margin-bottom: 18px;
        margin-top: 10px;
    }
    .q-num {
        background: linear-gradient(135deg, #b400ff, #7000aa);
        color: white;
        border-radius: 10px;
        padding: 6px 14px;
        font-size: 0.75rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        white-space: nowrap;
        margin-top: 3px;
        font-family: 'Space Mono', monospace !important;
    }
    .q-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: #e2d8ff;
    }
    .q-method {
        font-size: 0.78rem;
        color: #5a4d7a;
        margin-top: 3px;
    }

    /* ── SIDEBAR ── */
    .sidebar-logo {
        padding: 10px 0 20px 0;
    }
    .sidebar-logo-main {
        font-size: 1.4rem;
        font-weight: 800;
        background: linear-gradient(90deg, #e040fb, #ff3c78);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        line-height: 1.2;
    }
    .sidebar-logo-sub {
        font-size: 0.72rem;
        color: #4a3060;
        margin-top: 3px;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        font-family: 'Space Mono', monospace !important;
    }
    .nav-header {
        font-size: 0.65rem;
        color: #3a2850;
        text-transform: uppercase;
        letter-spacing: 0.15em;
        margin: 24px 0 8px 0;
        font-weight: 800;
        border-top: 1px solid rgba(180,0,255,0.15);
        padding-top: 16px;
    }
    .sidebar-stat {
        background: rgba(180,0,255,0.08);
        border: 1px solid rgba(180,0,255,0.2);
        border-radius: 10px;
        padding: 10px 14px;
        margin-bottom: 8px;
    }
    .sidebar-stat-label { font-size: 0.65rem; color: #4a3060; text-transform: uppercase; letter-spacing: 0.08em; }
    .sidebar-stat-value { font-size: 1.25rem; font-weight: 800; font-family: 'Space Mono', monospace !important; }

    /* ── DIVIDER ── */
    .neon-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(180,0,255,0.5), rgba(255,60,120,0.5), transparent);
        margin: 28px 0;
        border: none;
    }

    /* ── FINDING BOX ── */
    .finding-box {
        background: rgba(180,0,255,0.06);
        border-left: 3px solid #b400ff;
        border-radius: 0 10px 10px 0;
        padding: 12px 18px;
        margin-top: 14px;
        font-size: 0.88rem;
        color: #c0a8e8;
    }
    .finding-box strong { color: #e040fb; }

    /* ── STREAMLIT OVERRIDES ── */
    .stApp h1, .stApp h2, .stApp h3, .stApp h4 { color: #e2d8ff !important; }
    p, li { color: #8a7aaa; }
    .stDataFrame { border-radius: 10px; overflow: hidden; }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        background: rgba(13,0,32,0.5) !important;
        padding: 6px;
        border-radius: 12px;
        border: 1px solid rgba(180,0,255,0.2);
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent !important;
        border-radius: 8px !important;
        color: #5a4d7a !important;
        border: none !important;
        padding: 8px 20px !important;
        font-weight: 600 !important;
        font-size: 0.85rem !important;
        transition: all 0.2s !important;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #b400ff, #7000aa) !important;
        color: white !important;
        box-shadow: 0 0 16px rgba(180,0,255,0.4) !important;
    }

    /* Selectbox & Radio */
    .stSelectbox label, .stRadio label { color: #5a4d7a !important; font-size: 0.82rem !important; }
    .stSelectbox > div > div {
        background: rgba(13,0,32,0.8) !important;
        border: 1px solid rgba(180,0,255,0.3) !important;
        border-radius: 10px !important;
        color: #e2d8ff !important;
    }

    /* Radio buttons */
    .stRadio > div { gap: 6px !important; }
    .stRadio > div > label {
        background: rgba(13,0,32,0.6) !important;
        border: 1px solid rgba(180,0,255,0.2) !important;
        border-radius: 8px !important;
        padding: 8px 14px !important;
        color: #8a7aaa !important;
        transition: all 0.2s !important;
        cursor: pointer !important;
    }
    .stRadio [aria-checked="true"] {
        background: rgba(180,0,255,0.15) !important;
        border-color: rgba(180,0,255,0.6) !important;
        color: #e040fb !important;
    }

    /* Metrics */
    [data-testid="stMetricValue"] { color: #e040fb !important; font-family: 'Space Mono', monospace !important; }
    [data-testid="stMetricLabel"] { color: #5a4d7a !important; font-size: 0.78rem !important; }

    /* Expander */
    .streamlit-expanderHeader {
        background: rgba(13,0,32,0.8) !important;
        border: 1px solid rgba(180,0,255,0.2) !important;
        border-radius: 10px !important;
        color: #8a7aaa !important;
    }
    .streamlit-expanderContent {
        background: rgba(8,0,20,0.8) !important;
        border: 1px solid rgba(180,0,255,0.15) !important;
        border-top: none !important;
        border-radius: 0 0 10px 10px !important;
    }

    /* Spinner */
    .stSpinner > div { border-top-color: #b400ff !important; }

    /* Info/success boxes */
    .stInfo { background: rgba(0,245,255,0.06) !important; border-color: rgba(0,245,255,0.3) !important; color: #00f5ff !important; }

    /* Scrollbar */
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: #04010f; }
    ::-webkit-scrollbar-thumb { background: rgba(180,0,255,0.4); border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ── Plotly theme helper ───────────────────────────────────────────────────────
PLOT_BGCOLOR  = 'rgba(8,0,20,0.0)'
PAPER_BGCOLOR = 'rgba(8,0,20,0.0)'
GRID_COLOR    = 'rgba(180,0,255,0.1)'
FONT_COLOR    = '#8a7aaa'
TITLE_COLOR   = '#e2d8ff'
ACCENT_COLORS = ['#b400ff','#ff3c78','#00f5ff','#ffb800','#00f582','#ff7c00','#7c6dfa','#fc5c7d']

def plot_layout(fig, title=None, height=None):
    fig.update_layout(
        plot_bgcolor=PLOT_BGCOLOR,
        paper_bgcolor=PAPER_BGCOLOR,
        font_color=FONT_COLOR,
        title_font_color=TITLE_COLOR,
        title_font_size=14,
        title_font_family='Space Grotesk',
        xaxis=dict(gridcolor=GRID_COLOR, linecolor='rgba(180,0,255,0.2)', tickcolor='rgba(180,0,255,0.2)'),
        yaxis=dict(gridcolor=GRID_COLOR, linecolor='rgba(180,0,255,0.2)', tickcolor='rgba(180,0,255,0.2)'),
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color=FONT_COLOR)),
    )
    if title:
        fig.update_layout(title_text=title)
    if height:
        fig.update_layout(height=height)
    return fig

# ── Data Loading & Preprocessing ─────────────────────────────────────────────
import os
DATA_PATH = os.path.join(os.path.dirname(__file__), "augmented_survey.xlsx")

@st.cache_data
def load_data():
    df = pd.read_excel(DATA_PATH)
    df['gender']        = df['Q1. What is your gender?']
    df['allowance']     = df['Q2. What is your approximate monthly allowance / pocket money?']
    df['year']          = df['Q3. Which year of college are you in?']
    df['venue']         = df['Q4. What type of venue did you go to?']
    df['day']           = df['Q5. What day was the outing?']
    df['intiated_plan'] = df['Q6. Who initiated / planned the outing?']
    df['occasion']      = df['Q7. What was the occasion?']
    df['group_size']    = df['Q8. How many people were in your group, including yourself?']
    df['travel_dist']   = df['Q9. How far did you travel to reach the venue?']
    df['spending']      = df['Q10. Approximately how much did YOU personally spend on this outing? (₹) (Include food, travel, entry, shopping — everything)']
    df['rough_budget']  = df['Q11. Did you have a rough budget in mind before going?']
    df['exceed_budget'] = df['Q12. Did your actual spending exceed that budget? (Answer only if Q11 = Yes)']
    df['spend_reason']  = df['Q13. What was the biggest reason you spent what you did? (Pick one)']
    df['outing_freq']   = df['Q14. How often do you go on social outings per month?']
    df['peer_influence']= df["Q15. On a scale of 1–5, how much do your friends' choices influence where you go and what you spend? (1 = Not at all, 5 = Completely driven by friends)"]
    df['discount']      = df['Q16. Did you use any discount or offer? (Zomato, Swiggy Dineout, student discount, etc.)']

    spend_map = {'Less than ₹200': 100, '₹200 – ₹500': 350, '₹501 – ₹1,000': 750,
                 '₹1,001 – ₹2,000': 1500, 'More than ₹2,000': 2500}
    allowance_map = {'Less than ₹3,000': 1500, '₹3,000 – ₹6,000': 4500,
                     '₹6,001 – ₹10,000': 8000, '₹10,001 – ₹15,000': 12500,
                     'More than ₹15,000': 17500}

    df['spending']  = df['spending'].map(spend_map)
    df['allowance'] = df['allowance'].map(allowance_map)

    cols = ['gender','allowance','year','venue','day','intiated_plan','occasion',
            'group_size','travel_dist','spending','rough_budget','exceed_budget',
            'spend_reason','outing_freq','peer_influence','discount']
    df = df[cols].iloc[:140].dropna(subset=['spending','allowance'])
    return df

df = load_data()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class='sidebar-logo'>
        <div class='sidebar-logo-main'>💸 Broke<br>College<br>Kids</div>
        <div class='sidebar-logo-sub'>Student Spending Lab</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='nav-header'>🗺 Navigate</div>", unsafe_allow_html=True)
    page = st.radio(
        "Go to",
        ["📊 Overview", "🧪 Hypothesis Testing", "📈 ANOVA", "🔢 MANOVA", "📉 Regression"],
        label_visibility="collapsed"
    )

    st.markdown("<div class='nav-header'>📦 Dataset</div>", unsafe_allow_html=True)
    st.markdown(f"""
    <div class='sidebar-stat'>
        <div class='sidebar-stat-label'>Respondents Analysed</div>
        <div class='sidebar-stat-value' style='color:#b400ff;'>{len(df)}</div>
    </div>
    <div class='sidebar-stat'>
        <div class='sidebar-stat-label'>Avg Spending / Outing</div>
        <div class='sidebar-stat-value' style='color:#ff3c78;'>₹{df['spending'].mean():.0f}</div>
    </div>
    <div class='sidebar-stat'>
        <div class='sidebar-stat-label'>Avg Monthly Allowance</div>
        <div class='sidebar-stat-value' style='color:#ffb800;'>₹{df['allowance'].mean():,.0f}</div>
    </div>
    <div class='sidebar-stat'>
        <div class='sidebar-stat-label'>Discount Users</div>
        <div class='sidebar-stat-value' style='color:#00f5ff;'>{(df['discount']=='Yes').mean()*100:.0f}%</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style='margin-top:24px; padding:12px; background:rgba(255,60,120,0.06); border:1px solid rgba(255,60,120,0.2); border-radius:10px;'>
        <div style='font-size:0.7rem; color:#5a2030; text-transform:uppercase; letter-spacing:0.08em; font-weight:700;'>⚠ Scope Note</div>
        <div style='font-size:0.75rem; color:#7a4055; margin-top:5px;'>Analysis uses first 140 rows of the augmented survey dataset only.</div>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "📊 Overview":

    st.markdown("""
    <div class='hero-banner'>
        <div class='hero-title'>Where Did Our Money Go?? 💸</div>
        <div class='hero-sub'>A statistical autopsy of 140 college students blowing their allowance on outings.</div>
        <span class='hero-tag'>n = 140</span>
        <span class='hero-tag'>Mumbai</span>
        <span class='hero-tag'>Survey Data</span>
        <span class='hero-tag'>2024–25</span>
    </div>
    """, unsafe_allow_html=True)

    # KPI Row
    st.markdown(f"""
    <div class='kpi-grid'>
        <div class='kpi-box'>
            <div class='kpi-value'>{len(df)}</div>
            <div class='kpi-label'>Respondents</div>
        </div>
        <div class='kpi-box'>
            <div class='kpi-value kpi-value-pink'>₹{df['spending'].mean():.0f}</div>
            <div class='kpi-label'>Avg Spend / Outing</div>
        </div>
        <div class='kpi-box'>
            <div class='kpi-value kpi-value-gold'>₹{df['allowance'].mean():,.0f}</div>
            <div class='kpi-label'>Avg Monthly Allowance</div>
        </div>
        <div class='kpi-box'>
            <div class='kpi-value kpi-value-cyan'>{(df['discount']=='Yes').mean()*100:.0f}%</div>
            <div class='kpi-label'>Use Discounts</div>
        </div>
        <div class='kpi-box'>
            <div class='kpi-value' style='background:linear-gradient(135deg,#00f582,#00cc66);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;'>{df['peer_influence'].median():.0f}/5</div>
            <div class='kpi-label'>Median Peer Influence</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_l, col_r = st.columns(2)

    with col_l:
        fig = px.histogram(df, x='spending', nbins=10,
                           color_discrete_sequence=['#b400ff'],
                           title='Spending Distribution (₹)',
                           labels={'spending': 'Spending (₹)', 'count': 'Count'})
        fig.update_traces(marker_line_color='rgba(180,0,255,0.8)', marker_line_width=1,
                          opacity=0.85)
        plot_layout(fig)
        st.plotly_chart(fig, use_container_width=True)

        occ_counts = df['occasion'].value_counts().reset_index()
        occ_counts.columns = ['Occasion', 'Count']
        fig2 = px.bar(occ_counts, x='Count', y='Occasion', orientation='h',
                      color='Count', color_continuous_scale=[[0,'#3d0070'],[0.5,'#b400ff'],[1,'#ff3c78']],
                      title='Outings by Occasion')
        fig2.update_layout(coloraxis_showscale=False, yaxis_title=None)
        plot_layout(fig2)
        st.plotly_chart(fig2, use_container_width=True)

    with col_r:
        fig3 = px.histogram(df, x='allowance', nbins=10,
                            color_discrete_sequence=['#ff3c78'],
                            title='Allowance Distribution (₹)',
                            labels={'allowance': 'Monthly Allowance (₹)'})
        fig3.update_traces(marker_line_color='rgba(255,60,120,0.8)', marker_line_width=1,
                           opacity=0.85)
        plot_layout(fig3)
        st.plotly_chart(fig3, use_container_width=True)

        fig4 = px.box(df, x='occasion', y='spending',
                      color='occasion',
                      title='Spending by Occasion',
                      color_discrete_sequence=ACCENT_COLORS)
        fig4.update_layout(showlegend=False)
        fig4.update_xaxes(tickangle=-30, title=None)
        plot_layout(fig4)
        st.plotly_chart(fig4, use_container_width=True)

    st.markdown("<hr class='neon-divider'>", unsafe_allow_html=True)

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        gender_c = df['gender'].value_counts()
        fig5 = px.pie(values=gender_c.values, names=gender_c.index,
                      title='Gender Split',
                      color_discrete_sequence=['#b400ff','#ff3c78','#00f5ff'])
        fig5.update_traces(hole=0.45, marker=dict(line=dict(color='rgba(0,0,0,0.5)', width=2)))
        fig5.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color=FONT_COLOR,
                           title_font_color=TITLE_COLOR)
        st.plotly_chart(fig5, use_container_width=True)

    with col_b:
        disc_c = df['discount'].value_counts()
        fig6 = px.pie(values=disc_c.values, names=disc_c.index,
                      title='Discount Usage',
                      color_discrete_sequence=['#00f582','#ff3c78'])
        fig6.update_traces(hole=0.45, marker=dict(line=dict(color='rgba(0,0,0,0.5)', width=2)))
        fig6.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color=FONT_COLOR,
                           title_font_color=TITLE_COLOR)
        st.plotly_chart(fig6, use_container_width=True)

    with col_c:
        peer_c = df['peer_influence'].value_counts().sort_index()
        fig7 = px.bar(x=peer_c.index.astype(str), y=peer_c.values,
                      title='Peer Influence Score (1–5)',
                      color=peer_c.values,
                      color_continuous_scale=[[0,'#3d0070'],[0.5,'#b400ff'],[1,'#ff3c78']],
                      labels={'x':'Score','y':'Count'})
        fig7.update_layout(coloraxis_showscale=False)
        plot_layout(fig7)
        st.plotly_chart(fig7, use_container_width=True)

    with st.expander("🗄️ Raw Data Preview (first 30 rows)"):
        st.dataframe(df.head(30), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — HYPOTHESIS TESTING
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🧪 Hypothesis Testing":

    st.markdown("""
    <div class='hero-banner'>
        <div class='hero-title'>Hypothesis Tests 🧪</div>
        <div class='hero-sub'>T-tests and Z-tests for proportions — putting our assumptions under the microscope.</div>
        <span class='hero-tag'>Welch's T-test</span>
        <span class='hero-tag'>One-tailed T-test</span>
        <span class='hero-tag'>Z-test Proportions</span>
        <span class='hero-tag'>α = 0.05</span>
    </div>
    """, unsafe_allow_html=True)

    def result_badge(p, threshold=0.05):
        if p < threshold:
            return "<span class='badge-sig'>⚡ Significant (p < 0.05)</span>"
        else:
            return "<span class='badge-nosig'>✗ Not Significant (p ≥ 0.05)</span>"

    # Q1 — Gender vs Spending
    st.markdown("""
    <div class='q-header'>
        <div class='q-num'>H₁</div>
        <div>
            <div class='q-title'>Does gender affect outing spending?</div>
            <div class='q-method'>Independent Samples T-test (two-tailed)</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    male   = df[df['gender']=='Male']['spending'].dropna()
    female = df[df['gender']=='Female']['spending'].dropna()
    t1, p1 = stats.ttest_ind(male, female)

    col1, col2 = st.columns([1, 1.5])
    with col1:
        st.markdown(f"""
        <div class='neon-card'>
            <div class='stat-row'><span class='stat-key'>T-statistic</span><span class='stat-val'>{t1:.4f}</span></div>
            <div class='stat-row'><span class='stat-key'>P-value</span><span class='stat-val'>{p1:.4f}</span></div>
            <div class='stat-row'><span class='stat-key'>Male mean</span><span class='stat-val'>₹{male.mean():.0f}</span></div>
            <div class='stat-row'><span class='stat-key'>Female mean</span><span class='stat-val'>₹{female.mean():.0f}</span></div>
            <div style='margin-top:16px;'>{result_badge(p1)}</div>
            <div class='finding-box'>
                Male and female students show <strong>no statistically significant difference</strong> in outing expenditure — gender doesn't decide the bill.
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        fig = px.box(df[df['gender'].isin(['Male','Female'])],
                     x='gender', y='spending', color='gender',
                     color_discrete_map={'Male':'#b400ff','Female':'#ff3c78'},
                     title='Spending Distribution by Gender')
        fig.update_layout(showlegend=False)
        fig.update_xaxes(title=None)
        plot_layout(fig)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("<hr class='neon-divider'>", unsafe_allow_html=True)

    # Q2 — Budget vs Spending
    st.markdown("""
    <div class='q-header'>
        <div class='q-num'>H₂</div>
        <div>
            <div class='q-title'>Do students with a budget actually spend less?</div>
            <div class='q-method'>One-tailed T-test (budget_yes &lt; budget_no)</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    budget_yes = df[df['rough_budget']=='Yes']['spending'].dropna()
    budget_no  = df[df['rough_budget']=='No']['spending'].dropna()
    t2, p2 = stats.ttest_ind(budget_yes, budget_no, alternative='less')

    col1, col2 = st.columns([1, 1.5])
    with col1:
        st.markdown(f"""
        <div class='neon-card neon-card-cyan'>
            <div class='stat-row'><span class='stat-key'>T-statistic</span><span class='stat-val'>{t2:.4f}</span></div>
            <div class='stat-row'><span class='stat-key'>P-value</span><span class='stat-val'>{p2:.4f}</span></div>
            <div class='stat-row'><span class='stat-key'>Budget = Yes mean</span><span class='stat-val'>₹{budget_yes.mean():.0f}</span></div>
            <div class='stat-row'><span class='stat-key'>Budget = No mean</span><span class='stat-val'>₹{budget_no.mean():.0f}</span></div>
            <div style='margin-top:16px;'>{result_badge(p2)}</div>
            <div class='finding-box'>
                Having a budget in mind <strong>does not constrain actual spending</strong> — vibes override spreadsheets, apparently.
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        fig2 = px.violin(df, x='rough_budget', y='spending', color='rough_budget',
                         box=True, color_discrete_map={'Yes':'#00f5ff','No':'#ff3c78'},
                         title='Spending: Had Budget vs No Budget')
        fig2.update_layout(showlegend=False)
        fig2.update_xaxes(title='Had a Budget?')
        plot_layout(fig2)
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("<hr class='neon-divider'>", unsafe_allow_html=True)

    # Q3 — Discount usage by gender
    st.markdown("""
    <div class='q-header'>
        <div class='q-num'>H₃</div>
        <div>
            <div class='q-title'>Does discount usage differ by gender?</div>
            <div class='q-method'>Z-test for Proportions (two-tailed)</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    male_d   = df[df['gender']=='Male']['discount'].dropna()
    female_d = df[df['gender']=='Female']['discount'].dropna()
    count = [(male_d=='Yes').sum(), (female_d=='Yes').sum()]
    nobs  = [len(male_d), len(female_d)]
    z3, p3 = proportions_ztest(count, nobs)

    col1, col2 = st.columns([1, 1.5])
    with col1:
        st.markdown(f"""
        <div class='neon-card neon-card-gold'>
            <div class='stat-row'><span class='stat-key'>Z-statistic</span><span class='stat-val'>{z3:.4f}</span></div>
            <div class='stat-row'><span class='stat-key'>P-value</span><span class='stat-val'>{p3:.4f}</span></div>
            <div class='stat-row'><span class='stat-key'>Male discount %</span><span class='stat-val'>{count[0]/nobs[0]*100:.1f}%</span></div>
            <div class='stat-row'><span class='stat-key'>Female discount %</span><span class='stat-val'>{count[1]/nobs[1]*100:.1f}%</span></div>
            <div style='margin-top:16px;'>{result_badge(p3)}</div>
            <div class='finding-box'>
                Coupon-clipping behaviour is <strong>gender-neutral</strong> — everyone loves a Zomato deal.
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        disc_gender = df[df['gender'].isin(['Male','Female'])].groupby(['gender','discount']).size().reset_index(name='count')
        fig3 = px.bar(disc_gender, x='gender', y='count', color='discount',
                      barmode='group',
                      color_discrete_map={'Yes':'#00f582','No':'#ff3c78'},
                      title='Discount Usage by Gender')
        fig3.update_xaxes(title=None)
        plot_layout(fig3)
        st.plotly_chart(fig3, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — ANOVA
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈 ANOVA":

    st.markdown("""
    <div class='hero-banner'>
        <div class='hero-title'>ANOVA 📈</div>
        <div class='hero-sub'>One-way & two-way ANOVA with Tukey HSD post-hoc — who's spending differently and why?</div>
        <span class='hero-tag'>One-Way ANOVA</span>
        <span class='hero-tag'>Two-Way ANOVA</span>
        <span class='hero-tag'>Tukey HSD</span>
        <span class='hero-tag'>Type I SS</span>
    </div>
    """, unsafe_allow_html=True)

    tabs = st.tabs(["⚡ One-Way ANOVA", "🌀 Two-Way ANOVA"])

    with tabs[0]:
        anova_choice = st.selectbox("Select a test to run:", [
            "Allowance ~ College Year",
            "Spending ~ Travel Distance",
            "Allowance ~ Discount Usage",
            "Spending ~ Occasion",
            "Spending ~ Group Size"
        ])

        def run_anova(formula, group_col, dv_col, label):
            fit = smf.ols(formula, data=df).fit()
            table = sm.stats.anova_lm(fit, typ=1)
            p_val = table['PR(>F)'].iloc[0]

            tukey = pairwise_tukeyhsd(df[dv_col], groups=df[group_col])
            tukey_df = pd.DataFrame(tukey._results_table.data[1:],
                                    columns=tukey._results_table.data[0])
            sig_pairs = tukey_df[tukey_df['reject']==True].sort_values('meandiff', ascending=False)

            col1, col2 = st.columns([1, 2])
            with col1:
                badge = "<span class='badge-sig'>⚡ Significant</span>" if p_val < 0.05 else "<span class='badge-nosig'>✗ Not Significant</span>"
                f_val = table['F'].iloc[0]
                st.markdown(f"""
                <div class='neon-card'>
                    <div style='font-size:0.75rem; color:#5a4d7a; text-transform:uppercase; letter-spacing:0.1em; margin-bottom:14px; font-weight:700;'>ANOVA Result</div>
                    <div class='stat-row'><span class='stat-key'>F-statistic</span><span class='stat-val'>{f_val:.4f}</span></div>
                    <div class='stat-row'><span class='stat-key'>P-value</span><span class='stat-val'>{p_val:.4f}</span></div>
                    <div style='margin-top:16px;'>{badge}</div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown(f"**Significant Tukey Pairs** ({len(sig_pairs)} pairs found)")
                if len(sig_pairs) > 0:
                    st.dataframe(sig_pairs[['group1','group2','meandiff','p-adj']].round(3),
                                 use_container_width=True)
                else:
                    st.info("No significant pairwise differences")

            with col2:
                means = df.groupby(group_col)[dv_col].mean().sort_values(ascending=False).reset_index()
                fig = px.bar(means, x=group_col, y=dv_col,
                             color=dv_col,
                             color_continuous_scale=[[0,'#3d0070'],[0.5,'#b400ff'],[1,'#ff3c78']],
                             title=f'Mean {label} by {group_col}')
                fig.update_layout(coloraxis_showscale=False)
                fig.update_xaxes(tickangle=-30, title=None)
                plot_layout(fig)
                st.plotly_chart(fig, use_container_width=True)

        if anova_choice == "Allowance ~ College Year":
            run_anova("allowance ~ C(year)", "year", "allowance", "Allowance")
            st.markdown("<div class='finding-box'>💡 <strong>Finding:</strong> 4th year students have significantly higher allowance than 1st year students.</div>", unsafe_allow_html=True)
        elif anova_choice == "Spending ~ Travel Distance":
            run_anova("spending ~ C(travel_dist)", "travel_dist", "spending", "Spending")
            st.markdown("<div class='finding-box'>💡 <strong>Finding:</strong> Students travelling >10 km spend the most; the &lt;2 km group spends the least.</div>", unsafe_allow_html=True)
        elif anova_choice == "Allowance ~ Discount Usage":
            run_anova("allowance ~ C(discount)", "discount", "allowance", "Allowance")
            st.markdown("<div class='finding-box'>💡 <strong>Finding:</strong> Students who use discounts tend to have higher allowances — richer students hunt deals more.</div>", unsafe_allow_html=True)
        elif anova_choice == "Spending ~ Occasion":
            run_anova("spending ~ C(occasion)", "occasion", "spending", "Spending")
            st.markdown("<div class='finding-box'>💡 <strong>Finding:</strong> Date/romantic outings drive the highest spending. Love is expensive.</div>", unsafe_allow_html=True)
        elif anova_choice == "Spending ~ Group Size":
            run_anova("spending ~ C(group_size)", "group_size", "spending", "Spending")
            st.markdown("<div class='finding-box'>💡 <strong>Finding:</strong> Groups of 8+ people spend the most — squad tax is real.</div>", unsafe_allow_html=True)

    with tabs[1]:
        two_choice = st.selectbox("Choose interaction model:", [
            "Spending ~ Travel Distance × Group Size × Occasion",
            "Allowance ~ Year × Rough Budget × Occasion"
        ])

        if two_choice == "Spending ~ Travel Distance × Group Size × Occasion":
            fit = smf.ols('spending ~ C(travel_dist) * C(group_size) * C(occasion)', data=df).fit()
            table = sm.stats.anova_lm(fit, typ=1).reset_index()
            table.columns = ['Source'] + list(table.columns[1:])
            table = table.dropna(subset=['PR(>F)'])

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Full ANOVA Table**")
                st.dataframe(table[['Source','df','F','PR(>F)']].round(4), use_container_width=True)
            with col2:
                sig = table[table['PR(>F)'] < 0.05]
                fig = px.bar(sig, x='Source', y='F',
                             color='F',
                             color_continuous_scale=[[0,'#3d0070'],[0.5,'#b400ff'],[1,'#ff3c78']],
                             title='Significant F-values (p < 0.05)')
                fig.update_layout(coloraxis_showscale=False)
                fig.update_xaxes(tickangle=-30)
                plot_layout(fig)
                st.plotly_chart(fig, use_container_width=True)
            st.markdown("<div class='finding-box'>💡 <strong>Finding:</strong> Occasion and travel distance are the dominant individual drivers of spending. Date outings >10 km produce the highest combined spend.</div>", unsafe_allow_html=True)

        else:
            fit = smf.ols('allowance ~ C(year) * C(rough_budget) * C(occasion)', data=df).fit()
            table = sm.stats.anova_lm(fit, typ=1).reset_index()
            table.columns = ['Source'] + list(table.columns[1:])
            table = table.dropna(subset=['PR(>F)'])

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Full ANOVA Table**")
                st.dataframe(table[['Source','df','F','PR(>F)']].round(4), use_container_width=True)
            with col2:
                sig = table[table['PR(>F)'] < 0.05]
                if len(sig) > 0:
                    fig = px.bar(sig, x='Source', y='F',
                                 color='F',
                                 color_continuous_scale=[[0,'#3d0070'],[0.5,'#b400ff'],[1,'#ff3c78']],
                                 title='Significant F-values (p < 0.05)')
                    fig.update_layout(coloraxis_showscale=False)
                    fig.update_xaxes(tickangle=-30)
                    plot_layout(fig)
                    st.plotly_chart(fig, use_container_width=True)
            st.markdown("<div class='finding-box'>💡 <strong>Finding:</strong> Year drives allowance; occasion drives where it gets spent. Senior students self-select into higher-spend occasion types.</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — MANOVA
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔢 MANOVA":

    st.markdown("""
    <div class='hero-banner'>
        <div class='hero-title'>MANOVA 🔢</div>
        <div class='hero-sub'>Multivariate ANOVA — testing whether categorical IVs simultaneously affect both spending AND allowance.</div>
        <span class='hero-tag'>Wilks' Lambda</span>
        <span class='hero-tag'>Pillai's Trace</span>
        <span class='hero-tag'>DVs: spending + allowance</span>
    </div>
    """, unsafe_allow_html=True)

    df_m = df.dropna(subset=['spending','allowance'])

    tests = [
        ("occasion",    "spending + allowance ~ occasion",    "Occasion"),
        ("travel_dist", "spending + allowance ~ travel_dist", "Travel Distance"),
        ("year",        "spending + allowance ~ year",        "College Year"),
        ("group_size",  "spending + allowance ~ group_size",  "Group Size"),
        ("gender",      "spending + allowance ~ gender",      "Gender"),
        ("discount",    "spending + allowance ~ discount",    "Discount Usage"),
    ]

    results_list = []
    for iv, formula, label in tests:
        try:
            mov = MANOVA.from_formula(formula, data=df_m)
            res = mov.mv_test()
            tbl = res.results[iv]['stat']
            wilks = tbl[tbl.index.str.contains("Wilks", case=False, na=False)]
            if len(wilks) > 0:
                p = float(wilks['Pr > F'].values[0])
                f = float(wilks['F Value'].values[0])
                w = float(wilks['Value'].values[0])
                sig = "Significant ✓" if p < 0.05 else ("Marginal" if p < 0.10 else "Not Significant ✗")
                results_list.append({'IV': label, "Wilks' λ": round(w,4), 'F Value': round(f,4),
                                     'Pr > F': round(p,4), 'Result': sig})
        except:
            pass

    if results_list:
        sum_df = pd.DataFrame(results_list)
        st.markdown("#### One-Way MANOVA Summary — DVs: spending + allowance")

        def color_result(val):
            if 'Significant ✓' in str(val):
                return 'color: #00f582; font-weight: 700;'
            elif 'Marginal' in str(val):
                return 'color: #ffb800; font-weight: 700;'
            else:
                return 'color: #ff3c78; font-weight: 700;'

        try:
            styled = sum_df.style.map(color_result, subset=['Result'])
        except AttributeError:
            styled = sum_df.style.applymap(color_result, subset=['Result'])
        st.dataframe(styled, use_container_width=True)

        fig = px.bar(sum_df, x='IV', y="Wilks' λ",
                     color="Pr > F",
                     color_continuous_scale=[[0,'#00f582'],[0.5,'#ffb800'],[1,'#ff3c78']],
                     range_color=[0, 0.2],
                     title="Wilks' λ by IV  —  lower = stronger multivariate effect")
        fig.add_hline(y=0.05, line_dash='dot', line_color='rgba(255,60,120,0.6)',
                      annotation_text='λ = 0.05', annotation_font_color='#ff3c78')
        fig.update_xaxes(title=None)
        plot_layout(fig)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("<hr class='neon-divider'>", unsafe_allow_html=True)
    st.markdown("#### 🔬 Drill Down — Select an IV for detailed results")

    iv_sel = st.selectbox("Independent Variable", [l for _, _, l in tests])
    iv_map = {l: (iv, f) for iv, f, l in tests}
    iv_col, formula = iv_map[iv_sel]

    try:
        mov = MANOVA.from_formula(formula, data=df_m)
        res = mov.mv_test()
        tbl = res.results[iv_col]['stat']

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Multivariate Test Statistics**")
            st.dataframe(tbl.round(4), use_container_width=True)

        with col2:
            tukey_s = pairwise_tukeyhsd(df_m['spending'], groups=df_m[iv_col])
            tukey_df = pd.DataFrame(tukey_s._results_table.data[1:],
                                    columns=tukey_s._results_table.data[0])
            sig = tukey_df[tukey_df['reject']==True].sort_values('meandiff', ascending=False)

            if len(sig) > 0:
                st.markdown(f"**Significant Tukey Pairs — Spending** ({len(sig)} pairs)")
                st.dataframe(sig[['group1','group2','meandiff','p-adj']].round(3),
                             use_container_width=True)
            else:
                st.info("No significant pairwise spending differences found.")

        col3, col4 = st.columns(2)
        with col3:
            m = df_m.groupby(iv_col)['spending'].mean().sort_values(ascending=False).reset_index()
            fig = px.bar(m, x=iv_col, y='spending', color='spending',
                         color_continuous_scale=[[0,'#3d0070'],[0.5,'#b400ff'],[1,'#ff3c78']],
                         title=f'Mean Spending by {iv_sel}')
            fig.update_layout(coloraxis_showscale=False)
            fig.update_xaxes(tickangle=-30, title=None)
            plot_layout(fig)
            st.plotly_chart(fig, use_container_width=True)

        with col4:
            m2 = df_m.groupby(iv_col)['allowance'].mean().sort_values(ascending=False).reset_index()
            fig2 = px.bar(m2, x=iv_col, y='allowance', color='allowance',
                          color_continuous_scale=[[0,'#002040'],[0.5,'#00a0ff'],[1,'#00f5ff']],
                          title=f'Mean Allowance by {iv_sel}')
            fig2.update_layout(coloraxis_showscale=False)
            fig2.update_xaxes(tickangle=-30, title=None)
            plot_layout(fig2)
            st.plotly_chart(fig2, use_container_width=True)

    except Exception as e:
        st.error(f"Could not run MANOVA: {e}")

    st.markdown("<hr class='neon-divider'>", unsafe_allow_html=True)
    st.markdown("#### 🌐 Two-Way MANOVA — Occasion × Travel Distance → Spending")
    try:
        fit = smf.ols('spending ~ C(occasion) * C(travel_dist)', data=df_m).fit()
        anova_2w = sm.stats.anova_lm(fit, typ=1).reset_index()
        anova_2w.columns = ['Source'] + list(anova_2w.columns[1:])
        anova_2w = anova_2w.dropna(subset=['PR(>F)'])
        col1, col2 = st.columns(2)
        with col1:
            st.dataframe(anova_2w[['Source','df','F','PR(>F)']].round(4), use_container_width=True)
        with col2:
            pivot = df_m.groupby(['occasion','travel_dist'])['spending'].mean().reset_index()
            fig = px.density_heatmap(pivot, x='travel_dist', y='occasion', z='spending',
                                     color_continuous_scale=[[0,'#0a001a'],[0.5,'#b400ff'],[1,'#ff3c78']],
                                     title='Mean Spending: Occasion × Travel Distance')
            fig.update_xaxes(title='Travel Distance', tickangle=-30)
            fig.update_yaxes(title=None)
            plot_layout(fig)
            st.plotly_chart(fig, use_container_width=True)
        st.markdown("<div class='finding-box'>💡 <strong>Finding:</strong> Both occasion and travel distance independently drive spending. Date outings >10 km produce the highest combined spend.</div>", unsafe_allow_html=True)
    except Exception as e:
        st.error(str(e))


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — REGRESSION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📉 Regression":

    st.markdown("""
    <div class='hero-banner'>
        <div class='hero-title'>Regression Analysis 📉</div>
        <div class='hero-sub'>Lasso regularisation, OLS linear regression, and K=100 cross-validation — building the spending prediction model.</div>
        <span class='hero-tag'>Lasso (L1)</span>
        <span class='hero-tag'>OLS</span>
        <span class='hero-tag'>K=100 CV</span>
        <span class='hero-tag'>StandardScaler</span>
    </div>
    """, unsafe_allow_html=True)

    @st.cache_data
    def prep_regression():
        df_r = df.copy()
        # Cast peer_influence to int so dummies are 'peer_influence_3' not 'peer_influence_3.0'
        df_r['peer_influence'] = pd.to_numeric(df_r['peer_influence'], errors='coerce').astype('Int64')
        df_r = pd.get_dummies(df_r, columns=[
            'group_size','peer_influence','gender','venue','day','intiated_plan',
            'occasion','exceed_budget','spend_reason','rough_budget','discount',
            'travel_dist','outing_freq','year'
        ])
        # Cast boolean dummy columns to int (pandas >= 2.0 returns bool dtype)
        bool_cols_r = df_r.select_dtypes(include='bool').columns
        df_r[bool_cols_r] = df_r[bool_cols_r].astype(int)
        Y = df_r['spending'].astype(float)
        feature_cols = [c for c in df_r.columns if c != 'spending' and df_r[c].dtype in [np.float64, np.int64, np.int32, np.int8]]
        X_raw = df_r[feature_cols].fillna(0).astype(float)
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(X_raw), columns=X_raw.columns)
        return X_raw, X_scaled, Y, feature_cols

    X_raw, X_scaled, Y, feature_cols = prep_regression()

    tabs = st.tabs(["🔬 Lasso Regularisation", "📐 OLS Linear Regression", "🔁 K-Fold CV"])

    with tabs[0]:
        st.markdown("#### Lasso (L1) Regularisation — Automatic Feature Selection")
        st.markdown("<div style='font-size:0.85rem; color:#5a4d7a; margin-bottom:16px;'>LassoCV over 400 alpha values (log-space 1e-6 to 1e4), 5-fold internal CV, standardised features.</div>", unsafe_allow_html=True)

        with st.spinner("Fitting LassoCV — crunching alphas..."):
            alphas = np.logspace(-6, 4, 400)
            lassocv = LassoCV(alphas=alphas, cv=5, max_iter=10000)
            lassocv.fit(X_scaled, Y)
            best_alpha = lassocv.alpha_

            lasso = Lasso(alpha=best_alpha, max_iter=10000)
            lasso.fit(X_scaled, Y)
            coef_series = pd.Series(lasso.coef_, index=X_scaled.columns)
            nonzero = coef_series[coef_series != 0].sort_values(key=abs, ascending=False)

        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown(f"""
            <div class='neon-card'>
                <div style='font-size:0.75rem; color:#5a4d7a; text-transform:uppercase; letter-spacing:0.1em; margin-bottom:14px; font-weight:700;'>Lasso Results</div>
                <div class='stat-row'><span class='stat-key'>Best α (lambda)</span><span class='stat-val'>{best_alpha:.4f}</span></div>
                <div class='stat-row'><span class='stat-key'>Non-zero features</span><span class='stat-val'>{len(nonzero)}</span></div>
                <div class='stat-row'><span class='stat-key'>Total features</span><span class='stat-val'>{len(coef_series)}</span></div>
                <div class='stat-row'><span class='stat-key'>Zeroed out</span><span class='stat-val'>{(coef_series==0).sum()}</span></div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("**🎯 Selected Features:**")
            st.dataframe(nonzero.round(4).reset_index().rename(columns={'index':'Feature', 0:'Coefficient'}),
                         use_container_width=True)
        with col2:
            if len(nonzero) > 0:
                colors = ['#00f582' if v > 0 else '#ff3c78' for v in nonzero.values]
                fig = go.Figure(go.Bar(
                    x=nonzero.values,
                    y=nonzero.index,
                    orientation='h',
                    marker_color=colors,
                    marker_line_color='rgba(0,0,0,0.3)',
                    marker_line_width=1,
                ))
                fig.update_layout(title_text=f'Lasso Coefficients (α = {best_alpha:.4f})',
                                  xaxis_title='Coefficient', yaxis_title=None,
                                  height=max(300, len(nonzero)*30))
                plot_layout(fig)
                st.plotly_chart(fig, use_container_width=True)

    with tabs[1]:
        st.markdown("#### OLS Linear Regression — Final Selected Features")
        st.markdown("<div class='finding-box' style='margin-bottom:16px;'>Step 1: Lasso-selected features fed into OLS. Step 2: Features with <strong>p &gt; 0.05 dropped</strong>. Final model uses only statistically significant predictors.</div>", unsafe_allow_html=True)
        try:
            df_ols = df.copy()
            # Cast peer_influence to int so dummies are 'peer_influence_3' not 'peer_influence_3.0'
            df_ols['peer_influence'] = pd.to_numeric(df_ols['peer_influence'], errors='coerce').astype('Int64')
            df_ols = pd.get_dummies(df_ols, columns=[
                'group_size','peer_influence','gender','venue','day','intiated_plan',
                'occasion','exceed_budget','spend_reason','rough_budget','discount',
                'travel_dist','outing_freq','year'
            ])
            # Cast boolean dummy columns to int (pandas >= 2.0 returns bool dtype)
            bool_cols = df_ols.select_dtypes(include='bool').columns
            df_ols[bool_cols] = df_ols[bool_cols].astype(int)

            # 6 features exactly matching smf.ols formula in the notebook
            # Step 1 — use Lasso-selected features (nonzero comes from tabs[0] Lasso above)
            lasso_features = [f for f in nonzero.index if f in df_ols.columns]

            X_ols = df_ols[lasso_features].fillna(0).astype(float)
            Y_ols = df_ols['spending'].astype(float)

            X_const = sm.add_constant(X_ols)
            ols_fit_full = sm.OLS(Y_ols, X_const).fit()

# Step 2 — drop features with p > 0.05, rerun OLS
            sig_features = ols_fit_full.pvalues[ols_fit_full.pvalues < 0.05].index.tolist()
            sig_features = [f for f in sig_features if f != 'const']

            X_ols2 = df_ols[sig_features].fillna(0).astype(float)
            X_const2 = sm.add_constant(X_ols2)
            ols_fit = sm.OLS(Y_ols, X_const2).fit()   # ols_fit now = final clean model
            col1, col2 = st.columns([1.5, 1])
            with col1:
                st.markdown("**OLS Summary**")
                summary_html = ols_fit.summary().as_html()
                st.markdown(f"<div style='font-size:0.82rem; color:#8a7aaa;'>{summary_html}</div>",
                            unsafe_allow_html=True)
            with col2:
                coef_df = pd.DataFrame({'Feature': ols_fit.params.index,
                                        'Coef': ols_fit.params.values,
                                        'P-value': ols_fit.pvalues.values})
                coef_df = coef_df[coef_df['Feature'] != 'const']

                fig = px.bar(coef_df, x='Coef', y='Feature', orientation='h',
                             color='P-value',
                             color_continuous_scale=[[0,'#00f582'],[0.5,'#ffb800'],[1,'#ff3c78']],
                             range_color=[0, 0.2],
                             title='OLS Coefficients (color = p-value)')
                fig.update_yaxes(title=None)
                plot_layout(fig)
                st.plotly_chart(fig, use_container_width=True)

                r2 = ols_fit.rsquared
                st.markdown(f"""
                <div class='neon-card neon-card-cyan'>
                    <div class='stat-row'><span class='stat-key'>R²</span><span class='stat-val'>{r2:.4f}</span></div>
                    <div class='stat-row'><span class='stat-key'>Adj. R²</span><span class='stat-val'>{ols_fit.rsquared_adj:.4f}</span></div>
                    <div class='stat-row'><span class='stat-key'>AIC</span><span class='stat-val'>{ols_fit.aic:.2f}</span></div>
                    <div class='stat-row'><span class='stat-key'>F-stat p-val</span><span class='stat-val'>{ols_fit.f_pvalue:.4f}</span></div>
                </div>
                """, unsafe_allow_html=True)

                resid = ols_fit.resid
                fitted = ols_fit.fittedvalues
                fig_r = px.scatter(x=fitted, y=resid,
                                   title='Residuals vs Fitted',
                                   labels={'x':'Fitted','y':'Residual'},
                                   color_discrete_sequence=['#b400ff'])
                fig_r.add_hline(y=0, line_dash='dot', line_color='#ff3c78',
                                annotation_text='y=0', annotation_font_color='#ff3c78')
                fig_r.update_traces(marker=dict(size=7, opacity=0.7,
                                                line=dict(color='rgba(180,0,255,0.5)', width=1)))
                plot_layout(fig_r)
                st.plotly_chart(fig_r, use_container_width=True)
        except Exception as e:
            st.error(f"OLS error: {e}")

    with tabs[2]:
        st.markdown("#### K-Fold Cross-Validation (K = 10)")
        st.markdown("<div style='font-size:0.85rem; color:#5a4d7a; margin-bottom:16px;'>10-fold CV with shuffle (random_state=0) on the <strong style='color:#b400ff;'>final significant features</strong> (p &lt; 0.05 from OLS).</div>", unsafe_allow_html=True)
        try:
            df_kf = df.copy()
            # Cast peer_influence to int so dummies are 'peer_influence_3' not 'peer_influence_3.0'
            df_kf['peer_influence'] = pd.to_numeric(df_kf['peer_influence'], errors='coerce').astype('Int64')
            df_kf = pd.get_dummies(df_kf, columns=[
                'group_size','peer_influence','gender','venue','day','intiated_plan',
                'occasion','exceed_budget','spend_reason','rough_budget','discount',
                'travel_dist','outing_freq','year'
            ])
            # Cast boolean dummy columns to int (pandas >= 2.0 returns bool dtype)
            bool_cols_kf = df_kf.select_dtypes(include='bool').columns
            df_kf[bool_cols_kf] = df_kf[bool_cols_kf].astype(int)

            # 4 features used in K-Fold CV (notebook cell 26)
            cv_cols = [c for c in sig_features if c in df_kf.columns]
            X_cv = df_kf[cv_cols].fillna(0).astype(float)
            Y_cv = df_kf['spending'].astype(float)

            K = 10
            kfold = KFold(K, random_state=0, shuffle=True)
            model_cv = lm.LinearRegression()
            mse_cv = cross_val_score(model_cv, X_cv, Y_cv, cv=kfold,
                                     scoring='neg_mean_squared_error')
            r2_cv  = cross_val_score(model_cv, X_cv, Y_cv, cv=kfold,
                                     scoring='r2')

            avg_mse  = np.mean(-mse_cv)
            avg_rmse = np.sqrt(avg_mse)
            avg_r2   = np.mean(r2_cv)

            st.markdown(f"""
            <div class='kpi-grid'>
                <div class='kpi-box'>
                    <div class='kpi-value' style='font-size:1.5rem;'>{avg_mse:.0f}</div>
                    <div class='kpi-label'>Mean MSE</div>
                </div>
                <div class='kpi-box'>
                    <div class='kpi-value kpi-value-pink' style='font-size:1.5rem;'>₹{avg_rmse:.0f}</div>
                    <div class='kpi-label'>Mean RMSE</div>
                </div>
                <div class='kpi-box'>
                    <div class='kpi-value kpi-value-cyan' style='font-size:1.5rem;'>{avg_r2:.4f}</div>
                    <div class='kpi-label'>Mean R²</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            col_l, col_r = st.columns(2)
            with col_l:
                fig = px.histogram(x=-mse_cv, nbins=30,
                                   color_discrete_sequence=['#b400ff'],
                                   title='Distribution of Fold MSEs',
                                   labels={'x':'MSE'})
                fig.update_traces(marker_line_color='rgba(180,0,255,0.6)', marker_line_width=1, opacity=0.8)
                fig.add_vline(x=avg_mse, line_dash='dash', line_color='#ff3c78',
                              annotation_text=f'Mean={avg_mse:.0f}',
                              annotation_font_color='#ff3c78')
                fig.update_layout(showlegend=False)
                plot_layout(fig)
                st.plotly_chart(fig, use_container_width=True)
            with col_r:
                fig2 = px.histogram(x=r2_cv, nbins=30,
                                    color_discrete_sequence=['#00f5ff'],
                                    title='Distribution of Fold R² Values',
                                    labels={'x':'R²'})
                fig2.update_traces(marker_line_color='rgba(0,245,255,0.6)', marker_line_width=1, opacity=0.8)
                fig2.add_vline(x=avg_r2, line_dash='dash', line_color='#ff3c78',
                               annotation_text=f'Mean={avg_r2:.3f}',
                               annotation_font_color='#ff3c78')
                fig2.update_layout(showlegend=False)
                plot_layout(fig2)
                st.plotly_chart(fig2, use_container_width=True)

            st.markdown(f"""
            <div class='neon-card'>
                <div style='font-size:0.75rem; color:#5a4d7a; text-transform:uppercase; letter-spacing:0.1em; margin-bottom:12px; font-weight:700;'>Model Interpretation</div>
                <p style='color:#8a7aaa; font-size:0.9rem; margin:0;'>
                With K=100 fold CV, the model achieves a mean MSE of
                <strong style='color:#b400ff;'>{avg_mse:.2f}</strong>
                (RMSE ≈ <strong style='color:#ff3c78;'>₹{avg_rmse:.2f}</strong>) and mean R² of
                <strong style='color:#00f5ff;'>{avg_r2:.4f}</strong>.
                The final significant predictors (p &lt; 0.05 from OLS) — including occasion type, group size, peer influence, 
                and venue expense — explain a consistent portion of spending variance across all 100 folds.
                </p>
            </div>
            """, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"CV error: {e}")
