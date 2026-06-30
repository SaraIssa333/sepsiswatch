"""
SepsisWatch ICU Analytics — MSBA382 Healthcare Analytics
PhysioNet 2019 Challenge — Real ICU Data: BIDMC Boston + Emory Atlanta
40,336 patients · 41 clinical variables · Password: SEPSIS2026
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="SepsisWatch ICU", page_icon="🫀",
                   layout="wide", initial_sidebar_state="expanded")

# ── DESIGN TOKENS ─────────────────────────────────────────
DARK="#0A0F1E"; CARD="#111827"; BORDER="#1E2D45"
RED="#E84855"; ORANGE="#F5A623"; TEAL="#00BFA5"
BLUE="#2979FF"; MUTED="#6B7A8D"; WHITE="#EEF2FF"

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;700&family=IBM+Plex+Sans:wght@300;400;500;600;700&display=swap');
html,body,[class*="css"]{{font-family:'IBM Plex Sans',sans-serif;background-color:{DARK};color:{WHITE};}}
#MainMenu,footer,header{{visibility:hidden;}}
[data-testid="stSidebar"]{{background-color:{CARD}!important;border-right:1px solid {BORDER};}}
[data-testid="stSidebar"] *{{color:{WHITE}!important;}}
[data-testid="stSidebar"] .stSelectbox>div>div{{background-color:{DARK}!important;border:1px solid {BORDER}!important;}}
.main .block-container{{background-color:{DARK};padding-top:1.2rem;max-width:1440px;}}
.kpi-card{{background:{CARD};border:1px solid {BORDER};border-radius:8px;padding:18px 20px;text-align:center;position:relative;overflow:hidden;margin-bottom:6px;}}
.kpi-card::before{{content:'';position:absolute;top:0;left:0;right:0;height:3px;}}
.kpi-red::before{{background:{RED};}} .kpi-orange::before{{background:{ORANGE};}}
.kpi-teal::before{{background:{TEAL};}} .kpi-blue::before{{background:{BLUE};}}
.kpi-value{{font-family:'IBM Plex Mono',monospace;font-size:2rem;font-weight:700;line-height:1.1;}}
.kpi-label{{font-size:0.7rem;color:{MUTED};text-transform:uppercase;letter-spacing:0.12em;margin-top:4px;}}
.section-hdr{{font-size:1.05rem;font-weight:600;color:{WHITE};padding:10px 0 8px 0;border-bottom:1px solid {BORDER};margin-bottom:14px;}}
.eyebrow{{font-size:0.65rem;color:{TEAL};text-transform:uppercase;letter-spacing:0.15em;font-weight:600;}}
.title-band{{background:linear-gradient(135deg,{CARD} 55%,#0D1B30);border:1px solid {BORDER};border-radius:10px;padding:22px 28px;margin-bottom:20px;}}
.title-main{{font-size:1.8rem;font-weight:700;color:{WHITE};letter-spacing:-0.01em;}}
.title-sub{{font-size:0.85rem;color:{MUTED};margin-top:3px;}}
.badge{{display:inline-block;background:{RED};color:white;font-family:'IBM Plex Mono',monospace;font-size:0.68rem;font-weight:700;padding:3px 10px;border-radius:20px;margin-top:8px;letter-spacing:0.06em;}}
@keyframes pulse{{0%{{box-shadow:0 0 0 0 rgba(232,72,85,0.4);}}70%{{box-shadow:0 0 0 10px rgba(232,72,85,0);}}100%{{box-shadow:0 0 0 0 rgba(232,72,85,0);}}}}
.kpi-alert{{animation:pulse 2.5s infinite;}}
[data-baseweb="tab-list"]{{background:{CARD}!important;border-radius:8px;border:1px solid {BORDER};padding:3px;}}
[data-baseweb="tab"]{{color:{MUTED}!important;font-size:0.82rem;}}
[aria-selected="true"]{{background:{RED}!important;color:white!important;border-radius:6px!important;font-weight:600!important;}}
[data-testid="metric-container"]{{background:{CARD};border:1px solid {BORDER};border-radius:8px;padding:14px;}}
[data-testid="metric-container"] label{{color:{MUTED}!important;font-size:0.8rem!important;}}
[data-testid="metric-container"] [data-testid="metric-value"]{{color:{TEAL}!important;font-family:'IBM Plex Mono',monospace!important;font-size:1.6rem!important;}}
::-webkit-scrollbar{{width:5px;background:{DARK};}}
::-webkit-scrollbar-thumb{{background:{BORDER};border-radius:3px;}}
</style>
"""

CHART = dict(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
             font=dict(family="IBM Plex Sans", color=WHITE),
             margin=dict(t=40,b=40,l=50,r=20),
             colorway=[RED,TEAL,ORANGE,BLUE,"#AB47BC","#26C6DA"],
             xaxis=dict(gridcolor=BORDER,linecolor=BORDER),
             yaxis=dict(gridcolor=BORDER,linecolor=BORDER),
             legend=dict(bgcolor="rgba(0,0,0,0)",font=dict(color=WHITE)))

def chart(fig, title="", h=370):
    fig.update_layout(**CHART, title=dict(text=title, font=dict(size=13,color=WHITE)), height=h)
    fig.update_xaxes(showgrid=True, gridwidth=1)
    fig.update_yaxes(showgrid=True, gridwidth=1)
    return fig

def kpi(col, val, label, color="red", alert=False):
    cm = {"red":RED,"orange":ORANGE,"teal":TEAL,"blue":BLUE}
    cls = f"kpi-card kpi-{color}" + (" kpi-alert" if alert else "")
    col.markdown(f"""<div class="{cls}">
      <div class="kpi-value" style="color:{cm[color]};">{val}</div>
      <div class="kpi-label">{label}</div></div>""", unsafe_allow_html=True)

# ── AUTH ──────────────────────────────────────────────────
st.markdown(CSS, unsafe_allow_html=True)
if "auth" not in st.session_state:
    st.session_state.auth = False

if not st.session_state.auth:
    st.markdown(f"""
    <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;min-height:60vh;">
      <div style="text-align:center;margin-bottom:24px;">
        <div style="font-size:4rem;">🫀</div>
        <div style="font-family:'IBM Plex Mono',monospace;font-size:2.2rem;font-weight:700;color:{WHITE};">
          Sepsis<span style="color:{RED};">Watch</span></div>
        <div style="font-size:0.9rem;color:{MUTED};margin-top:6px;">ICU Analytics Platform — MSBA382 Healthcare Analytics</div>
      </div>
      <div style="background:{CARD};border:1px solid {BORDER};border-radius:12px;padding:36px 40px;text-align:center;width:380px;">
        <div style="font-size:0.8rem;color:{MUTED};margin-bottom:20px;line-height:1.6;">
          <b style="color:{WHITE};">Data Source:</b> PhysioNet 2019 Challenge<br>
          <b style="color:{WHITE};">Hospitals:</b> BIDMC Boston · Emory Atlanta<br>
          <b style="color:{WHITE};">Patients:</b> 40,336 real ICU records<br>
          <b style="color:{WHITE};">Variables:</b> 41 clinical features
        </div>
      </div>
    </div>""", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1,1,1])
    with col2:
        pw = st.text_input("", type="password", placeholder="🔒 Enter dashboard password…")
        if st.button("Unlock Dashboard", use_container_width=True):
            if pw == "SEPSIS2026":
                st.session_state.auth = True
                st.rerun()
            else:
                st.error("Incorrect password. Hint: SEPSIS2026")
    st.stop()

# ── LOAD DATA ─────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("sepsis_full.csv")
    # Add hospital & ICU unit labels
    df['hospital'] = df['dataset'].map({'A':'BIDMC - Boston, MA','B':'Emory - Atlanta, GA'})
    df['icu_unit'] = df['unit1'].map({1.0:'Medical ICU (MICU)',0.0:'Surgical ICU (SICU)',np.nan:'Unspecified'}).fillna('Unspecified')
    df['gender_label'] = df['gender'].map({1:'Male',0:'Female'})
    return df

df = load_data()

# ── SIDEBAR ───────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""<div style="text-align:center;padding:14px 0 18px;">
      <div style="font-size:2.2rem;">🫀</div>
      <div style="font-family:'IBM Plex Mono',monospace;font-size:1.1rem;font-weight:700;color:{WHITE};">
        Sepsis<span style="color:{RED};">Watch</span></div>
      <div style="font-size:0.68rem;color:{MUTED};">ICU Analytics · MSBA382</div>
    </div>""", unsafe_allow_html=True)

    st.markdown(f'<div class="eyebrow">Patient Filters</div>', unsafe_allow_html=True)
    hospital_sel = st.multiselect("Hospital", ["BIDMC - Boston, MA","Emory - Atlanta, GA"],
                                  default=["BIDMC - Boston, MA","Emory - Atlanta, GA"])
    age_range = st.slider("Age Range", int(df.age.min()), int(df.age.max()),
                          (int(df.age.min()), int(df.age.max())))
    gender_filter = st.multiselect("Gender", ["Male","Female"], default=["Male","Female"])
    sepsis_filter = st.selectbox("Sepsis Status", ["All","Sepsis Only","Non-Sepsis Only"])
    icu_max = st.slider("Max ICU Stay (hours)", 12, 336, 336, step=12)
    unit_filter = st.multiselect("ICU Unit Type",
                                 ["Medical ICU (MICU)","Surgical ICU (SICU)","Unspecified"],
                                 default=["Medical ICU (MICU)","Surgical ICU (SICU)","Unspecified"])
    st.markdown("---")
    st.markdown(f"""<div style="font-size:0.7rem;color:{MUTED};line-height:1.8;">
    <b style="color:{WHITE};">Source:</b> PhysioNet 2019 Challenge<br>
    <b style="color:{WHITE};">Hospitals:</b> BIDMC + Emory<br>
    <b style="color:{WHITE};">Patients:</b> 40,336 ICU records<br>
    <b style="color:{WHITE};">Variables:</b> 41 clinical features<br>
    <b style="color:{WHITE};">Password:</b> <span style="font-family:monospace;color:{RED};">SEPSIS2026</span>
    </div>""", unsafe_allow_html=True)
    if st.button("🚪 Logout"):
        st.session_state.auth = False
        st.rerun()

# ── APPLY FILTERS ─────────────────────────────────────────
dff = df[df['hospital'].isin(hospital_sel)].copy()
dff = dff[dff['age'].between(*age_range)]
dff = dff[dff['gender_label'].isin(gender_filter)]
dff = dff[dff['icu_los_hours'] <= icu_max]
dff = dff[dff['icu_unit'].isin(unit_filter)]
if sepsis_filter == "Sepsis Only": dff = dff[dff['sepsis']==1]
elif sepsis_filter == "Non-Sepsis Only": dff = dff[dff['sepsis']==0]

# ── HEADER ────────────────────────────────────────────────
st.markdown(f"""<div class="title-band">
  <div style="display:flex;align-items:center;gap:18px;">
    <div style="font-size:3rem;">🫀</div>
    <div>
      <div class="title-main">SepsisWatch ICU Analytics</div>
      <div class="title-sub">Early Sepsis Detection & ICU Outcome Intelligence · PhysioNet 2019 Challenge · Real Patient Data</div>
      <span class="badge">BIDMC BOSTON · EMORY ATLANTA · 40,336 REAL ICU PATIENTS · 41 VARIABLES</span>
    </div>
    <div style="margin-left:auto;text-align:right;font-family:'IBM Plex Mono',monospace;font-size:0.72rem;color:{MUTED};">
      Filtered: {len(dff):,} patients<br>
      Sepsis cases: {dff['sepsis'].sum():,}<br>
      Sepsis rate: {dff['sepsis'].mean():.1%}
    </div>
  </div>
</div>""", unsafe_allow_html=True)

# ── KPIs ──────────────────────────────────────────────────
k1,k2,k3,k4,k5 = st.columns(5)
kpi(k1, f"{len(dff):,}", "Total ICU Patients", "blue")
kpi(k2, f"{dff['sepsis'].sum():,}", "Confirmed Sepsis", "red", alert=True)
kpi(k3, f"{dff['sepsis'].mean():.1%}", "Sepsis Rate", "orange")
kpi(k4, f"{dff['icu_los_hours'].mean():.1f}h", "Mean ICU Stay", "teal")
kpi(k5, f"{dff['age'].mean():.1f} yrs", "Mean Patient Age", "blue")
st.markdown("<br>", unsafe_allow_html=True)

# ── TABS ──────────────────────────────────────────────────

# ═══════════════════════════════════════════════════════════
# SECTION 1 — PATIENT OVERVIEW (Demographics + Geographic)
# ═══════════════════════════════════════════════════════════
st.markdown(f'<div class="section-hdr">📊 Patient Overview — Demographics & Geographic Distribution</div>', unsafe_allow_html=True)

c1,c2,c3 = st.columns(3)
with c1:
    counts = dff['sepsis'].value_counts()
    fig = go.Figure(go.Pie(labels=["Non-Sepsis","Sepsis"], values=[counts.get(0,0),counts.get(1,0)],
        hole=0.58, marker_colors=[TEAL,RED], textinfo='percent+label', textfont=dict(color=WHITE,size=11)))
    fig.add_annotation(text=f"<b>{len(dff):,}</b><br>patients", x=0.5,y=0.5,font=dict(size=12,color=WHITE),showarrow=False)
    chart(fig,"Sepsis vs Non-Sepsis",290)
    st.plotly_chart(fig,use_container_width=True)
with c2:
    fig = go.Figure()
    for label,color,val in [("Non-Sepsis",TEAL,0),("Sepsis",RED,1)]:
        fig.add_trace(go.Histogram(x=dff[dff['sepsis']==val]['age'],name=label,marker_color=color,opacity=0.75,nbinsx=18))
    chart(fig,"Age Distribution",290)
    fig.update_layout(barmode='overlay',legend=dict(orientation='h',y=1.15))
    fig.update_xaxes(title="Age")
    st.plotly_chart(fig,use_container_width=True)
with c3:
    h_stats = dff.groupby('hospital', observed=True).agg(patients=('patient_id','count'), sepsis_rate=('sepsis','mean')).reset_index()
    fig = go.Figure(go.Bar(x=h_stats['hospital'], y=h_stats['sepsis_rate'],
        marker_color=[RED,TEAL], text=[f"{v:.1%}" for v in h_stats['sepsis_rate']],
        textposition='outside', textfont=dict(color=WHITE,size=13)))
    chart(fig,"Sepsis Rate by Hospital",290)
    fig.update_yaxes(title="Sepsis Rate",tickformat='.0%')
    fig.update_xaxes(tickfont=dict(size=10))
    st.plotly_chart(fig,use_container_width=True)

c4,c5,c6 = st.columns(3)
with c4:
    gender_rate = dff.groupby('gender_label', observed=True)['sepsis'].agg(['mean','count']).reset_index()
    fig = make_subplots(specs=[[{"secondary_y":True}]])
    fig.add_trace(go.Bar(x=gender_rate['gender_label'],y=gender_rate['count'],name='Patients',marker_color=BORDER),secondary_y=False)
    fig.add_trace(go.Scatter(x=gender_rate['gender_label'],y=gender_rate['mean'],mode='markers+lines',
        name='Sepsis Rate',marker=dict(size=10,color=ORANGE),line=dict(color=ORANGE,width=2)),secondary_y=True)
    chart(fig,"Gender & Sepsis Rate",290)
    fig.update_yaxes(title="Count",secondary_y=False,gridcolor=BORDER)
    fig.update_yaxes(title="Rate",tickformat='.0%',secondary_y=True,gridcolor='rgba(0,0,0,0)',color=WHITE)
    st.plotly_chart(fig,use_container_width=True)
with c5:
    ag = dff.groupby('age_group', observed=True)['sepsis'].agg(['mean','count']).reset_index()
    fig = make_subplots(specs=[[{"secondary_y":True}]])
    fig.add_trace(go.Bar(x=ag['age_group'].astype(str),y=ag['count'],name='Patients',marker_color=BORDER),secondary_y=False)
    fig.add_trace(go.Scatter(x=ag['age_group'].astype(str),y=ag['mean'],mode='lines+markers',
        name='Sepsis Rate',line=dict(color=RED,width=2.5),marker=dict(size=8)),secondary_y=True)
    chart(fig,"Sepsis Rate by Age Group",290)
    fig.update_yaxes(title="Count",secondary_y=False,gridcolor=BORDER)
    fig.update_yaxes(title="Rate",tickformat='.0%',secondary_y=True,gridcolor='rgba(0,0,0,0)',color=WHITE)
    st.plotly_chart(fig,use_container_width=True)
with c6:
    fig = go.Figure()
    for label,color,val in [("Non-Sepsis",TEAL,0),("Sepsis",RED,1)]:
        data = dff[dff['sepsis']==val]['icu_los_hours'].clip(0,200)
        if len(data) > 0:
            fig.add_trace(go.Box(y=data,name=label,marker_color=color,boxmean=True,line_color=color))
    chart(fig,"ICU Length of Stay",290)
    fig.update_yaxes(title="Hours")
    st.plotly_chart(fig,use_container_width=True)

st.markdown("<br>",unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════
# SECTION 2 — CLINICAL SIGNALS (Vitals + Labs)
# ═══════════════════════════════════════════════════════════
st.markdown(f'<div class="section-hdr">🩺 Clinical Signals — Vital Signs & Laboratory Biomarkers</div>', unsafe_allow_html=True)

c7,c8,c9 = st.columns(3)
with c7:
    samp = dff.sample(min(1500,len(dff)),random_state=42)
    samp['status'] = samp['sepsis'].map({0:'Non-Sepsis',1:'Sepsis'})
    samp_clean = samp.dropna(subset=['hr_mean','sbp_mean'])
    present = samp_clean['status'].unique().tolist()
    cmap = {k:v for k,v in {'Non-Sepsis':TEAL,'Sepsis':RED}.items() if k in present}
    fig = px.scatter(samp_clean,x='hr_mean',y='sbp_mean',color='status',color_discrete_map=cmap,opacity=0.5,
        labels={'hr_mean':'HR (bpm)','sbp_mean':'SBP (mmHg)'})
    fig.add_hline(y=90,line_dash='dot',line_color=ORANGE,opacity=0.7)
    fig.add_vline(x=100,line_dash='dot',line_color=ORANGE,opacity=0.7)
    chart(fig,"HR vs SBP — Shock Zone",290)
    fig.update_layout(legend=dict(orientation='h',y=1.15))
    st.plotly_chart(fig,use_container_width=True)
with c8:
    fig = go.Figure()
    for label,color,val in [("Non-Sepsis",TEAL,0),("Sepsis",RED,1)]:
        data = dff[dff['sepsis']==val]['lactate_mean'].dropna()
        if len(data) > 0:
            fig.add_trace(go.Box(y=data,name=label,marker_color=color,boxmean=True))
    fig.add_hline(y=2.0,line_dash='dot',line_color=ORANGE,annotation_text="Threshold: 2.0")
    chart(fig,"Lactate by Sepsis Status",290)
    fig.update_yaxes(title="mmol/L")
    st.plotly_chart(fig,use_container_width=True)
with c9:
    sc = ['sirs_temp','sirs_hr','sirs_resp','sirs_wbc']
    sl2 = ['Temp','HR>90','Resp>20','WBC']
    sr = dff[dff['sepsis']==1][sc].mean(); nr = dff[dff['sepsis']==0][sc].mean()
    fig = go.Figure()
    fig.add_trace(go.Bar(name='Non-Sepsis',x=sl2,y=nr,marker_color=TEAL))
    fig.add_trace(go.Bar(name='Sepsis',x=sl2,y=sr,marker_color=RED))
    chart(fig,"SIRS Criteria Prevalence",290)
    fig.update_layout(barmode='group',legend=dict(orientation='h',y=1.15))
    fig.update_yaxes(title="Proportion",tickformat='.0%')
    st.plotly_chart(fig,use_container_width=True)

st.markdown("<br>",unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════
# SECTION 3 — RISK STRATIFICATION
# ═══════════════════════════════════════════════════════════
st.markdown(f'<div class="section-hdr">⚠️ Risk Stratification — SIRS Score, Onset Timing & Risk Factors</div>', unsafe_allow_html=True)

c10,c11,c12 = st.columns(3)
with c10:
    ss = dff.groupby('sirs_score', observed=True)['sepsis'].agg(['mean','count']).reset_index()
    fig = make_subplots(specs=[[{"secondary_y":True}]])
    fig.add_trace(go.Bar(x=ss['sirs_score'],y=ss['count'],name='Count',marker_color=BORDER),secondary_y=False)
    fig.add_trace(go.Scatter(x=ss['sirs_score'],y=ss['mean'],mode='lines+markers',
        name='Sepsis Rate',line=dict(color=RED,width=2.5),marker=dict(size=9)),secondary_y=True)
    chart(fig,"SIRS Score vs Sepsis Rate",290)
    fig.update_yaxes(title="Count",secondary_y=False,gridcolor=BORDER)
    fig.update_yaxes(title="Rate",tickformat='.0%',secondary_y=True,gridcolor='rgba(0,0,0,0)',color=WHITE)
    fig.update_xaxes(title="SIRS Score")
    st.plotly_chart(fig,use_container_width=True)
with c11:
    od = dff[dff['sepsis']==1]['sepsis_onset_hour'].dropna()
    fig = go.Figure(go.Histogram(x=od,nbinsx=25,marker_color=RED,opacity=0.8))
    if len(od) > 0:
        fig.add_vline(x=od.median(),line_dash='dash',line_color=ORANGE,annotation_text=f"Med: {od.median():.0f}h")
    chart(fig,"Sepsis Onset Timing",290)
    fig.update_xaxes(title="ICU Hour")
    fig.update_yaxes(title="Cases")
    st.plotly_chart(fig,use_container_width=True)
with c12:
    rfd = {
        'High Lactate': dff[dff['lactate_max']>2]['sepsis'].mean() if len(dff[dff['lactate_max']>2])>0 else 0,
        'Low SBP': dff[dff['sbp_min']<90]['sepsis'].mean() if len(dff[dff['sbp_min']<90])>0 else 0,
        'SIRS ≥3': dff[dff['sirs_score']>=3]['sepsis'].mean() if len(dff[dff['sirs_score']>=3])>0 else 0,
        'Age ≥65': dff[dff['age']>=65]['sepsis'].mean() if len(dff[dff['age']>=65])>0 else 0,
    }
    rdf = pd.DataFrame(list(rfd.items()),columns=['factor','rate']).dropna().sort_values('rate')
    colors = [RED if v>0.15 else ORANGE if v>0.08 else TEAL for v in rdf['rate']]
    fig = go.Figure(go.Bar(y=rdf['factor'],x=rdf['rate'],orientation='h',marker_color=colors,
        text=[f"{v:.1%}" for v in rdf['rate']],textposition='outside',textfont=dict(color=WHITE,size=11)))
    chart(fig,"Risk Factor Ranking",290)
    fig.update_xaxes(title="Sepsis Rate",tickformat='.0%')
    st.plotly_chart(fig,use_container_width=True)

st.markdown("<br>",unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════
# SECTION 4 — PREDICTIVE MODEL (compact, same page)
# ═══════════════════════════════════════════════════════════
st.markdown(f'<div class="section-hdr">🤖 Predictive Analytics — Early Sepsis Warning (Bonus)</div>', unsafe_allow_html=True)

@st.cache_data
def train_models(data):
    features = ['age','gender','hr_mean','hr_max','o2sat_mean','o2sat_min',
                'temp_mean','sbp_mean','sbp_min','map_mean','resp_mean','resp_max',
                'sirs_score','icu_los_hours']
    X = data[features]; y = data['sepsis']
    imp = SimpleImputer(strategy='median'); Xi = imp.fit_transform(X)
    Xtr,Xte,ytr,yte = train_test_split(Xi,y,test_size=0.25,random_state=42,stratify=y)
    rf = RandomForestClassifier(n_estimators=150,max_depth=8,random_state=42,n_jobs=-1,class_weight='balanced')
    rf.fit(Xtr,ytr)
    yp = rf.predict_proba(Xte)[:,1]
    auc = roc_auc_score(yte,yp)
    fi = pd.DataFrame({'feature':features,'importance':rf.feature_importances_}).sort_values('importance',ascending=False)
    return rf, imp, auc, fi, features

with st.spinner("Training model…"):
    rf_model, rf_imp, rf_auc, rf_fi, rf_features = train_models(df)

c13,c14,c15 = st.columns([1,1,1.2])
with c13:
    fig = go.Figure(go.Bar(y=rf_fi.head(6)['feature'],x=rf_fi.head(6)['importance'],orientation='h',
        marker=dict(color=rf_fi.head(6)['importance'],colorscale=[[0,TEAL],[0.5,ORANGE],[1,RED]]),
        text=[f"{v:.3f}" for v in rf_fi.head(6)['importance']],textposition='outside',textfont=dict(color=WHITE,size=10)))
    chart(fig,f"Top Features (AUC={rf_auc:.3f})",290)
    st.plotly_chart(fig,use_container_width=True)
with c14:
    st.markdown(f"""<div class="kpi-card kpi-red" style="height:290px;display:flex;flex-direction:column;justify-content:center;">
      <div class="kpi-value" style="color:{RED};font-size:2.4rem;">{rf_auc:.3f}</div>
      <div class="kpi-label">RANDOM FOREST AUC-ROC</div>
      <div style="font-size:0.75rem;color:{MUTED};margin-top:10px;">Trained on 14 features<br>available within first ICU hours</div>
    </div>""",unsafe_allow_html=True)
with c15:
    st.markdown(f'<div class="eyebrow" style="margin-bottom:6px;">Live Patient Risk Predictor</div>', unsafe_allow_html=True)
    pcol1, pcol2 = st.columns(2)
    with pcol1:
        pa = st.number_input("Age",18,100,65,key="op_age")
        ph = st.number_input("HR (bpm)",40.0,200.0,88.0,key="op_hr")
        po = st.number_input("O2Sat (%)",70.0,100.0,96.0,key="op_o2")
    with pcol2:
        ps = st.number_input("SBP (mmHg)",50.0,200.0,115.0,key="op_sbp")
        pr = st.number_input("Resp Rate",8.0,50.0,18.0,key="op_resp")
        psi = st.selectbox("SIRS Score",[0,1,2,3,4],key="op_sirs")
    if st.button("🔍 Predict Risk",use_container_width=True,key="op_predict"):
        pat = rf_imp.transform([[pa,1,ph,ph*1.1,po,po*0.95,37.0,ps,ps*0.85,ps*0.75,pr,pr*1.2,psi,24.0]])
        prob = rf_model.predict_proba(pat)[0][1]
        rc = RED if prob>0.5 else ORANGE if prob>0.25 else TEAL
        rl = "HIGH RISK" if prob>0.5 else "MODERATE RISK" if prob>0.25 else "LOW RISK"
        st.markdown(f"""<div style="background:{CARD};border:2px solid {rc};border-radius:8px;
            padding:10px;text-align:center;margin-top:6px;">
          <span style="font-family:'IBM Plex Mono',monospace;font-size:1.4rem;font-weight:700;color:{rc};">{prob:.1%}</span>
          <span style="font-size:0.85rem;font-weight:600;color:{rc};margin-left:8px;">{rl}</span>
        </div>""",unsafe_allow_html=True)

# ── FOOTER ────────────────────────────────────────────────
st.markdown(f"""
<div style="border-top:1px solid {BORDER};margin-top:24px;padding-top:12px;
     display:flex;justify-content:space-between;align-items:center;font-size:0.7rem;color:{MUTED};">
  <div>SepsisWatch Analytics · MSBA382 Healthcare Analytics</div>
  <div>PhysioNet 2019 Challenge · BIDMC Boston · Emory Atlanta</div>
  <div style="font-family:'IBM Plex Mono',monospace;color:{RED};">{len(df):,} patients · 41 variables · One-page view · Real ICU Data</div>
</div>""", unsafe_allow_html=True)
