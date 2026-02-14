import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans # <-- NEW AI IMPORT

# ----------------------------------------
# 1. Page Configuration & Custom CSS
# ----------------------------------------
st.set_page_config(page_title="Surya Chemicals | Skill Analytics", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    div[data-testid="metric-container"] {
        background-color: #FFFFFF;
        border: 1px solid #E0E4E8;
        padding: 20px;
        border-radius: 8px;
        box-shadow: 0px 2px 6px rgba(0, 0, 0, 0.04);
        border-left: 5px solid #2A59D4;
    }
    .problem-box {
        background-color: #F8F9FA;
        border-left: 5px solid #E63946;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# ----------------------------------------
# 2. High-Performance Data Integration
# ----------------------------------------
@st.cache_data
def load_data():
    file_path = 'Surya_Chemicals_Data_Final.xlsx'
    df_op = pd.read_excel(file_path, sheet_name='Operator_Master')
    df_skill = pd.read_excel(file_path, sheet_name='Skill_Assessment')
    df_perf = pd.read_excel(file_path, sheet_name='Operational_Performance')
    
    df_merged = pd.merge(df_op, df_skill, on='operator_id', how='inner')
    df_merged = pd.merge(df_merged, df_perf, on='operator_id', how='inner')
    return df_merged

try:
    df = load_data()
except FileNotFoundError:
    st.error("⚠️ Data file not found! Ensure 'Surya_Chemicals_Data_Final.xlsx' is in the directory.")
    st.stop()

# Universal Chart Theming 
chart_theme = dict(template="plotly_white", margin=dict(l=20, r=20, t=40, b=20), plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')

def clean_axes(fig):
    fig.update_xaxes(showgrid=False, title_font=dict(size=13, color='#555'))
    fig.update_yaxes(showgrid=True, gridcolor='#EFEFEF', title_font=dict(size=13, color='#555'))
    return fig

# ----------------------------------------
# 3. Premium Sidebar & Filters
# ----------------------------------------
st.sidebar.markdown("## ⚡ Surya Chemicals")
st.sidebar.markdown("### Decision Cockpit")
st.sidebar.markdown("---")

page = st.sidebar.radio("Navigation", [
    "Business Problem Context", 
    "Executive Overview", 
    "Skill Matrix Analysis", 
    "Safety & Risk Profiling"
])
st.sidebar.markdown("---")

st.sidebar.subheader("Global Filters")
selected_dept = st.sidebar.multiselect("Department", df['department'].unique(), default=df['department'].unique())
selected_shift = st.sidebar.multiselect("Shift Type", df['shift_type'].unique(), default=df['shift_type'].unique())
exp_range = st.sidebar.slider("Experience (Years)", int(df['experience_years'].min()), int(df['experience_years'].max()), (0, 25))

filtered_df = df[
    (df['department'].isin(selected_dept)) & 
    (df['shift_type'].isin(selected_shift)) &
    (df['experience_years'].between(exp_range[0], exp_range[1]))
]

# ==========================================
# PAGE 1: BUSINESS PROBLEM CONTEXT
# ==========================================
if page == "Business Problem Context":
    st.title("Business Problem Definitions & Project Scope")
    st.markdown("This dashboard transitions Surya Chemicals from manual, periodic evaluations to AI-driven continuous skill intelligence to address three critical operational challenges.")
    st.markdown("---")

    # Problem 1
    st.markdown("""
    <div class="problem-box">
        <h3 style='margin-top:0px; color:#1E1E1E;'>Problem 1: Inconsistent Operator Competency Levels</h3>
        <p>Current operator evaluations are manual and supervisor-dependent, leading to vast skill discrepancies across shifts and departments.</p>
        <b>Target KPIs:</b> Skill Score Standardization Index ≥ 90% | 20% reduction in operator-related operational deviations.
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("#### Visual Evidence: Variance in Operator Skill Scores")
    fig_p1 = px.violin(filtered_df, x="department", y="final_skill_score", color="department", box=True, points=False, color_discrete_sequence=px.colors.qualitative.Pastel)
    fig_p1.update_layout(**chart_theme, yaxis_title="Final Skill Score", xaxis_title="", showlegend=False)
    fig_p1 = clean_axes(fig_p1)
    st.plotly_chart(fig_p1, width='stretch')
    st.markdown("<br>", unsafe_allow_html=True)

    # Problem 2
    st.markdown("""
    <div class="problem-box" style="border-left-color: #F4A261;">
        <h3 style='margin-top:0px; color:#1E1E1E;'>Problem 2: High Human-Error-Based Incidents</h3>
        <p>A lack of real-time competency tracking has increased operational risk, leading to safety incidents and unplanned shutdowns.</p>
        <b>Target KPIs:</b> 25% reduction in operator error-related near misses | 15% reduction in unplanned shutdowns linked to human intervention.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### Visual Evidence: Operational Risk Exposure by Shift")
    risk_df = filtered_df.groupby(['shift_type', 'department'])[['sop_deviation_count', 'near_miss_count']].sum().reset_index()
    risk_df['Total Incidents'] = risk_df['sop_deviation_count'] + risk_df['near_miss_count']
    fig_p2 = px.bar(risk_df, x="shift_type", y="Total Incidents", color="department", barmode="stack", text="Total Incidents")
    fig_p2.update_layout(**chart_theme, xaxis_title="Shift Type", yaxis_title="Total Logged Incidents")
    fig_p2 = clean_axes(fig_p2)
    st.plotly_chart(fig_p2, width='stretch')
    st.markdown("<br>", unsafe_allow_html=True)

    # Problem 3 (UPDATED WITH AI CLUSTERING)
    st.markdown("""
    <div class="problem-box" style="border-left-color: #2A9D8F;">
        <h3 style='margin-top:0px; color:#1E1E1E;'>Problem 3: Ineffective Targeted Training Programs</h3>
        <p>Training interventions are currently reactive rather than proactive. Operators are assigned training hours without data confirming it addresses their specific skill gaps.</p>
        <b>Target KPIs:</b> 30% improvement in post-training assessment scores | 20% reduction in time-to-competency for new operators.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### AI Model Evidence: Skill Gap Clustering (K-Means)")
    
    # ---------------------------------------------------------
    # Perform K-Means Clustering for Skill Segmentation
    # ---------------------------------------------------------
    cluster_features = filtered_df[['training_hours', 'final_skill_score']].copy()
    
    if len(cluster_features) > 3:
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        filtered_df['Skill_Segment'] = kmeans.fit_predict(cluster_features)
        
        # Categorize the segments
        filtered_df['Skill_Segment'] = filtered_df['Skill_Segment'].astype(str).map(
            {'0': 'Segment A (Routine)', '1': 'Segment B (At-Risk)', '2': 'Segment C (Optimized)'}
        )
        
        fig_p3 = px.scatter(filtered_df, x="training_hours", y="final_skill_score", 
                            color="Skill_Segment", opacity=0.5, # Lowered opacity to reduce dot clutter
                            hover_data=['Name', 'department'],
                            color_discrete_sequence=px.colors.qualitative.Safe)
        
        # Add the Centroids (Large black X marks) so the eye focuses on the groups, not the dots
        centroids = kmeans.cluster_centers_
        fig_p3.add_trace(go.Scatter(x=centroids[:, 0], y=centroids[:, 1], mode='markers',
                                    marker=dict(color='black', symbol='x', size=14, line=dict(width=2, color='white')),
                                    name='Cluster Center', hoverinfo='skip'))

        fig_p3.update_layout(**chart_theme, xaxis_title="Total Training Hours Logged", yaxis_title="Final Assessment Score")
        fig_p3 = clean_axes(fig_p3)
        st.plotly_chart(fig_p3, width='stretch')
        
        st.info("💡 **AI Insight:** The K-Means algorithm above segments the 1,000 operators into 3 distinct behavioral clusters based on their training volume vs. actual skill. The large 'X' marks the center of each segment. This proves we need a **Training Recommendation Engine** rather than just blindly assigning hours.")
    else:
        st.warning("Not enough data points to perform clustering with the current filters.")

# ==========================================
# PAGE 2: EXECUTIVE OVERVIEW
# ==========================================
elif page == "Executive Overview":
    st.title("Executive Overview")
    st.markdown("High-level operational performance and competency tracking.")
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Avg Competency Score", f"{filtered_df['final_skill_score'].mean():.1f}/100")
    c2.metric("Avg Turbine Efficiency", f"{filtered_df['turbine_efficiency'].mean():.1f}%")
    c3.metric("Total SOP Deviations", int(filtered_df['sop_deviation_count'].sum()))
    c4.metric("Total Near Misses", int(filtered_df['near_miss_count'].sum()))
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Average Competency by Department")
        agg_comp = filtered_df.groupby('department', as_index=False)['final_skill_score'].mean().sort_values('final_skill_score', ascending=True)
        fig1 = px.bar(agg_comp, x="final_skill_score", y="department", orientation='h', text_auto='.1f', color='final_skill_score', color_continuous_scale='Blues')
        fig1.update_layout(**chart_theme, coloraxis_showscale=False, xaxis_title="Avg Skill Score", yaxis_title="")
        fig1 = clean_axes(fig1)
        st.plotly_chart(fig1, width='stretch')
        
    with col2:
        st.markdown("#### Operational Efficiency by Shift")
        fig2 = px.histogram(filtered_df, x="shift_type", y="turbine_efficiency", color="department", barmode="group", histfunc="avg", text_auto='.1f', color_discrete_sequence=px.colors.qualitative.Pastel)
        fig2.update_layout(**chart_theme, xaxis_title="Shift Type", yaxis_title="Avg Turbine Efficiency (%)", legend_title="Dept")
        fig2 = clean_axes(fig2)
        fig2.update_yaxes(range=[80, 100]) 
        st.plotly_chart(fig2, width='stretch')

# ==========================================
# PAGE 3: SKILL MATRIX ANALYSIS
# ==========================================
elif page == "Skill Matrix Analysis":
    st.title("Skill Matrix Analysis")
    st.markdown("Granular breakdown of operator theoretical and practical assessments.")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("#### Theory vs. Simulation by Certification")
        melted_skills = filtered_df.melt(id_vars=['certification_level'], value_vars=['theory_score', 'simulation_score'], var_name='Assessment_Type', value_name='Score')
        melted_skills['Assessment_Type'] = melted_skills['Assessment_Type'].str.replace('_score', '').str.title()
        fig3 = px.histogram(melted_skills, x='certification_level', y='Score', color='Assessment_Type', barmode='group', histfunc='avg', text_auto='.1f', color_discrete_sequence=['#2A59D4', '#85A5FF'], category_orders={"certification_level": ["Level 1", "Level 2", "Level 3"]})
        fig3.update_layout(**chart_theme, xaxis_title="Certification Level", yaxis_title="Average Score", legend_title="Test Type")
        fig3 = clean_axes(fig3)
        fig3.update_yaxes(range=[50, 100])
        st.plotly_chart(fig3, width='stretch')
        
    with col2:
        st.markdown("#### Certification Distribution")
        cert_counts = filtered_df['certification_level'].value_counts().reset_index()
        cert_counts.columns = ['Level', 'Count']
        fig4 = px.pie(cert_counts, names="Level", values="Count", hole=0.5, color_discrete_sequence=px.colors.sequential.Blues_r)
        fig4.update_traces(textposition='inside', textinfo='percent+label')
        fig4.update_layout(**chart_theme, showlegend=False)
        st.plotly_chart(fig4, width='stretch')
        
    st.markdown("#### Operator Competency Master List")
    st.dataframe(filtered_df[['operator_id', 'Name', 'department', 'experience_years', 'certification_level', 'theory_score', 'simulation_score', 'final_skill_score']], width='stretch')

# ==========================================
# PAGE 4: SAFETY & RISK PROFILING
# ==========================================
elif page == "Safety & Risk Profiling":
    st.title("Safety & Risk Profiling")
    st.markdown("Correlating behavioral metrics with operational safety incidents.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Reaction Time Impact on Near Misses")
        agg_reaction = filtered_df.groupby('near_miss_count', as_index=False)['reaction_time_sec'].mean()
        fig5 = px.bar(agg_reaction, x="near_miss_count", y="reaction_time_sec", text_auto='.2f', color="near_miss_count", color_continuous_scale="Reds")
        fig5.update_layout(**chart_theme, xaxis_title="Total Near Misses Logged", yaxis_title="Avg Reaction Time (Seconds)", coloraxis_showscale=False)
        fig5.update_xaxes(type='category') 
        fig5 = clean_axes(fig5)
        st.plotly_chart(fig5, width='stretch')
        
    with col2:
        st.markdown("#### Heatmap: Deviations by Shift & Department")
        pivot_df = filtered_df.pivot_table(values='sop_deviation_count', index='department', columns='shift_type', aggfunc='sum')
        fig6 = px.imshow(pivot_df, text_auto=True, color_continuous_scale='Reds', aspect="auto")
        fig6.update_layout(**chart_theme, xaxis_title="Shift Type", yaxis_title="")
        st.plotly_chart(fig6, width='stretch')