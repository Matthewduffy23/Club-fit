# app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import math

st.set_page_config(page_title="Club Fit", layout="wide")
st.title("🏟️ Club Fit Finder")

# ---------- Load data ----------
@st.cache_data
def load_df_from_repo(csv_name: str = "WORLDJUNE25.csv"):
    p = Path(__file__).with_name(csv_name)
    if p.exists():
        return pd.read_csv(p)
    return None

df = load_df_from_repo()
if df is None:
    uploaded = st.file_uploader("Upload WORLDJUNE25.csv", type=["csv"])
    if not uploaded:
        st.warning("Add WORLDJUNE25.csv to your repo root or upload it here.")
        st.stop()
    df = pd.read_csv(uploaded)

# ---------- Config ----------
included_leagues = [
    'England 1.', 'England 2.', 'England 3.', 'England 4.', 'England 5.',
    'England 6.', 'England 7.', 'England 8.', 'England 9.', 'England 10.',
    'Albania 1.', 'Algeria 1.', 'Andorra 1.', 'Argentina 1.', 'Armenia 1.',
    'Australia 1.', 'Austria 1.', 'Austria 2.', 'Azerbaijan 1.', 'Belgium 1.',
    'Belgium 2.', 'Bolivia 1.', 'Bosnia 1.', 'Brazil 1.', 'Brazil 2.', 'Brazil 3.',
    'Bulgaria 1.', 'Canada 1.', 'Chile 1.', 'Colombia 1.', 'Costa Rica 1.',
    'Croatia 1.', 'Cyprus 1.', 'Czech 1.', 'Czech 2.', 'Denmark 1.', 'Denmark 2.',
    'Ecuador 1.', 'Egypt 1.', 'Estonia 1.', 'Finland 1.', 'France 1.', 'France 2.',
    'France 3.', 'Georgia 1.', 'Germany 1.', 'Germany 2.', 'Germany 3.', 'Germany 4.',
    'Greece 1.', 'Hungary 1.', 'Iceland 1.', 'Israel 1.', 'Israel 2.', 'Italy 1.',
    'Italy 2.', 'Italy 3.', 'Japan 1.', 'Japan 2.', 'Kazakhstan 1.', 'Korea 1.',
    'Latvia 1.', 'Lithuania 1.', 'Malta 1.', 'Mexico 1.', 'Moldova 1.', 'Morocco 1.',
    'Netherlands 1.', 'Netherlands 2.', 'North Macedonia 1.', 'Northern Ireland 1.',
    'Norway 1.', 'Norway 2.', 'Paraguay 1.', 'Peru 1.', 'Poland 1.', 'Poland 2.',
    'Portugal 1.', 'Portugal 2.', 'Portugal 3.', 'Qatar 1.', 'Ireland 1.', 'Romania 1.',
    'Russia 1.', 'Saudi 1.', 'Scotland 1.', 'Scotland 2.', 'Scotland 3.', 'Serbia 1.',
    'Serbia 2.', 'Slovakia 1.', 'Slovakia 2.', 'Slovenia 1.', 'Slovenia 2.', 'South Africa 1.',
    'Spain 1.', 'Spain 2.', 'Spain 3.', 'Sweden 1.', 'Sweden 2.', 'Switzerland 1.',
    'Switzerland 2.', 'Tunisia 1.', 'Turkey 1.', 'Turkey 2.', 'Ukraine 1.', 'UAE 1.',
    'USA 1.', 'USA 2.', 'Uruguay 1.', 'Uzbekistan 1.', 'Venezuela 1.', 'Wales 1.'
]

# Presets for the CANDIDATE POOL
PRESETS = {
    "Top 5 Europe": [
        'England 1.', 'France 1.', 'Germany 1.', 'Italy 1.', 'Spain 1.'
    ],
    "Top 20 Europe": [
        'England 1.', 'Italy 1.', 'Spain 1.', 'Germany 1.', 'France 1.',
        'England 2.', 'Portugal 1.', 'Belgium 1.',
        'Turkey 1.', 'Germany 2.', 'Spain 2.', 'France 2.',
        'Netherlands 1.', 'Austria 1.', 'Switzerland 1.', 'Denmark 1.', 'Croatia 1.',
        'Italy 2.', 'Czech 1.', 'Norway 1.'
    ],
    "EFL (England 2–4)": ['England 2.', 'England 3.', 'England 4.'],
    "All listed leagues": included_leagues,
    "Custom": None,
}

features = [
    'Defensive duels per 90', 'Aerial duels per 90', 'Aerial duels won, %', 'PAdj Interceptions',
    'Non-penalty goals per 90', 'xG per 90', 'Shots per 90', 'Shots on target, %', 'Goal conversion, %',
    'Crosses per 90', 'Accurate crosses, %', 'Dribbles per 90', 'Successful dribbles, %',
    'Offensive duels per 90', 'Touches in box per 90', 'Progressive runs per 90', 'Accelerations per 90',
    'Passes per 90', 'Accurate passes, %', 'xA per 90', 'Smart passes per 90', 'Key passes per 90',
    'Passes to final third per 90', 'Passes to penalty area per 90', 'Accurate passes to penalty area, %',
    'Deep completions per 90'
]

# Default weights (anything not listed defaults to 1.0 in the UI)
default_weight_factors = {
    'Passes per 90': 2,
    'Accurate passes, %': 2,
    'Dribbles per 90': 2,
    'Non-penalty goals per 90': 2,
    'Shots per 90': 2,
    'Successful dribbles, %': 2,
    'Aerial duels won, %': 2,
    'xA per 90': 2,
    'xG per 90': 2,
    'Touches in box per 90': 2,
}

league_strengths = {
    'England 1.': 100.00, 'Italy 1.': 97.14, 'Spain 1.': 94.29, 'Germany 1.': 94.29, 'France 1.': 91.43,
    'Brazil 1.': 82.86, 'England 2.': 71.43, 'Portugal 1.': 71.43, 'Argentina 1.': 71.43, 'Belgium 1.': 68.57,
    'Mexico 1.': 68.57, 'Turkey 1.': 65.71, 'Germany 2.': 65.71, 'Spain 2.': 65.71, 'France 2.': 65.71,
    'USA 1.': 65.71, 'Russia 1.': 65.71, 'Colombia 1.': 62.86, 'Netherlands 1.': 62.86, 'Austria 1.': 62.86,
    'Switzerland 1.': 62.86, 'Denmark 1.': 62.86, 'Croatia 1.': 62.86, 'Japan 1.': 62.86, 'Korea 1.': 62.86,
    'Italy 2.': 62.86, 'Czech 1.': 57.14, 'Norway 1.': 57.14, 'Poland 1.': 57.14, 'Romania 1.': 57.14,
    'Israel 1.': 57.14, 'Algeria 1.': 57.14, 'Paraguay 1.': 57.14, 'Saudi 1.': 57.14, 'Uruguay 1.': 57.14,
    'Morocco 1.': 57.00, 'Brazil 2.': 56.00, 'Ukraine 1.': 54.29, 'Ecuador 1.': 54.29, 'Spain 3.': 54.29,
    'Scotland 1.': 54.29, 'Chile 1.': 51.43, 'Cyprus 1.': 51.43, 'Portugal 2.': 51.43, 'Slovakia 1.': 51.43,
    'Australia 1.': 51.43, 'Hungary 1.': 51.43, 'Egypt 1.': 51.43, 'England 3.': 51.43, 'France 3.': 48.00,
    'Japan 2.': 48.00, 'Bulgaria 1.': 48.57, 'Slovenia 1.': 48.57, 'Venezuela 1.': 48.00, 'Germany 3.': 45.71,
    'Albania 1.': 44.00, 'Serbia 1.': 42.86, 'Belgium 2.': 42.86, 'Bosnia 1.': 42.86, 'Kosovo 1.': 42.86,
    'Nigeria 1.': 42.86, 'Azerbaijan 1.': 50.00, 'Bolivia 1.': 50.00, 'Costa Rica 1.': 50.00,
    'South Africa 1.': 50.00, 'UAE 1.': 50.00, 'Georgia 1.': 40.00, 'Finland 1.': 40.00, 'Italy 3.': 40.00,
    'Peru 1.': 40.00, 'Tunisia 1.': 40.00, 'USA 2.': 40.00, 'Armenia 1.': 40.00, 'North Macedonia 1.': 40.00,
    'Qatar 1.': 40.00, 'Uzbekistan 1.': 42.00, 'Norway 2.': 42.00, 'Kazakhstan 1.': 42.00, 'Poland 2.': 38.00,
    'Denmark 2.': 37.00, 'Czech 2.': 37.14, 'Israel 2.': 37.14, 'Netherlands 2.': 37.14, 'Switzerland 2.': 37.14,
    'Iceland 1.': 34.29, 'Ireland 1.': 34.29, 'Sweden 2.': 34.29, 'Germany 4.': 34.29, 'Malta 1.': 30.00,
    'Turkey 2.': 35.00, 'Canada 1.': 28.57, 'England 4.': 28.57, 'Scotland 2.': 28.57, 'Moldova 1.': 28.57,
    'Austria 2.': 25.71, 'Lithuania 1.': 25.71, 'Brazil 3.': 25.00, 'England 7.': 25.00, 'Slovenia 2.': 22.00,
    'Latvia 1.': 22.86, 'Serbia 2.': 20.00, 'Slovakia 2.': 20.00, 'England 9.': 20.00, 'England 8.': 15.00,
    'Montenegro 1.': 14.29, 'Wales 1.': 12.00, 'Portugal 3.': 11.43, 'Northern Ireland 1.': 11.43,
    'England 5.': 11.43, 'Andorra 1.': 10.00, 'Estonia 1.': 8.57, 'England 10.': 5.00, 'Scotland 3.': 0.00,
    'England 6.': 0.00
}

DEFAULT_LEAGUE_WEIGHT = 0.4
DEFAULT_MARKET_WEIGHT = 0.2

# ---------- Sidebar ----------
with st.sidebar:
    st.header("Player & Pools")

    leagues_available = sorted(set(included_leagues) | set(df.get('League', pd.Series([])).dropna().unique()))

    # Target leagues (for picking the target player)
    target_leagues = st.multiselect(
        "Target leagues (choose target from here)",
        leagues_available,
        default=leagues_available
    )

    # Candidate pool leagues via preset + extras
    if 'candidate_leagues' not in st.session_state:
        st.session_state.candidate_leagues = included_leagues.copy()

    preset_name = st.selectbox("Candidate pool preset", list(PRESETS.keys()), index=0)
    if st.button("Apply preset"):
        preset = PRESETS[preset_name]
        if preset is not None:
            st.session_state.candidate_leagues = preset

    extra_candidate_leagues = st.multiselect(
        "Extra leagues to add to candidate pool",
        leagues_available,
        default=[]
    )
    # Final candidate set
    leagues_selected = sorted(set(st.session_state.candidate_leagues) | set(extra_candidate_leagues))
    st.caption(f"Candidate pool leagues: **{len(leagues_selected)}** selected.")

    pos_scope = st.text_input("Position startswith", "CF")

    # Target player comes from target_leagues only (not restricted by candidate pool)
    target_pool = df[df['League'].isin(target_leagues)]
    target_pool = target_pool[target_pool['Position'].astype(str).str.startswith(tuple([pos_scope]))]
    target_player = st.selectbox("Target player", sorted(target_pool['Player'].dropna().unique()))

    st.header("Filters (for candidate teams)")
    # Filters applied ONLY to the candidate pool that builds team profiles
    # Set a big upper bound for minutes so you can include everyone by default
    max_minutes_in_data = int(pd.to_numeric(df.get('Minutes played', pd.Series([0])), errors='coerce').fillna(0).max())
    min_minutes, max_minutes = st.slider("Minutes filter", 0, max(1000, max_minutes_in_data), (500, max(1000, max_minutes_in_data)))
    # Age bounds
    age_series = pd.to_numeric(df.get('Age', pd.Series([16, 45])), errors='coerce')
    age_min_data = int(np.nanmin(age_series)) if age_series.notna().any() else 14
    age_max_data = int(np.nanmax(age_series)) if age_series.notna().any() else 45
    min_age, max_age = st.slider("Age filter", age_min_data, age_max_data, (16, 33))

    min_strength, max_strength = st.slider("League quality (strength)", 0, 101, (0, 101))

    st.header("Weights")
    league_weight = st.slider("League weight", 0.0, 1.0, DEFAULT_LEAGUE_WEIGHT, 0.05)
    market_value_weight = st.slider("Market value weight", 0.0, 1.0, DEFAULT_MARKET_WEIGHT, 0.05)

    manual_override = st.number_input("Target market value override (€)", min_value=0, value=0, step=100000)

    # Advanced feature weights
    st.subheader("Advanced feature weights")
    st.caption("Unlisted features default to weight = 1.")
    weights_ui = {}
    for f in features:
        default_val = default_weight_factors.get(f, 1)
        weights_ui[f] = st.slider(f"• {f}", 0, 5, int(default_val))

    top_n = st.number_input("Show top N teams", 5, 100, 20, 5)

# ---------- Data checks ----------
required_cols = {'Player','Team','League','Age','Position','Minutes played','Market value', *features}
missing = [c for c in required_cols if c not in df.columns]
if missing:
    st.error(f"Missing required columns: {missing}")
    st.stop()

# ---------- Build candidate pool (teams) ----------
df_candidates = df[df['League'].isin(leagues_selected)].copy()
df_candidates = df_candidates[df_candidates['Position'].astype(str).str.startswith(tuple([pos_scope]))]

# Coerce numerics for filters
df_candidates['Minutes played'] = pd.to_numeric(df_candidates['Minutes played'], errors='coerce')
df_candidates['Age'] = pd.to_numeric(df_candidates['Age'], errors='coerce')
df_candidates['Market value'] = pd.to_numeric(df_candidates['Market value'], errors='coerce')

df_candidates = df_candidates[
    df_candidates['Minutes played'].between(min_minutes, max_minutes, inclusive='both')
]
df_candidates = df_candidates[
    df_candidates['Age'].between(min_age, max_age, inclusive='both')
]
df_candidates = df_candidates.dropna(subset=features)

if df_candidates.empty:
    st.warning("No candidate players after filters. Widen candidate leagues or relax filters.")
    st.stop()

# ---------- Target vector (from target_leagues pool only) ----------
df_target_pool = df[df['League'].isin(target_leagues)].copy()
df_target_pool = df_target_pool[df_target_pool['Position'].astype(str).str.startswith(tuple([pos_scope]))]
if target_player not in df_target_pool['Player'].values:
    st.warning("Target player not found in selected target leagues. Adjust 'Target leagues'.")
    st.stop()

# Ensure MV numeric for target info
df_target_pool['Market value'] = pd.to_numeric(df_target_pool['Market value'], errors='coerce')

target_row = df_target_pool.loc[df_target_pool['Player'] == target_player].iloc[0]
target_vector = target_row[features].values
target_league_strength = league_strengths.get(target_row['League'], 1.0)

if manual_override and manual_override > 0:
    target_market_value = float(manual_override)
else:
    mv = target_row['Market value']
    target_market_value = float(mv) if pd.notna(mv) and mv > 0 else 2_000_000.0

# ---------- Build club profiles (unique per team) from candidates ----------
club_profiles = df_candidates.groupby(['Team'])[features].mean().reset_index()

# Team's dominant league & average team market value (from candidate pool)
team_league = df_candidates.groupby('Team')['League'].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else x.iloc[0])
team_market = df_candidates.groupby('Team')['Market value'].mean()

club_profiles['League'] = club_profiles['Team'].map(team_league)
club_profiles['Avg Team Market Value'] = club_profiles['Team'].map(team_market)
club_profiles = club_profiles.dropna(subset=['Avg Team Market Value'])

# ---------- Similarity (feature scaling + weights) ----------
weights_vec = np.array([weights_ui.get(f, 1) for f in features], dtype=float)

scaler = StandardScaler()
scaled_team = scaler.fit_transform(club_profiles[features])
target_scaled = scaler.transform([target_vector])[0]

# distance -> base fit (0..100)
dist = np.linalg.norm((scaled_team - target_scaled) * weights_vec, axis=1)
rng = dist.max() - dist.min()
base_fit = (1 - (dist - dist.min()) / (rng if rng > 0 else 1)) * 100
club_profiles['Club Fit %'] = base_fit.round(2)

# ---------- League strength adjustment ----------
club_profiles['League strength'] = club_profiles['League'].map(league_strengths).fillna(0.0)
club_profiles = club_profiles[
    (club_profiles['League strength'] >= float(min_strength)) &
    (club_profiles['League strength'] <= float(max_strength))
]

ratio = (club_profiles['League strength'] / target_league_strength).clip(0.5, 1.2)
club_profiles['Adjusted Fit %'] = (
    club_profiles['Club Fit %'] * (1 - league_weight) +
    club_profiles['Club Fit %'] * ratio * league_weight
)

# small penalty if league significantly stronger than target
league_gap = (club_profiles['League strength'] - target_league_strength).clip(lower=0)
penalty = (1 - (league_gap / 100)).clip(lower=0.7)
club_profiles['Adjusted Fit %'] = club_profiles['Adjusted Fit %'] * penalty

# ---------- Market value fit ----------
value_fit_ratio = (club_profiles['Avg Team Market Value'] / target_market_value).clip(0.5, 1.5)
value_fit_score = (1 - abs(1 - value_fit_ratio)) * 100

club_profiles['Final Fit %'] = (
    club_profiles['Adjusted Fit %'] * (1 - market_value_weight) +
    value_fit_score * market_value_weight
)

# ---------- Results ----------
results = club_profiles[['Team','League','League strength','Avg Team Market Value',
                         'Club Fit %','Adjusted Fit %','Final Fit %']].copy()

results = results.sort_values('Final Fit %', ascending=False).reset_index(drop=True)
results.insert(0, 'Rank', np.arange(1, len(results)+1))

# ---------- UI ----------
st.subheader(
    f"Target: {target_player} — {target_row.get('Team','Unknown')} ({target_row['League']})"
)
st.caption(
    f"Target market value used: €{target_market_value:,.0f} • "
    f"Target league strength {target_league_strength:.2f} • "
    f"Candidate leagues: {len(leagues_selected)} (preset: {preset_name})"
)

st.dataframe(results.head(int(top_n)), use_container_width=True)

csv = results.to_csv(index=False).encode('utf-8')
st.download_button("⬇️ Download all results (CSV)", data=csv, file_name="club_fit_results.csv", mime="text/csv")

with st.expander("Debug / Repro details"):
    st.write({
        "preset": preset_name,
        "candidate_leagues_count": len(leagues_selected),
        "target_leagues_count": len(target_leagues),
        "league_weight": float(league_weight),
        "market_value_weight": float(market_value_weight),
        "target_market_value": float(target_market_value),
        "minutes_range": (int(min_minutes), int(max_minutes)),
        "age_range": (int(min_age), int(max_age)),
        "strength_range": (int(min_strength), int(max_strength)),
    })

