import streamlit as st
import pandas as pd
import numpy as np
import random
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
import json

# Page configuration
st.set_page_config(
    page_title="Sports Team & Tournament Generator",
    page_icon="🏆",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sport-header {
        font-size: 1.8rem;
        font-weight: bold;
        color: #2ca02c;
        margin-top: 1.5rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 3rem;
        padding: 0 2rem;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Standardized Column Names (for internal use) ---
LOC_COL = 'Base Location (Bengaluru/Pune/Hyderabad/Chennai)'
CRICKET_SKILL_COL = 'Primary Skill(Batsman/ Bowler)'
CARROM_COL = 'Carroms (Doubles Only)' # Updated
TUG_COL = 'Tug of War'
CRICKET_COL = 'Cricket'
GENDER_COL = 'Gender'
NAME_COL = 'Employee Name'
CHESS_COL = 'Chess'

# --- New list of available sports ---
AVAILABLE_SPORTS = [
    CRICKET_COL, 'Sack Race', TUG_COL, 'Cup Stack Relay', 
    'Three Legged Race', 'Push Ups', 'Planks', 'Squats', 
    CHESS_COL, CARROM_COL
]

# --- Source Column Names (from your new file) ---
SOURCE_NAME_COL = 'Please Enter your Full Name (In CAPITAL Letters)'
SOURCE_GENDER_COL = 'Select Your Gender'
SOURCE_LOC_COL = 'Select Your Base Location'
SOURCE_SPORTS_COL = 'Please select Sporting Event you would like to take part during the TENTHPIN INDIA SPORTS FEST 2025.'
SOURCE_CRICKET_SKILL_COL = 'If you Chose Cricket. Please Select your Primary Skill'


# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'cricket_teams' not in st.session_state:
    st.session_state.cricket_teams = None
if 'carroms_teams' not in st.session_state:  # Added Carroms
    st.session_state.carroms_teams = None
if 'chess_fixtures' not in st.session_state:
    st.session_state.chess_fixtures = None
if 'tug_of_war_teams' not in st.session_state:
    st.session_state.tug_of_war_teams = None
if 'cricket_fixtures' not in st.session_state:
    st.session_state.cricket_fixtures = None
if 'carroms_fixtures' not in st.session_state: # Added Carroms
    st.session_state.carroms_fixtures = None
if 'tug_of_war_fixtures' not in st.session_state:
    st.session_state.tug_of_war_fixtures = None


# Helper Functions
def clean_column_names(df):
    """Clean column names by removing extra spaces and newlines"""
    df.columns = df.columns.str.strip().str.replace(r'\s+', ' ', regex=True).str.replace('\n', ' ', regex=True)
    return df

def transform_sports_column(df):
    """Splits the single 'Sports' column into multiple game columns"""
    if SOURCE_SPORTS_COL in df.columns:
        # Use get_dummies to split the column by ';'
        try:
            sports_dummies = df[SOURCE_SPORTS_COL].astype(str).str.get_dummies(sep=';')
            
            # Clean the new column names (e.g., "Tug of War " -> "Tug of War")
            sports_dummies.columns = sports_dummies.columns.str.strip()

            # Rename dummy columns to our standard names
            sports_dummies = sports_dummies.rename(columns={
                "Carroms (Doubles Only)": CARROM_COL,
                "Tug of War": TUG_COL,
                # Add any other renames if the text doesn't match AVAILABLE_SPORTS
            })

            # Add these new columns to the original dataframe
            df = pd.concat([df, sports_dummies], axis=1)
            
            # Now, standardize the 'yes'/'no' values in these new columns
            for col in AVAILABLE_SPORTS:
                if col in df.columns:
                    # If the column exists (from get_dummies), 1 becomes 'yes', 0 becomes 'no'
                    df[col] = df[col].apply(lambda x: 'yes' if x == 1 else 'no')
                else:
                    # If the sport wasn't in anyone's list, create a 'no' column
                    df[col] = 'no'
                    
        except Exception as e:
            st.error(f"Error splitting sports column: {e}")
            st.write("Could not find or split the main sports column. Please check the column name and format.")
    else:
        st.warning(f"Could not find the sports selection column: '{SOURCE_SPORTS_COL}'")
    return df


def clean_data(df):
    """Clean and standardize data values"""
    if GENDER_COL in df.columns:
        df[GENDER_COL] = df[GENDER_COL].astype(str).str.strip()
        df[GENDER_COL] = df[GENDER_COL].str.replace('Men', 'Male', case=False, regex=False)
        df[GENDER_COL] = df[GENDER_COL].str.replace('Women', 'Female', case=False, regex=False)
    
    # Standardize Yes/No values (this is now handled in transform_sports_column)
    # But we keep this for the other individual sports
    for col in AVAILABLE_SPORTS:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.lower()
            # Handle cases where it's already 'yes' or '1'
            df[col] = df[col].apply(lambda x: 'yes' if x in ['yes', '1'] else 'no')
    
    if CRICKET_SKILL_COL in df.columns:
        df[CRICKET_SKILL_COL] = df[CRICKET_SKILL_COL].astype(str).str.strip()
        df[CRICKET_SKILL_COL] = df[CRICKET_SKILL_COL].str.replace('Both', 'All Rounder', case=False, regex=False)
        df[CRICKET_SKILL_COL] = df[CRICKET_SKILL_COL].str.replace('Allrounder', 'All Rounder', case=False, regex=False)
    
    return df

def validate_columns(df):
    """Validate that required columns exist in the dataframe *after* mapping"""
    # These are the *internal, standardized* names
    required_cols = [NAME_COL, GENDER_COL, LOC_COL]
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    # Check if *any* sport column was created
    if not any(sport in df.columns for sport in AVAILABLE_SPORTS):
         st.warning("Note: No sport columns were found or created. Team generation will be disabled.")
         missing_cols.append("Any Sport Column")

    return len(missing_cols) == 0, missing_cols

def get_yes_count(df, sport_col):
    if sport_col not in df.columns:
        return 0
    return df[sport_col].astype(str).str.lower().str.contains('yes', na=False).sum()

def create_cricket_teams(df, random_seed=None):
    if random_seed:
        random.seed(random_seed)
        np.random.seed(random_seed)
    
    if CRICKET_COL not in df.columns:
        return None, "Cricket column not found in data."
    if CRICKET_SKILL_COL not in df.columns:
        return None, "Cricket Skill column not found in data."
        
    cricket_df = df[df[CRICKET_COL].astype(str).str.lower().str.contains('yes', na=False)].copy()
    
    if len(cricket_df) < 11:
        return None, "Not enough players for at least one team (minimum 11 required)"
    
    females = cricket_df[cricket_df[GENDER_COL].astype(str).str.lower().str.contains('female', na=False)]
    all_players = cricket_df.sample(frac=1).index.tolist()
    
    num_teams = len(all_players) // 11
    if num_teams == 0:
         return None, "Not enough players for at least one team (minimum 11 required)"

    teams = []
    for i in range(num_teams):
        team_indices = all_players[i*11 : (i+1)*11]
        
        team_df = cricket_df.loc[team_indices]
        num_females = team_df[GENDER_COL].str.lower().str.contains('female', na=False).sum()
        
        if num_females == 0 and len(females) > num_teams:
            remaining_females = [f for f in females.index if f not in all_players[:num_teams*11]]
            if remaining_females:
                female_to_add = remaining_females[0]
                males_in_team = [p for p in team_indices if p not in females.index]
                if males_in_team:
                    male_to_remove = males_in_team[0]
                    team_indices.remove(male_to_remove)
                    team_indices.append(female_to_add)
                    all_players.remove(female_to_add)
                    all_players.append(male_to_remove)
        
        teams.append(team_indices)

    if len(teams) == 0:
        return None, "Unable to form balanced teams with given constraints"
    
    team_dfs = []
    for i, team_indices in enumerate(teams, 1):
        team_df = df.loc[team_indices][[NAME_COL, GENDER_COL, LOC_COL, CRICKET_SKILL_COL]].copy()
        team_df.insert(0, 'Team', f'Team {i}')
        team_dfs.append(team_df)
    
    return pd.concat(team_dfs, ignore_index=True), None


def create_fixtures(num_teams, team_prefix="Team"):
    if num_teams < 2:
        return None
    
    teams = [f"{team_prefix} {i+1}" for i in range(num_teams)]
    
    next_power = 2 ** int(np.ceil(np.log2(num_teams)))
    byes = next_power - num_teams
    
    fixtures = {"rounds": []}
    current_teams = teams.copy()
    
    for i in range(byes):
        current_teams.append("BYE")
    
    random.shuffle(current_teams) # Shuffle teams for random matchups
    
    round_num = 1
    
    while len(current_teams) > 1:
        matches = []
        next_round_teams = []
        
        for i in range(0, len(current_teams), 2):
            if i+1 < len(current_teams):
                team1, team2 = current_teams[i], current_teams[i+1]
                if team2 == "BYE":
                    matches.append(f"{team1} (BYE - Auto Advance)")
                    next_round_teams.append(team1)
                elif team1 == "BYE":
                    matches.append(f"{team2} (BYE - Auto Advance)")
                    next_round_teams.append(team2)
                else:
                    matches.append(f"{team1} vs {team2}")
                    next_round_teams.append(f"Winner(Match {len(matches)+1})")
        
        if len(current_teams) == 2:
            round_name = "Final"
        elif len(current_teams) == 4:
            round_name = "Semi Finals"
        elif len(current_teams) == 8:
            round_name = "Quarter Finals"
        else:
            round_name = f"Round of {len(current_teams)}"
        
        fixtures["rounds"].append({
            "name": round_name,
            "matches": matches
        })
        
        current_teams = next_round_teams
        round_num += 1
    
    return fixtures

# --- NEW FUNCTION for Carroms ---
def create_carroms_pairs(df, random_seed=None):
    """Create random doubles pairs for Carroms"""
    if random_seed:
        random.seed(random_seed)
        np.random.seed(random_seed)
    
    if CARROM_COL not in df.columns:
        return None, "Carroms column not found in data."
        
    carroms_df = df[df[CARROM_COL].astype(str).str.lower().str.contains('yes', na=False)].copy()
    
    if len(carroms_df) < 2:
        return None, "Not enough players for at least one team (minimum 2 required)"
    
    # Shuffle all players
    players = carroms_df.sample(frac=1).reset_index(drop=True)
    
    pairs = []
    num_pairs = len(players) // 2
    
    for i in range(num_pairs):
        player1 = players.iloc[i*2]
        player2 = players.iloc[i*2 + 1]
        
        pairs.append({
            'Team': f'Team {i+1}',
            'Player 1': player1[NAME_COL],
            'Player 2': player2[NAME_COL],
            'Location 1': player1[LOC_COL],
            'Location 2': player2[LOC_COL]
        })
    
    # Inform about the leftover player
    leftover_player = None
    if len(players) % 2 == 1:
        leftover_player = players.iloc[-1][NAME_COL]
        
    return pd.DataFrame(pairs) if pairs else None, leftover_player

def create_chess_fixtures(df, random_seed=None):
    if random_seed:
        random.seed(random_seed)
        np.random.seed(random_seed)
    
    if CHESS_COL not in df.columns:
        return None
        
    chess_df = df[df[CHESS_COL].astype(str).str.lower().str.contains('yes', na=False)].copy()
    players = chess_df[NAME_COL].tolist()
    
    if len(players) < 2:
        return None
    
    random.shuffle(players)
    
    fixtures = {"rounds": []}
    round_num = 1
    current_players = players.copy()
    
    while len(current_players) > 1:
        matches = []
        next_round = []
        
        for i in range(0, len(current_players), 2):
            if i+1 < len(current_players):
                matches.append(f"{current_players[i]} vs {current_players[i+1]}")
                next_round.append(f"Winner(Match {len(matches)+1})")
            else:
                matches.append(f"{current_players[i]} (BYE - Auto Advance)")
                next_round.append(current_players[i])
        
        if len(current_players) == 2:
            round_name = "Final"
        elif len(current_players) <= 4:
            round_name = "Semi Finals"
        elif len(current_players) <= 8:
            round_name = "Quarter Finals"
        else:
            round_name = f"Round {round_num}"
        
        fixtures["rounds"].append({
            "name": round_name,
            "matches": matches
        })
        
        current_players = next_round
        round_num += 1
    
    return fixtures

def create_tug_of_war_teams(df, team_size=10, random_seed=None):
    if random_seed:
        random.seed(random_seed)
        np.random.seed(random_seed)
    
    if TUG_COL not in df.columns:
        return None, "Tug of War column not found in data."
        
    tug_df = df[df[TUG_COL].astype(str).str.lower().str.contains('yes', na=False)].copy()
    
    if len(tug_df) < team_size:
        return None, f"Not enough players for at least one team (minimum {team_size} required)"
    
    all_players = tug_df.sample(frac=1).index.tolist()
    num_teams = len(all_players) // team_size
    
    if num_teams == 0:
        return None, f"Not enough players for at least one team (minimum {team_size} required)"

    teams = []
    for i in range(num_teams):
        team_indices = all_players[i*team_size : (i+1)*team_size]
        teams.append(team_indices)
        
    if len(teams) == 0:
        return None, "Unable to form teams"
    
    team_dfs = []
    for i, team_indices in enumerate(teams, 1):
        team_df = df.loc[team_indices][[NAME_COL, GENDER_COL, LOC_COL]].copy()
        team_df.insert(0, 'Team', f'Team {i}')
        team_dfs.append(team_df)
    
    return pd.concat(team_dfs, ignore_index=True), None

def to_excel(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Sheet1')
    return output.getvalue()

# Sidebar
with st.sidebar:
    st.markdown("### 🏆 Sports Tournament App")
    st.markdown("---")
    
    st.markdown("### 1. File Settings")
    header_row = st.number_input("Header Row in Excel/CSV", min_value=1, value=1, step=1,
                                 help="Enter the row number where your column headers (e.g., 'Employee Name') are located. This is usually 1.")
    header_index = header_row - 1
    
    uploaded_file = st.file_uploader("📁 Upload Excel or CSV File", type=['xlsx', 'xls', 'csv'])
    
    if uploaded_file:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file, header=header_index)
            else:
                df = pd.read_excel(uploaded_file, header=header_index)
            
            df = clean_column_names(df)
            df = df.dropna(how='all')
            
            # ---!!! KEY CHANGE: COLUMN MAPPING !!!---
            column_mapping = {
                SOURCE_NAME_COL: NAME_COL,
                SOURCE_GENDER_COL: GENDER_COL,
                SOURCE_LOC_COL: LOC_COL,
                SOURCE_CRICKET_SKILL_COL: CRICKET_SKILL_COL,
            }
            
            df = df.rename(columns=column_mapping)
            
            # ---!!! KEY CHANGE: TRANSFORM SPORTS COLUMN !!!---
            df = transform_sports_column(df)
            
            # Now validate the *standardized* columns
            valid, missing = validate_columns(df)
            
            if valid:
                df = clean_data(df)
                st.session_state.data = df
                st.success(f"✅ File uploaded! {len(df)} participants found.")
                
                with st.expander("📋 Processed Columns (What the app sees)"):
                    for i, col in enumerate(df.columns):
                        st.text(f"{i}: {col}")
            else:
                st.error(f"❌ Missing required columns: {', '.join(missing)}")
                st.info("📋 Detected columns (Original from your file):")
                
                # Re-read for debugging
                uploaded_file.seek(0) # Reset file pointer
                if uploaded_file.name.endswith('.csv'):
                    original_df = pd.read_csv(uploaded_file, header=header_index)
                else:
                    original_df = pd.read_excel(uploaded_file, header=header_index)
                original_df = clean_column_names(original_df)
                
                for i, col in enumerate(original_df.columns):
                    st.text(f"{i}: {col}")
                
                st.warning(f"💡 Tip: The app is looking for headers in Row {header_row}. It also *needs* the column '{SOURCE_SPORTS_COL}' to find the games.")
        
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")
            import traceback
            with st.expander("🔍 Error Details"):
                st.code(traceback.format_exc())
    
    st.markdown("---")
    
    if st.session_state.data is not None:
        st.markdown("### ⚙️ Settings")
        random_seed = st.number_input("Random Seed (optional)", min_value=0, value=42, step=1)
        
        st.markdown("### 🎯 Team Sizes")
        tug_of_war_size = st.slider("Tug of War Team Size", min_value=8, max_value=15, value=10, step=1)
        
        st.markdown("### 📊 Quick Stats")
        df = st.session_state.data
        st.metric("Total Participants", len(df))
        if GENDER_COL in df.columns:
            males = df[GENDER_COL].astype(str).str.lower().str.contains('male', na=False).sum()
            females = df[GENDER_COL].astype(str).str.lower().str.contains('female', na=False).sum()
            st.metric("Male/Female", f"{males}/{females}")

# Main Content
st.markdown('<p class="main-header">🏆 Sports Team & Tournament Generator</p>', unsafe_allow_html=True)

if st.session_state.data is None:
    st.info("👈 Please upload an Excel or CSV file from the sidebar to get started.")
    
    st.markdown("### 📋 Expected Excel/CSV Format")
    st.markdown("**Make sure your file has headers in the correct row (set in sidebar):**")
    sample_data = {
        SOURCE_NAME_COL: ['John Doe', 'Jane Smith', 'Bob Wilson'],
        SOURCE_GENDER_COL: ['Male', 'Female', 'Male'],
        SOURCE_LOC_COL: ['Bengaluru', 'Pune', 'Chennai'],
        SOURCE_SPORTS_COL: ['Cricket;Sack Race', 'Chess;Planks', 'Cricket;Tug of War;Squats'],
        SOURCE_CRICKET_SKILL_COL: ['Batsman', '', 'All Rounder']
    }
    st.dataframe(pd.DataFrame(sample_data), use_container_width=True)
    
    st.markdown(f"""
    **Important Notes:**
    - **Use the "Header Row in Excel/CSV" setting in the sidebar** (usually 1).
    - The sports column **must** be named: `{SOURCE_SPORTS_COL}`
    - Games in that column **must** be separated by a semi-colon (`;`).
    """)
else:
    df = st.session_state.data
    
    # ---!!! KEY CHANGE: NEW TAB LAYOUT !!!---
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Dashboard", 
        "📋 Participant Lists", 
        "👥 Team Generator", 
        "🏆 Fixture Creator"
    ])
    
    # TAB 1: Dashboard
    with tab1:
        st.markdown("## 📊 Data Summary")
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### 📄 Data Preview")
            display_cols = [NAME_COL, GENDER_COL, LOC_COL, 
                            CRICKET_COL, CHESS_COL, TUG_COL, CARROM_COL]
            available_cols = [col for col in display_cols if col in df.columns]
            if available_cols:
                st.dataframe(df[available_cols].head(10), use_container_width=True)
            else:
                st.warning("No displayable columns found. Check column mapping.")
        
        with col2:
            st.markdown("### 🎯 Participation Overview")
            # Use the new AVAILABLE_SPORTS list
            sports = AVAILABLE_SPORTS
            
            participation = {}
            for sport in sports:
                if sport in df.columns:
                    count = get_yes_count(df, sport)
                    if count > 0:
                        display_name = sport.split('(')[0].strip() if '(' in sport else sport
                        participation[display_name] = count
            
            if participation:
                fig = px.bar(
                    x=list(participation.keys()), y=list(participation.values()),
                    labels={'x': 'Sport', 'y': 'Participants'}, title='Participation by Sport',
                    color=list(participation.values()), color_continuous_scale='viridis'
                )
                fig.update_layout(showlegend=False, height=400, xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No participation data found. Check the sports column in your file.")
        
        st.markdown("---")
        col1, col2 = st.columns(2)
        
        with col1:
            if GENDER_COL in df.columns:
                gender_counts = df[GENDER_COL].value_counts()
                if not gender_counts.empty:
                    fig = px.pie(
                        values=gender_counts.values, names=gender_counts.index,
                        title='Gender Distribution', hole=0.4
                    )
                    st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if LOC_COL in df.columns:
                location_counts = df[LOC_COL].value_counts()
                if not location_counts.empty:
                    fig = px.bar(
                        x=location_counts.index, y=location_counts.values,
                        title='Participants by Location', labels={'x': 'Location', 'y': 'Count'},
                        color=location_counts.values, color_continuous_scale='blues'
                    )
                    st.plotly_chart(fig, use_container_width=True)
    
    # ---!!! UPDATED TAB 2: Participant Lists with Filter & Download !!!---
    with tab2:
        st.markdown("## 📋 Participant Lists by Game")
        st.markdown("Select an event from the dropdown to see all registered participants and download the list.")

        # Filter list to only show sports that are actually in the dataframe
        sports_in_data = [sport for sport in AVAILABLE_SPORTS if sport in df.columns]
        
        if not sports_in_data:
            st.error("No valid sport columns were found in the data. Check your source file.")
        else:
            selected_sport = st.selectbox("Select an Event", options=sports_in_data)
            
            if selected_sport:
                participants_df = df[df[selected_sport].str.lower() == 'yes']
                display_df = participants_df[[NAME_COL, GENDER_COL, LOC_COL]].copy()
                
                st.metric(label=f"Total Participants for {selected_sport}", value=len(display_df))
                
                if not display_df.empty:
                    st.dataframe(display_df, use_container_width=True, hide_index=True)
                    
                    # Add download button for the filtered list
                    excel_data = to_excel(display_df)
                    st.download_button(
                        label=f"📥 Download {selected_sport} List (Excel)",
                        data=excel_data,
                        file_name=f"{selected_sport.lower().replace(' ', '_')}_participants.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                else:
                    st.info(f"No participants found for {selected_sport}.")

    
    # TAB 3: Team Generator
    with tab3:
        st.markdown("## 👥 Team Generator")
        
        # --- Updated Tabs (Removed Badminton, Added Carroms) ---
        sport_tab1, sport_tab2, sport_tab3 = st.tabs([
            "🏏 Cricket Teams", 
            "🕳️ Carroms Pairs", 
            "💪 Tug of War Teams"
        ])
        
        # Cricket Teams
        with sport_tab1:
            st.markdown("### 🏏 Cricket Team Generator")
            st.markdown("*Teams will have 11 players each.*")
            
            if st.button("🎲 Generate Cricket Teams", type="primary", key="gen_cricket"):
                with st.spinner("Generating teams..."):
                    teams, error = create_cricket_teams(df, random_seed)
                    if teams is not None:
                        st.session_state.cricket_teams = teams
                        st.success(f"✅ Created {len(teams['Team'].unique())} teams!")
                    else:
                        st.error(f"❌ {error}")
            
            if st.session_state.cricket_teams is not None:
                teams_df = st.session_state.cricket_teams
                st.markdown("#### 📋 Team Compositions")
                
                for team in teams_df['Team'].unique():
                    with st.expander(f"👥 {team} ({len(teams_df[teams_df['Team'] == team])} players)"):
                        team_data = teams_df[teams_df['Team'] == team]
                        skill_dist = team_data[CRICKET_SKILL_COL].value_counts()
                        st.markdown(f"**Skill Distribution:** {dict(skill_dist)}")
                        st.dataframe(team_data[[NAME_COL, GENDER_COL, LOC_COL, CRICKET_SKILL_COL]], 
                                     use_container_width=True, hide_index=True)
                
                st.markdown("---")
                col1, col2 = st.columns(2)
                with col1:
                    excel_data = to_excel(teams_df)
                    st.download_button(label="📥 Download Teams (Excel)", data=excel_data,
                                       file_name="cricket_teams.xlsx",
                                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
                with col2:
                    csv_data = teams_df.to_csv(index=False)
                    st.download_button(label="📥 Download Teams (CSV)", data=csv_data,
                                       file_name="cricket_teams.csv", mime="text/csv")
        
        # --- NEW TAB: Carroms Pairs ---
        with sport_tab2:
            st.markdown("### 🕳️ Carroms Doubles Generator")
            st.markdown("*Pairs will have 2 players each.*")
            
            if st.button("🎲 Generate Carroms Pairs", type="primary", key="gen_carroms"):
                with st.spinner("Creating pairs..."):
                    pairs, leftover = create_carroms_pairs(df, random_seed)
                    if pairs is not None and not pairs.empty:
                        st.session_state.carroms_teams = pairs
                        st.success(f"✅ Created {len(pairs)} pairs!")
                        if leftover:
                            st.info(f"Note: {leftover} is left without a partner.")
                    else:
                         st.error(f"❌ {leftover or 'Not enough players to create pairs.'}")
            
            if st.session_state.carroms_teams is not None:
                pairs_df = st.session_state.carroms_teams
                st.markdown("#### 🕳️ Carroms Pairs")
                st.dataframe(pairs_df, use_container_width=True, hide_index=True)
                
                st.markdown("---")
                col1, col2 = st.columns(2)
                with col1:
                    excel_data = to_excel(pairs_df)
                    st.download_button(label="📥 Download Pairs (Excel)", data=excel_data,
                                       file_name="carroms_pairs.xlsx",
                                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
                with col2:
                    csv_data = pairs_df.to_csv(index=False)
                    st.download_button(label="📥 Download Pairs (CSV)", data=csv_data,
                                       file_name="carroms_pairs.csv", mime="text/csv")
        
        # Tug of War Teams
        with sport_tab3:
            st.markdown("### 💪 Tug of War Team Generator")
            st.markdown(f"*Teams will have {tug_of_war_size} players each.*")
            
            if st.button("🎲 Generate Tug of War Teams", type="primary", key="gen_tug"):
                with st.spinner("Generating teams..."):
                    teams, error = create_tug_of_war_teams(df, tug_of_war_size, random_seed)
                    if teams is not None:
                        st.session_state.tug_of_war_teams = teams
                        st.success(f"✅ Created {len(teams['Team'].unique())} teams!")
                    else:
                        st.error(f"❌ {error}")
            
            if st.session_state.tug_of_war_teams is not None:
                teams_df = st.session_state.tug_of_war_teams
                st.markdown("#### 💪 Team Compositions")
                
                for team in teams_df['Team'].unique():
                    with st.expander(f"👥 {team} ({len(teams_df[teams_df['Team'] == team])} players)"):
                        team_data = teams_df[teams_df['Team'] == team]
                        gender_dist = team_data[GENDER_COL].value_counts()
                        st.markdown(f"**Gender Distribution:** {dict(gender_dist)}")
                        st.dataframe(team_data[[NAME_COL, GENDER_COL, LOC_COL]], 
                                     use_container_width=True, hide_index=True)
                
                st.markdown("---")
                col1, col2 = st.columns(2)
                with col1:
                    excel_data = to_excel(teams_df)
                    st.download_button(label="📥 Download Teams (Excel)", data=excel_data,
                                       file_name="tug_of_war_teams.xlsx",
                                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
                with col2:
                    csv_data = teams_df.to_csv(index=False)
                    st.download_button(label="📥 Download Teams (CSV)", data=csv_data,
                                       file_name="tug_of_war_teams.csv", mime="text/csv")
    
    # TAB 4: Fixture Creator
    with tab4:
        st.markdown("## 🏆 Fixture Creator")
        
        # --- Updated Tabs (Removed Badminton, Added Carroms) ---
        fixture_tab1, fixture_tab2, fixture_tab3, fixture_tab4 = st.tabs([
            "🏏 Cricket Fixtures", 
            "🕳️ Carroms Fixtures", 
            "♟️ Chess Fixtures", 
            "💪 Tug of War Fixtures"
        ])
        
        def display_fixtures(fixture_data, sport_name):
            """Helper function to show fixture UI"""
            for round_data in fixture_data['rounds']:
                st.markdown(f"#### 🏆 {round_data['name']}")
                for i, match in enumerate(round_data['matches'], 1):
                    st.markdown(f"**Match {i}:** {match}")
                st.markdown("---")
            
            fixture_text = f"{sport_name.upper()} TOURNAMENT FIXTURES\n" + "="*50 + "\n"
            for round_data in fixture_data['rounds']:
                # --- CORRECTED LINE BELOW ---
                fixture_text += f"\n{round_data['name']}\n{'='*50}\n" 
                for i, match in enumerate(round_data['matches'], 1):
                    fixture_text += f"Match {i}: {match}\n"
                fixture_text += "\n"
            
            st.download_button(label="📥 Download Fixtures (TXT)", data=fixture_text,
                               file_name=f"{sport_name.lower().replace(' ', '_')}_fixtures.txt",
                               mime="text/plain")

        # Cricket Fixtures
        with fixture_tab1:
            st.markdown("### 🏏 Cricket Tournament Fixtures")
            if st.session_state.cricket_teams is not None:
                num_teams = len(st.session_state.cricket_teams['Team'].unique())
                if st.button("📋 Generate Cricket Fixtures", type="primary", key="fix_cricket"):
                    fixtures = create_fixtures(num_teams, "Team")
                    st.session_state.cricket_fixtures = fixtures
                if st.session_state.cricket_fixtures:
                    display_fixtures(st.session_state.cricket_fixtures, "cricket")
            else:
                st.info("👈 Please generate cricket teams first in the Team Generator tab")
        
        # --- NEW TAB: Carroms Fixtures ---
        with fixture_tab2:
            st.markdown("### 🕳️ Carroms Tournament Fixtures")
            if st.session_state.carroms_teams is not None:
                num_teams = len(st.session_state.carroms_teams)
                if st.button("📋 Generate Carroms Fixtures", type="primary", key="fix_carroms"):
                    fixtures = create_fixtures(num_teams, "Team")
                    st.session_state.carroms_fixtures = fixtures
                if st.session_state.carroms_fixtures:
                    display_fixtures(st.session_state.carroms_fixtures, "carroms")
            else:
                st.info("👈 Please generate carroms pairs first in the Team Generator tab")
        
        # Chess Fixtures
        with fixture_tab3:
            st.markdown("### ♟️ Chess Tournament Fixtures")
            if st.button("📋 Generate Chess Fixtures", type="primary", key="fix_chess"):
                fixtures = create_chess_fixtures(df, random_seed)
                if fixtures:
                    st.session_state.chess_fixtures = fixtures
                    st.success(f"✅ Fixtures generated!")
                else:
                    st.error("❌ Not enough chess players (minimum 2 required)")
            if st.session_state.chess_fixtures:
                display_fixtures(st.session_state.chess_fixtures, "chess")
        
        # Tug of War Fixtures
        with fixture_tab4:
            st.markdown("### 💪 Tug of War Tournament Fixtures")
            if st.session_state.tug_of_war_teams is not None:
                num_teams = len(st.session_state.tug_of_war_teams['Team'].unique())
                if st.button("📋 Generate Tug of War Fixtures", type="primary", key="fix_tug"):
                    fixtures = create_fixtures(num_teams, "Team")
                    st.session_state.tug_of_war_fixtures = fixtures
                if st.session_state.tug_of_war_fixtures:
                    display_fixtures(st.session_state.tug_of_war_fixtures, "tug_of_war")
            else:
                st.info("👈 Please generate tug of war teams first in the Team Generator tab")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>Made with ❤️ using Streamlit | Sports Tournament Generator v2.1</div>",
    unsafe_allow_html=True
)