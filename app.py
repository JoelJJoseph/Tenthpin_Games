import streamlit as st
import pandas as pd
import numpy as np
import random
import plotly.express as px
# import plotly.graph_objects as go # Not currently used
from io import BytesIO
# import json # Not currently used
import re # Import regex for parsing winner strings
import math # Import math for log2

# --- Page configuration ---
st.set_page_config(
    page_title="Sports Team & Tournament Generator",
    page_icon="🏆",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS ---
st.markdown("""
    <style>
    .main-header {font-size: 2.5rem; font-weight: bold; color: #1f77b4; text-align: center; margin-bottom: 2rem;}
    .sport-header {font-size: 1.8rem; font-weight: bold; color: #2ca02c; margin-top: 1.5rem;}
    .stTabs [data-baseweb="tab-list"] {gap: 2rem;}
    .stTabs [data-baseweb="tab"] {height: 3rem; padding: 0 2rem;}
    .stRadio > label { /* Make radio buttons horizontal */
        display: inline-block;
        margin-right: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Constants ---
# Standardized Column Names
LOC_COL = 'Base Location (Bengaluru/Pune/Hyderabad/Chennai)'
CRICKET_SKILL_COL = 'Primary Skill(Batsman/ Bowler)'
CARROM_COL = 'Carroms (Doubles Only)'
TUG_COL = 'Tug of War'
CRICKET_COL = 'Cricket'
GENDER_COL = 'Gender'
NAME_COL = 'Employee Name'
CHESS_COL = 'Chess'

AVAILABLE_SPORTS = [
    CRICKET_COL, 'Sack Race', TUG_COL, 'Cup Stack Relay',
    'Three Legged Race', 'Push Ups', 'Planks', 'Squats',
    CHESS_COL, CARROM_COL
]

# Source Column Names
SOURCE_NAME_COL = 'Please Enter your Full Name (In CAPITAL Letters)'
SOURCE_GENDER_COL = 'Select Your Gender'
SOURCE_LOC_COL = 'Select Your Base Location'
SOURCE_SPORTS_COL = 'Please select Sporting Event you would like to take part during the TENTHPIN INDIA SPORTS FEST 2025.'
SOURCE_CRICKET_SKILL_COL = 'If you Chose Cricket. Please Select your Primary Skill'

# --- Initialize Session State ---
# Basic app state
if 'data' not in st.session_state: st.session_state.data = None
if 'random_seed' not in st.session_state: st.session_state.random_seed = 42

# Team states (Initialize all keys that might be checked or deleted)
if 'cricket_teams_df' not in st.session_state: st.session_state.cricket_teams_df = None
if 'cricket_unassigned' not in st.session_state: st.session_state.cricket_unassigned = []
if 'carroms_teams_df' not in st.session_state: st.session_state.carroms_teams_df = None
if 'carroms_unassigned' not in st.session_state: st.session_state.carroms_unassigned = []
if 'tug_teams_df' not in st.session_state: st.session_state.tug_teams_df = None
if 'tug_unassigned' not in st.session_state: st.session_state.tug_unassigned = []

# Fixture states
if 'cricket_fixtures_data' not in st.session_state: st.session_state.cricket_fixtures_data = None
if 'carroms_fixtures_data' not in st.session_state: st.session_state.carroms_fixtures_data = None
if 'chess_fixtures_data' not in st.session_state: st.session_state.chess_fixtures_data = None
if 'tug_fixtures_data' not in st.session_state: st.session_state.tug_fixtures_data = None


# --- Helper Functions ---

def clean_column_names(df):
    df.columns = df.columns.str.strip().str.replace(r'\s+', ' ', regex=True).str.replace('\n', ' ', regex=True)
    return df

def transform_sports_column(df):
    if SOURCE_SPORTS_COL in df.columns:
        try:
            # Handle potential NaN values before splitting
            df[SOURCE_SPORTS_COL] = df[SOURCE_SPORTS_COL].fillna('')
            sports_dummies = df[SOURCE_SPORTS_COL].astype(str).str.get_dummies(sep=';')
            sports_dummies.columns = sports_dummies.columns.str.strip()
            sports_dummies = sports_dummies.rename(columns={
                "Carroms (Doubles Only)": CARROM_COL, "Tug of War": TUG_COL,
            })
            # Ensure no empty column names ('') are created
            sports_dummies = sports_dummies.loc[:, sports_dummies.columns != '']

            df = pd.concat([df, sports_dummies], axis=1)
            for col in AVAILABLE_SPORTS:
                if col in df.columns:
                    # Make sure the column exists after potential renaming before applying
                    if col in sports_dummies.columns:
                         df[col] = df[col].apply(lambda x: 'yes' if x == 1 else 'no')
                else:
                    df[col] = 'no' # Ensure all AVAILABLE_SPORTS columns exist
        except Exception as e:
            st.error(f"Error splitting sports column: {e}")
    else:
        st.warning(f"Could not find the sports selection column: '{SOURCE_SPORTS_COL}'")
    return df


def clean_data(df):
    if GENDER_COL in df.columns:
        df[GENDER_COL] = df[GENDER_COL].astype(str).str.strip().replace({'Men': 'Male', 'Women': 'Female'}, regex=False)
        df[GENDER_COL] = df[GENDER_COL].apply(lambda x: x if x in ['Male', 'Female'] else 'Unknown')

    for col in AVAILABLE_SPORTS:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.lower()
            df[col] = df[col].apply(lambda x: 'yes' if x in ['yes', '1'] else 'no')

    if CRICKET_SKILL_COL in df.columns:
        df[CRICKET_SKILL_COL] = df[CRICKET_SKILL_COL].astype(str).str.strip().replace({'Both': 'All Rounder', 'Allrounder': 'All Rounder'}, regex=False)
        df[CRICKET_SKILL_COL] = df[CRICKET_SKILL_COL].apply(lambda x: x if x in ['Batsman', 'Bowler', 'All Rounder'] else '')

    return df

def validate_columns(df):
    required_cols = [NAME_COL, GENDER_COL, LOC_COL]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if not any(sport in df.columns for sport in AVAILABLE_SPORTS):
         st.warning("Note: No sport columns were found or created. Team generation will be disabled.")
         missing_cols.append("Any Sport Column")
    return len(missing_cols) == 0, missing_cols

def get_yes_count(df, sport_col):
    if sport_col not in df.columns: return 0
    return df[sport_col].astype(str).str.lower().str.contains('yes', na=False).sum()

def to_excel(df):
    output = BytesIO()
    try:
        # Use xlsxwriter if available for better formatting, fallback to openpyxl
        import xlsxwriter
        engine = 'xlsxwriter'
    except ImportError:
        engine = 'openpyxl'
    with pd.ExcelWriter(output, engine=engine) as writer:
        df.to_excel(writer, index=False, sheet_name='Sheet1')
    return output.getvalue()


# --- Team Generation Functions ---

def create_cricket_teams(df, random_seed=None):
    if random_seed: random.seed(random_seed); np.random.seed(random_seed)
    if CRICKET_COL not in df.columns: return None, [], "Cricket column not found."
    if CRICKET_SKILL_COL not in df.columns: return None, [], "Cricket Skill column not found."

    cricket_df = df[df[CRICKET_COL].astype(str).str.lower() == 'yes'].copy()
    if len(cricket_df) < 11: return None, [], "Not enough players (min 11)."

    all_players_indices = cricket_df.index.tolist()
    random.shuffle(all_players_indices)
    num_teams = len(all_players_indices) // 11
    if num_teams == 0: return None, [], "Not enough players (min 11)."

    teams = []; players_used_indices = []
    for i in range(num_teams):
        team_indices = all_players_indices[i*11 : (i+1)*11]
        players_used_indices.extend(team_indices)
        teams.append(team_indices)
        # Note: Balancing logic omitted for brevity as per previous version
         
    if not teams: return None, [], "Failed to form teams."

    team_dfs = []
    for i, team_indices in enumerate(teams, 1):
        team_data = df.loc[team_indices][[NAME_COL, GENDER_COL, LOC_COL, CRICKET_SKILL_COL]].copy()
        team_data['Team'] = f'Team {i}'
        team_dfs.append(team_data)

    final_teams_df = pd.concat(team_dfs, ignore_index=True)
    unassigned_indices = list(set(cricket_df.index) - set(players_used_indices))
    unassigned_players = df.loc[unassigned_indices][[NAME_COL, GENDER_COL, LOC_COL, CRICKET_SKILL_COL]].to_dict('records')
    return final_teams_df, unassigned_players, None

def create_carroms_pairs(df, random_seed=None):
    if random_seed: random.seed(random_seed); np.random.seed(random_seed)
    if CARROM_COL not in df.columns: return None, [], "Carroms column not found."

    carroms_df = df[df[CARROM_COL].astype(str).str.lower() == 'yes'].copy()
    if len(carroms_df) < 2: return None, [], "Not enough players (min 2)."

    player_indices = carroms_df.index.tolist()
    random.shuffle(player_indices)
    pairs_list = []; num_pairs = len(player_indices) // 2; players_used_indices = []

    for i in range(num_pairs):
        idx1 = player_indices[i*2]; idx2 = player_indices[i*2 + 1]
        players_used_indices.extend([idx1, idx2])
        player1 = df.loc[idx1]; player2 = df.loc[idx2]
        pairs_list.append({
            'Team': f'Team {i+1}', 'Player 1': player1[NAME_COL], 'Player 2': player2[NAME_COL],
            'Location 1': player1[LOC_COL], 'Location 2': player2[LOC_COL],
            '_Player1_idx': idx1, '_Player2_idx': idx2 # Store index
        })

    final_pairs_df = pd.DataFrame(pairs_list)
    unassigned_indices = list(set(carroms_df.index) - set(players_used_indices))
    unassigned_players = df.loc[unassigned_indices][[NAME_COL, GENDER_COL, LOC_COL]].to_dict('records')
    leftover_msg = f"Note: {df.loc[player_indices[-1]][NAME_COL]} is unassigned." if len(player_indices) % 2 == 1 else None
    return final_pairs_df if not final_pairs_df.empty else None, unassigned_players, leftover_msg

def create_tug_of_war_teams(df, random_seed=None):
    team_size = 9; num_teams_required = 6; total_players_needed = team_size * num_teams_required
    if random_seed: random.seed(random_seed); np.random.seed(random_seed)
    if TUG_COL not in df.columns: return None, [], "Tug of War column not found."

    tug_df = df[df[TUG_COL].astype(str).str.lower() == 'yes'].copy()
    if len(tug_df) < total_players_needed:
        return None, [], f"Need {total_players_needed}, found {len(tug_df)}."

    all_players_indices = tug_df.index.tolist()
    random.shuffle(all_players_indices)
    selected_players_indices = all_players_indices[:total_players_needed]
    unassigned_indices = all_players_indices[total_players_needed:]

    teams = []
    for i in range(num_teams_required):
        teams.append(selected_players_indices[i*team_size : (i+1)*team_size])

    if len(teams) != num_teams_required: return None, [], "Error forming teams."

    team_dfs = []
    for i, team_indices in enumerate(teams, 1):
        team_data = df.loc[team_indices][[NAME_COL, GENDER_COL, LOC_COL]].copy()
        team_data['Team'] = f'Team {i}'
        team_dfs.append(team_data)

    final_teams_df = pd.concat(team_dfs, ignore_index=True)
    unassigned_players = df.loc[unassigned_indices][[NAME_COL, GENDER_COL, LOC_COL]].to_dict('records')
    leftover_msg = f"{len(unassigned_players)} players unassigned." if unassigned_players else None
    return final_teams_df, unassigned_players, leftover_msg


# --- Fixture Generation & Update Functions ---

# --- *** NEW FUNCTION: create_structured_fixtures (v2.5) *** ---
def create_structured_fixtures(participants, is_teams=True):
    """
    Generates fixture data structure using a naive single-elimination bracket
    that handles non-power-of-2 numbers by adding byes *as needed* in each round.
    This ensures all players play in Round 1 (if even).
    e.g., 24 -> 12 -> 6 -> 3 -> 2 (1 bye) -> 1
    """
    if not participants or len(participants) < 2:
        return None
    
    current_round_participants = list(participants)
    random.shuffle(current_round_participants)
    num_participants = len(current_round_participants)

    fixtures = {"rounds": [], "matches": {}}
    round_num = 1
    
    while len(current_round_participants) > 1:
        round_matches_data = [] # Store dicts for this round
        next_round_participants = []
        current_size = len(current_round_participants)
        
        round_name = "Final" if current_size == 2 else \
                     "Semi Finals" if current_size == 3 else \
                     f"Round of {current_size}"

        # Handle odd number of participants in this round
        bye_player = None
        if current_size % 2 != 0:
            # Pop one player to give them a bye
            bye_player = current_round_participants.pop()
            next_round_participants.append(bye_player) # They advance automatically
        
        # Pair up the remaining (even number)
        round_match_counter = 1
        for i in range(0, len(current_round_participants), 2):
            match_id = f"R{round_num}M{round_match_counter}"
            p1 = current_round_participants[i]
            p2 = current_round_participants[i+1]
            
            match_data = {'id': match_id, 'round': round_num, 'team1': p1, 'team2': p2, 'winner': None, 'next_match_id': None}
            winner_placeholder = f"Winner({match_id})"
            next_round_participants.append(winner_placeholder)
            
            # Link previous round matches to this one
            if isinstance(p1, str) and p1.startswith("Winner("):
                prev_match_id = re.search(r'\((.*?)\)', p1).group(1)
                if prev_match_id in fixtures["matches"]:
                    fixtures["matches"][prev_match_id]["next_match_id"] = match_id
            
            if isinstance(p2, str) and p2.startswith("Winner("):
                prev_match_id = re.search(r'\((.*?)\)', p2).group(1)
                if prev_match_id in fixtures["matches"]:
                    fixtures["matches"][prev_match_id]["next_match_id"] = match_id

            round_matches_data.append(match_data)
            fixtures["matches"][match_id] = match_data
            round_match_counter += 1

        # Add the bye player match data (as a non-playable "match")
        if bye_player:
            match_id = f"R{round_num}MBYE"
            match_data = {'id': match_id, 'round': round_num, 'team1': bye_player, 'team2': "BYE", 'winner': bye_player, 'next_match_id': None}
            fixtures["matches"][match_id] = match_data
            round_matches_data.append(match_data) # Add to round for display

        fixtures["rounds"].append({"name": round_name, "matches": [m['id'] for m in round_matches_data]})
        
        # Shuffle next round participants so the bye isn't always in the same slot
        random.shuffle(next_round_participants)
        current_round_participants = next_round_participants
        round_num += 1
        if round_num > 10: break # Safety break

    return fixtures


def update_fixtures(fixtures_data, sport_key_prefix):
    """Updates placeholders based on selected winners in session state."""
    if not fixtures_data or "matches" not in fixtures_data: return fixtures_data, False
    matches = fixtures_data["matches"]
    something_updated = False # Flag to check if a rerun is needed

    for match_id, match in matches.items():
        # --- KEY FIX ---
        # Make the key unique by adding the sport_key_prefix
        winner_key = f"winner_{sport_key_prefix}_{match_id}" 
        
        if winner_key in st.session_state and st.session_state[winner_key]:
            selected_winner = st.session_state[winner_key]
            if match['winner'] != selected_winner: # Only update if changed
                 match['winner'] = selected_winner
                 something_updated = True

                 # Update the next match if needed
                 next_match_id = match.get('next_match_id')
                 if next_match_id and next_match_id in matches:
                      next_match = matches[next_match_id]
                      placeholder = f"Winner({match_id})"
                      if next_match['team1'] == placeholder: next_match['team1'] = selected_winner
                      elif next_match['team2'] == placeholder: next_match['team2'] = selected_winner

    # If any winner was updated, trigger a rerun in the calling function
    return fixtures_data, something_updated


def display_fixture_ui(fixtures_data, sport_key_prefix):
    """Renders the interactive fixture UI."""
    if not fixtures_data or not fixtures_data.get("rounds"):
        st.info("Fixtures not generated yet.")
        return

    matches = fixtures_data.get("matches", {}) # Use .get for safety

    for i, round_data in enumerate(fixtures_data["rounds"]):
        st.markdown(f"#### 🏆 {round_data['name']}")
        match_ids_in_round = round_data.get("matches", []) # Use .get

        for match_id in match_ids_in_round:
            match = matches.get(match_id) # Use .get
            if not match:
                 st.warning(f"Match data for {match_id} not found.")
                 continue

            team1, team2, winner = match['team1'], match['team2'], match['winner']
            display_id = match_id.replace("R", "Rd ").replace("M", " M ")

            if winner: # Match decided
                # Handle BYE display for the new logic
                if team2 == "BYE": st.markdown(f"**{display_id}:** {team1} (BYE - Auto Advance)")
                elif team1 == "BYE": st.markdown(f"**{display_id}:** {team2} (BYE - Auto Advance)")
                else:
                    win_style = "color: green; font-weight: bold;"
                    t1_disp = f"<span style='{win_style}'>{team1}</span>" if team1 == winner else team1
                    t2_disp = f"<span style='{win_style}'>{team2}</span>" if team2 == winner else team2
                    st.markdown(f"**{display_id}:** {t1_disp} vs {t2_disp} -> **Winner: {winner}**", unsafe_allow_html=True)

            elif isinstance(team1, str) and team1.startswith("Winner(") or \
                 isinstance(team2, str) and team2.startswith("Winner("): # Depends on previous round
                st.markdown(f"**{display_id}:** {team1} vs {team2}")

            else: # Playable match
                st.markdown(f"**{display_id}:** {team1} vs {team2}")
                
                # --- KEY FIX ---
                # Make the key unique by adding the sport_key_prefix
                winner_key = f"winner_{sport_key_prefix}_{match_id}"
                
                # Get current winner from state for default radio selection
                current_winner_selection = st.session_state.get(winner_key, None)
                st.radio("Select Winner:", [team1, team2], index=([team1, team2].index(current_winner_selection) if current_winner_selection in [team1, team2] else None),
                         key=winner_key, horizontal=True, label_visibility="collapsed")
        st.markdown("---")

    if st.button("Update Fixtures", key=f"update_{sport_key_prefix}"):
        
        # --- KEY FIX ---
        # Pass the sport_key_prefix to the update function
        updated_data, updated = update_fixtures(fixtures_data, sport_key_prefix)
        
        if updated:
            st.session_state[f"{sport_key_prefix}_fixtures_data"] = updated_data
            st.rerun() # Rerun ONLY if something changed
        else:
             st.toast("No new winners selected.") # Feedback if nothing selected


    # Download Button
    fixture_text = f"{sport_key_prefix.upper()} TOURNAMENT FIXTURES\n{'='*50}\n"
    for round_data in fixtures_data["rounds"]:
        fixture_text += f"\n{round_data['name']}\n{'='*50}\n"
        for match_id in round_data.get("matches", []):
            match = matches.get(match_id)
            if match:
                 t1, t2, w = match['team1'], match['team2'], match['winner']
                 status = f" -> Winner: {w}" if w else ""
                 if t2 == "BYE": line = f"{t1} (BYE - Auto Advance)"
                 elif t1 == "BYE": line = f"{t2} (BYE - Auto Advance)"
                 else: line = f"{t1} vs {t2}{status}"
                 fixture_text += f"{match['id']}: {line}\n"
        fixture_text += "\n"

    st.download_button(label="📥 Download Fixtures (TXT)", data=fixture_text.encode('utf-8'),
                       file_name=f"{sport_key_prefix}_fixtures.txt", mime="text/plain")



# --- Streamlit App UI ---
st.markdown('<p class="main-header">🏆 Sports Team & Tournament Generator</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### 🏆 Sports Tournament App")
    st.markdown("---")
    st.markdown("### 1. File Settings")
    header_row = st.number_input("Header Row in Excel/CSV", min_value=1, value=1, step=1,
                                 help="Row number containing column headers (usually 1).")
    header_index = header_row - 1
    uploaded_file = st.file_uploader("📁 Upload Excel or CSV File", type=['xlsx', 'xls', 'csv'], key="file_uploader")

    if uploaded_file:
        try:
            file_buffer = BytesIO(uploaded_file.getvalue())
            df_input = pd.read_excel(file_buffer, header=header_index, engine='openpyxl') if uploaded_file.name.lower().endswith(('.xlsx', '.xls')) else pd.read_csv(file_buffer, header=header_index)
            df_input = clean_column_names(df_input)
            df_input = df_input.dropna(how='all')
            df_input = df_input.rename(columns={
                SOURCE_NAME_COL: NAME_COL, SOURCE_GENDER_COL: GENDER_COL,
                SOURCE_LOC_COL: LOC_COL, SOURCE_CRICKET_SKILL_COL: CRICKET_SKILL_COL,
            })
            df_input = transform_sports_column(df_input)
            valid, missing = validate_columns(df_input)

            if valid:
                current_data = st.session_state.get('data')
                new_data = clean_data(df_input)
                # Check if data actually changed before resetting state
                if current_data is None or not current_data.equals(new_data):
                    st.session_state.data = new_data
                    st.success(f"✅ File loaded! {len(st.session_state.data)} participants.")
                    # Clear old team/fixture data ONLY if data changed
                    keys_to_clear = [k for k in st.session_state if '_teams_df' in k or '_unassigned' in k or '_fixtures_data' in k]
                    if keys_to_clear:
                        for k in keys_to_clear: 
                            if k in st.session_state: del st.session_state[k]
                        st.info("Data changed. Previous teams/fixtures cleared.")
                    st.rerun() # Rerun to reflect new data and cleared state
                else:
                     st.success("File processed. No changes detected in data.")

            else:
                 st.error(f"❌ Missing required columns: {', '.join(missing)}")
                 file_buffer.seek(0)
                 original_df = pd.read_excel(file_buffer, header=header_index, engine='openpyxl') if uploaded_file.name.lower().endswith(('.xlsx', '.xls')) else pd.read_csv(file_buffer, header=header_index)
                 st.dataframe(original_df.head(2))
                 st.warning(f"Looking for headers in Row {header_row}. Need column: '{SOURCE_SPORTS_COL}'.")
                 st.session_state.data = None

        except Exception as e:
            st.error(f"💥 Error reading file: {e}")
            st.session_state.data = None

    st.markdown("---")

    if st.session_state.get('data') is not None: # Use get here too
        st.markdown("### ⚙️ Settings")
        st.session_state.random_seed = st.number_input("Random Seed", min_value=0, value=st.session_state.random_seed, step=1, key="random_seed_input")
        st.markdown("### 📊 Quick Stats")
        df_quick = st.session_state.data
        st.metric("Total Participants", len(df_quick))
        if GENDER_COL in df_quick.columns:
            m, f = df_quick[GENDER_COL].str.contains('Male', na=False).sum(), df_quick[GENDER_COL].str.contains('Female', na=False).sum()
            st.metric("Male / Female", f"{m} / {f}")


# Main Content Area
if st.session_state.get('data') is None: # Use get
    st.info("👈 Upload participant data from the sidebar to begin.")
    st.markdown("### 📋 Expected Format")
    st.markdown(f"Your Excel/CSV needs columns like: `{SOURCE_NAME_COL}`, `{SOURCE_GENDER_COL}`, `{SOURCE_LOC_COL}`, and importantly: **`{SOURCE_SPORTS_COL}`** where sports are separated by semicolons (`;`). Set the correct header row in the sidebar.")

else: # Data is loaded
    df_main = st.session_state.data
    active_sports = [s for s in AVAILABLE_SPORTS if s in df_main.columns and get_yes_count(df_main, s) > 0]

    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Dashboard", "📋 Participant Lists", "👥 Team Generator", "🏆 Fixture Creator"
    ])

    # --- TAB 1: Dashboard ---
    with tab1:
        st.markdown("## 📊 Data Summary")
        col1, col2 = st.columns([1, 1])
        with col1:
             st.markdown("### 📄 Data Preview")
             display_cols = [NAME_COL, GENDER_COL, LOC_COL, CRICKET_COL, CHESS_COL, TUG_COL, CARROM_COL]
             available_display_cols = [col for col in display_cols if col in df_main.columns]
             if available_display_cols: st.dataframe(df_main[available_display_cols].head(10))
             else: st.warning("No standard columns found for preview.")
        with col2:
             st.markdown("### 🎯 Participation Overview")
             participation = {s: get_yes_count(df_main, s) for s in active_sports}
             if participation:
                 sorted_p = dict(sorted(participation.items(), key=lambda item: item[1], reverse=True))
                 fig = px.bar(x=list(sorted_p.keys()), y=list(sorted_p.values()), title="Participation by Sport")
                 fig.update_layout(xaxis_title="Sport", yaxis_title="Participants", xaxis_tickangle=-45)
                 st.plotly_chart(fig, use_container_width=True)
             else: st.info("No participants found for available sports.")
        st.markdown("---")
        col3, col4 = st.columns(2)
        with col3:
             if GENDER_COL in df_main.columns:
                gender_counts = df_main[GENDER_COL].value_counts()
                if not gender_counts.empty:
                    fig_gender = px.pie(values=gender_counts.values, names=gender_counts.index, title='Gender Distribution', hole=0.4)
                    st.plotly_chart(fig_gender, use_container_width=True)
        with col4:
             if LOC_COL in df_main.columns:
                loc_counts = df_main[LOC_COL].value_counts()
                if not loc_counts.empty:
                    fig_loc = px.bar(x=loc_counts.index, y=loc_counts.values, title='Participants by Location', labels={'x':'Location', 'y':'Count'})
                    st.plotly_chart(fig_loc, use_container_width=True)


    # --- TAB 2: Participant Lists ---
    with tab2:
        st.markdown("## 📋 Participant Lists by Game")
        st.markdown("Select an event to view and download the participant list.")
        if not active_sports: st.warning("No participants found for any tracked sport.")
        else:
            selected_sport = st.selectbox("Select Event:", options=active_sports, key="participant_sport_select")
            if selected_sport:
                participants_df = df_main[df_main[selected_sport].astype(str).str.lower() == 'yes'].copy()
                cols_to_show = [NAME_COL, GENDER_COL, LOC_COL]
                if selected_sport == CRICKET_COL and CRICKET_SKILL_COL in df_main.columns: cols_to_show.append(CRICKET_SKILL_COL)
                display_df = participants_df[[col for col in cols_to_show if col in participants_df.columns]]

                st.metric(f"Participants in {selected_sport}", len(display_df))
                if not display_df.empty:
                    st.dataframe(display_df, hide_index=True, use_container_width=True)
                    excel_bytes = to_excel(display_df)
                    st.download_button( f"📥 Download {selected_sport} List (Excel)", excel_bytes,
                        file_name=f"{selected_sport.replace(' ', '_')}_participants.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key=f"dl_{selected_sport}")
                else: st.info("No participants for this event.")


    # --- TAB 3: Team Generator ---
    with tab3:
        st.markdown("## 👥 Team Generator")
        tgen_tab1, tgen_tab2, tgen_tab3 = st.tabs(["🏏 Cricket", "🕳️ Carroms", "💪 Tug of War"])

        # --- Cricket ---
        with tgen_tab1:
            st.markdown("### 🏏 Cricket Teams (11 players)")
            if CRICKET_COL not in active_sports: st.warning("No participants registered for Cricket.")
            else:
                if st.button("🎲 Generate Cricket Teams", key="gen_cricket_btn"):
                    teams_df, unassigned, error = create_cricket_teams(df_main, st.session_state.random_seed)
                    if error: st.error(error)
                    else:
                        st.session_state.cricket_teams_df = teams_df
                        st.session_state.cricket_unassigned = unassigned
                        st.success(f"Generated {teams_df['Team'].nunique()} teams.")
                        if unassigned: st.info(f"{len(unassigned)} players unassigned.")
                        st.rerun()

                # --- Use .get() for checking existence ---
                if st.session_state.get("cricket_teams_df") is not None:
                    teams_df = st.session_state.cricket_teams_df
                    unassigned_list = st.session_state.get("cricket_unassigned", [])
                    unassigned_names = [p[NAME_COL] for p in unassigned_list if NAME_COL in p] # Safer access

                    st.markdown("#### 📋 Team Compositions")
                    for team_name in sorted(teams_df['Team'].unique()):
                        with st.expander(f"👥 {team_name} ({len(teams_df[teams_df['Team'] == team_name])} players)"):
                            team_data = teams_df[teams_df['Team'] == team_name]
                            st.dataframe(team_data[[NAME_COL, GENDER_COL, LOC_COL, CRICKET_SKILL_COL]], hide_index=True)

                            st.markdown("---"); st.markdown("**Edit Team:**")
                            current_player_names = team_data[NAME_COL].tolist()
                            players_to_remove = st.multiselect("Select players to REMOVE:", current_player_names, key=f"remove_cricket_{team_name}")
                            players_to_add = st.multiselect("Select players to ADD:", unassigned_names, key=f"add_cricket_{team_name}")

                            if st.button(f"Update {team_name}", key=f"update_cricket_{team_name}"):
                                current_df = st.session_state.cricket_teams_df
                                current_unassigned = st.session_state.cricket_unassigned
                                idx_to_remove = current_df[(current_df['Team'] == team_name) & (current_df[NAME_COL].isin(players_to_remove))].index
                                removed_players_data = current_df.loc[idx_to_remove].to_dict('records')
                                current_df = current_df.drop(index=idx_to_remove)
                                players_to_add_data = [p for p in current_unassigned if p[NAME_COL] in players_to_add]
                                for p_data in players_to_add_data:
                                    p_data['Team'] = team_name
                                    current_df = pd.concat([current_df, pd.DataFrame([p_data])], ignore_index=True)
                                new_unassigned = [p for p in current_unassigned if p[NAME_COL] not in players_to_add]
                                for p_removed in removed_players_data:
                                     new_unassigned.append({k:v for k,v in p_removed.items() if k != 'Team'})

                                st.session_state.cricket_teams_df = current_df.sort_values(by=['Team', NAME_COL]).reset_index(drop=True)
                                st.session_state.cricket_unassigned = new_unassigned
                                st.success(f"{team_name} updated.")
                                st.rerun()

                    st.markdown("---"); col1, col2 = st.columns(2)
                    with col1:
                        excel_bytes = to_excel(st.session_state.cricket_teams_df)
                        st.download_button("📥 All Teams (Excel)", excel_bytes, "cricket_teams_edited.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
                    with col2:
                        csv_bytes = st.session_state.cricket_teams_df.to_csv(index=False).encode('utf-8')
                        st.download_button("📥 All Teams (CSV)", csv_bytes, "cricket_teams_edited.csv", mime="text/csv")

        # --- Carroms ---
        with tgen_tab2:
            st.markdown("### 🕳️ Carroms Doubles (2 players)")
            if CARROM_COL not in active_sports: st.warning("No participants registered for Carroms.")
            else:
                if st.button("🎲 Generate Carroms Pairs", key="gen_carroms_btn"):
                    pairs_df, unassigned, error_msg = create_carroms_pairs(df_main, st.session_state.random_seed)
                    if error_msg: st.info(error_msg) # Show leftover msg as info
                    if pairs_df is not None:
                        st.session_state.carroms_teams_df = pairs_df
                        st.session_state.carroms_unassigned = unassigned
                        st.success(f"Generated {len(pairs_df)} pairs.")
                        st.rerun()
                    else: st.error("Failed to generate pairs (likely not enough players).")

                # --- Use .get() ---
                if st.session_state.get("carroms_teams_df") is not None:
                    teams_df = st.session_state.carroms_teams_df
                    st.markdown("#### 📋 Pair Compositions")
                    st.dataframe(teams_df[['Team', 'Player 1', 'Player 2', 'Location 1', 'Location 2']], hide_index=True)
                    st.info("Pair editing not implemented. Regenerate or edit downloaded file.")
                    st.markdown("---"); col1, col2 = st.columns(2)
                    with col1:
                         excel_bytes = to_excel(teams_df[['Team', 'Player 1', 'Player 2', 'Location 1', 'Location 2']])
                         st.download_button("📥 Pairs (Excel)", excel_bytes, "carroms_pairs.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
                    with col2:
                         csv_bytes = teams_df[['Team', 'Player 1', 'Player 2', 'Location 1', 'Location 2']].to_csv(index=False).encode('utf-8')
                         st.download_button("📥 Pairs (CSV)", csv_bytes, "carroms_pairs.csv", mime="text/csv")

        # --- Tug of War ---
        with tgen_tab3:
            st.markdown("### 💪 Tug of War Teams (9 players, 6 teams)")
            if TUG_COL not in active_sports: st.warning("No participants registered for Tug of War.")
            else:
                if st.button("🎲 Generate Tug of War Teams", key="gen_tug_btn"):
                    teams_df, unassigned, error_msg = create_tug_of_war_teams(df_main, st.session_state.random_seed)
                    if error_msg: st.info(error_msg) # Show unassigned msg as info
                    if teams_df is not None:
                        st.session_state.tug_teams_df = teams_df
                        st.session_state.tug_unassigned = unassigned # List of dicts
                        st.success(f"Generated {teams_df['Team'].nunique()} teams.")
                        st.rerun()
                    else: st.error("Failed to generate teams (check player count).")

                # --- Use .get() ---
                if st.session_state.get("tug_teams_df") is not None:
                    teams_df = st.session_state.tug_teams_df
                    unassigned_list = st.session_state.get("tug_unassigned", [])
                    unassigned_names = [p[NAME_COL] for p in unassigned_list if NAME_COL in p]

                    st.markdown("#### 📋 Team Compositions")
                    for team_name in sorted(teams_df['Team'].unique()):
                         with st.expander(f"👥 {team_name} ({len(teams_df[teams_df['Team'] == team_name])} players)"):
                             team_data = teams_df[teams_df['Team'] == team_name]
                             st.dataframe(team_data[[NAME_COL, GENDER_COL, LOC_COL]], hide_index=True)

                             st.markdown("---"); st.markdown("**Edit Team:**")
                             current_player_names = team_data[NAME_COL].tolist()
                             players_to_remove = st.multiselect("Select players to REMOVE:", current_player_names, key=f"remove_tug_{team_name}")
                             players_to_add = st.multiselect("Select players to ADD:", unassigned_names, key=f"add_tug_{team_name}")

                             if st.button(f"Update {team_name}", key=f"update_tug_{team_name}"):
                                 current_df = st.session_state.tug_teams_df
                                 current_unassigned = st.session_state.tug_unassigned
                                 idx_to_remove = current_df[(current_df['Team'] == team_name) & (current_df[NAME_COL].isin(players_to_remove))].index
                                 removed_players_data = current_df.loc[idx_to_remove].to_dict('records')
                                 current_df = current_df.drop(index=idx_to_remove)
                                 players_to_add_data = [p for p in current_unassigned if p[NAME_COL] in players_to_add]
                                 for p_data in players_to_add_data:
                                     p_data['Team'] = team_name
                                     current_df = pd.concat([current_df, pd.DataFrame([p_data])], ignore_index=True)
                                 new_unassigned = [p for p in current_unassigned if p[NAME_COL] not in players_to_add]
                                 for p_removed in removed_players_data: new_unassigned.append({k:v for k,v in p_removed.items() if k != 'Team'})

                                 st.session_state.tug_teams_df = current_df.sort_values(by=['Team', NAME_COL]).reset_index(drop=True)
                                 st.session_state.tug_unassigned = new_unassigned
                                 st.success(f"{team_name} updated.")
                                 st.warning("Note: Team size/count constraints may be broken after editing.")
                                 st.rerun()

                    st.markdown("---"); col1, col2 = st.columns(2)
                    with col1:
                        excel_bytes = to_excel(st.session_state.tug_teams_df)
                        st.download_button("📥 All Teams (Excel)", excel_bytes, "tug_teams_edited.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
                    with col2:
                        csv_bytes = st.session_state.tug_teams_df.to_csv(index=False).encode('utf-8')
                        st.download_button("📥 All Teams (CSV)", csv_bytes, "tug_teams_edited.csv", mime="text/csv")


    # --- TAB 4: Fixture Creator ---
    with tab4:
        st.markdown("## 🏆 Fixture Creator")
        st.info("Select winners using the radio buttons and click 'Update Fixtures' below each bracket to advance them.")

        fix_tab1, fix_tab2, fix_tab3, fix_tab4 = st.tabs([
            "🏏 Cricket", "🕳️ Carroms", "♟️ Chess", "💪 Tug of War"
        ])

        with fix_tab1: # Cricket
            st.markdown("### 🏏 Cricket Fixtures")
            # --- Use .get() ---
            if st.session_state.get("cricket_teams_df") is not None:
                team_names = st.session_state.cricket_teams_df['Team'].unique().tolist()
                if st.button("Generate Cricket Fixtures", key="gen_fix_cricket"):
                     st.session_state.cricket_fixtures_data = create_structured_fixtures(team_names, is_teams=True)
                     st.rerun()
                # --- Use .get() ---
                if st.session_state.get("cricket_fixtures_data"):
                    display_fixture_ui(st.session_state.cricket_fixtures_data, "cricket")
            else: st.info("Generate Cricket teams first.")

        with fix_tab2: # Carroms
            st.markdown("### 🕳️ Carroms Fixtures")
            # --- Use .get() ---
            if st.session_state.get("carroms_teams_df") is not None:
                team_names = st.session_state.carroms_teams_df['Team'].unique().tolist()
                if st.button("Generate Carroms Fixtures", key="gen_fix_carroms"):
                     st.session_state.carroms_fixtures_data = create_structured_fixtures(team_names, is_teams=True)
                     st.rerun()
                # --- Use .get() ---
                if st.session_state.get("carroms_fixtures_data"):
                    display_fixture_ui(st.session_state.carroms_fixtures_data, "carroms")
            else: st.info("Generate Carroms pairs first.")

        with fix_tab3: # Chess
            st.markdown("### ♟️ Chess Fixtures")
            if CHESS_COL in active_sports:
                player_names = df_main[df_main[CHESS_COL]=='yes'][NAME_COL].tolist()
                if len(player_names) < 2: st.warning("Need at least 2 Chess players.")
                else:
                     if st.button("Generate Chess Fixtures", key="gen_fix_chess"):
                          st.session_state.chess_fixtures_data = create_structured_fixtures(player_names, is_teams=False)
                          st.rerun()
                     # --- Use .get() ---
                     if st.session_state.get("chess_fixtures_data"):
                          display_fixture_ui(st.session_state.chess_fixtures_data, "chess")
            else: st.info("No Chess participants found.")


        with fix_tab4: # Tug of War
             st.markdown("### 💪 Tug of War Fixtures")
             # --- Use .get() ---
             if st.session_state.get("tug_teams_df") is not None:
                  team_names = st.session_state.tug_teams_df['Team'].unique().tolist()
                  if st.button("Generate Tug of War Fixtures", key="gen_fix_tug"):
                       st.session_state.tug_fixtures_data = create_structured_fixtures(team_names, is_teams=True)
                       st.rerun()
                  # --- Use .get() ---
                  if st.session_state.get("tug_fixtures_data"):
                      display_fixture_ui(st.session_state.tug_fixtures_data, "tug")
             else: st.info("Generate Tug of War teams first.")


# --- Footer ---
st.markdown("---")
st.markdown("<div style='text-align: center; color: gray;'>Sports Tournament Generator v2.5</div>", unsafe_allow_html=True)