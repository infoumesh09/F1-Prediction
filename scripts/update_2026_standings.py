"""
Update 2026 F1 Season Standings
Fetches the latest race results from FastF1 and updates f1_2026_grid.csv
"""

import pandas as pd
import numpy as np
import fastf1
from fastf1.ergast import Ergast
import sys
import os
from datetime import datetime

# Add parent to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 2026 driver/team mapping (from F1-2026-Prediction-main/Experiments/Race/ETL.py)
DRIVER_TEAM_MAP = {
    "NOR": "McLaren",
    "PIA": "McLaren",
    "VER": "Red Bull Racing",
    "HAD": "Red Bull Racing",
    "LEC": "Ferrari",
    "HAM": "Ferrari",
    "RUS": "Mercedes",
    "ANT": "Mercedes",
    "ALO": "Aston Martin",
    "STR": "Aston Martin",
    "ALB": "Williams",
    "SAI": "Williams",
    "OCO": "Haas F1 Team",
    "BEA": "Haas F1 Team",
    "GAS": "Alpine",
    "COL": "Alpine",
    "HUL": "Audi",
    "BOR": "Audi",
    "PER": "Cadillac",
    "BOT": "Cadillac",
    "LAW": "Racing Bulls",
    "LIN": "Racing Bulls",
}

# Driver code to full name mapping
DRIVER_NAMES = {
    "NOR": "Lando Norris",
    "PIA": "Oscar Piastri",
    "VER": "Max Verstappen",
    "HAD": "Isack Hadjar",
    "LEC": "Charles Leclerc",
    "HAM": "Lewis Hamilton",
    "RUS": "George Russell",
    "ANT": "Andrea Kimi Antonelli",
    "ALO": "Fernando Alonso",
    "STR": "Lance Stroll",
    "ALB": "Alexander Albon",
    "SAI": "Carlos Sainz",
    "OCO": "Esteban Ocon",
    "BEA": "Oliver Bearman",
    "GAS": "Pierre Gasly",
    "COL": "Franco Colapinto",
    "HUL": "Nico Hulkenberg",
    "BOR": "Gabriel Bortoleto",
    "PER": "Sergio Perez",
    "BOT": "Valtteri Bottas",
    "LAW": "Liam Lawson",
    "LIN": "Arvid Lindblad",
}

# Constructor ID mapping (for CSV consistency)
CONSTRUCTOR_IDS = {
    "McLaren": "mclaren",
    "Red Bull Racing": "red_bull",
    "Ferrari": "ferrari",
    "Mercedes": "mercedes",
    "Aston Martin": "aston_martin",
    "Williams": "williams",
    "Haas F1 Team": "haas",
    "Alpine": "alpine",
    "Audi": "audi",
    "Cadillac": "cadillac",
    "Racing Bulls": "racing_bulls",
}

DRIVER_IDS = {
    "Lando Norris": "norris",
    "Oscar Piastri": "piastri",
    "Max Verstappen": "verstappen",
    "Isack Hadjar": "hadjar",
    "Charles Leclerc": "leclerc",
    "Lewis Hamilton": "hamilton",
    "George Russell": "russell",
    "Andrea Kimi Antonelli": "antonelli",
    "Fernando Alonso": "alonso",
    "Lance Stroll": "stroll",
    "Alexander Albon": "albon",
    "Carlos Sainz": "sainz",
    "Esteban Ocon": "ocon",
    "Oliver Bearman": "bearman",
    "Pierre Gasly": "gasly",
    "Franco Colapinto": "colapinto",
    "Nico Hulkenberg": "hulkenberg",
    "Gabriel Bortoleto": "bortoleto",
    "Sergio Perez": "perez",
    "Valtteri Bottas": "bottas",
    "Liam Lawson": "lawson",
    "Arvid Lindblad": "lindblad",
}

NATIONALITIES = {
    "Lando Norris": "British",
    "Oscar Piastri": "Australian",
    "Max Verstappen": "Dutch",
    "Isack Hadjar": "French",
    "Charles Leclerc": "Monegasque",
    "Lewis Hamilton": "British",
    "George Russell": "British",
    "Andrea Kimi Antonelli": "Italian",
    "Fernando Alonso": "Spanish",
    "Lance Stroll": "Canadian",
    "Alexander Albon": "Thai",
    "Carlos Sainz": "Spanish",
    "Esteban Ocon": "French",
    "Oliver Bearman": "British",
    "Pierre Gasly": "French",
    "Franco Colapinto": "Argentine",
    "Nico Hulkenberg": "German",
    "Gabriel Bortoleto": "Brazilian",
    "Sergio Perez": "Mexican",
    "Valtteri Bottas": "Finnish",
    "Liam Lawson": "New Zealander",
    "Arvid Lindblad": "British",
}


def sprint_points_map(season=2026, round_num=24):
    """Return {driver_code: sprint_points} for a sprint weekend, {} otherwise."""
    try:
        spr = fastf1.get_session(season, round_num, 'S')
        spr.load(telemetry=False, weather=False, messages=False)
        res = spr.results
        if res is None or len(res) == 0 or 'Points' not in res.columns:
            return {}
        out = {}
        for _, row in res.iterrows():
            code = row.get('Abbreviation', row.get('DriverNumber', 'UNK'))
            pts = row.get('Points', 0)
            if pd.notna(pts) and pts > 0:
                out[code] = out.get(code, 0) + int(pts)
        return out
    except Exception:
        return {}


def add_sprint_points(driver_data, constructor_data, season, round_num):
    """Credit sprint points (drivers + constructors) for a completed round.

    Sprint weekends award official championship points separate from the Grand
    Prix, and the official standings include them.  Race-only stats (wins,
    podiums, races_started) are intentionally left untouched.
    """
    for code, pts in sprint_points_map(season, round_num).items():
        if code in driver_data:
            driver_data[code]['points'] += pts
        team_mapped = DRIVER_TEAM_MAP.get(code)
        if team_mapped and team_mapped in constructor_data:
            constructor_data[team_mapped]['points'] += pts


def aggregate_standings_from_races(season=2026, max_round=24):
    """Fallback: aggregate standings by summing points from all race results"""
    print(f"Aggregating standings from individual race results...")
    
    driver_data = {}
    constructor_data = {}
    
    completed_rounds = 0
    
    for round_num in range(1, max_round + 1):
        try:
            session = fastf1.get_session(season, round_num, 'R')
            session.load(telemetry=False, weather=False, messages=False)
            
            if len(session.results) == 0:
                break
            
            completed_rounds = round_num
            
            for _, result in session.results.iterrows():
                driver_code = result.get('Abbreviation', result.get('DriverNumber', 'UNK'))
                team_name = result.get('TeamName', 'Unknown')
                position = result.get('Position', 0)
                points = result.get('Points', 0)
                
                # Map team name to our standard names
                team_mapped = DRIVER_TEAM_MAP.get(driver_code, team_name)
                driver_name = DRIVER_NAMES.get(driver_code, driver_code)
                
                # Initialize driver if not seen
                if driver_code not in driver_data:
                    driver_data[driver_code] = {
                        'driver_name': driver_name,
                        'team': team_mapped,
                        'points': 0,
                        'wins': 0,
                        'podiums': 0,
                        'races': 0,
                        'dnfs': 0,
                        'finishes': [],
                        'grids': [],
                    }
                
                # Aggregate driver stats
                driver_data[driver_code]['points'] += points
                driver_data[driver_code]['races'] += 1
                
                if pd.notna(position) and position > 0:
                    driver_data[driver_code]['finishes'].append(int(position))
                    if position == 1:
                        driver_data[driver_code]['wins'] += 1
                    if position <= 3:
                        driver_data[driver_code]['podiums'] += 1
                else:
                    driver_data[driver_code]['dnfs'] += 1
                    driver_data[driver_code]['finishes'].append(20)  # DNF
                
                grid_pos = result.get('GridPosition', 0)
                if pd.notna(grid_pos) and grid_pos > 0:
                    driver_data[driver_code]['grids'].append(int(grid_pos))
                
                # Aggregate constructor stats
                if team_mapped not in constructor_data:
                    constructor_data[team_mapped] = {
                        'points': 0,
                        'wins': 0,
                    }
                
                constructor_data[team_mapped]['points'] += points
                if position == 1:
                    constructor_data[team_mapped]['wins'] += 1

            add_sprint_points(driver_data, constructor_data, season, round_num)
        
        except Exception as e:
            print(f"  Round {round_num}: No data or error")
            break
    
    if completed_rounds == 0:
        return None, None, None
    
    print(f"Aggregated data from {completed_rounds} completed rounds")
    return driver_data, constructor_data, completed_rounds


def build_grid_csv(driver_data, constructor_data, last_round, season=2026):
    """Build the f1_2026_grid.csv from aggregated data"""
    
    rows = []
    
    # Sort drivers by points
    sorted_drivers = sorted(
        driver_data.items(),
        key=lambda x: x[1]['points'],
        reverse=True
    )
    
    # Sort constructors by points
    sorted_constructors = sorted(
        constructor_data.items(),
        key=lambda x: x[1]['points'],
        reverse=True
    )
    constructor_positions = {team: idx + 1 for idx, (team, _) in enumerate(sorted_constructors)}
    
    for driver_code, data in sorted_drivers:
        driver_name = data['driver_name']
        team = data['team']
        
        # Calculate stats
        races_started = data['races']
        races_completed = races_started - data['dnfs']
        completion_rate = races_completed / races_started if races_started > 0 else 1.0
        
        finishes = [f for f in data['finishes'] if f <= 20]  # Valid positions
        avg_finish = np.mean(finishes) if finishes else 20.0
        avg_grid = np.mean(data['grids']) if data['grids'] else 15.0
        
        points_finishes = sum(1 for f in finishes if f <= 10)
        points_rate = points_finishes / races_started if races_started > 0 else 0.0
        
        podium_rate = data['podiums'] / races_started if races_started > 0 else 0.0
        dnf_rate = data['dnfs'] / races_started if races_started > 0 else 0.0
        
        constructor_points = constructor_data.get(team, {}).get('points', 0)
        constructor_wins = constructor_data.get(team, {}).get('wins', 0)
        constructor_position = constructor_positions.get(team, 11)
        
        row = {
            'year': season,
            'driver_id': DRIVER_IDS.get(driver_name, driver_code.lower()),
            'driver_code': driver_code,
            'driver_name': driver_name,
            'nationality': NATIONALITIES.get(driver_name, 'Unknown'),
            'constructor_id': CONSTRUCTOR_IDS.get(team, team.lower().replace(' ', '_')),
            'constructor_name': team,
            'final_position': '',  # Empty for in-progress season
            'final_points': int(data['points']),
            'wins': int(data['wins']),
            'races_completed': int(races_completed),
            'races_started': int(races_started),
            'completion_rate': round(completion_rate, 3),
            'podiums': int(data['podiums']),
            'podium_rate': round(podium_rate, 3),
            'points_finishes': int(points_finishes),
            'points_rate': round(points_rate, 3),
            'dnfs': int(data['dnfs']),
            'dnf_rate': round(dnf_rate, 3),
            'fastest_laps': 0,  # Would need additional FastF1 query
            'avg_finish_position': round(avg_finish, 1),
            'avg_grid_position': round(avg_grid, 1),
            'constructor_final_position': constructor_position,
            'constructor_final_points': int(constructor_points),
            'constructor_wins': int(constructor_wins),
        }
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    return df, last_round


def read_current_grid():
    """Read the current grid CSV to determine what round it's updated through"""
    grid_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        'data', 'f1_2026_grid.csv'
    )
    
    if not os.path.exists(grid_path):
        print("No existing grid CSV found - will create from scratch")
        return None, 0
    
    try:
        df = pd.read_csv(grid_path)
        if len(df) == 0:
            return None, 0
        
        # Infer the last updated round from races_started (assumes all drivers have same value)
        last_round = int(df['races_started'].max())
        print(f"Current grid is updated through Round {last_round}")
        return df, last_round
    except Exception as e:
        print(f"Error reading existing grid: {e}")
        return None, 0


def update_incremental(current_df, current_round, new_round, season=2026):
    """Incrementally update standings by only fetching new races"""
    
    if new_round <= current_round:
        print(f"Already up-to-date (Round {current_round})")
        return None, None, current_round
    
    print(f"\nFetching ONLY NEW race data: Rounds {current_round + 1} to {new_round}...")
    
    # Convert current grid to driver_data format
    driver_data = {}
    for _, row in current_df.iterrows():
        code = row['driver_code']
        driver_data[code] = {
            'driver_name': row['driver_name'],
            'team': row['constructor_name'],
            'points': int(row['final_points']),
            'wins': int(row['wins']),
            'podiums': int(row['podiums']),
            'races': int(row['races_started']),
            'dnfs': int(row['dnfs']),
            'points_finishes': int(row['points_finishes']),
            'finishes': [],  # Will accumulate from new races
            'grids': [],
        }
    
    constructor_data = {}
    for constructor in current_df['constructor_name'].unique():
        constructor_rows = current_df[current_df['constructor_name'] == constructor]
        constructor_data[constructor] = {
            'points': int(constructor_rows['constructor_final_points'].iloc[0]),
            'wins': int(constructor_rows['constructor_wins'].iloc[0]),
        }
    
    # Fetch only the NEW races
    cache_dir = os.path.join(os.path.dirname(__file__), '.fastf1_cache')
    os.makedirs(cache_dir, exist_ok=True)
    fastf1.Cache.enable_cache(cache_dir)
    
    final_round = current_round
    
    for round_num in range(current_round + 1, new_round + 1):
        try:
            print(f"  Round {round_num}...", end=" ")
            session = fastf1.get_session(season, round_num, 'R')
            session.load(telemetry=False, weather=False, messages=False)
            
            if len(session.results) == 0:
                print("NO DATA")
                break
            
            final_round = round_num
            
            for _, result in session.results.iterrows():
                driver_code = result.get('Abbreviation', result.get('DriverNumber', 'UNK'))
                team_name = result.get('TeamName', 'Unknown')
                position = result.get('Position', 0)
                points = result.get('Points', 0)
                
                team_mapped = DRIVER_TEAM_MAP.get(driver_code, team_name)
                driver_name = DRIVER_NAMES.get(driver_code, driver_code)
                
                # Initialize if new driver
                if driver_code not in driver_data:
                    driver_data[driver_code] = {
                        'driver_name': driver_name,
                        'team': team_mapped,
                        'points': 0,
                        'wins': 0,
                        'podiums': 0,
                        'races': 0,
                        'dnfs': 0,
                        'points_finishes': 0,
                        'finishes': [],
                        'grids': [],
                    }
                
                # Update driver stats incrementally
                driver_data[driver_code]['points'] += points
                driver_data[driver_code]['races'] += 1
                
                if pd.notna(position) and position > 0:
                    driver_data[driver_code]['finishes'].append(int(position))
                    if position == 1:
                        driver_data[driver_code]['wins'] += 1
                    if position <= 3:
                        driver_data[driver_code]['podiums'] += 1
                    if position <= 10:
                        driver_data[driver_code]['points_finishes'] += 1
                else:
                    driver_data[driver_code]['dnfs'] += 1
                    driver_data[driver_code]['finishes'].append(20)
                
                grid_pos = result.get('GridPosition', 0)
                if pd.notna(grid_pos) and grid_pos > 0:
                    driver_data[driver_code]['grids'].append(int(grid_pos))
                
                # Update constructor
                if team_mapped not in constructor_data:
                    constructor_data[team_mapped] = {'points': 0, 'wins': 0}
                
                constructor_data[team_mapped]['points'] += points
                if position == 1:
                    constructor_data[team_mapped]['wins'] += 1

            add_sprint_points(driver_data, constructor_data, season, round_num)
            print("DONE")
        
        except Exception as e:
            print(f"ERROR: {str(e)[:50]}")
            break
    
    if final_round == current_round:
        # No new data was added
        return None, None, current_round
    
    return driver_data, constructor_data, final_round


def main():
    print("=" * 70)
    print("F1 2026 Season Standings Updater (Incremental)")
    print("=" * 70)
    
    # Read current grid to see what round we're at
    current_df, current_round = read_current_grid()
    
    # Check latest available race
    print("\nChecking for new races...")
    cache_dir = os.path.join(os.path.dirname(__file__), '.fastf1_cache')
    os.makedirs(cache_dir, exist_ok=True)
    fastf1.Cache.enable_cache(cache_dir)
    
    schedule = fastf1.get_event_schedule(2026)
    latest_round = current_round
    latest_race_name = ""
    
    # Quick check: just try to load the next few rounds after current
    print(f"Checking rounds after {current_round}...")
    for round_num in range(current_round + 1, min(current_round + 5, 25)):
        try:
            session = fastf1.get_session(2026, round_num, 'R')
            session.load(telemetry=False, weather=False, messages=False)
            if len(session.results) > 0:
                latest_round = round_num
                event = schedule[schedule['RoundNumber'] == round_num].iloc[0]
                latest_race_name = event['EventName']
                print(f"  Round {round_num}: {latest_race_name} [DATA AVAILABLE]")
            else:
                print(f"  Round {round_num}: No data yet")
                break
        except:
            break
    
    if latest_round == current_round:
        print(f"\n[INFO] Already up-to-date at Round {current_round}")
        print("No new race data to fetch.")
        return
    
    print(f"\n[INFO] New races found: Rounds {current_round + 1} to {latest_round}")
    print(f"[INFO] Latest race: Round {latest_round} - {latest_race_name}")
    
    # Update incrementally
    if current_df is not None and current_round > 0:
        driver_data, constructor_data, final_round = update_incremental(
            current_df, current_round, latest_round, 2026
        )
        
        if driver_data is None:
            # No update happened
            print("\n[INFO] No changes made")
            return
    else:
        # No existing data - full aggregation
        print("\nNo existing grid - fetching all data from scratch...")
        driver_data, constructor_data, final_round = aggregate_standings_from_races(2026, latest_round)
    
    if driver_data is None:
        print("\n[ERROR] Failed to fetch standings data")
        return
    
    # Build the CSV
    print("\nBuilding updated grid CSV...")
    grid_df, _ = build_grid_csv(driver_data, constructor_data, final_round, 2026)
    
    # Save to file
    output_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        'data', 'f1_2026_grid.csv'
    )
    
    grid_df.to_csv(output_path, index=False)
    
    print(f"\n[SUCCESS] Updated {output_path}")
    print(f"  Season: 2026")
    print(f"  Last round: {final_round} ({latest_race_name})")
    print(f"  Drivers: {len(grid_df)}")
    
    # Show top 5
    print("\n  Top 5 drivers:")
    for idx, row in grid_df.head(5).iterrows():
        print(f"    {idx+1}. {row['driver_name']:<25} {row['final_points']:>3} pts  ({row['constructor_name']})")
    
    # Update COMPLETED_ROUNDS in predictor
    predictor_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        'src', 'f1_2026_predictor.py'
    )
    
    if os.path.exists(predictor_path):
        with open(predictor_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Update constants
        import re
        content = re.sub(
            r'COMPLETED_ROUNDS = \d+',
            f'COMPLETED_ROUNDS = {final_round}',
            content
        )
        remaining = 24 - final_round
        content = re.sub(
            r'REMAINING_ROUNDS = \d+',
            f'REMAINING_ROUNDS = {remaining}',
            content
        )
        
        with open(predictor_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"\n[SUCCESS] Updated {predictor_path}")
        print(f"  COMPLETED_ROUNDS: {final_round}")
        print(f"  REMAINING_ROUNDS: {remaining}")
    
    print("\n" + "=" * 70)
    print("Update complete! You can now run the predictor with latest data:")
    print("  cd src")
    print("  python f1_2026_predictor.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
