"""
Calibrate Clean Sheet Probability Predictions

Compares predicted clean sheet probabilities to actual results
and suggests parameter tuning to improve accuracy.
"""

import sys
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from collections import defaultdict
import json

# Add parent directory to path
sys.path.insert(0, '..')
from bot.core.data_collector import DataCollector
from bot.core.predictor import PointsPredictor


def get_actual_clean_sheets(fixtures_df: pd.DataFrame) -> Dict[Tuple[int, int], bool]:
    """
    Extract actual clean sheet results from fixtures
    
    Returns:
        Dict mapping (team_id, fixture_id) -> bool (True if clean sheet)
    """
    clean_sheets = {}
    
    for _, fixture in fixtures_df.iterrows():
        if not fixture.get('finished', False):
            continue  # Skip unfinished fixtures
        
        fixture_id = fixture.get('id')
        team_h = fixture.get('team_h')
        team_a = fixture.get('team_a')
        team_h_score = fixture.get('team_h_score', 0)
        team_a_score = fixture.get('team_a_score', 0)
        
        if team_h and team_a_score is not None:
            # Home team got clean sheet if away team scored 0
            clean_sheets[(team_h, fixture_id)] = (team_a_score == 0)
        
        if team_a and team_h_score is not None:
            # Away team got clean sheet if home team scored 0
            clean_sheets[(team_a, fixture_id)] = (team_h_score == 0)
    
    return clean_sheets


def analyze_cs_predictions(
    predictor: PointsPredictor,
    players_df: pd.DataFrame,
    fixtures_df: pd.DataFrame,
    team_strengths_df: pd.DataFrame,
    actual_cs: Dict[Tuple[int, int], bool]
) -> Dict:
    """
    Analyze predicted vs actual clean sheet probabilities
    
    Returns analysis with calibration metrics
    """
    # Build team stats
    current_gw = fixtures_df['event'].max() if not fixtures_df.empty else 1
    team_stats = predictor._build_team_stats(players_df, fixtures_df, current_gw)
    
    # Group predictions by different factors
    predictions_by_factor = {
        'base_cs_rate': defaultdict(list),
        'opponent_strength': defaultdict(list),
        'home_away': defaultdict(list),
        'defensive_strength': defaultdict(list),
    }
    
    # Collect predictions and actuals
    for _, fixture in fixtures_df.iterrows():
        if not fixture.get('finished', False):
            continue
        
        fixture_id = fixture.get('id')
        event = fixture.get('event')
        team_h = fixture.get('team_h')
        team_a = fixture.get('team_a')
        
        if not team_h or not team_a:
            continue
        
        # Analyze home team
        if (team_h, fixture_id) in actual_cs:
            is_home = True
            team_id = team_h
            opponent_id = team_a
            
            fixture_info = {
                'team_id': team_id,
                'opponent_id': opponent_id,
                'is_home': is_home,
                'team_h': team_h,
                'team_a': team_a,
            }
            
            # Get predicted CS probability
            predicted_cs = predictor._predict_clean_sheet_probability(fixture_info, team_stats)
            actual_cs_result = actual_cs[(team_h, fixture_id)]
            
            # Get team stats for analysis
            team_data = team_stats.get(team_id, {})
            opponent_data = team_stats.get(opponent_id, {})
            
            base_cs_rate = team_data.get('clean_sheet_rate', 0.35)
            opponent_attack = opponent_data.get('avg_goals_scored_away', 
                                                opponent_data.get('avg_goals_scored', 1.4))
            avg_attack = predictor.AVG_GOALS_PER_GAME / 2
            attack_ratio = opponent_attack / avg_attack
            
            # Categorize by factors
            cs_rate_bucket = round(base_cs_rate * 10) / 10  # Round to 0.1
            attack_ratio_bucket = round(attack_ratio * 10) / 10  # Round to 0.1
            
            predictions_by_factor['base_cs_rate'][cs_rate_bucket].append({
                'predicted': predicted_cs,
                'actual': 1.0 if actual_cs_result else 0.0,
                'fixture_id': fixture_id,
                'event': event
            })
            
            predictions_by_factor['opponent_strength'][attack_ratio_bucket].append({
                'predicted': predicted_cs,
                'actual': 1.0 if actual_cs_result else 0.0,
                'fixture_id': fixture_id,
                'event': event
            })
            
            predictions_by_factor['home_away']['home'].append({
                'predicted': predicted_cs,
                'actual': 1.0 if actual_cs_result else 0.0,
                'fixture_id': fixture_id,
                'event': event
            })
        
        # Analyze away team
        if (team_a, fixture_id) in actual_cs:
            is_home = False
            team_id = team_a
            opponent_id = team_h
            
            fixture_info = {
                'team_id': team_id,
                'opponent_id': opponent_id,
                'is_home': is_home,
                'team_h': team_h,
                'team_a': team_a,
            }
            
            # Get predicted CS probability
            predicted_cs = predictor._predict_clean_sheet_probability(fixture_info, team_stats)
            actual_cs_result = actual_cs[(team_a, fixture_id)]
            
            # Get team stats for analysis
            team_data = team_stats.get(team_id, {})
            opponent_data = team_stats.get(opponent_id, {})
            
            base_cs_rate = team_data.get('clean_sheet_rate', 0.35)
            opponent_attack = opponent_data.get('avg_goals_scored_home',
                                                opponent_data.get('avg_goals_scored', 1.4))
            avg_attack = predictor.AVG_GOALS_PER_GAME / 2
            attack_ratio = opponent_attack / avg_attack
            
            # Categorize by factors
            cs_rate_bucket = round(base_cs_rate * 10) / 10
            attack_ratio_bucket = round(attack_ratio * 10) / 10
            
            predictions_by_factor['base_cs_rate'][cs_rate_bucket].append({
                'predicted': predicted_cs,
                'actual': 1.0 if actual_cs_result else 0.0,
                'fixture_id': fixture_id,
                'event': event
            })
            
            predictions_by_factor['opponent_strength'][attack_ratio_bucket].append({
                'predicted': predicted_cs,
                'actual': 1.0 if actual_cs_result else 0.0,
                'fixture_id': fixture_id,
                'event': event
            })
            
            predictions_by_factor['home_away']['away'].append({
                'predicted': predicted_cs,
                'actual': 1.0 if actual_cs_result else 0.0,
                'fixture_id': fixture_id,
                'event': event
            })
    
    # Calculate calibration metrics
    analysis = {
        'overall': calculate_calibration_metrics(
            [p for factor_data in predictions_by_factor.values() 
             for bucket_data in factor_data.values() 
             for p in bucket_data]
        ),
        'by_base_cs_rate': {},
        'by_opponent_strength': {},
        'by_home_away': {},
    }
    
    # Analyze by base CS rate
    for cs_rate, data in sorted(predictions_by_factor['base_cs_rate'].items()):
        if len(data) >= 5:  # Need at least 5 samples
            analysis['by_base_cs_rate'][cs_rate] = calculate_calibration_metrics(data)
    
    # Analyze by opponent strength
    for attack_ratio, data in sorted(predictions_by_factor['opponent_strength'].items()):
        if len(data) >= 5:
            analysis['by_opponent_strength'][attack_ratio] = calculate_calibration_metrics(data)
    
    # Analyze by home/away
    for venue, data in predictions_by_factor['home_away'].items():
        if len(data) >= 5:
            analysis['by_home_away'][venue] = calculate_calibration_metrics(data)
    
    return analysis


def calculate_calibration_metrics(data: List[Dict]) -> Dict:
    """
    Calculate calibration metrics for a set of predictions
    
    Returns:
        Dict with calibration metrics
    """
    if not data:
        return {}
    
    predicted = [d['predicted'] for d in data]
    actual = [d['actual'] for d in data]
    
    # Overall metrics
    mean_predicted = np.mean(predicted)
    mean_actual = np.mean(actual)
    n_samples = len(data)
    
    # Brier score (lower is better, 0 = perfect)
    brier_score = np.mean([(p - a) ** 2 for p, a in zip(predicted, actual)])
    
    # Calibration error (how far off are predictions on average)
    calibration_error = abs(mean_predicted - mean_actual)
    
    # Reliability (how well-calibrated)
    # Group predictions into bins and compare
    bins = np.linspace(0, 1, 11)  # 0.0, 0.1, 0.2, ..., 1.0
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_actuals = []
    bin_predicted = []
    bin_counts = []
    
    for i in range(len(bins) - 1):
        bin_mask = (np.array(predicted) >= bins[i]) & (np.array(predicted) < bins[i+1])
        if i == len(bins) - 2:  # Last bin includes upper bound
            bin_mask = (np.array(predicted) >= bins[i]) & (np.array(predicted) <= bins[i+1])
        
        if np.sum(bin_mask) > 0:
            bin_actuals.append(np.mean(np.array(actual)[bin_mask]))
            bin_predicted.append(bin_centers[i])
            bin_counts.append(np.sum(bin_mask))
    
    # Calculate reliability (mean squared error in bins)
    if bin_actuals:
        reliability = np.mean([(p - a) ** 2 for p, a in zip(bin_predicted, bin_actuals)])
    else:
        reliability = 0.0
    
    return {
        'n_samples': n_samples,
        'mean_predicted': mean_predicted,
        'mean_actual': mean_actual,
        'calibration_error': calibration_error,
        'brier_score': brier_score,
        'reliability': reliability,
        'bins': {
            'predicted': bin_predicted,
            'actual': bin_actuals,
            'counts': bin_counts
        }
    }


def suggest_parameter_adjustments(analysis: Dict) -> List[str]:
    """
    Suggest parameter adjustments based on calibration analysis
    
    Returns:
        List of suggested adjustments
    """
    suggestions = []
    
    overall = analysis.get('overall', {})
    if overall:
        error = overall.get('calibration_error', 0)
        if error > 0.05:  # More than 5% off
            if overall['mean_predicted'] > overall['mean_actual']:
                suggestions.append(
                    f"[WARNING] Overall predictions are {error:.1%} too high. "
                    f"Consider reducing base CS rates or increasing opponent penalties."
                )
            else:
                suggestions.append(
                    f"[WARNING] Overall predictions are {error:.1%} too low. "
                    f"Consider increasing base CS rates or reducing opponent penalties."
                )
    
    # Check home/away bias
    home_away = analysis.get('by_home_away', {})
    if 'home' in home_away and 'away' in home_away:
        home_error = home_away['home'].get('calibration_error', 0)
        away_error = home_away['away'].get('calibration_error', 0)
        
        if home_error > 0.05 or away_error > 0.05:
            if home_away['home']['mean_predicted'] > home_away['home']['mean_actual']:
                suggestions.append(
                    f"[WARNING] Home predictions are {home_error:.1%} too high. "
                    f"Consider reducing home advantage factor (currently +10%)."
                )
            if home_away['away']['mean_predicted'] > home_away['away']['mean_actual']:
                suggestions.append(
                    f"[WARNING] Away predictions are {away_error:.1%} too high. "
                    f"Consider reducing away penalty (currently -8%)."
                )
    
    # Check opponent strength calibration
    opponent_strength = analysis.get('by_opponent_strength', {})
    weak_opponents = {k: v for k, v in opponent_strength.items() if k < 1.0}
    strong_opponents = {k: v for k, v in opponent_strength.items() if k >= 1.3}
    
    if weak_opponents:
        avg_error = np.mean([v.get('calibration_error', 0) for v in weak_opponents.values()])
        if avg_error > 0.05:
            suggestions.append(
                f"[WARNING] Predictions vs weak opponents are off by {avg_error:.1%}. "
                f"Consider adjusting weak opponent boost (currently +8% max)."
            )
    
    if strong_opponents:
        avg_error = np.mean([v.get('calibration_error', 0) for v in strong_opponents.values()])
        if avg_error > 0.05:
            suggestions.append(
                f"[WARNING] Predictions vs strong opponents are off by {avg_error:.1%}. "
                f"Consider adjusting strong opponent penalty (currently -25% base)."
            )
    
    return suggestions


def print_analysis_report(analysis: Dict):
    """Print a formatted analysis report"""
    print("\n" + "=" * 80)
    print("CLEAN SHEET PREDICTION CALIBRATION ANALYSIS")
    print("=" * 80)
    
    # Overall metrics
    overall = analysis.get('overall', {})
    if overall:
        print(f"\n[OVERALL CALIBRATION]")
        print(f"  Samples: {overall['n_samples']}")
        print(f"  Mean Predicted: {overall['mean_predicted']:.1%}")
        print(f"  Mean Actual: {overall['mean_actual']:.1%}")
        print(f"  Calibration Error: {overall['calibration_error']:.1%}")
        print(f"  Brier Score: {overall['brier_score']:.4f} (lower is better, 0 = perfect)")
        print(f"  Reliability: {overall['reliability']:.4f} (lower is better)")
        
        if overall['calibration_error'] < 0.02:
            print("  [OK] Excellent calibration!")
        elif overall['calibration_error'] < 0.05:
            print("  [OK] Good calibration")
        elif overall['calibration_error'] < 0.10:
            print("  [WARNING] Moderate calibration issues")
        else:
            print("  [ERROR] Significant calibration issues")
    
    # By base CS rate
    by_cs_rate = analysis.get('by_base_cs_rate', {})
    if by_cs_rate:
        print(f"\n[BY BASE CLEAN SHEET RATE]")
        print(f"  {'CS Rate':<10} {'Samples':<10} {'Predicted':<12} {'Actual':<12} {'Error':<10}")
        print("  " + "-" * 54)
        for cs_rate in sorted(by_cs_rate.keys()):
            data = by_cs_rate[cs_rate]
            error = data['calibration_error']
            error_str = f"{error:.1%}"
            if error < 0.05:
                error_str = f"[OK] {error_str}"
            elif error < 0.10:
                error_str = f"[WARN] {error_str}"
            else:
                error_str = f"[ERROR] {error_str}"
            
            print(f"  {cs_rate:<10.1f} {data['n_samples']:<10} "
                  f"{data['mean_predicted']:<12.1%} {data['mean_actual']:<12.1%} {error_str}")
    
    # By opponent strength
    by_opponent = analysis.get('by_opponent_strength', {})
    if by_opponent:
        print(f"\n[BY OPPONENT ATTACK STRENGTH (ratio vs league average)]")
        print(f"  {'Ratio':<10} {'Samples':<10} {'Predicted':<12} {'Actual':<12} {'Error':<10}")
        print("  " + "-" * 54)
        for ratio in sorted(by_opponent.keys()):
            data = by_opponent[ratio]
            error = data['calibration_error']
            error_str = f"{error:.1%}"
            if error < 0.05:
                error_str = f"[OK] {error_str}"
            elif error < 0.10:
                error_str = f"[WARN] {error_str}"
            else:
                error_str = f"[ERROR] {error_str}"
            
            print(f"  {ratio:<10.2f} {data['n_samples']:<10} "
                  f"{data['mean_predicted']:<12.1%} {data['mean_actual']:<12.1%} {error_str}")
    
    # By home/away
    by_venue = analysis.get('by_home_away', {})
    if by_venue:
        print(f"\n[BY HOME/AWAY]")
        print(f"  {'Venue':<10} {'Samples':<10} {'Predicted':<12} {'Actual':<12} {'Error':<10}")
        print("  " + "-" * 54)
        for venue in ['home', 'away']:
            if venue in by_venue:
                data = by_venue[venue]
                error = data['calibration_error']
                error_str = f"{error:.1%}"
                if error < 0.05:
                    error_str = f"[OK] {error_str}"
                elif error < 0.10:
                    error_str = f"[WARN] {error_str}"
                else:
                    error_str = f"[ERROR] {error_str}"
                
                print(f"  {venue:<10} {data['n_samples']:<10} "
                      f"{data['mean_predicted']:<12.1%} {data['mean_actual']:<12.1%} {error_str}")
    
    # Suggestions
    suggestions = suggest_parameter_adjustments(analysis)
    if suggestions:
        print(f"\n[SUGGESTED PARAMETER ADJUSTMENTS]")
        for i, suggestion in enumerate(suggestions, 1):
            print(f"  {i}. {suggestion}")
    else:
        print(f"\n[OK] No major calibration issues detected!")


def main():
    """Main calibration function"""
    print("Calibrating Clean Sheet Predictions...")
    print("=" * 80)
    
    # Initialize components
    data_collector = DataCollector()
    predictor = PointsPredictor()
    
    # Get data
    print("\n1. Collecting data...")
    players_df = data_collector.get_player_data(include_set_pieces=True)
    fixtures_df = data_collector.get_fixtures()
    team_strengths_df = data_collector.get_team_strengths()
    
    print(f"   - Players: {len(players_df)}")
    print(f"   - Fixtures: {len(fixtures_df)}")
    
    # Get actual clean sheet results
    print("\n2. Extracting actual clean sheet results...")
    actual_cs = get_actual_clean_sheets(fixtures_df)
    finished_fixtures = fixtures_df[fixtures_df['finished'] == True]
    print(f"   - Finished fixtures: {len(finished_fixtures)}")
    print(f"   - Clean sheet records: {len(actual_cs)}")
    
    # Analyze predictions
    print("\n3. Analyzing predictions vs actuals...")
    analysis = analyze_cs_predictions(
        predictor, players_df, fixtures_df, team_strengths_df, actual_cs
    )
    
    # Print report
    print_analysis_report(analysis)
    
    # Save detailed results
    output_file = 'clean_sheet_calibration_results.json'
    with open(output_file, 'w') as f:
        # Convert numpy types to native Python types for JSON
        def convert_to_native(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_native(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_native(item) for item in obj]
            return obj
        
        json.dump(convert_to_native(analysis), f, indent=2)
    
    print(f"\n[SAVED] Detailed results saved to: {output_file}")
    print("\n" + "=" * 80)


if __name__ == '__main__':
    main()

