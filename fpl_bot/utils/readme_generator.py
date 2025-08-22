"""
Utility for generating README reports from team prediction and composition data.
"""

import os
import json
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path
from .file_utils import safe_file_operation
from .constants import POSITION_NAMES


class FPLReadmeGenerator:
    """
    Generates README reports from FPL season data
    """
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize the README generator
        
        Parameters:
        -----------
        data_dir : str
            Data directory path
        """
        self.data_dir = Path(data_dir)
        self.iterative_state_file = self.data_dir / "iterative_season_state.json"
        self.team_composition_file = self.data_dir / "team_composition_analysis.json"
    
    def load_season_data(self) -> tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """
        Load season data from JSON files
        
        Returns:
        --------
        tuple
            (iterative_season_data, team_composition_data)
        """
        iterative_data = None
        composition_data = None
        
        # Load iterative season state
        if self.iterative_state_file.exists():
            try:
                with open(self.iterative_state_file, 'r') as f:
                    iterative_data = json.load(f)
            except Exception as e:
                print(f"Warning: Could not load iterative season data: {e}")
        
        # Load team composition analysis
        if self.team_composition_file.exists():
            try:
                with open(self.team_composition_file, 'r') as f:
                    composition_data = json.load(f)
            except Exception as e:
                print(f"Warning: Could not load team composition data: {e}")
        
        return iterative_data, composition_data
    
    def generate_season_summary_report(self) -> str:
        """
        Generate a comprehensive season summary report
        
        Returns:
        --------
        str
            Formatted markdown report
        """
        iterative_data, composition_data = self.load_season_data()
        
        if not iterative_data and not composition_data:
            return "\n# FPL Season Team Analysis Report\n\n**No data available yet.** Run some predictions first!\n\n"
        
        report = []
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Header
        report.append("# FPL Season Team Analysis Report")
        report.append("")
        report.append(f"**Generated on:** {current_time}")
        
        if iterative_data and iterative_data.get("predictions_history"):
            last_prediction = iterative_data["predictions_history"][-1]
            report.append(f"**Last Updated:** {last_prediction.get('prediction_time', 'N/A')}")
            report.append(f"**Total Gameweeks Analyzed:** {len(iterative_data['predictions_history'])}")
        
        report.append("")
        
        # Season Summary
        if composition_data:
            report.extend(self._generate_season_summary_section(composition_data, iterative_data))
        
        # Formation Analysis
        if composition_data and composition_data.get("formations_used"):
            report.extend(self._generate_formation_section(composition_data))
        
        # Captain Analysis
        if composition_data and composition_data.get("captain_choices"):
            report.extend(self._generate_captain_section(composition_data))
        
        # Player Loyalty Analysis
        if composition_data and composition_data.get("players_selected"):
            report.extend(self._generate_player_loyalty_section(composition_data))
        
        # Team Distribution
        if composition_data and composition_data.get("teams_represented"):
            report.extend(self._generate_team_distribution_section(composition_data))
        
        # Position Analysis
        if iterative_data and iterative_data.get("teams_history"):
            report.extend(self._generate_position_analysis_section(iterative_data))
        
        return "\n".join(report)
    
    def generate_team_composition_report(self) -> str:
        """
        Generate a detailed team composition analysis report
        
        Returns:
        --------
        str
            Formatted markdown report
        """
        iterative_data, composition_data = self.load_season_data()
        
        if not composition_data:
            return "\n# FPL Season Team Composition Analysis\n\n**No team composition data available yet.**\n\n"
        
        report = []
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Header
        report.append("# FPL Season Team Composition Analysis")
        report.append("")
        report.append(f"*Analysis generated on {current_time}*")
        report.append("")
        
        # Season Overview
        report.extend(self._generate_composition_overview(composition_data))
        
        # Formation Analysis
        if composition_data.get("formations_used"):
            report.extend(self._generate_formation_analysis(composition_data))
        
        # Player Consistency
        if composition_data.get("players_selected"):
            report.extend(self._generate_player_consistency_section(composition_data))
        
        # Captaincy Analysis
        if composition_data.get("captain_choices"):
            report.extend(self._generate_captaincy_analysis(composition_data))
        
        # Team Distribution Analysis
        if composition_data.get("teams_represented"):
            report.extend(self._generate_team_analysis(composition_data))
        
        # Position Distribution
        if composition_data.get("position_distribution"):
            report.extend(self._generate_position_distribution(composition_data))
        
        # Gameweek Summary Table
        if iterative_data and iterative_data.get("predictions_history"):
            report.extend(self._generate_gameweek_summary_table(iterative_data, composition_data))
        
        return "\n".join(report)
    
    def _generate_season_summary_section(self, composition_data: Dict[str, Any], iterative_data: Optional[Dict[str, Any]]) -> List[str]:
        """Generate season summary section"""
        section = ["## Season Summary", ""]
        
        # Basic stats from composition data
        avg_cost = composition_data.get("average_team_cost", 0)
        avg_points = composition_data.get("average_predicted_points", 0)
        total_gws = composition_data.get("total_gameweeks", 0)
        
        section.append(f"- **Average Team Cost:** £{avg_cost:.1f}M")
        section.append(f"- **Average Predicted Points:** {avg_points:.1f}")
        
        # Budget utilization from latest gameweek
        if composition_data.get("budget_utilization"):
            latest_budget = composition_data["budget_utilization"][-1]
            utilization = latest_budget.get("budget_utilization", 0)
            section.append(f"- **Budget Utilization:** {utilization:.1f}% (Latest GW)")
        
        # Transfer count
        transfer_count = 0
        if iterative_data and iterative_data.get("transfers_history"):
            transfer_count = len(iterative_data["transfers_history"])
        section.append(f"- **Total Transfers Made:** {transfer_count}")
        
        section.append("")
        return section
    
    def _generate_formation_section(self, composition_data: Dict[str, Any]) -> List[str]:
        """Generate formation usage section"""
        section = ["## Formation Usage", ""]
        
        formations = composition_data["formations_used"]
        total_gws = sum(formations.values())
        
        for formation, count in sorted(formations.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total_gws) * 100
            section.append(f"- **{formation}:** {count} gameweeks ({percentage:.1f}%)")
        
        section.append("")
        return section
    
    def _generate_captain_section(self, composition_data: Dict[str, Any]) -> List[str]:
        """Generate captain choices section"""
        section = ["## Captain Choices", ""]
        
        captains = composition_data["captain_choices"]
        total_gws = sum(captains.values())
        
        for captain, count in sorted(captains.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total_gws) * 100
            section.append(f"- **{captain}:** {count} gameweeks ({percentage:.1f}%)")
        
        section.append("")
        return section
    
    def _generate_player_loyalty_section(self, composition_data: Dict[str, Any]) -> List[str]:
        """Generate player loyalty section"""
        section = ["## Most Loyal Players (Top 10)", ""]
        
        players = composition_data["players_selected"]
        total_gws = composition_data["total_gameweeks"]
        
        # Sort by selection count and take top 10
        sorted_players = sorted(players.items(), key=lambda x: x[1], reverse=True)[:10]
        
        for player, count in sorted_players:
            percentage = (count / total_gws) * 100
            section.append(f"- **{player}:** {count}/{total_gws} gameweeks ({percentage:.1f}%)")
        
        section.append("")
        return section
    
    def _generate_team_distribution_section(self, composition_data: Dict[str, Any]) -> List[str]:
        """Generate team distribution section"""
        section = ["## Team Distribution (Most Represented)", ""]
        
        teams = composition_data["teams_represented"]
        
        # Sort by selection count
        sorted_teams = sorted(teams.items(), key=lambda x: x[1], reverse=True)
        
        for team, count in sorted_teams:
            section.append(f"- **{team}:** {count} player selections")
        
        section.append("")
        return section
    
    def _generate_position_analysis_section(self, iterative_data: Dict[str, Any]) -> List[str]:
        """Generate position analysis section"""
        section = ["## Position Analysis", ""]
        
        # Aggregate position data from teams history
        position_players = {}
        
        for team_history in iterative_data.get("teams_history", []):
            team = team_history.get("team", {})
            playing_xi = team.get("playing_xi", [])
            
            for player in playing_xi:
                position = player.get("position")
                name = player.get("name")
                if position and name:
                    if position not in position_players:
                        position_players[position] = {}
                    if name not in position_players[position]:
                        position_players[position][name] = 0
                    position_players[position][name] += 1
        
        # Generate report by position
        for pos_key in ["GKP", "DEF", "MID", "FWD"]:
            if pos_key in position_players:
                section.append(f"### {pos_key}")
                players = position_players[pos_key]
                total_gws = len(iterative_data.get("teams_history", []))
                
                sorted_players = sorted(players.items(), key=lambda x: x[1], reverse=True)
                for player, count in sorted_players:
                    percentage = (count / total_gws) * 100
                    section.append(f"- **{player}:** {count} appearances ({percentage:.1f}%)")
                section.append("")
        
        return section
    
    def _generate_composition_overview(self, composition_data: Dict[str, Any]) -> List[str]:
        """Generate composition overview section"""
        section = ["## Season Overview", ""]
        
        total_gws = composition_data.get("total_gameweeks", 0)
        avg_cost = composition_data.get("average_team_cost", 0)
        avg_points = composition_data.get("average_predicted_points", 0)
        
        # Calculate average budget utilization
        avg_budget_util = 0
        if composition_data.get("budget_utilization"):
            budget_utils = [bu.get("budget_utilization", 0) for bu in composition_data["budget_utilization"]]
            avg_budget_util = sum(budget_utils) / len(budget_utils) if budget_utils else 0
        
        section.append(f"- **Total Gameweeks Analyzed:** {total_gws}")
        section.append(f"- **Average Team Cost:** £{avg_cost:.1f}m")
        section.append(f"- **Average Predicted Points:** {avg_points:.1f}")
        section.append(f"- **Average Budget Utilization:** {avg_budget_util:.1f}%")
        section.append("")
        
        return section
    
    def _generate_formation_analysis(self, composition_data: Dict[str, Any]) -> List[str]:
        """Generate formation analysis section"""
        section = ["## Formation Analysis", "", "### Most Used Formations", ""]
        
        formations = composition_data["formations_used"]
        total_gws = sum(formations.values())
        
        for formation, count in sorted(formations.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total_gws) * 100
            section.append(f"- **{formation}:** {count} times ({percentage:.1f}%)")
        
        section.append("")
        return section
    
    def _generate_player_consistency_section(self, composition_data: Dict[str, Any]) -> List[str]:
        """Generate player consistency section"""
        section = ["## Most Consistent Players", "", "*Players selected in 50% or more gameweeks*", ""]
        
        players = composition_data["players_selected"]
        total_gws = composition_data["total_gameweeks"]
        
        # Filter for players selected in 50% or more gameweeks
        consistent_players = {player: count for player, count in players.items() 
                            if count / total_gws >= 0.5}
        
        # Sort by selection count
        sorted_players = sorted(consistent_players.items(), key=lambda x: x[1], reverse=True)
        
        for player, count in sorted_players:
            percentage = (count / total_gws) * 100
            section.append(f"- **{player}:** {count}/{total_gws} gameweeks ({percentage:.1f}%)")
        
        section.append("")
        return section
    
    def _generate_captaincy_analysis(self, composition_data: Dict[str, Any]) -> List[str]:
        """Generate captaincy analysis section"""
        section = ["## Captaincy Analysis", "", "### Most Captained Players", ""]
        
        captains = composition_data.get("captain_choices", {})
        total_gws = sum(captains.values())
        
        for captain, count in sorted(captains.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total_gws) * 100
            section.append(f"- **{captain}:** {count} times ({percentage:.1f}%)")
        
        section.append("")
        
        # Vice captains
        section.append("### Most Vice-Captained Players")
        section.append("")
        
        vice_captains = composition_data.get("vice_captain_choices", {})
        total_gws_vc = sum(vice_captains.values())
        
        for vice_captain, count in sorted(vice_captains.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total_gws_vc) * 100
            section.append(f"- **{vice_captain}:** {count} times ({percentage:.1f}%)")
        
        section.append("")
        return section
    
    def _generate_team_analysis(self, composition_data: Dict[str, Any]) -> List[str]:
        """Generate team distribution analysis section"""
        section = ["## Team Distribution Analysis", "", "### Most Represented Teams", ""]
        
        teams = composition_data["teams_represented"]
        total_gws = composition_data["total_gameweeks"]
        
        # Sort by selection count
        sorted_teams = sorted(teams.items(), key=lambda x: x[1], reverse=True)
        
        for team, count in sorted_teams:
            avg_per_gw = count / total_gws
            section.append(f"- **{team}:** {count} player selections (avg {avg_per_gw:.1f} per GW)")
        
        section.append("")
        return section
    
    def _generate_position_distribution(self, composition_data: Dict[str, Any]) -> List[str]:
        """Generate position distribution section"""
        section = ["## Position Distribution", ""]
        
        positions = composition_data.get("position_distribution", {})
        total_selections = sum(positions.values())
        total_gws = composition_data["total_gameweeks"]
        
        # Sort by selection count
        sorted_positions = sorted(positions.items(), key=lambda x: x[1], reverse=True)
        
        for position, count in sorted_positions:
            percentage = (count / total_selections) * 100
            avg_per_gw = count / total_gws
            section.append(f"- **{position}:** {count} selections ({percentage:.1f}%, avg {avg_per_gw:.1f} per GW)")
        
        section.append("")
        return section
    
    def _generate_gameweek_summary_table(self, iterative_data: Dict[str, Any], composition_data: Dict[str, Any]) -> List[str]:
        """Generate gameweek summary table"""
        section = ["## Gameweek Summary", ""]
        
        section.append("| GW | Formation | Cost | Predicted Pts | Teams Used | Captain | Vice-Captain |")
        section.append("|----|-----------|------|---------------|------------|---------|--------------|")
        
        predictions = iterative_data.get("predictions_history", [])
        teams_history = iterative_data.get("teams_history", [])
        
        for i, prediction in enumerate(predictions):
            gw = prediction.get("gameweek", i+1)
            formation = prediction.get("formation", "N/A")
            cost = prediction.get("team_cost", 0)
            predicted_pts = prediction.get("predicted_points", 0)
            captain = prediction.get("captain", "N/A")
            vice_captain = prediction.get("vice_captain", "N/A")
            
            # Count unique teams used
            teams_used = 0
            if i < len(teams_history):
                team_data = teams_history[i].get("team", {})
                playing_xi = team_data.get("playing_xi", [])
                unique_teams = set(player.get("team", "") for player in playing_xi)
                teams_used = len(unique_teams)
            
            section.append(f"| {gw} | {formation} | £{cost:.1f}m | {predicted_pts:.1f} | {teams_used} | {captain} | {vice_captain} |")
        
        section.append("")
        return section
    
    def update_readme_with_analysis(self, readme_path: str = "README.md") -> bool:
        """
        Update the README file with fresh team analysis
        
        Parameters:
        -----------
        readme_path : str
            Path to the README file
            
        Returns:
        --------
        bool
            True if update successful, False otherwise
        """
        try:
            # Generate new reports
            season_report = self.generate_season_summary_report()
            composition_report = self.generate_team_composition_report()
            
            # Read current README
            readme_file = Path(readme_path)
            if not readme_file.exists():
                print(f"Warning: README file {readme_path} not found")
                return False
            
            with open(readme_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Find and replace existing analysis sections
            content = self._replace_or_append_section(content, season_report, "# FPL Season Team Analysis Report")
            content = self._replace_or_append_section(content, composition_report, "# FPL Season Team Composition Analysis")
            
            # Write updated content back
            with open(readme_file, 'w', encoding='utf-8') as f:
                f.write(content)
            
            print(f"✅ README updated successfully with fresh team analysis")
            return True
            
        except Exception as e:
            print(f"❌ Failed to update README: {e}")
            return False
    
    def _replace_or_append_section(self, content: str, new_section: str, section_header: str) -> str:
        """
        Replace existing section or append new section to content
        
        Parameters:
        -----------
        content : str
            Original content
        new_section : str
            New section content
        section_header : str
            Section header to look for
            
        Returns:
        --------
        str
            Updated content
        """
        # Find the start of the section
        start_idx = content.find(section_header)
        
        if start_idx == -1:
            # Section doesn't exist, append it
            if not content.endswith('\n'):
                content += '\n'
            content += '\n' + new_section
            return content
        
        # Find the end of the section (next header or end of file)
        section_start = start_idx
        next_header_idx = content.find('\n# ', start_idx + 1)
        
        if next_header_idx == -1:
            # This is the last section, replace until end
            content = content[:section_start] + new_section
        else:
            # Replace until the next header
            content = content[:section_start] + new_section + '\n' + content[next_header_idx+1:]
        
        return content
