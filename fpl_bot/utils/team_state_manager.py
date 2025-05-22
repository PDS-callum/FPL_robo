import os
import json
from datetime import datetime
import pandas as pd


class TeamStateManager:
    def __init__(self, state_dir="data/team_state"):
        self.state_dir = state_dir
        os.makedirs(state_dir, exist_ok=True)
        self.current_state_path = os.path.join(state_dir, "current_team_state.json")
    
    def save_team_state(self, team_dict, gameweek):
        """Save the current team state with timestamp"""
        state = {
            "gameweek": gameweek,
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "team": team_dict,
            "free_transfers": 1  # Default to 1 free transfer for next week
        }
        
        # Check if we already have a state file
        if os.path.exists(self.current_state_path):
            previous_state = self.load_team_state()
            
            # If this is a consecutive gameweek and we didn't use our transfer last week
            if previous_state and previous_state.get("gameweek") == gameweek - 1:
                unused_transfers = previous_state.get("free_transfers", 1)
                # Max 2 free transfers
                state["free_transfers"] = min(unused_transfers + 1, 2)
        
        # Save both as current state and as gameweek-specific state
        with open(self.current_state_path, 'w') as f:
            json.dump(state, f, indent=2)
        
        # Also save gameweek-specific version for history
        gw_state_path = os.path.join(self.state_dir, f"team_state_gw{gameweek}.json")
        with open(gw_state_path, 'w') as f:
            json.dump(state, f, indent=2)
            
        return state
    
    def load_team_state(self):
        """Load the most recent team state"""
        if not os.path.exists(self.current_state_path):
            return None
            
        with open(self.current_state_path, 'r') as f:
            return json.load(f)
    
    def get_player_ids(self):
        """Get list of player IDs from current team"""
        state = self.load_team_state()
        if not state:
            return []
            
        return [player["id"] for player in state["team"]["squad"]]
    
    def get_free_transfers(self):
        """Get number of free transfers available"""
        state = self.load_team_state()
        if not state:
            return 1  # Default to 1 if no previous state
        return state.get("free_transfers", 1)