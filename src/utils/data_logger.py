import os
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd

class DataLogger:
    """Handles logging of attention and emotion data to CSV and JSON formats."""
    
    def __init__(self, output_dir: str = "sessions"):
        """
        Initialize the data logger.
        
        Args:
            output_dir: Directory to store session data
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.current_session = None
        self.session_data = []
        
    def start_session(self, user_id: str = "default_user"):
        """Start a new logging session."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.current_session = f"{user_id}_{timestamp}"
        self.session_data = []
        return self.current_session
        
    def log_data(
        self,
        attention_score: float,
        emotion: str,
        emotion_confidence: float,
        stress_level: float,
        timestamp: Optional[float] = None
    ):
        """
        Log attention and emotion data for the current session.
        
        Args:
            attention_score: Current attention score (0-1)
            emotion: Detected emotion label
            emotion_confidence: Confidence of emotion detection (0-1)
            stress_level: Current stress level (0-1)
            timestamp: Optional custom timestamp
        """
        if not self.current_session:
            self.start_session()
            
        entry = {
            'timestamp': timestamp or datetime.now().timestamp(),
            'datetime': datetime.now().isoformat(),
            'attention_score': float(attention_score),
            'emotion': emotion,
            'emotion_confidence': float(emotion_confidence),
            'stress_level': float(stress_level)
        }
        self.session_data.append(entry)
        
    def save_session(self, format: str = 'both') -> str:
        """
        Save the current session data to disk.
        
        Args:
            format: Output format ('csv', 'json', or 'both')
            
        Returns:
            str: Path to the saved session file(s)
        """
        if not self.current_session or not self.session_data:
            return ""
            
        base_path = self.output_dir / self.current_session
        saved_files = []
        
        if format in ('csv', 'both'):
            csv_path = f"{base_path}.csv"
            df = pd.DataFrame(self.session_data)
            df.to_csv(csv_path, index=False)
            saved_files.append(csv_path)
            
        if format in ('json', 'both'):
            json_path = f"{base_path}.json"
            with open(json_path, 'w') as f:
                json.dump(self.session_data, f, indent=2)
            saved_files.append(json_path)
            
        return saved_files[0] if len(saved_files) == 1 else saved_files
    
    def get_session_summary(self) -> Dict:
        """Generate a summary of the current session."""
        if not self.session_data:
            return {}
            
        df = pd.DataFrame(self.session_data)
        
        return {
            'session_id': self.current_session,
            'start_time': df['datetime'].min(),
            'end_time': df['datetime'].max(),
            'avg_attention': df['attention_score'].mean(),
            'focus_percentage': (df['attention_score'] > 0.7).mean() * 100,
            'dominant_emotion': df['emotion'].mode()[0] if not df.empty else None,
            'avg_stress_level': df['stress_level'].mean(),
            'total_duration_seconds': (pd.to_datetime(df['datetime']).max() - 
                                     pd.to_datetime(df['datetime']).min()).total_seconds()
        }

# Singleton instance
session_logger = DataLogger()
