from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Union

import pandas as pd

from bist_quant.settings import get_output_dir

logger = logging.getLogger(__name__)


class SignalExporter:
    """Export raw or composite signal/factor scores to CSV for external consumption."""

    def __init__(self, signal_name: str, output_dir: Union[str, Path, None] = None):
        """
        Initialize the SignalExporter for a specific signal or factor suite.
        
        Args:
            signal_name: Name of the signal (e.g., 'momentum', 'five_factor')
            output_dir: Optional override for the output directory. If None,
                       uses the standard get_output_dir("signals", signal_name).
        """
        self.signal_name = signal_name
        
        if output_dir is None:
            self.output_dir = get_output_dir("signals", signal_name)
        else:
            self.output_dir = Path(output_dir)
            
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export_factor_scores(
        self, 
        scores: Union[pd.DataFrame, pd.Series, Dict[str, Union[pd.DataFrame, pd.Series]]], 
        filename: str = "raw_scores.csv"
    ) -> Path:
        """
        Export a DataFrame, Series, or Dictionary of signal scores to CSV.
        
        Args:
            scores: The scores to export. Can be a DataFrame (where rows/cols are dates/tickers),
                   a Series, or a Dictionary mapping factor names to DataFrames.
            filename: The name of the file to save within the output directory.
                      
        Returns:
            The absolute Path to the exported file.
        """
        filepath = self.output_dir / filename
        
        try:
            if isinstance(scores, (pd.DataFrame, pd.Series)):
                scores.to_csv(filepath)
            elif isinstance(scores, dict):
                # If it's a dictionary of dataframes, we can concatenate or export them as one
                # For simplicity, we create a multi-index mapping or simply merge them if they share an index.
                # However, usually we might want to iterate over the items and save separately if requested,
                # but to fulfill a single file request, we concat them.
                combined = pd.concat(scores, axis=1) if scores else pd.DataFrame()
                combined.to_csv(filepath)
            else:
                raise ValueError("`scores` must be a DataFrame, Series, or Dictionary of such.")
                
            logger.info(f"💾 Extracted raw signal score to: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Failed to export signal scores for {self.signal_name}: {e}")
            raise
