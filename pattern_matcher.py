import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine
from fastdtw import fastdtw
import logging

logger = logging.getLogger(__name__)

class PatternMatcher:
    def __init__(self, window_size: int = 20):
        self.window_size = window_size

    def _cosine_similarity(self, ts1, ts2):
        """Calculates cosine similarity (1 - cosine distance)."""
        return 1 - cosine(ts1, ts2)

    def find_similar_patterns(self, target_series: np.array, historical_series: np.array, top_n: int = 5, candidate_pool_size: int = 50):
        """
        Two-stage similarity search:
        1. Filter candidates using Cosine Similarity (Fast).
        2. Re-rank top candidates using FastDTW (Precise).
        """
        if len(target_series) < self.window_size or len(historical_series) < self.window_size:
            return []

        # Sliding window candidates
        candidates = []
        for i in range(len(historical_series) - self.window_size + 1):
            window = historical_series[i : i + self.window_size]
            # Normalize window to handle scale differences
            window_norm = (window - np.mean(window)) / (np.std(window) + 1e-9)
            target_norm = (target_series - np.mean(target_series)) / (np.std(target_series) + 1e-9)
            
            cos_sim = self._cosine_similarity(target_norm, window_norm)
            candidates.append({
                'start_idx': i,
                'cos_sim': cos_sim,
                'window': window_norm
            })

        # Stage 1: Filter using Cosine Similarity
        candidates.sort(key=lambda x: x['cos_sim'], reverse=True)
        top_candidates = candidates[:candidate_pool_size]

        # Stage 2: Re-rank using FastDTW
        results = []
        target_norm = (target_series - np.mean(target_series)) / (np.std(target_series) + 1e-9)
        
        for cand in top_candidates:
            distance, path = fastdtw(target_norm, cand['window'], dist=2) # Euclidean distance
            results.append({
                'start_idx': cand['start_idx'],
                'dtw_distance': float(distance),
                'cos_sim': float(cand['cos_sim'])
            })

        # Re-rank by DTW distance (lower is better)
        results.sort(key=lambda x: x['dtw_distance'])
        return results[:top_n]
