"""Timing utilities for benchmarking inference stages."""

import time
import numpy as np


class Timer:
    """Tracks wall-clock time for embedding, matching-score, and ranking stages.

    Attributes:
        text2graph_text_embedding_time: Durations for text embedding calls.
        text2graph_text_embedding_iter: Iteration counts for embedding calls.
        text2graph_text_embedding_matching_score_time: Durations for score computation.
        text2graph_text_embedding_matching_score_iter: Iteration counts for score computation.
        text2graph_matching_time: Durations for the ranking/sorting step.
        text2graph_matching_iter: Iteration counts for the ranking step.
    """

    def __init__(self) -> None:
        self.start_time = time.time()
        self.total_time = 0

        self.text2graph_text_embedding_time = []
        self.text2graph_text_embedding_iter = []

        self.text2graph_text_embedding_matching_score_time = []
        self.text2graph_text_embedding_matching_score_iter = []

        self.text2graph_matching_time = []
        self.text2graph_matching_iter = []

    def save(self, path: str, cfg: object) -> None:
        """Writes aggregated timing statistics to a text file.

        Args:
            path: Destination file path.
            cfg: Hydra DictConfig (uses ``cfg.eval.eval_iters``,
                ``cfg.eval.eval_iter_count``, ``cfg.eval.out_of``).
        """
        eval_iters = cfg.eval.eval_iters
        eval_iter_count = cfg.eval.eval_iter_count
        out_of = cfg.eval.out_of

        with open(path, 'w') as f:
            assert len(self.text2graph_text_embedding_matching_score_time) == len(self.text2graph_text_embedding_matching_score_iter), \
                "Mismatch between score times and score iteration counts"
            assert len(self.text2graph_matching_time) == len(self.text2graph_matching_iter), \
                "Mismatch between matching times and matching iteration counts"
            assert sum(self.text2graph_matching_iter) == eval_iters * eval_iter_count, \
                "Total matching iterations does not match eval_iters * eval_iter_count"

            f.write(f'start_time: {self.start_time}\n')
            f.write(f'total_time: {self.total_time}\n')
            f.write(f'text2graph_text_embedding_time: {sum(self.text2graph_text_embedding_time)}\n')
            f.write(f'text2graph_text_embedding_iter: {sum(self.text2graph_text_embedding_iter)}\n')
            f.write(f'text2graph_text_embedding_matching_score_time: {sum(self.text2graph_text_embedding_matching_score_time)}\n')
            f.write(f'text2graph_text_embedding_matching_score_iter: {sum(self.text2graph_text_embedding_matching_score_iter)}\n')
            f.write(f'text2graph_matching_time: {sum(self.text2graph_matching_time)}\n')
            f.write(f'text2graph_matching_iter: {sum(self.text2graph_matching_iter)}\n')

            time_for_embedding = sum(self.text2graph_text_embedding_time) / sum(self.text2graph_text_embedding_iter)
            time_for_matching_score = sum(self.text2graph_text_embedding_matching_score_time) / sum(self.text2graph_text_embedding_matching_score_iter)
            time_for_matching = sum(self.text2graph_matching_time) / sum(self.text2graph_matching_iter)

            f.write(f'Embedding time, avg time for 1 encode_text(str): {time_for_embedding}\n')
            f.write(f'Std of embedding time: {np.std(self.text2graph_text_embedding_time)}\n')
            f.write(f'Matching score time, avg time for 1 matching score: {time_for_matching_score}\n')
            f.write(f'Std of matching score time: {np.std(self.text2graph_text_embedding_matching_score_time)}\n')
            f.write(f'Matching time, avg time for 1 matching, or sorting within {out_of}: {time_for_matching}\n')
            f.write(f'Std of matching time: {np.std(self.text2graph_matching_time)}\n')

            calc_time = time_for_embedding + time_for_matching_score * out_of + time_for_matching
            f.write(f'Total run time for 1 text matching against {out_of} database scenes: {calc_time}\n')
