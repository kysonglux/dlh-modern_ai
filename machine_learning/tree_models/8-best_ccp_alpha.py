#!/usr/bin/env python3
"""selects the best pruning value ccp_alpha"""


def get_best_alpha(clfs, train_scores, test_scores,
                   ccp_alphas):
    """get_best_alpha(clfs, train_scores, test_scores, ccp_alphas)"""
    max_test = max(test_scores)

    best_indices = [i for i, score in enumerate(test_scores)
                    if score == max_test]

    gaps = [(abs(train_scores[i] - test_scores[i]), i) for i in best_indices]
    min_gap = min(gaps)[0]

    gap_indices = [i for gap, i in gaps if gap == min_gap]

    alpha_candidates = [(ccp_alphas[i], i) for i in gap_indices]
    best_alpha, best_index = max(alpha_candidates)

    return best_alpha, clfs[best_index]
