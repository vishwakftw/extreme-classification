import numpy as np


def precision_at_k(ground_truth, predictions, k=5, pos_label=1):
    """
    Function to evaluate the precision @ k for a given
    ground truth vector and a list of predictions (between 0 and 1).

    Args:
        ground_truth : np.array consisting of multi-hot encoding of
                       label vector
        predictions : np.array consisting of predictive probabilities
                      for every label.
        k : Value of k. Default: 5
        pos_label : Value to consider as positive. Default: 1

    Returns:
        precision @ k for a given ground truth - prediction pair.
    """
    assert len(ground_truth) == len(predictions), "P@k: Length mismatch"

    n_pos_vals = (ground_truth == pos_label).sum()
    desc_order = np.argsort(predictions)[::-1]  # ::-1 reverses array
    ground_truth = np.take(ground_truth, desc_order[:k])  # the top indices
    relevant_preds = (ground_truth == pos_label).sum()

    return relevant_preds / min(n_pos_vals, k)


def dcg_score_at_k(ground_truth, predictions, k=5, pos_label=1):
    """
    Function to evaluate the Discounted Cumulative Gain @ k for a given
    ground truth vector and a list of predictions (between 0 and 1).

    Args:
        ground_truth : np.array consisting of multi-hot encoding of label
                       vector
        predictions : np.array consisting of predictive probabilities for
                      every label.
        k : Value of k. Default: 5
        pos_label : Value to consider as positive. Default: 1

    Returns:
        DCG @ k for a given ground truth - prediction pair.
    """
    assert len(ground_truth) == len(predictions), "DCG@k: Length mismatch"

    desc_order = np.argsort(predictions)[::-1]  # ::-1 reverses array
    ground_truth = np.take(ground_truth, desc_order[:k])  # the top indices
    gains = 2 ** ground_truth - 1

    discounts = np.log2(np.arange(1, len(ground_truth) + 1) + 1)
    return np.sum(gains / discounts)


def ndcg_score_at_k(ground_truth, predictions, k=5, pos_label=1):
    """
    Function to evaluate the Discounted Cumulative Gain @ k for a given
    ground truth vector and a list of predictions (between 0 and 1).

    Args:
        ground_truth : np.array consisting of multi-hot encoding of label
                       vector
        predictions : np.array consisting of predictive probabilities for
                      every label.
        k : Value of k. Default: 5
        pos_label : Value to consider as positive. Default: 1

    Returns:
        NDCG @ k for a given ground truth - prediction pair.
    """
    dcg_at_k = dcg_score_at_k(ground_truth, predictions, k, pos_label)
    best_dcg_at_k = dcg_score_at_k(ground_truth, ground_truth, k, pos_label)
    return dcg_at_k / best_dcg_at_k
