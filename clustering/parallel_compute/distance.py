import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
from sklearn.cluster import DBSCAN

def calculate_overlap_percentage_vectorized(bboxes1, bboxes2):

    # Expand dimensions to (N1, 1, 4) and (1, N2, 4) to enable broadcasting
    bboxes1_exp = np.expand_dims(bboxes1, axis=1)
    bboxes2_exp = np.expand_dims(bboxes2, axis=0)

    # Calculate the coordinates of the intersection rectangles
    inter_x1 = np.maximum(bboxes1_exp[:, :, 0], bboxes2_exp[:, :, 0])
    inter_y1 = np.maximum(bboxes1_exp[:, :, 1], bboxes2_exp[:, :, 1])
    inter_x2 = np.minimum(bboxes1_exp[:, :, 2], bboxes2_exp[:, :, 2])
    inter_y2 = np.minimum(bboxes1_exp[:, :, 3], bboxes2_exp[:, :, 3])

    # Compute the width and height of the intersection rectangles
    inter_width = np.maximum(0, inter_x2 - inter_x1)
    inter_height = np.maximum(0, inter_y2 - inter_y1)

    # Compute the area of the intersection rectangles
    inter_area = inter_width * inter_height

    # Compute the area of bbox1
    bbox1_area = (bboxes1_exp[:, :, 2] - bboxes1_exp[:, :, 0]) * (
        bboxes1_exp[:, :, 3] - bboxes1_exp[:, :, 1]
    )

    # Calculate the overlapping percentages
    # overlap_percentages = np.where(bbox1_area > 0, inter_area / bbox1_area, 0)

    with np.errstate(divide="ignore", invalid="ignore"):
        overlap_percentages = np.where(bbox1_area > 0, inter_area / bbox1_area, 0)

    return overlap_percentages


def penalized_mse_from_one_hot_vectorized(
    vectors, penalty_weight=1.0, min_value_threshold=0.5
):
    # Identify the index and value of the maximum element for each vector
    max_indices = np.argmax(vectors, axis=1)
    max_values = np.max(vectors, axis=1)

    # Create one-hot encoded vectors with the same shape
    one_hot_vectors = np.zeros_like(vectors)
    one_hot_vectors[np.arange(vectors.shape[0]), max_indices] = 1

    # Calculate the mean squared error for each vector
    mse = np.mean((vectors - one_hot_vectors) ** 2, axis=1)

    # Calculate the penalty for low maximum values
    penalty = penalty_weight * np.maximum(0, min_value_threshold - max_values)

    # Combine MSE with the penalty
    penalized_mses = mse + penalty

    return penalized_mses


def distance_function_vectorized(reference_json: np.ndarray, other_json: np.ndarray):
    if len(reference_json) != len(other_json):
        return 1.0
    overlaps = calculate_overlap_percentage_vectorized(reference_json, other_json)

    # Determine if there are any matches
    has_matches = np.sum(overlaps, axis=1) > 0

    # Apply penalized_mse_from_one_hot to rows with matches
    loss_with_matches = penalized_mse_from_one_hot_vectorized(overlaps[has_matches])

    # Penalize non-matching blocks heavily
    loss_without_matches = np.ones(
        np.sum(~has_matches)
    )  # Assuming 1.0 is the maximum penalty

    # Combine the losses
    if len(loss_with_matches) > 0:
        loss = np.concatenate([loss_with_matches, loss_without_matches])
    else:
        loss = loss_without_matches

    if len(loss) == 0:
        loss = np.array([1.0])

    return float(np.mean(loss))




def compute_distance(i, j, text_i, text_j):
    dist1 = distance_function_vectorized(text_i, text_j)
    return (i, j, dist1)



def create_cluster(batch,cluster_count):
    eps = 0.01
    min_samples = 2
    n = len(batch)
    unclustered = []
    distance_matrix = np.zeros((n, n))
    matrix = [
        (i, j, batch[i][1], batch[j][1])
        for i in range(n)
        for j in range(i + 1, n)
    ]
    # each value of text_i and text_j is a n*4 matrix which distance with another n*4 matrix will be computed
    results = Parallel(n_jobs=8)(
        delayed(compute_distance)(i, j, text_i, text_j)
        for i, j, text_i, text_j in tqdm(matrix)
    )

    # fill matrix with upper triangle values distance text_i -> text_j = text_j -> text_i
    for i, j, dist in results:
        distance_matrix[i, j] = dist
        distance_matrix[j, i] = dist
