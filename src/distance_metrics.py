import numpy as np

def euclidean(x, y):
    return np.linalg.norm(x - y)

def mahalanobis(x, y, cov_inv):
    diff = x - y
    return np.sqrt(diff.T @ cov_inv @ diff)

def adaptive_distance(x, y, weights):
    # Your MPhil idea: weighted distance based on feature importance
    return np.sqrt(np.sum(weights * (x - y)**2))

if __name__ == "__main__":
    print("Distance metrics ready")