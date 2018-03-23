import numpy as np
from q2e_word2vec import normalizeRows


def knn(vector, matrix, k=10):
    """
    Finds the k-nearest rows in the matrix with comparison to the vector.
    Use the cosine similarity as a distance metric.

    Arguments:
    vector -- A D dimensional vector
    matrix -- V x D dimensional numpy matrix.

    Return:
    nearest_idx -- A numpy vector consists of the rows indices of the k-nearest neighbors in the matrix
    """

    norm_vector = np.zeros([matrix.shape[0]])
    nearest_idx = []

    ### YOUR CODE

    for idx in range(matrix.shape[0]):
        #calc cosine distance
        dot_prod = np.dot(vector,matrix[idx,:])
        vec_norm = np.linalg.norm(vector)
        mat_norm = np.linalg.norm(matrix[idx,:])

        if dot_prod != 0 :
            norm_vector[idx] = dot_prod/vec_norm/mat_norm
        else:
            norm_vector[idx] = 0

    #find K-nn
    nearest_idx = np.ndarray.tolist(np.argsort(1-norm_vector)[:k]) # find the k indexes that are closest to 1

    ### END YOUR CODE

    return nearest_idx

def test_knn():
    """
    Use this space to test your knn implementation by running:
        python knn.py
    This function will not be called by the autograder, nor will
        your tests be graded.
    """
    print "Running your tests..."
    ### YOUR CODE HERE

    indices = knn(np.array([0.2,0.5]), np.array([[0,0.5],[0.1,0.1],[0,0.5],[2,2],[4,4],[3,3]]), k=2)
    assert 0 in indices and 2 in indices and len(indices) == 2

    ### END YOUR CODE

if __name__ == "__main__":
    test_knn()


