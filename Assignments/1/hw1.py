##print('hello world')
'''
hw1.py
Author: murt

Tufts COMP 135 Intro ML
'''
##print('start of part 1')
import numpy as np

def split_into_train_and_test(x_all_LF, frac_test=0.5, random_state=None):
    ''' Divide provided array into train and test set along first dimension

    User can provide a random number generator object to ensure reproducibility.

    Args
    ----
    x_all_LF : 2D array, shape = (n_total_examples, n_features) (L, F)
        Each row is a feature vector
    frac_test : float, fraction between 0 and 1
        Indicates fraction of all L examples to allocate to the "test" set
    random_state : np.random.RandomState instance or integer or None
        If int, code will create RandomState instance with provided value as seed
        If None, defaults to the current numpy random number generator np.random

    Returns
    -------
    x_train_MF : 2D array, shape = (n_train_examples, n_features) (M, F)
        Each row is a feature vector
        Should be a separately allocated array, NOT a view of any input array

    x_test_NF : 2D array, shape = (n_test_examples, n_features) (N, F)
        Each row is a feature vector
        Should be a separately allocated array, NOT a view of any input array

    Post Condition
    --------------
    This function should be side-effect free. The provided input array x_all_LF
    should not change at all (not be shuffled, etc.)

    Examples
    --------
    >>> x_LF = np.eye(10)
    >>> xcopy_LF = x_LF.copy() # preserve what input was before the call
    >>> train_MF, test_NF = split_into_train_and_test(
    ...     x_LF, frac_test=0.3, random_state=np.random.RandomState(0))
    >>> train_MF.shape
    (7, 10)
    >>> test_NF.shape
    (3, 10)
    >>> print(train_MF)
    [[0. 0. 1. 0. 0. 0. 0. 0. 0. 0.]
     [0. 0. 0. 0. 0. 0. 0. 0. 1. 0.]
     [0. 0. 0. 0. 1. 0. 0. 0. 0. 0.]
     [0. 0. 0. 0. 0. 0. 0. 0. 0. 1.]
     [0. 1. 0. 0. 0. 0. 0. 0. 0. 0.]
     [0. 0. 0. 0. 0. 0. 1. 0. 0. 0.]
     [0. 0. 0. 0. 0. 0. 0. 1. 0. 0.]]
    >>> print(test_NF)
    [[0. 0. 0. 1. 0. 0. 0. 0. 0. 0.]
     [1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
     [0. 0. 0. 0. 0. 1. 0. 0. 0. 0.]]

    ## Verify that input array did not change due to function call
    >>> np.allclose(x_LF, xcopy_LF)
    True

    References
    ----------
    For more about RandomState, see:
    https://stackoverflow.com/questions/28064634/random-state-pseudo-random-numberin-scikit-learn
    
if random_state is None:
random_state = np.random
    '''
## TODO fixme Murt's actual work

    L = int(x_all_LF.shape[0])
    N = int(np.ceil(frac_test * L))
    M = L - N

    if random_state is None:
        random_state = np.random
    elif isinstance(random_state, int):
        random_state = np.random.RandomState(random_state)

    xcopy_all_LF = x_all_LF.copy()

    random_state.shuffle(xcopy_all_LF)

    x_train_MF = xcopy_all_LF[:M, :]
    x_test_NF = xcopy_all_LF[M:, :]

    return x_train_MF, x_test_NF

##print('end of part 1')


##print('start of part 2')
def calc_k_nearest_neighbors(data_NF, query_QF, K=1):
    ''' Compute and return k-nearest neighbors under Euclidean distance

    Any ties in distance may be broken arbitrarily.

    Args
    ----
    data_NF : 2D array, shape = (n_examples, n_features) aka (N, F)
        Each row is a feature vector for one example in dataset
    query_QF : 2D array, shape = (n_queries, n_features) aka (Q, F)
        Each row is a feature vector whose neighbors we want to find
    K : int, positive (must be >= 1)
        Number of neighbors to find per query vector

    Returns
    -------
    neighb_QKF : 3D array, (n_queries, n_neighbors, n_feats) (Q, K, F)
        Entry q,k is feature vector of the k-th neighbor of the q-th query
    '''
    # TODO fixme Murt's actual work

 

    if K < 1:
        print('K must be a value >= 1')
        return None

    Q = int(query_QF.shape[0])
    F = int(query_QF.shape[1])
    N = int(data_NF.shape[0])

    neighb_QKF = np.empty([Q, K, F])

    for i in range(Q):
        querying = query_QF[i]
        distances = np.empty(N)
        for j in range(N):
            neighbor = data_NF[j]
            dist = 0
            for f in range(F):
                dist += ((neighbor[f]-querying[f])**2)
            dist = np.sqrt(dist)
            distances[j] = dist
        indices = []
        for k in range(K):
            index = np.argmin(distances)
            indices.append(index)
            distances[index] = float('inf')

        neighbors = data_NF[indices, :]
        neighb_QKF[i] = neighbors

    return neighb_QKF

##print('end of part 2')
