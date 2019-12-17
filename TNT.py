
import numpy as np
from numpy.linalg import norm


def check_result(ternary, target):
    '''
    Calculating the cosine similarity between ternary and target
    ==============
    Parameters
    ternary: a numpy array and its elements are -1, 0 or 1
    target: a numpy array whose elements are floating numbers.

    Output
    return the cosine similarity
    '''

    if norm(ternary, ord=2) == 0 or norm(target, ord=2) == 0:
        return 0
    else:
        return np.dot(ternary, target) / (norm(ternary, ord=2) * norm(target, ord=2))


def order_vector(target_vector, num):
    '''
    Ordering a list whose elements are -1, 0, 1 follows
    the order of target_vector's.
    -----------
    Parameter
    ===========
    binary_vector: all elements are -1, 0, 1
    target_vector: the targeterved floating type list
    num: a int, indicating how many -1 and 1 the binary_vector should have
    -----------
    Return
    ===========
    binary_vector: whose elements are all -1, 0 or 1 and have
                   the same ordering with target_vector.
    '''
    # sort the target_vector in a decreasing order,
    # and return the index of each
    # elements after sorting
    binary_vector = signalization(target_vector)
    x_sorted_index = np.argsort(np.abs(target_vector))[::-1]
    for elem in x_sorted_index[(num + 2)::]:
        binary_vector[elem] = 0

    return binary_vector


def normalize_rows(x):
    """
    function that normalizes each row of the matrix x to have unit length.

    Args:
     ``x``: A numpy matrix of shape (n, m)

    Returns:
     ``x``: The normalized (by row) numpy matrix.
    """
    return x / norm(x, ord=2, keepdims=True)


def similar_cos(target_vector):
    # norm_scalar = norm(target_vector)
    # target_l2 = target_vector / norm_scalar
    target_hat = normalize_rows(target_vector)
    target_hat_sorted = sorted(np.abs(target_hat), reverse=True)
    similar_value = []
    temp = target_hat_sorted[0]
    for i in range(1, len(target_hat_sorted)):
        temp += target_hat_sorted[i]
        similar_value.append(temp / np.sqrt(i + 1))
    return similar_value, max(similar_value), np.argmax(similar_value)


def signalization(target_vector):
    '''
    Parameters
    ============
    target_vector: the vector that will be converted to a ternary vector.
    ------------
    Return
    ============
    binary_vector: all elements convert to -1 or 1
    '''

    binary_vector = target_vector.copy()
    binary_vector[binary_vector < 0] = -1
    binary_vector[binary_vector >= 0] = 1

    return binary_vector


def TNT_convert(weights, name=False):
    '''
    ternary_T_filter: the cluster results of the weights of a filter.
    n_clusters_: The number of clusters, in Ternary it should be 3.
    '''

    kernel_flatten = weights.flatten()
    # print(np.shape(kernel_flatten))
    # binary_type = signalization(kernel_flatten)
    similarValue_conTerv, maxValue_conv, maxInde_conv = similar_cos(kernel_flatten)
    restultVector_conv = order_vector(kernel_flatten, maxInde_conv)
    # result_conv = check_result(restultVector_conv, kernel_flatten)
    '''
    if name:
        print('[INFO] the cosine similarity is {}'.
              format(result_conv))
    else:
        print('[INFO] the similarity of the layer {} is {}'
              .format(name, result_conv))
    '''
    ternary_weights = np.array(restultVector_conv).reshape(weights.shape)

    return ternary_weights


def inner(a_, t_):
    return np.dot(a_, t_.reshape(-1, 1)) / (norm(a_) * (norm(t_) + 0.00001))


def scaling(weights_, ternary_):

    a = weights_.flatten()
    t = ternary_.flatten()
    ap = a.copy()
    an = a.copy()
    tp = t.copy()
    tn = t.copy()

    ap[a < 0.] = 0.
    an[a > 0.] = 0.
    tp[t < 0.] = 0.
    tn[t > 0.] = 0.

    rp = 0
    rn = 0
    if sum(ap) == 0:
        rp = 0
    else:
        rp = (norm(ap) / (norm(tp) + 0.00001)) * inner(ap, tp)

    if sum(an) == 0:
        rn = 0
    else:
        rn = (norm(an) / (norm(tn) + 0.00001)) * inner(an, tn)
    t_result = tp * rp + tn * rn
    return t_result.reshape(weights_.shape)


def kernels_cluster(weights_):
    if weights_.ndim == 4:
        r, c, in_sample, out_sample = weights_.shape
        kernels = np.zeros(weights_.shape)
        for i in range(out_sample):
            for j in range(in_sample):
                xi = weights_[:, :, j, i]
                temp_T = TNT_convert(xi)
                t_ = scaling(xi, temp_T)
                kernels[:, :, j, i] = t_
        return kernels
    elif weights_.ndim == 2:
        temp_T = TNT_convert(weights_)
        t_ = scaling(weights_, temp_T)
        return t_
    elif weights_.ndim == 1:
        temp_T = TNT_convert(weights_)
        t_ = scaling(weights_, temp_T)
        return t_
