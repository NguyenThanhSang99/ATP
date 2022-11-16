import nimfa


def run_nmf(V,rank = 12, max_iter = 5000):
    """
    Run standard nonnegative matrix factorization.
    
    :param V: Target matrix to estimate.
    :type V: :class:`numpy.matrix`
    """
    # Euclidean
    
    nmf = nimfa.Nmf(V, seed="random_vcol", rank=rank, max_iter=max_iter, update='euclidean',
                      objective='fro')
    fit = nmf()
    # divergence
    nmf = nimfa.Nmf(V, seed="random_vcol", rank=rank, max_iter=max_iter, initialize_only=True,
                    update='divergence', objective='div')
    fit = nmf()
    return fit.basis(),fit.coef()