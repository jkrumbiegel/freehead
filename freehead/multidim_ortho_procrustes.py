import numpy as np


def multidim_ortho_procrustes(As, Bs):
    # from As to Bs, reference onto measured
    # As & Bs : N x M x 3

    u, w, vt = np.linalg.svd(np.einsum('nmi,nmj->nij', Bs, As))
    # we flip vt to v and u to ut by using the indices in reverse, which is like transposing before multiplication
    Rs = np.einsum('nji,nkj->nik', vt, u)
    determinants = np.linalg.det(Rs)
    negative_determinants = np.isclose(determinants, -1)

    J = np.diag([1, 1, -1])

    # same reversal here
    Rs[negative_determinants, ...] = np.einsum(
        'nji,njk->nik',
        vt[negative_determinants],
        np.einsum(
            'ij,nkj->nik',
            J,
            u[negative_determinants]))

    return Rs