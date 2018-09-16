import numpy as np


def multidim_ortho_procrustes(measured, reference):
    # from reference onto measured
    # both N x M x 3
    # from https://www.cse.iitb.ac.in/~ajitvr/CS763_Spring2017/procrustes.pdf

    u, w, vt = np.linalg.svd(np.einsum('nmi,nmj->nij', reference, measured))
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