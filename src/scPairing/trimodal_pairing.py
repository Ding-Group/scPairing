import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def generate_trimodal_pairing(latents1, latents2, latents3, cutoff=0.8, total_cutoff=2.7, seed=0):
    """Greedy randomized algorithm to generate an trimodal pairing
    
    Parameters
    ----------
    latents1
        Latent representations of the first modality.
    latents2
        Latent representations of the second modality.
    latents3
        Latent representations of the third modality.
    cutoff
        Minimum similarity between two modalities for them to be paired together.
    total_cutoff
        The total combined similarity between three modalities for them to be paired together
    seed
        Random seed.
    """
    np.random.seed(seed)

    n_cells = min([latents1.shape[0], latents2.shape[0], latents3.shape[0]])
    rna_ind = [i for i in range(latents1.shape[0])]
    atac_ind = [i for i in range(latents2.shape[0])]
    prot_ind = [i for i in range(latents3.shape[0])]

    pairings = []

    dist_matrix_1 = cosine_similarity(latents1, latents2)
    dist_matrix_2 = cosine_similarity(latents1, latents3)
    dist_matrix_3 = cosine_similarity(latents2, latents3)

    dist_matrix_1[dist_matrix_1 < cutoff] = 0
    dist_matrix_2[dist_matrix_2 < cutoff] = 0
    dist_matrix_3[dist_matrix_3 < cutoff] = 0

    skip = set()

    for _ in range(n_cells):
        ind = np.arange(len(rna_ind))
        np.random.shuffle(ind)
        for d in range(len(ind)):
            i = ind[d]
            if rna_ind[i] in skip:
                continue
            N = dist_matrix_1[i, :, None] + dist_matrix_3 + dist_matrix_2[i, None, :]
            j, k = np.unravel_index(N.argmax(), N.shape)
            if N[j, k] >= total_cutoff:
                pairings.append((rna_ind[i], atac_ind[j], prot_ind[k]))

                dist_matrix_1 = np.delete(dist_matrix_1, i, axis=0)
                dist_matrix_1 = np.delete(dist_matrix_1, j, axis=1)
                dist_matrix_2 = np.delete(dist_matrix_2, i, axis=0)
                dist_matrix_2 = np.delete(dist_matrix_2, k, axis=1)
                dist_matrix_3 = np.delete(dist_matrix_3, j, axis=0)
                dist_matrix_3 = np.delete(dist_matrix_3, k, axis=1)

                del rna_ind[i], atac_ind[j], prot_ind[k]
                break
            else:
                skip.add(rna_ind[i])

        if d == len(ind):
            break

    return pairings


def generate_trimodal_pairing2(latents1, latents2, latents3, cutoff=0.8, total_cutoff=2.7, seed=0):
    """Greedy randomized algorithm to generate an trimodal pairing
    
    Parameters
    ----------
    latents1
        Latent representations of the first modality.
    latents2
        Latent representations of the second modality.
    latents3
        Latent representations of the third modality.
    cutoff
        Minimum similarity between two modalities for them to be paired together.
    total_cutoff
        The total combined similarity between three modalities for them to be paired together
    seed
        Random seed.
    """
    np.random.seed(seed)

    n_cells = min([latents1.shape[0], latents2.shape[0], latents3.shape[0]])
    rna_ind = [i for i in range(latents1.shape[0])]

    pairings = []

    dist_matrix_1 = cosine_similarity(latents1, latents2)
    dist_matrix_2 = cosine_similarity(latents1, latents3)
    dist_matrix_3 = cosine_similarity(latents2, latents3)

    dist_matrix_1[dist_matrix_1 < cutoff] = 0
    dist_matrix_2[dist_matrix_2 < cutoff] = 0
    dist_matrix_3[dist_matrix_3 < cutoff] = 0

    ind = np.arange(len(rna_ind))
    np.random.shuffle(ind)

    for d in range(n_cells):
    
        i = ind[d]
        N = dist_matrix_1[i, :, None] + dist_matrix_3 + dist_matrix_2[i, None, :]
        j, k = np.unravel_index(N.argmax(), N.shape)
        if N[j, k] >= total_cutoff:
            pairings.append((i, j, k))

            dist_matrix_1[i, :] = 0
            dist_matrix_1[:, j] = 0
            dist_matrix_2[i, :] = 0
            dist_matrix_2[:, k] = 0
            dist_matrix_3[j, :] = 0
            dist_matrix_3[:, k] = 0


    return pairings
