import numpy as np

from scipy import spatial
import scipy.cluster.hierarchy as sch

from itertools import permutations

def get_one_hot(targets, nb_classes):
    # TODO: test this
    res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
    return res.reshape(list(targets.shape)+[nb_classes])

def draw_data(pop_allele_freq, ind_admix_propn):
    # pop_allele_freq is n_loci x n_population
    # ind_admix_propn is n_obs x n_population

    n_obs = ind_admix_propn.shape[0]
    n_pop = ind_admix_propn.shape[1]

    n_loci = pop_allele_freq.shape[0]
    assert pop_allele_freq.shape[1] == n_pop

    # population belongings for each loci
    z_a = np.array([np.random.choice(n_pop, p=row, size = (n_loci))
              for row in ind_admix_propn])
    z_b = np.array([np.random.choice(n_pop, p=row, size = (n_loci))
              for row in ind_admix_propn])
    z_a_onehot = get_one_hot(z_a, nb_classes=n_pop)
    z_b_onehot = get_one_hot(z_b, nb_classes=n_pop)


    # allele frequencies for each individual at each loci
    ind_allele_freq_a = np.einsum('nlk, lk -> nl', z_a_onehot, pop_allele_freq)
    ind_allele_freq_b = np.einsum('nlk, lk -> nl', z_b_onehot, pop_allele_freq)

    # draw genotypes at each chromosome
    genotype_a = (np.random.random((n_obs, n_loci)) < ind_allele_freq_a).astype(int)
    genotype_b = (np.random.random((n_obs, n_loci)) < ind_allele_freq_b).astype(int)

    # we only observe their sum
    g_obs = genotype_a + genotype_b
    g_obs = get_one_hot(g_obs, nb_classes=3)

    return g_obs


####################
# Other utils for
# permuting / clustering matrices
####################
def find_min_perm(x, y, axis = 0):
    # perumutes array x along axis to find closest
    # match to y

    perms = list(permutations(np.arange(x.shape[axis])))

    i = 0
    diff_best = np.Inf
    for perm in perms:

        x_perm = x.take(perm, axis)

        diff = np.sum((x_perm - y)**2)

        if diff < diff_best:
            diff_best = diff
            i_best = i

        i += 1

    return perms[i_best]

def cluster_admix_get_indx(ind_admix_propn):
    # clusters the individual admixtures for better plotting
    y = sch.linkage(ind_admix_propn, method='average')
    indx = sch.dendrogram(y, no_plot=True)["leaves"]

    return indx
