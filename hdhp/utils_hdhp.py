import numpy as np
import scipy.stats
from scipy.special import erfc, gammaln
import pickle
from numpy import exp
from numpy.random import RandomState
from scipy import *
from numpy import log as ln
from scipy.misc import logsumexp
from copy import deepcopy, copy
from collections import defaultdict, OrderedDict


def memoize(f):
    class memodict(dict):
        __slots__ = ()
        def __missing__(self, key):
            self[key] = ret = f(key)
            return ret
    return memodict().__getitem__


@memoize
def _gammaln(x):
    return gammaln(x)


@memoize
def _ln(x):
    return ln(x)


def copy_dict(original):
    new = {k: copy(original[k]) for k in original}
    return new


def parse_newsitem_2_doc(news_item, words_idx, time_unit):
    index = news_item[0]
    timestamp = news_item[1] / time_unit  # unix time in hour
    word_id = news_item[2][0]
    count = news_item[2][1]
    word_distribution = defaultdict(int)
    if words_idx:
        for i in range(len(word_id)):
            if word_id[i] in words_idx:
                word_distribution[words_idx[word_id[i]]] = count[i]
    else:
        for i in range(len(word_id)):
            word_distribution[word_id[i]] = count[i]
    word_count = sum(list(word_distribution.values()))
    doc = Document(index, timestamp, word_distribution, word_count)
    return doc


def get_df_words(data, num=10000):
    def doc_freq(data):
        doc_size = len(data)
        freq = defaultdict(int)
        for doc in data:
            for word in doc[2][0]:
                freq[word] += 1
        values = np.array(list(freq.values())) / doc_size
        keys = freq.keys()
        return keys, values

    k, v = doc_freq(data)
    idx = dict(zip(k, v))
    sorted_idx = sorted(idx.items(), key=lambda x: x[1], reverse=True)
    idx = {sorted_idx[i][0]: i for i in range(num)}
    ridx = {i: sorted_idx[i][0] for i in range(num)}
    return idx, ridx


class Document(object):
    def __init__(self, index, timestamp, word_distribution, word_count):
        super(Document, self).__init__()
        self.index = index
        self.timestamp = timestamp
        self.word_distribution = word_distribution
        self.word_count = word_count


class Particle(object):
    """docstring for Particle"""
    def __init__(self, weight, index=0):
        super(Particle, self).__init__()
        self.alpha = dict() # topic level
        self.index = index
        self.weight = weight
        self.log_update_prob = 0
        self.docs2cluster_ID = []  # the element is the cluster index of a sequence of document ordered by the index of document
        self.active_clusters = {}  # dict key = cluster_index, value = list of timestamps in specific cluster (queue)
        self.cluster_num_by_now = 0
        self.cls_2_topic = {}

        self.topic_num_by_now = 0
        self.docs2topic_ID = []
        self.topic_count = dict()
        self.topic_word_distribution = {}
        self.topic_word_count = defaultdict(int)

        self._Q0 = 0

    def add_doc(self, topic_index, doc):
        self.topic_word_count[topic_index] += doc.word_count
        if topic_index not in self.topic_word_distribution:
            self.topic_word_distribution[topic_index] = defaultdict(int)
        for word in doc.word_distribution:
            self.topic_word_distribution[topic_index][word] += doc.word_distribution[word]

    def copy(self, particle):
        self.index = particle.index
        self.weight = particle.weight
        self.cluster_num_by_now = particle.cluster_num_by_now
        self.log_update_prob = particle.log_update_prob
        self.docs2cluster_ID = particle.docs2cluster_ID.copy()
        self.docs2topic_ID = particle.docs2topic_ID.copy()

        self.topic_num_by_now = particle.topic_num_by_now
        self.active_clusters = deepcopy(particle.active_clusters)
        self.topic_word_count = particle.topic_word_count.copy()
        self.topic_word_distribution = copy_dict(particle.topic_word_distribution)
        self.alpha = particle.alpha.copy()
        self.topic_count = particle.topic_count.copy()
        self.cls_2_topic = particle.cls_2_topic.copy()
        self._Q0 = particle._Q0

    def __repr__(self):
        return 'particle document list to cluster IDs: ' + str(self.docs2cluster_ID) + '\n' + 'weight: ' + str(
            self.weight)


def perplexity(particle, test_set, idx, theta0, reference_time, bandwidth, base_intensity, beta):
    log_llh = []
    prng = RandomState(1024)
    total_count = 0
    # particle = inferer.particles[0]
    for doc in test_set:
        doc = parse_newsitem_2_doc(news_item=doc, words_idx=idx, time_unit=3600)

        active_cluster_indexes = [0]
        active_cluster_rates = [base_intensity]
        base_textual_prob = log_dirichlet_multinomial_distribution_dict(dict(), doc.word_distribution, 0, doc.word_count, theta0)
        active_cluster_textual_probs = [base_textual_prob]
        topic_llhood = {}
        for active_cluster_index, timeseq in particle.active_clusters.items():
            topic_index = particle.cls_2_topic[active_cluster_index]
            active_cluster_indexes.append(active_cluster_index)
            time_intervals = doc.timestamp - np.array(timeseq)
            time_intervals = time_intervals[np.where(time_intervals > 0)]
            alpha = particle.alpha[active_cluster_index]
            rate = triggering_kernel(alpha, reference_time, time_intervals, bandwidth)
            # print(rate)
            active_cluster_rates.append(rate)
            cls_word_count = particle.topic_word_count[topic_index]
            cls_word_distribution = particle.topic_word_distribution[topic_index]
            if topic_index in topic_llhood:
                cls_log_dirichlet_multinomial_distribution = topic_llhood[topic_index]
            else:
                cls_log_dirichlet_multinomial_distribution = log_dirichlet_multinomial_distribution_dict(
                    cls_word_distribution, doc.word_distribution, cls_word_count, doc.word_count, theta0)
            active_cluster_textual_probs.append(cls_log_dirichlet_multinomial_distribution)

        active_cluster_logrates = np.log(active_cluster_rates/np.sum(active_cluster_rates))
        active_cluster_logrates = np.nan_to_num(active_cluster_logrates)

        cluster_selection_probs = active_cluster_logrates + active_cluster_textual_probs  # in log scale
        # print(cluster_selection_probs)
        selected_cluster_array = weight_choice_log(cluster_selection_probs, prng)

        if selected_cluster_array == 0:
            active_topic_index = []
            topic_pop = []
            topic_textual_prob = []
            for topic_index in range(1, particle.topic_num_by_now + 1):
                active_topic_index.append(topic_index)
                topic_pop.append(particle.topic_count[topic_index]/(particle.cluster_num_by_now + beta))
                topic_word_count = particle.topic_word_count[topic_index]
                topic_word_distribution = particle.topic_word_distribution[topic_index]
                if topic_index in topic_llhood:
                    tt_likelihood = topic_llhood[topic_index]
                else:
                    tt_likelihood = log_dirichlet_multinomial_distribution_dict(topic_word_distribution,
                                                                        doc.word_distribution,
                                                                        topic_word_count,
                                                                        doc.word_count,
                                                                        theta0)
                topic_textual_prob.append(tt_likelihood)
            topic_logpop = np.log(topic_pop / np.sum(topic_pop))
            topic_log_intensities = topic_logpop + topic_textual_prob
            tii = weight_choice_log(topic_log_intensities, prng)
            log_llh.append(topic_textual_prob[tii])
        else:
            log_llh.append(active_cluster_textual_probs[selected_cluster_array])
        total_count += doc.word_count
    return sum(log_llh)/total_count

def KL(p, q):
    P = np.ones([10000])
    P[list(p.keys())] = list(p.values())
    Q = np.ones([10000])
    Q[list(q.keys())] = list(q.values())
    temp = P/Q
    temp[np.isinf(temp)] = 0
    temp[np.isnan(temp)] = 0
    temp = P * log(temp)
    temp[np.isinf(temp)] = 0
    temp[np.isnan(temp)] = 0
    return exp(-sum(temp))

def dirichlet(prior):
    ''' Draw 1-D samples from a dirichlet distribution to multinomial distritbution. Return a multinomial probability distribution.
        @param:
            1.prior: Parameter of the distribution (k dimension for sample of dimension k).
        @rtype: 1-D numpy array
    '''
    return np.random.dirichlet(prior).squeeze()

def A(KL, xi, sigma):
    if xi.shape[0] > 1:
        return (KL * np.expand_dims(xi, 1) + sigma)
    else:
        return (KL  * xi + sigma)


def multinomial(exp_num, probabilities):
    ''' Draw samples from a multinomial distribution.
        @param:
            1. exp_num: Number of experiments.
            2. probabilities: multinomial probability distribution (sequence of floats).
        @rtype: 1-D numpy array
    '''
    np.random.seed()
    return np.random.multinomial(exp_num, probabilities).squeeze()

def EfficientImplementation(tn, reference_time, bandwidth, epsilon=1e-5):
    # TODO: What's the effect of this function
    ''' return the time we need to compute to update the triggering kernel
        @param:
            1.tn: float, current document time
            2.reference_time: list, reference_time for triggering_kernel
            3.bandwidth: int, bandwidth for triggering_kernel
            4.epsilon: float, error tolerance
        @rtype: float
    '''
    max_ref_time = max(reference_time)
    max_bandwidth = max(bandwidth)
    tu = tn - (max_ref_time + np.sqrt(
        -2 * max_bandwidth * np.log(0.5 * epsilon * np.sqrt(2 * np.pi * max_bandwidth ** 2))))
    return tu


def log_Dirichlet_CDF(outcomes, prior):
    ''' the function only applies to the symmetry case when all prior equals to 1.
        @param:
            1.outcomes: output variables vector
            2.prior: must be list of 1's in our case, avoiding the integrals.
        @rtype:
    '''
    return np.sum(np.log(outcomes)) + scipy.stats.dirichlet.logpdf(outcomes, prior)

def log_normal_CDF(outcomes, prior):
    scipy.stats.norm.logpdf()

def RBF_kernel(reference_time, time_interval, bandwidth):
    ''' RBF kernel for Hawkes process.
        @param:
            1.reference_time: np.array, entries larger than 0.
            2.time_interval: float/np.array, entry must be the same.
            3. bandwidth: np.array, entries larger than 0.
        @rtype: np.array
    '''
    numerator = - (time_interval - reference_time) ** 2 / (2 * bandwidth ** 2)
    denominator = (2 * np.pi * bandwidth ** 2) ** 0.5
    return np.exp(numerator) / denominator


'''
def mutual_kernel(mu, beta, kl, reference_time, time_intervals, bandwidth):
    time_intervals = time_intervals.reshape(-1, 1)
    return np.sum(np.sum((mu * kl + beta) * RBF_kernel(reference_time, time_intervals, bandwidth), axis=0), axis=0)
'''


def exp_kernel(omega, t_i, t_j):
    t_j = np.array(t_j)
    return exp(-omega * (t_i - t_j))


def triggering_kernel(alpha, reference_time, time_intervals, bandwidth):
    ''' triggering kernel for Hawkes porcess.
        @param:
            1. alpha: np.array, entres larger than 0
            2. reference_time: np.array, entries larger than 0.
            3. time_intervals: float/np.array, entry must be the same.
            4. bandwidth: np.array, entries larger than 0.
        @rtype: np.array
    '''
    # if len(alpha) != len(reference_time):
    # raise Exception("length of alpha and length of reference time must equal")
    time_intervals = time_intervals.reshape(-1, 1)
    # print((alpha * RBF_kernel(reference_time, time_intervals, bandwidth)).shape)
    if len(alpha.shape) == 3:
        return np.sum(np.sum(alpha * RBF_kernel(reference_time, time_intervals, bandwidth), axis=1), axis=1)
    else:
        return np.sum(np.sum(alpha * RBF_kernel(reference_time, time_intervals, bandwidth), axis=0), axis=0)


def g_theta(timeseq, reference_time, bandwidth, max_time):
    ''' g_theta for DHP
        @param:
            2. timeseq: 1-D np array time sequence before current time
            3. base_intensity: float
            4. reference_time: 1-D np.array
            5. bandwidth: 1-D np.array
        @rtype: np.array, shape(3,)
    '''
    timeseq = timeseq.reshape(-1, 1)
    results = 0.5 * (erfc(- reference_time / (2 * bandwidth ** 2) ** 0.5) - erfc(
        (max_time - timeseq - reference_time) / (2 * bandwidth ** 2) ** 0.5))
    return np.sum(results, axis=0)


def weighted_choice(weights, prng):
    """Samples from a discrete distribution.


    Parameters
    ----------
    weights : list
        A list of floats that identifies the distribution.

    prng : numpy.random.RandomState
        A pseudorandom number generator object.


    Returns
    -------
    int
    """
    rnd = prng.rand() * sum(weights)
    n = len(weights)
    i = -1
    while i < n - 1 and rnd >= 0:
        i += 1
        rnd -= weights[i]
    return i


def weight_choice_log(weights, prng):
    logsum = np.logaddexp.accumulate(weights)
    u = np.log(prng.rand()) + logsum[-1]
    for i, v in enumerate(logsum):
        if logsum[i] > u:
            break
    return i


def update_triggering_kernel(timeseq, alphas, reference_time, bandwidth, base_intensity, max_time, log_priors):
    ''' procedure of triggering kernel for SMC
        @param:
            1. timeseq: list, time sequence including current time
            2. alphas: 2-D np.array with shape (sample number, length of alpha)
            3. reference_time: np.array
            4. bandwidth: np.array
            5. log_priors: 1-D np.array with shape (sample number,), p(alpha, alpha_0)
            6. base_intensity: float
            7. max_time: float
        @rtype: 1-D numpy array with shape (length of alpha0,)
    '''
    # print(alphas.shape)
    logLikelihood = log_likelihood(timeseq, alphas, reference_time, bandwidth, base_intensity, max_time)
    log_update_weight = log_priors + logLikelihood
    log_update_weight = log_update_weight - np.max(log_update_weight)  # avoid overflow
    update_weight = np.exp(log_update_weight);
    update_weight = update_weight / np.sum(update_weight)
    update_weight = update_weight.reshape(-1, 1)
    alpha = np.sum(update_weight * alphas, axis=0)
    return alpha


def update_xi(likelihood, xis, log_priors):
    log_update_weight = log_priors + likelihood
    log_update_weight = log_update_weight - np.max(log_update_weight)
    update_weight = np.exp(log_update_weight);
    update_weight = update_weight / np.sum(update_weight)
    update_weight = update_weight.squeeze()

    xi = np.sum(update_weight * xis)
    return np.array([xi])



def log_likelihood(timeseq, alphas, reference_time, bandwidth, base_intensity, max_time):
    ''' compute log_likelihood for a time sequence for a cluster for SMC
        @param:
            1. timeseq: list, time sequence including current time
            2. alphas: 2-D np.array with shape (sample number, length of alpha)
            3. reference_time: np.array
            4. bandwidth: np.array
            5. log_priors: 1-D np.array, p(alpha, alpha_0)
            6. base_intensity: float
            7. max_time: float
        @rtype: 1-D numpy array with shape (sample number,)
    '''
    Lambda_0 = base_intensity * max_time

    if len(alphas.shape) == 2:
        alphas_times_gtheta = np.sum(alphas * g_theta(timeseq, reference_time, bandwidth, max_time),
                                 axis=1)  # shape = (sample number,)
    else:
        alphas_times_gtheta = np.sum(alphas * g_theta(timeseq, reference_time, bandwidth, max_time))
    if len(timeseq) == 1:
        raise Exception('The length of time sequence must be larger than 1.')
    time_intervals = timeseq[-1] - timeseq[:-1]
    alphas = alphas.reshape(-1, 1, alphas.shape[-1])

    triggers = np.log(triggering_kernel(alphas, reference_time, time_intervals, bandwidth))
    return -Lambda_0 - alphas_times_gtheta + triggers



def log_dirichlet_multinomial_distribution_dict(cls_word_distribution, doc_word_distribution, cls_word_count, doc_word_count, priors):
    priors_sum = np.sum(priors)
    log_prob = 0

    log_prob += _gammaln(cls_word_count + priors_sum)
    log_prob -= _gammaln(cls_word_count + doc_word_count + priors_sum)

    rest = [_gammaln(cls_word_distribution[word] + doc_word_distribution[word] + priors[0]) - _gammaln(cls_word_distribution[word] + priors[0])
            if word in cls_word_distribution else _gammaln(doc_word_distribution[word] + priors[0]) - _gammaln(priors[0])
            for word in doc_word_distribution]
    return log_prob + sum(rest)
'''

def log_dirichlet_multinomial_distribution_dict(cls_word_distribution, doc_word_distribution, cls_word_count, doc_word_count, priors):
    priors_sum = np.sum(priors)
    log_prob = 0

    log_prob += _gammaln(cls_word_count + priors_sum)
    log_prob -= _gammaln(cls_word_count + doc_word_count + priors_sum)
    unique_words = len(doc_word_distribution) == doc_word_count

    is_old_topic = False
    if cls_word_distribution:
        is_old_topic = True
    if unique_words:
        rest = [_ln(cls_word_distribution[word] + priors[0])
                    if is_old_topic and word in cls_word_distribution
                    else _ln(priors[0])
                    for word in doc_word_distribution]
    else:
        rest = [_gammaln(cls_word_distribution[word] + doc_word_distribution[word] + priors[0]) - _gammaln(cls_word_distribution[word] + priors[0])
                if is_old_topic and word in cls_word_distribution
                else _gammaln(doc_word_distribution[word] + priors[0]) - _gammaln(priors[0])
                for word in doc_word_distribution]
    return log_prob + sum(rest)
'''


def log_dirichlet_multinomial_distribution(cls_word_distribution, doc_word_distribution, cls_word_count, doc_word_count,
                                           vocabulary_size, priors, new_cls = False):
    ''' compute the log dirichlet multinomial distribution
        @param:
            1. cls_word_distribution: 1-D numpy array, including document word_distribution
            2. doc_word_distribution: 1-D numpy array
            3. cls_word_count: int, including document word_distribution
            4. doc_word_count: int
            5. vocabulary_size: int
            6. priors: 1-d np.array
        @rtype: float
    '''
    priors_sum = np.sum(priors)
    log_prob = 0
    # print(cls_word_count - doc_word_count + priors_sum)
    log_prob += gammaln(cls_word_count - doc_word_count + priors_sum)
    log_prob -= gammaln(cls_word_count + priors_sum)

    # gammaln_this_time = np.sum(gammaln(cls_word_distribution + priors))
    # one_hot = doc_word_distribution.copy()
    # one_hot[np.nonzero(one_hot)] = 1

    log_prob += np.sum(gammaln(cls_word_distribution + priors))
    log_prob -= np.sum(gammaln(cls_word_distribution - doc_word_distribution + priors))


    # log_prob += gammaln_this_time
    # log_prob -= np.sum(gammaln(cls_word_distribution - doc_word_distribution + priors))
    # log_prob -= gammaln_last_time
    # log_prob += np.sum([_gammaln(cls_word_distribution[i] + priors[i]) for i in range(len(cls_word_distribution))])
    # log_prob -= np.sum([_gammaln(cls_word_distribution[i] - doc_word_distribution[i] + priors[i]) for i in range(len(cls_word_distribution))])
    return log_prob


def test_dirichlet():
    alpha = dirichlet(np.array([1] * 10))
    sample_alpha_list = [dirichlet([1] * 10) for _ in range(3000)]
    print('len(sample_alpha_list)', len(sample_alpha_list))
    print(np.sum(sample_alpha_list[0]))


def test_multinomial():
    probabilities = dirichlet(np.array([1] * 10))
    result = multinomial(5, probabilities)
    print(result)


def test_EfficientImplementation():
    tu = EfficientImplementation(100, [3, 7, 11], [2, 5, 10])
    print(tu)


def test_log_Dirichlet_CDF():
    prior = np.array([1] * 10)
    outcomes = dirichlet(prior)
    print(outcomes)
    print(log_Dirichlet_CDF(outcomes, prior))


def test_RBF_kernel():
    refernce_time = np.array([3, 7, 11])
    bandwidth = np.array([5, 5, 5])
    time_intervals = 3
    print(RBF_kernel(refernce_time, time_intervals, bandwidth))
    print(RBF_kernel(11, 3, 5))


def test_triggering_kernel():
    reference_time = np.array([3, 7, 11])
    bandwidth = np.array([5, 5, 5])
    time_intervals = np.array([1, 3])
    time_intervals = time_intervals.reshape(-1, 1)
    print(time_intervals.shape)
    alpha = dirichlet([1] * 3)
    print(alpha)
    print(RBF_kernel(reference_time, time_intervals, bandwidth))
    print(triggering_kernel(alpha, reference_time, time_intervals, bandwidth))
    time_intervals = np.array([1, 3, 50])
    print(triggering_kernel(alpha, reference_time, time_intervals, bandwidth))


def test_g_theta():
    timeseq = np.arange(0.2, 1000000, 0.6)
    bandwidth = np.array([5, 5, 5])
    reference_time = np.array([3, 7, 11])
    current_time = timeseq[-1]
    T = current_time + 1
    output = g_theta(timeseq, reference_time, bandwidth, T)


def test_log_likelihood():
    timeseq = np.arange(0.2, 1000, 0.6)
    alpha0 = np.array([1, 1, 1])
    bandwidth = np.array([5, 5, 5])
    reference_time = np.array([3, 7, 11])
    sample_num = 1000
    current_time = timeseq[-1]
    T = current_time + 1
    base_intensity = 1

    alphas = []
    log_priors = []
    for _ in range(sample_num):
        alpha = dirichlet(alpha0)
        log_prior = log_Dirichlet_CDF(alpha, alpha0)
        alphas.append(alpha)
        log_priors.append(log_prior)

    alphas = np.array(alphas)
    log_priors = np.array(log_priors)

    logLikelihood = log_likelihood(timeseq, alphas, reference_time, bandwidth, base_intensity, T)
    print(logLikelihood)


def test_update_triggering_kernel():
    # generate parameters
    timeseq = np.arange(0.2, 1000, 0.1)
    alpha0 = np.array([1, 1, 1])
    bandwidth = np.array([5, 5, 5])
    reference_time = np.array([3, 7, 11])
    sample_num = 3000
    base_intensity = 1
    current_time = timeseq[-1]
    T = current_time + 1

    alphas = []
    log_priors = []
    for _ in range(sample_num):
        alpha = dirichlet(alpha0)
        log_prior = log_Dirichlet_CDF(alpha, alpha0)
        alphas.append(alpha)
        log_priors.append(log_prior)

    alphas = np.array(alphas)
    log_priors = np.array(log_priors)

    alpha = update_triggering_kernel(timeseq, alphas, reference_time, bandwidth, base_intensity, T, log_priors)
    print(alpha)


def test_log_dirichlet_multinomial_distribution():
    with open('./data/meme/meme_docs.pkl', 'rb') as r:
        documents = pickle.load(r)

    cls_word_distribution = documents[0].word_distribution + documents[1].word_distribution
    doc_word_distribution = documents[1].word_distribution
    cls_word_count = documents[0].word_count + documents[1].word_count
    doc_word_count = documents[1].word_count
    vocabulary_size = len(documents[0].word_distribution)
    priors = np.array([1] * vocabulary_size)
    print('cls_word_count', cls_word_count)
    print('doc_word_count', doc_word_count)
    logprob = log_dirichlet_multinomial_distribution(cls_word_distribution, doc_word_distribution, cls_word_count,
                                                     doc_word_count, vocabulary_size, priors)
    print(logprob)

def weighted_choice(weights, prng):
    """Samples from a discrete distribution.


    Parameters
    ----------
    weights : list
        A list of floats that identifies the distribution.

    prng : numpy.random.RandomState
        A pseudorandom number generator object.


    Returns
    -------
    int
    """
    rnd = prng.rand() * sum(weights)
    n = len(weights)
    i = -1
    while i < n - 1 and rnd >= 0:
        i += 1
        rnd -= weights[i]
    return i

def main():
    # test_update_triggering_kernel()
    # test_log_likelihood()

    '''
    tu = 75
    active_clusters = {1:[50, 60, 76, 100], 2:[10,20,30,40], 3:[100,2000]}
    print(active_clusters)

    for cluster_index in active_clusters.keys():
        timeseq = active_clusters[cluster_index]
        active_timeseq = [t for t in timeseq if t > tu]
        if not active_timeseq:
            del active_clusters[cluster_index]
        else:
            active_clusters[cluster_index] = active_timeseq

    print(active_clusters)
    '''

    test_log_dirichlet_multinomial_distribution()


if __name__ == '__main__':
    '''
    a = np.array([0, 0, 1, 2, 3, 4])
    b = np.array([1, 2, 0, 0, 1, 2])

    a_dict = {2:1, 3:2, 4:3, 5:4}
    b_dict = {0:1, 1:2, 4:1, 5:2}
    print(log_dirichlet_multinomial_distribution(a+b, b, sum(a+b), sum(b), 0, [0.01] * 6))
    print(log_dirichlet_multinomial_distribution_dict(a_dict, b_dict, sum(list(a_dict.values())), sum(list(b_dict.values())), [0.01] * 6))
    '''
    pass
