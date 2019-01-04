from __future__ import print_function
from __future__ import division
from utils_hdhp import *
import pickle
import IPython
import numpy as np
from scipy.misc import logsumexp
from numpy import log as ln
import json
from numpy.random import RandomState
from sklearn.utils import check_random_state
import gc


def memoize(f):
    class memodict(dict):
        __slots__ = ()

        def __missing__(self, key):
            self[key] = ret = f(key)
            return ret

    return memodict().__getitem__


def _pickle_method(m):
    if m.im_self is None:
        return getattr, (m.im_class, m.im_func.func_name)
    else:
        return getattr, (m.im_self, m.im_func.func_name)


@memoize
def _ln(x):
    return ln(x)


class Hierarchical_Dirichlet_Hawkes_Process(object):
    """docstring for Dirichlet Hawkes Prcess"""

    def __init__(self, particle_num, base_intensity, theta0, alpha0, reference_time, vocabulary_size, bandwidth,
                 sample_num, beta0):
        super(Hierarchical_Dirichlet_Hawkes_Process, self).__init__()
        self.particle_num = particle_num
        self.base_intensity = base_intensity
        self.theta0 = theta0
        self.alpha0 = alpha0
        self.reference_time = reference_time
        self.vocabulary_size = vocabulary_size
        self.bandwidth = bandwidth
        self.sample_num = sample_num
        self.beta = beta0
        self.prng = RandomState(1024)
        self.particles = []
        for i in range(particle_num):
            self.particles.append(Particle(weight=1.0 / self.particle_num))
        alphas = []
        log_priors = []
        for _ in range(sample_num):
            alpha = dirichlet(alpha0)
            log_prior = log_Dirichlet_CDF(alpha, alpha0)
            alphas.append(alpha)
            log_priors.append(log_prior)
        self.alphas = np.array(alphas)
        self.log_priors = np.array(log_priors)
        self.active_interval = None  # [tu, tn]

    def sequential_monte_carlo(self, doc, threshold, pred=False):
        # print('\n\nhandling document %d' % doc.index)
        if isinstance(doc, Document):  # deal with the case of exact timing
            # get active interval (globally)
            tu = EfficientImplementation(doc.timestamp, self.reference_time, self.bandwidth)
            self.active_interval = [tu, doc.timestamp]
            # print('active_interval', self.active_interval)

            particles = []
            for particle in self.particles:
                particles.append(self.particle_sampler(particle, doc))

            self.particles = particles
            self.particles = self.particles_normal_resampling(self.particles, threshold)
            if (doc.index + 1) % 100 == 0:
                gc.collect()
        else:  # deal with the case of exact timing
            print('deal with the case of exact timing')

    def particle_sampler(self, particle, doc):
        particle, selected_cluster_index, selected_topic_index = self.sampling_cluster_label(particle, doc)
        particle.docs2topic_ID.append(selected_topic_index)
        particle.docs2cluster_ID.append(selected_cluster_index)

        particle.alpha[selected_cluster_index] = self.parameter_estimation(particle,
                                                                                    selected_cluster_index)
        # particle.topics[selected_topic_index].alpha = particle.clusters[selected_cluster_index].alpha

        T = self.active_interval[1] + 1
        likelihood = []
        for index in particle.active_clusters:
            timeseq = particle.active_clusters[index]
            # topic_index = particle.cls_2_topic[index]
            if len(timeseq) == 1:
                continue
            alpha = particle.alpha[index]
            likelihood.append(log_likelihood(np.array(timeseq), np.array(alpha),
                                         self.reference_time, self.bandwidth, self.base_intensity, T)[0])
        if likelihood:
            particle.log_update_prob = particle._Q0 + logsumexp(likelihood)
        else:
            particle.log_update_prob = particle._Q0
        return particle

    def sampling_cluster_label(self, particle, doc):
        if particle.cluster_num_by_now == 0:
            # sample cluster label
            particle.cluster_num_by_now += 1
            selected_cluster_index = particle.cluster_num_by_now
            particle.active_clusters = self.update_active_clusters(particle)

            # create new topic label
            particle.topic_num_by_now += 1
            selected_topic_index = particle.topic_num_by_now
            particle.add_doc(selected_topic_index, doc)
            particle.topic_count[selected_topic_index] = 1
            particle.cls_2_topic[selected_cluster_index] = selected_topic_index
        else:
            active_cluster_indexes = [0]
            active_cluster_rates = [self.base_intensity]
            cls0_log_dirichlet_multinomial_distribution = log_dirichlet_multinomial_distribution_dict(dict(),
                                                                                                 doc.word_distribution,
                                                                                                 0,
                                                                                                 doc.word_count,
                                                                                                 self.theta0)

            active_cluster_textual_probs = [cls0_log_dirichlet_multinomial_distribution]
            # first update the active cluster
            particle.active_clusters = self.update_active_clusters(particle)

            total_topic_init = self.base_intensity
            topic_likelihood = defaultdict(float)

            for active_cluster_index, timeseq in particle.active_clusters.items():
                topic_index = particle.cls_2_topic[active_cluster_index]
                active_cluster_indexes.append(active_cluster_index)
                time_intervals = doc.timestamp - np.array(timeseq)
                alpha = particle.alpha[active_cluster_index]
                rate = triggering_kernel(alpha, self.reference_time, time_intervals, self.bandwidth)
                total_topic_init += rate
                active_cluster_rates.append(rate)

                topic_word_count = particle.topic_word_count[topic_index]
                topic_word_distribution = particle.topic_word_distribution[topic_index]
                if topic_index in topic_likelihood:
                    topic_log_dirichlet_multinomial_distribution = topic_likelihood[topic_index]
                else:
                    topic_log_dirichlet_multinomial_distribution = log_dirichlet_multinomial_distribution_dict(
                                                                    topic_word_distribution, doc.word_distribution, topic_word_count,
                                                                    doc.word_count, self.theta0)

                topic_likelihood[topic_index] = topic_log_dirichlet_multinomial_distribution

                active_cluster_textual_probs.append(topic_log_dirichlet_multinomial_distribution)

            active_cluster_logrates = np.log(np.array(active_cluster_rates) / total_topic_init)
            cluster_selection_probs = active_cluster_logrates + np.array(active_cluster_textual_probs)
            log_intensities = list(cluster_selection_probs)

            cluster_index_index = weight_choice_log(cluster_selection_probs, self.prng)
            selected_cluster_index = active_cluster_indexes[cluster_index_index]
            if selected_cluster_index == 0:
                particle.cluster_num_by_now += 1
                selected_cluster_index = particle.cluster_num_by_now
                particle.active_clusters[selected_cluster_index] = [
                    self.active_interval[1]]  # create a new list containing the current time

                active_topic_index = [0]
                new_topic_intensity = [
                    self.base_intensity * self.beta / (total_topic_init * (particle.cluster_num_by_now - 1 + self.beta))]

                new_topic_log_likelihood = [cls0_log_dirichlet_multinomial_distribution]

                for topic_index in range(1, particle.topic_num_by_now + 1):
                    active_topic_index.append(topic_index)
                    new_topic_intensity.append(self.base_intensity * particle.topic_count[topic_index] / (
                                total_topic_init * (particle.cluster_num_by_now - 1 + self.beta)))

                    topic_word_count = particle.topic_word_count[topic_index]
                    topic_word_distribution = particle.topic_word_distribution[topic_index]
                    if topic_index in topic_likelihood:
                        new_topic_log_likelihood.append(topic_likelihood[topic_index])
                    else:
                        new_topic_log_likelihood.append(log_dirichlet_multinomial_distribution_dict(topic_word_distribution,
                                                                        doc.word_distribution,
                                                                        topic_word_count,
                                                                        doc.word_count,
                                                                        self.theta0))
                new_topic_intensity_1 = np.log(new_topic_intensity / np.sum(new_topic_intensity))
                intensities = new_topic_intensity_1 + new_topic_log_likelihood
                new_log_intensities = new_topic_log_likelihood + np.log(new_topic_intensity)
                log_intensities.extend(new_log_intensities)

                selected_topic_index = weight_choice_log(intensities, self.prng)
                selected_topic_index = active_topic_index[selected_topic_index]
                if selected_topic_index == 0:
                    # TODO: create a new topic
                    particle.topic_num_by_now += 1
                    selected_topic_index = particle.topic_num_by_now
                    particle.add_doc(selected_topic_index, doc)
                    particle.cls_2_topic[selected_cluster_index] = selected_topic_index
                    particle.topic_count[selected_topic_index] = 1
                else:
                    particle.add_doc(selected_topic_index, doc)
                    particle.cls_2_topic[selected_cluster_index] = selected_topic_index
                    particle.topic_count[selected_topic_index] += 1
            else:  # the case of the previous used cluster, update active cluster and add document to cluster
                selected_topic_index = particle.cls_2_topic[selected_cluster_index]
                particle.add_doc(selected_topic_index, doc)
                particle.active_clusters[selected_cluster_index].append(self.active_interval[1])
            particle._Q0 = logsumexp(log_intensities)
        # print('selected topic', selected_topic_index)
        return particle, selected_cluster_index, selected_topic_index

    def parameter_estimation(self, particle, selected_cluster_index):
        timeseq = np.array(particle.active_clusters[selected_cluster_index])
        if len(timeseq) == 1:  # the case of first document in a brand new cluster
            np.random.seed()
            alpha = dirichlet(self.alpha0)
            return alpha
        T = self.active_interval[1] + 1
        alpha = update_triggering_kernel(timeseq, self.alphas, self.reference_time, self.bandwidth, self.base_intensity,
                                         T, self.log_priors)
        return alpha

    def update_active_clusters(self, particle):
        if not particle.active_clusters:  # the case of the first document comes
            particle.active_clusters[1] = [self.active_interval[1]]
        else:  # update the active clusters
            tu = self.active_interval[0]
            wait_del = []
            for cluster_index in particle.active_clusters.keys():
                timeseq = particle.active_clusters[cluster_index]
                active_timeseq = [t for t in timeseq if t > tu]
                if not active_timeseq:
                    # del particle.active_clusters[cluster_index]
                    wait_del.append(cluster_index)
                else:
                    particle.active_clusters[cluster_index] = active_timeseq
            for index in wait_del:
                del particle.active_clusters[index]
        return particle.active_clusters

    def calculate_particle_log_update_prob(self, particle, selected_cluster_index, doc):
        cls_word_distribution = particle.clusters[selected_cluster_index].word_distribution
        cls_word_count = particle.clusters[selected_cluster_index].word_count
        doc_word_distribution = doc.word_distribution
        doc_word_count = doc.word_count
        assert doc_word_count == np.sum(doc.word_distribution)
        assert cls_word_count == np.sum(particle.clusters[selected_cluster_index].word_distribution)
        log_update_prob = log_dirichlet_multinomial_distribution(cls_word_distribution, doc_word_distribution, \
                                                                 cls_word_count, doc_word_count, self.vocabulary_size,
                                                                 self.theta0)
        return log_update_prob

    def particles_normal_resampling(self, particles, threshold):
        # print('\nparticles_normal_resampling')
        weights = []
        log_update_probs = []
        for particle in particles:
            weights.append(particle.weight)
            log_update_probs.append(particle.log_update_prob)
        weights = np.array(weights)
        log_update_probs = np.array(log_update_probs)
        log_update_probs = log_update_probs - np.max(log_update_probs)  # prevent overflow
        update_probs = np.exp(log_update_probs)  # print('update_probs',update_probs)
        weights = weights * update_probs  # update
        weights = weights / np.sum(weights)  # normalization
        resample_num = len(np.where(weights + 1e-5 < threshold)[0])
        if resample_num == 0:
            for i, particle in enumerate(particles):
                particle.weight = weights[i]
            return particles
        else:
            remaining_particles = [particle for i, particle in enumerate(particles) if weights[i] + 1e-5 > threshold]
            resample_particles = [particle for i, particle in enumerate(particles) if weights[i] + 1e-5 < threshold ]
            resample_probs = weights[np.where(weights + 1e-5 > threshold)]
            resample_probs = resample_probs / np.sum(resample_probs)
            remaining_particle_weights = weights[np.where(weights + 1e-5 > threshold)]
            for i, _ in enumerate(remaining_particles):
                remaining_particles[i].weight = remaining_particle_weights[i]
            np.random.seed()
            resample_distribution = multinomial(exp_num=resample_num,
                                                probabilities=resample_probs)  # print('len(remaining_particles)', len(remaining_particles)) #print('resample_probs', resample_probs)
            if not resample_distribution.shape:  # the case of only one particle left
                for particle in resample_particles:
                    particle.copy(remaining_particles[0])
                    remaining_particles.append(particle)
            else:  # the case of more than one particle left
                resample_index = 0
                for i, resample_times in enumerate(resample_distribution):
                    for _ in range(resample_times):
                        resample_particles[resample_index].copy(remaining_particles[i])
                        remaining_particles.append(resample_particles[resample_index])
                        resample_index += 1
            # normalize the particle weight again
            update_weights = np.array([particle.weight for particle in remaining_particles])
            update_weights = update_weights / np.sum(update_weights)
            for i, particle in enumerate(remaining_particles):
                particle.weight = update_weights[i]
            assert np.abs(np.sum(update_weights) - 1) < 1e-5
            assert len(remaining_particles) == self.particle_num
            self.particles = None
            return remaining_particles

def main():
    with open('../data/all_the_news_2017.json') as f:
        news_items = json.load(f)
    print('finish extracting news from json...')

    # parameter initialization
    # vocabulary_size = 40000
    # idx, ridx = get_df_words(news_items, vocabulary_size)
    vocabulary_size = 10000
    idx, ridx = get_df_words(news_items, vocabulary_size)

    # vocabulary_size = 57620
    particle_num = 8
    base_intensity = 0.1
    theta0 = np.array([0.1] * vocabulary_size)
    alpha0 = np.array([0.1] * 4)
    reference_time = np.array([3, 7, 11, 24])
    bandwidth = np.array([2, 3, 5, 5])
    sample_num = 2000
    threshold = 1.0 / particle_num
    beta0 = 2

    HDHP = Hierarchical_Dirichlet_Hawkes_Process(particle_num=particle_num, base_intensity=base_intensity, theta0=theta0,
                                   alpha0=alpha0, reference_time=reference_time, vocabulary_size=vocabulary_size, bandwidth=bandwidth,
                                   sample_num=sample_num, beta0=beta0)

    # begin sampling
    # for simple experiment
    news_items = news_items[:1000]
    for news_item in news_items:
        doc = parse_newsitem_2_doc(news_item=news_item, words_idx=idx, time_unit=3600)
        HDHP.sequential_monte_carlo(doc, threshold)
    IPython.embed()
    # with open('./result/particles_test_hdhp_full.pkl', 'wb') as w:
    #     pickle.dump(DHP.particles, w)


if __name__ == '__main__':
    main()