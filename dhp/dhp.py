from __future__ import print_function
from __future__ import division
import pickle
import numpy as np
from utils_dhp import *
import concurrent.futures
from functools import partial
import types
from copy import deepcopy
from numpy.random import RandomState
import json
import gc


class Dirichlet_Hawkes_Process(object):
    """docstring for Dirichlet Hawkes Prcess"""

    def __init__(self, particle_num, base_intensity, theta0, alpha0, reference_time, vocabulary_size, bandwidth,
                 sample_num):
        super(Dirichlet_Hawkes_Process, self).__init__()
        self.particle_num = particle_num
        self.base_intensity = base_intensity
        self.theta0 = theta0
        self.alpha0 = alpha0
        self.reference_time = reference_time
        self.vocabulary_size = vocabulary_size
        self.bandwidth = bandwidth
        self.sample_num = sample_num
        self.particles = []
        self.prng = RandomState(1024)
        for i in range(particle_num):
            self.particles.append(Particle(weight=1.0 / self.particle_num))
        alphas = [];
        log_priors = []
        for _ in range(sample_num):
            alpha = dirichlet(alpha0);
            log_prior = log_Dirichlet_CDF(alpha, alpha0)
            alphas.append(alpha);
            log_priors.append(log_prior)
        self.alphas = np.array(alphas)
        self.log_priors = np.array(log_priors)
        self.active_interval = None  # [tu, tn]

    def sequential_monte_carlo(self, doc, threshold):
        print('\n\nhandling document %d' % doc.index)
        if isinstance(doc, Document):  # deal with the case of exact timing
            # get active interval (globally)
            tu = EfficientImplementation(doc.timestamp, self.reference_time, self.bandwidth)
            self.active_interval = [tu, doc.timestamp]
            print('active_interval', self.active_interval)

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
        # sampling cluster label
        particle, selected_cluster_index = self.sampling_cluster_label(particle, doc)  # print(selected_cluster_index)
        # update the triggering kernel

        print('Selected cluster index: ', selected_cluster_index)
        particle.alpha[selected_cluster_index] = self.parameter_estimation(particle, selected_cluster_index)  # ;print('selected_cluster_index',selected_cluster_index,'alpha', particle.clusters[selected_cluster_index].alpha)
        # calculate the weight update probability
        particle.log_update_prob = self.calculate_particle_log_update_prob(particle, selected_cluster_index,
                                                                           doc)  # ; print('particle.log_update_prob',particle.log_update_prob)
        return particle

    def sampling_cluster_label(self, particle, doc):
        if particle.cluster_num_by_now == 0:  # the case of the first document comes
            # sample cluster label
            particle.cluster_num_by_now += 1
            selected_cluster_index = particle.cluster_num_by_now
            particle.add_doc(selected_cluster_index, doc)
            particle.docs2cluster_ID.append(selected_cluster_index)
            particle.active_clusters = self.update_active_clusters(particle)

        else:  # the case of the following document to come
            active_cluster_indexes = [0]  # zero for new cluster
            active_cluster_rates = [self.base_intensity]
            cls0_log_dirichlet_multinomial_distribution = log_dirichlet_multinomial_distribution_dict(dict(), doc.word_distribution,
                                                                                                 0, doc.word_count, self.theta0)
            active_cluster_textual_probs = [cls0_log_dirichlet_multinomial_distribution]
            # first update the active cluster
            particle.active_clusters = self.update_active_clusters(particle)
            # then calculate rates for each cluster in active interval
            for active_cluster_index, timeseq in particle.active_clusters.items():
                active_cluster_indexes.append(active_cluster_index)
                time_intervals = doc.timestamp - np.array(timeseq)
                alpha = particle.alpha[active_cluster_index]
                rate = triggering_kernel(alpha, self.reference_time, time_intervals, self.bandwidth)
                active_cluster_rates.append(rate)
                cls_word_distribution = particle.topic_word_distribution[active_cluster_index]
                cls_word_count = particle.topic_word_count[active_cluster_index]
                cls_log_dirichlet_multinomial_distribution = log_dirichlet_multinomial_distribution_dict(
                    cls_word_distribution, doc.word_distribution, cls_word_count, doc.word_count, self.theta0)
                active_cluster_textual_probs.append(cls_log_dirichlet_multinomial_distribution)

            active_cluster_logrates = np.log(active_cluster_rates)
            cluster_selection_probs = active_cluster_logrates + active_cluster_textual_probs  # in log scale
            selected_cluster_array = weight_choice_log(cluster_selection_probs, self.prng)
            # print(selected_cluster_array)

            selected_cluster_index = active_cluster_indexes[selected_cluster_array]
            # print(active_cluster_indexes)
            # print(selected_cluster_index)
            if selected_cluster_index == 0:  # the case of new cluster
                particle.cluster_num_by_now += 1
                selected_cluster_index = particle.cluster_num_by_now
                particle.add_doc(selected_cluster_index, doc)
                particle.docs2cluster_ID.append(selected_cluster_index)
                particle.active_clusters[selected_cluster_index] = [self.active_interval[1]]
            else:  # the case of the previous used cluster, update active cluster and add document to cluster
                particle.add_doc(selected_cluster_index, doc)
                particle.docs2cluster_ID.append(selected_cluster_index)
                particle.active_clusters[selected_cluster_index].append(self.active_interval[1])
        return particle, selected_cluster_index

    def parameter_estimation(self, particle, selected_cluster_index):
        # print('updating triggering kernel ...')
        # print(particle.active_clusters[selected_cluster_index])
        timeseq = np.array(particle.active_clusters[selected_cluster_index])
        if len(timeseq) == 1:  # the case of first document in a brand new cluster
            np.random.seed()
            alpha = dirichlet(self.alpha0)
            return alpha
        T = self.active_interval[1] + 1  # ;print('updating triggering kernel ..., len(timeseq)', len(timeseq))
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
                    wait_del.append(cluster_index)
                    # del particle.active_clusters[cluster_index]
                else:
                    particle.active_clusters[cluster_index] = active_timeseq
            for index in wait_del:
                del particle.active_clusters[index]
        return particle.active_clusters

    def calculate_particle_log_update_prob(self, particle, selected_cluster_index, doc):
        # print('calculate_particle_log_update_prob') #print('id(particle.clusters[selected_cluster_index])', id(particle.clusters[selected_cluster_index]));#print('id(particle.clusters[selected_cluster_index]).word_distribution', id(particle.clusters[selected_cluster_index].word_distribution))
        cls_word_distribution = particle.topic_word_distribution[selected_cluster_index]
        cls_word_count = particle.topic_word_count[selected_cluster_index]
        doc_word_distribution = doc.word_distribution
        doc_word_count = doc.word_count
        log_update_prob = log_dirichlet_multinomial_distribution_dict(cls_word_distribution, doc_word_distribution, \
                                                                 cls_word_count, doc_word_count, self.theta0)  # print('particle.log_update_probs',particle.log_update_probs)
        return log_update_prob

    def particles_normal_resampling(self, particles, threshold):
        print('\nparticles_normal_resampling')
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
        if resample_num == 0:  # no need to resample particle, but still need to assign the updated weights to paricle weight
            for i, particle in enumerate(particles):
                particle.weight = weights[i]
            return particles
        else:
            remaining_particles = [particle for i, particle in enumerate(particles) if weights[i]  > threshold + 1e-5]
            resample_particles = [particle for i, particle in enumerate(particles) if weights[i] <= threshold + 1e-5]
            resample_probs = weights[np.where(weights > threshold + 1e-5)]
            resample_probs = resample_probs / np.sum(resample_probs)
            remaining_particle_weights = weights[np.where(weights > threshold + 1e-5)]
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

    vocabulary_size = 10000
    # parameter initialization
    idx, ridx = get_df_words(news_items, vocabulary_size)

    # vocabulary_size =
    particle_num = 8
    base_intensity = 0.1
    theta0 = np.array([0.1] * vocabulary_size)
    alpha0 = np.array([0.1] * 4)
    reference_time = np.array([3, 7, 11, 24])
    bandwidth = np.array([5, 5, 5, 5])
    sample_num = 2000
    threshold = 1.0 / particle_num

    DHP = Dirichlet_Hawkes_Process(particle_num=particle_num, base_intensity=base_intensity, theta0=theta0,
                                   alpha0=alpha0, \
                                   reference_time=reference_time, vocabulary_size=vocabulary_size, bandwidth=bandwidth,
                                   sample_num=sample_num)

    # begin sampling
    # for simple experiment
    news_items = news_items[:1000]
    for news_item in news_items:
        doc = parse_newsitem_2_doc(news_item=news_item, words_idx=idx, time_unit=3600)
        DHP.sequential_monte_carlo(doc, threshold)

    with open('../result/particles.pkl', 'wb') as w:
        pickle.dump(DHP.particles, w)


if __name__ == '__main__':
    main()