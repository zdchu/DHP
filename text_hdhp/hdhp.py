from __future__ import print_function
from __future__ import division
import pickle
import numpy as np
from utils_hdhp import *
import concurrent.futures
from functools import partial
import copy_reg
import time
from scipy.misc import logsumexp
from numpy import log as ln, exp
import types
from copy import deepcopy
import json
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

copy_reg.pickle(types.MethodType, _pickle_method)


class Dirichlet_Hawkes_Process(object):
    """docstring for Dirichlet Hawkes Prcess"""

    def __init__(self, particle_num, base_intensity, theta0, alpha0, reference_time, vocabulary_size, bandwidth,
                 sample_num, beta0):
        super(Dirichlet_Hawkes_Process, self).__init__()
        self.particle_num = particle_num
        self.base_intensity = base_intensity
        self.theta0 = theta0
        self.alpha0 = alpha0
        self.reference_time = reference_time
        self.vocabulary_size = vocabulary_size
        self.bandwidth = bandwidth
        self.sample_num = sample_num
        self.beta = beta0
        # initilize particles
        self.particles = []
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
        self.gammaln_prior = np.sum(gammaln(theta0))

    def sequential_monte_carlo(self, doc, threshold):
        print('\n\nhandling document %d' % doc.index)
        if isinstance(doc, Document):  # deal with the case of exact timing
            # get active interval (globally)
            tu = EfficientImplementation(doc.timestamp, self.reference_time, self.bandwidth)
            self.active_interval = [tu, doc.timestamp]
            print('active_interval', self.active_interval)

            # sequential
            particles = []
            for particle in self.particles:
                particles.append(self.particle_sampler(particle, doc))

            self.particles = particles

            '''
            partial_particle_sampler = partial(self.particle_sampler, doc = doc)
            with concurrent.futures.ProcessPoolExecutor(max_workers = self.particle_num) as executor:
                self.particles = list(executor.map(partial_particle_sampler, self.particles))
            '''

            '''
            particle_index = 0
            partial_particle_sampler = partial(self.particle_sampler, doc = doc)
            executor = concurrent.futures.ProcessPoolExecutor(max_workers = self.particle_num)
            wait_for = [executor.submit(partial_particle_sampler, particle) for particle in self.particles]
            concurrent.futures.wait(wait_for)
            particles = []
            for f in concurrent.futures.as_completed(wait_for):
                particle = f.result()
                particles.append(particle)
            self.particles = particles
            '''

            # particle_generator = executor.map(partial_particle_sampler, self.particles)
            # begin particles normalization and resampling
            # for i, particle in enumerate(particle_generator):
            # self.particles[i] = particle
            self.particles = self.particles_normal_resampling(self.particles, threshold)
            if (doc.index + 1) % 100 == 0:
                gc.collect()
        else:  # deal with the case of exact timing
            print('deal with the case of exact timing')

    def particle_sampler(self, particle, doc):

        particle, selected_cluster_index, selected_topic_index = self.sampling_cluster_label(particle, doc)
        particle.docs2topic_ID.append(selected_topic_index)

        particle.clusters[selected_cluster_index].alpha = self.parameter_estimation(particle,

                                                                                    selected_cluster_index)
        particle.topics[selected_topic_index].alpha = particle.clusters[selected_cluster_index].alpha
        T = self.active_interval[1] + 1

        timeseq = particle.active_clusters[selected_cluster_index]
        if len(timeseq) == 1:
            likelihood = 0
        else:
            likelihood = log_likelihood(np.array(timeseq), np.array(particle.topics[selected_topic_index].alpha),
                                    self.reference_time, self.bandwidth, self.base_intensity, T)[0]

        particle.log_update_prob = particle._Q0 + likelihood
        # print('log_update_prob: ', particle.log_update_prob)
        '''
        particle.log_update_prob = self.calculate_particle_log_update_prob(particle, selected_cluster_index,
                                                                           doc)  # ; print('particle.log_update_prob',particle.log_update_prob)
        '''
        return particle

    def sampling_cluster_label(self, particle, doc):
        if len(particle.clusters) == 0:  # the case of the first document comes
            # sample cluster label
            particle.cluster_num_by_now += 1
            selected_cluster_index = particle.cluster_num_by_now
            selected_cluster = Cluster(index=selected_cluster_index)
            selected_cluster.add_document(doc)
            particle.clusters[selected_cluster_index] = selected_cluster
            particle.docs2cluster_ID.append(selected_cluster_index)
            # update active cluster
            particle.active_clusters = self.update_active_clusters(particle)

            particle.topic_num_by_now += 1
            selected_topic_index = particle.topic_num_by_now
            selected_topic = Topic(index=selected_topic_index)
            selected_topic.add_document(doc)
            particle.topics[selected_topic_index] = selected_topic
            # particle.topics[selected_topic_index].gammaln_last_time = np.sum(gammaln(selected_topic.word_distribution))
            particle.cluster2topic_ID[selected_cluster_index] = selected_topic_index
        else:  # the case of the following document to come
            active_cluster_indexes = [0]  # zero for new cluster
            active_cluster_rates = [self.base_intensity]
            cls0_log_dirichlet_multinomial_distribution = log_dirichlet_multinomial_distribution(doc.word_distribution,
                                                                                                 doc.word_distribution, \
                                                                                                 doc.word_count,
                                                                                                 doc.word_count,
                                                                                                 self.vocabulary_size,
                                                                                                 # self.theta0, self.gammaln_prior)
                                                                                                self.theta0)

            active_cluster_textual_probs = [cls0_log_dirichlet_multinomial_distribution]
            # first update the active cluster
            particle.active_clusters = self.update_active_clusters(particle)
            # then calculate rates for each cluster in active interval
            
            total_topic_init = self.base_intensity
            log_time_likelihood = ln(self.base_intensity)

            #gammaln_every = [gammaln_this_time]
            for active_cluster_index, timeseq in particle.active_clusters.iteritems():
                topic = particle.topics[particle.cluster2topic_ID[active_cluster_index]]
                active_cluster_indexes.append(active_cluster_index)
                time_intervals = doc.timestamp - np.array(timeseq)
                # alpha = particle.clusters[active_cluster_index].alpha
                alpha = topic.alpha[0]
                rate = triggering_kernel(alpha, self.reference_time, time_intervals, self.bandwidth)

                total_topic_init += rate


                '''
                if len(timeseq) > 1:
                    log_time_likelihood += log_likelihood(np.array(timeseq), np.array(alpha),
                                          self.reference_time, self.bandwidth, self.base_intensity, T)
                '''
                active_cluster_rates.append(rate)

                cls_word_distribution = topic.word_distribution + doc.word_distribution

                cls_word_count = topic.word_count + doc.word_count

                cls_log_dirichlet_multinomial_distribution = log_dirichlet_multinomial_distribution(
                    cls_word_distribution, doc.word_distribution, cls_word_count, doc.word_count, self.vocabulary_size, self.theta0)
                    #cls_word_count, doc.word_count, self.vocabulary_size, self.theta0, topic.gammaln_last_time)

                active_cluster_textual_probs.append(cls_log_dirichlet_multinomial_distribution)
                # gammaln_every.append(gammaln_this_time)

            particle.time_likelihood = log_time_likelihood
            active_cluster_logrates = np.log(np.array(active_cluster_rates)/total_topic_init)

            cluster_selection_probs = active_cluster_logrates + active_cluster_textual_probs  # in log scale

            log_intensities = list(cluster_selection_probs)

            cluster_selection_probs = cluster_selection_probs - np.max(cluster_selection_probs)  # prevent overflow
            cluster_selection_probs = np.exp(cluster_selection_probs)
            cluster_selection_probs = cluster_selection_probs / np.sum(cluster_selection_probs)

            # print('cluster_selection_probs', cluster_selection_probs)

            # cluster_selection_probs = np.array(active_cluster_rates)/np.sum(active_cluster_rates)
            # print('cluster_selection_probs', cluster_selection_probs)
            np.random.seed()
            selected_cluster_array = multinomial(exp_num=1, probabilities=cluster_selection_probs)
            cluster_index_index = np.nonzero(selected_cluster_array)
            selected_cluster_index = np.array(active_cluster_indexes)[cluster_index_index][0]
            # selected_gammaln = np.array(gammaln_every)[cluster_index_index][0]
            # print('np.nonzero(selected_cluster_array)',np.nonzero(selected_cluster_array))
            # print('selected_cluster_array',selected_cluster_array)
            # print('selected_cluster_index', selected_cluster_index)
            # print('type(selected_cluster_index)', type(selected_cluster_index))
            if selected_cluster_index == 0:  # the case of new cluster
                particle.cluster_num_by_now += 1
                selected_cluster_index = particle.cluster_num_by_now
                selected_cluster = Cluster(index=selected_cluster_index)
                selected_cluster.add_document(doc)
                particle.clusters[selected_cluster_index] = selected_cluster
                particle.docs2cluster_ID.append(selected_cluster_index)
                particle.active_clusters[selected_cluster_index] = [
                    self.active_interval[1]]  # create a new list containing the current time

                # sample a topic for the new cluster
                active_topic_index = [0]

                new_log_intensities = []

                new_topic_intensity = self.base_intensity * self.beta / (total_topic_init * (len(particle.topics) + self.beta))
                new_topic_intensity = ln(new_topic_intensity)
                new_topic_log_likelihood = log_dirichlet_multinomial_distribution(doc.word_distribution,
                                                                                                 doc.word_distribution, \
                                                                                                 doc.word_count,
                                                                                                 doc.word_count,
                                                                                                 self.vocabulary_size,
                                                                                                 #   self.theta0, self.gammaln_prior)
                                                                                                self.theta0)
                # gammaln_every = [gammaln_this_time]

                new_topic_intensity += new_topic_log_likelihood
                new_log_intensities.append(new_topic_intensity)

                for topic in particle.topics:
                    topic = particle.topics[topic]
                    active_topic_index.append(topic.index)
                    topic_intensity = (self.base_intensity * topic.count / (total_topic_init * (len(particle.topics) + self.beta)))
                    topic_intensity = ln(topic_intensity)
                    topic_word_distribution = topic.word_distribution + doc.word_distribution
                    topic_word_count = topic.word_count + doc.word_count
                    topic_log_likelihood = log_dirichlet_multinomial_distribution(topic_word_distribution,
                                                                                  doc.word_distribution,
                                                                                  topic_word_count,
                                                                                  doc.word_count,
                                                                                  self.vocabulary_size,
                                                                                  # self.theta0, topic.gammaln_last_time)
                                                                                  self.theta0)
                    topic_intensity += topic_log_likelihood
                    new_log_intensities.append(topic_intensity)
                    # gammaln_every.append(gammaln_this_time)
                log_intensities.extend(new_log_intensities)
                normalizing_log_intensity = logsumexp(new_log_intensities)
                intensities = [exp(log_intensity - normalizing_log_intensity)
                               for log_intensity in new_log_intensities]

                np.random.seed()
                selected_topic_array = multinomial(exp_num=1, probabilities=intensities)
                topic_index_index = np.nonzero(selected_topic_array)
                selected_topic_index = np.array(active_topic_index)[topic_index_index][0]
                # selected_gammaln = np.array(gammaln_every)[topic_index_index][0]

                if selected_topic_index == 0:
                    # TODO: create a new topic
                    particle.topic_num_by_now += 1
                    selected_topic_index = particle.topic_num_by_now
                    selected_topic = Topic(index=selected_topic_index)
                    selected_topic.add_document(doc)
                    particle.topics[selected_topic_index] = selected_topic
                    particle.cluster2topic_ID[selected_cluster_index] = selected_topic_index
                    # particle.topics[selected_topic_index].gammaln_last_time = selected_gammaln
                else:
                    # TODO: update the former topic
                    selected_topic = particle.topics[selected_topic_index]
                    selected_topic.add_document(doc)
                    particle.cluster2topic_ID[selected_cluster_index] = selected_topic_index
                    # particle.topics[selected_topic_index].gammaln_last_time = selected_gammaln

            # print('active_clusters', particle.active_clusters); print('cluster_num_by_now', particle.cluster_num_by_now) # FOR DEBUG
            else:  # the case of the previous used cluster, update active cluster and add document to cluster
                selected_cluster = particle.clusters[selected_cluster_index]
                selected_cluster.add_document(doc)
                selected_topic_index = particle.cluster2topic_ID[selected_cluster_index]
                selected_topic = particle.topics[selected_topic_index]
                selected_topic.add_document(doc)
                particle.docs2cluster_ID.append(selected_cluster_index)
                particle.active_clusters[selected_cluster_index].append(self.active_interval[1])
                # particle.topics[particle.cluster2topic_ID[selected_cluster_index]].gamma_last_time = selected_gammaln
            # print('active_clusters', particle.active_clusters); print('cluster_num_by_now', particle.cluster_num_by_now) # FOR DEBUG

            particle._Q0 = logsumexp(log_intensities)
        print('selected topic', selected_topic_index)
        particle.topics[selected_topic_index].count += 1
        return particle, selected_cluster_index, selected_topic_index

    def parameter_estimation(self, particle, selected_cluster_index):
        # print('updating triggering kernel ...')
        # print(particle.active_clusters[selected_cluster_index])
        timeseq = np.array(particle.active_clusters[selected_cluster_index])
        if len(timeseq) == 1:  # the case of first document in a brand new cluster
            np.random.seed()
            alpha = dirichlet(self.alpha0)
            return alpha, 0
        T = self.active_interval[1] + 1  # ;print('updating triggering kernel ..., len(timeseq)', len(timeseq))
        alpha = update_triggering_kernel(timeseq, self.alphas, self.reference_time, self.bandwidth, self.base_intensity,
                                         T, self.log_priors)
        return alpha

    def update_active_clusters(self, particle):
        if not particle.active_clusters:  # the case of the first document comes
            particle.active_clusters[1] = [self.active_interval[1]]
        else:  # update the active clusters
            tu = self.active_interval[0]
            for cluster_index in particle.active_clusters.keys():
                timeseq = particle.active_clusters[cluster_index]
                active_timeseq = [t for t in timeseq if t > tu]
                if not active_timeseq:
                    del particle.active_clusters[cluster_index]
                else:
                    particle.active_clusters[cluster_index] = active_timeseq
        return particle.active_clusters

    def calculate_particle_log_update_prob(self, particle, selected_cluster_index, doc):
        # print('calculate_particle_log_update_prob') #print('id(particle.clusters[selected_cluster_index])', id(particle.clusters[selected_cluster_index]));#print('id(particle.clusters[selected_cluster_index]).word_distribution', id(particle.clusters[selected_cluster_index].word_distribution))
        cls_word_distribution = particle.clusters[selected_cluster_index].word_distribution
        cls_word_count = particle.clusters[selected_cluster_index].word_count
        doc_word_distribution = doc.word_distribution
        doc_word_count = doc.word_count
        assert doc_word_count == np.sum(doc.word_distribution)
        assert cls_word_count == np.sum(particle.clusters[selected_cluster_index].word_distribution)
        log_update_prob = log_dirichlet_multinomial_distribution(cls_word_distribution, doc_word_distribution, \
                                                                 cls_word_count, doc_word_count, self.vocabulary_size,
                                                                 self.theta0)  # print('particle.log_update_probs',particle.log_update_probs)
        # print('log_update_prob', log_update_prob)
        return log_update_prob

    def particles_normal_resampling(self, particles, threshold):
        print('\nparticles_normal_resampling')
        weights = [];
        log_update_probs = []
        for particle in particles:
            weights.append(particle.weight)
            log_update_probs.append(particle.log_update_prob)
        weights = np.array(weights);
        log_update_probs = np.array(log_update_probs);
        # print('weights before update:', weights);
        # print('log_update_probs', log_update_probs)
        # print(log_update_probs)
        log_update_probs = log_update_probs - np.max(log_update_probs)  # prevent overflow

        update_probs = np.exp(log_update_probs);  # print('update_probs',update_probs)
        weights = weights * update_probs  # update
        weights = weights / np.sum(weights)  # normalization
        resample_num = len(np.where(weights + 1e-5 < threshold)[0])
        # print('weights:', weights)  # ; print('log_update_probs',log_update_probs);
        # print('resample_num:', resample_num)
        if resample_num == 0:  # no need to resample particle, but still need to assign the updated weights to paricle weight
            for i, particle in enumerate(particles):
                particle.weight = weights[i]
            return particles
        else:
            remaining_particles = [particle for i, particle in enumerate(particles) if weights[i] + 1e-5 > threshold]
            resample_probs = weights[np.where(weights + 1e-5 > threshold)];
            resample_probs = resample_probs / np.sum(resample_probs)
            remaining_particle_weights = weights[np.where(weights + 1e-5 > threshold)]
            for i, _ in enumerate(remaining_particles):
                remaining_particles[i].weight = remaining_particle_weights[i]
            np.random.seed()
            resample_distribution = multinomial(exp_num=resample_num,
                                                probabilities=resample_probs)  # print('len(remaining_particles)', len(remaining_particles)) #print('resample_probs', resample_probs)
            if not resample_distribution.shape:  # the case of only one particle left
                for _ in range(resample_num):
                    new_particle = deepcopy(remaining_particles[0])
                    remaining_particles.append(new_particle)
            else:  # the case of more than one particle left
                for i, resample_times in enumerate(resample_distribution):
                    for _ in range(resample_times):
                        new_particle = deepcopy(remaining_particles[i])
                        remaining_particles.append(new_particle)
            # normalize the particle weight again
            update_weights = np.array([particle.weight for particle in remaining_particles]);
            update_weights = update_weights / np.sum(update_weights)
            for i, particle in enumerate(remaining_particles):
                particle.weight = update_weights[i]
            # print('update_weights aftering resampling', update_weights)
            assert np.abs(np.sum(update_weights) - 1) < 1e-5
            assert len(remaining_particles) == self.particle_num
            self.particles = None
            return remaining_particles


def parse_newsitem_2_doc(news_item, vocabulary_size):
    ''' convert (id, timestamp, word_distribution, word_count) to the form of document
    '''
    # print(news_item)
    index = news_item[0]
    timestamp = news_item[1] / 3600.0  # unix time in hour
    word_id = news_item[2][0]
    count = news_item[2][1]
    word_distribution = np.zeros(vocabulary_size)
    word_distribution[word_id] = count
    word_count = news_item[3]
    doc = Document(index, timestamp, word_distribution, word_count)
    # assert doc.word_count == np.sum(doc.word_distribution)
    return doc


def main():
    with open('./data/dup.json') as f:
        news_items = json.load(f)
    print('finish extracting news from json...')

    # parameter initialization
    vocabulary_size = 56720
    particle_num = 8
    base_intensity = 0.1
    theta0 = np.array([0.01] * vocabulary_size)
    alpha0 = np.array([0.1] * 4)
    reference_time = np.array([3, 7, 11, 24])
    bandwidth = np.array([5, 5, 5, 5])
    sample_num = 2000
    threshold = 1.0 / particle_num
    mu0 = 0.1
    beta0 =1

    DHP = Dirichlet_Hawkes_Process(particle_num=particle_num, base_intensity=base_intensity, theta0=theta0,
                                   alpha0=alpha0, \
                                   reference_time=reference_time, vocabulary_size=vocabulary_size, bandwidth=bandwidth,
                                   sample_num=sample_num, beta0=beta0)

    # begin sampling
    # for simple experiment
    news_items = news_items[:200]
    for news_item in news_items:
        doc = parse_newsitem_2_doc(news_item=news_item, vocabulary_size=vocabulary_size)
        DHP.sequential_monte_carlo(doc, threshold)

    with open('./result/particles_dup_news.pkl', 'wb') as w:
        pickle.dump(DHP.particles, w)


if __name__ == '__main__':
    main()