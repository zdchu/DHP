import warnings
warnings.filterwarnings('ignore')
import pickle
from hdhp import *
from tqdm import tqdm
from utils_hdhp import *
import IPython
import json
from sklearn.model_selection import train_test_split


def hdhp_run(news_items, train, test, save_model=True):
    vocabulary_size = 10000

    particle_num = 8
    base_intensity = 0.1
    theta0 = np.array([0.1] * vocabulary_size)
    alpha0 = np.array([0.1] * 4)
    reference_time = np.array([3, 7, 11, 24])
    bandwidth = np.array([2, 3, 5, 5])
    sample_num = 2000
    threshold = 1.0 / particle_num
    beta0 = 2
    perplex = []

    idx, ridx = get_df_words(news_items, vocabulary_size)
    HDHP = Hierarchical_Dirichlet_Hawkes_Process(particle_num=particle_num, base_intensity=base_intensity,
                                                 theta0=theta0,
                                                 alpha0=alpha0, reference_time=reference_time,
                                                 vocabulary_size=vocabulary_size, bandwidth=bandwidth,
                                                 sample_num=sample_num, beta0=beta0)

    for train_item in tqdm(train):
        for news_item in train_item:
            doc = parse_newsitem_2_doc(news_item=news_item, words_idx=idx, time_unit=3600)
            HDHP.sequential_monte_carlo(doc, threshold)
        max_weight = -1
        max_particle = None
        for p in HDHP.particles:
            if p.weight > max_weight:
                max_particle = p
        perplex.append(perplexity(max_particle, test, idx, theta0=theta0, reference_time=reference_time, bandwidth=bandwidth, beta=beta0, base_intensity=base_intensity))
    if save_model:
        with open('../result/particles_hdhp.pkl', 'wb') as w:
            pickle.dump(HDHP.particles, w)
    return perplex


def train_test_creater(data, batch=100, test_rate=0.05):
    data_size = len(data)
    train_set = []
    test_set = []
    for i in range(int(data_size/batch)):
        train_items = data[i*batch: (i+1) * batch]
        train, test = train_test_split(train_items, test_size=test_rate)
        train = sorted(train, key=lambda x: x[1])
        test = sorted(test, key=lambda x: x[1])
        train_set.append(train)
        test_set.extend(test)
    return train_set, test_set


if __name__ == '__main__':
    with open('../data/all_the_news_2017.json') as f:
        data = json.load(f)
    with open('../data/train.json') as f:
        train = json.load(f)
    with open('../data/test.json') as f:
        test = json.load(f)
    perplex = hdhp_run(news_items=data, train=train, test=test)
    np.save('perplex.npy', perplex)
    IPython.embed()
