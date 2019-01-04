import pickle
import utils_hdhp
import json
import numpy as np
import datetime
from collections import Counter
import matplotlib.pyplot as plt

def topic_word(p, ridx=False):
    cluster_num = max(p[0].docs2cluster_ID)

    with open('./result/word_reidx.json', 'r') as f:
        ridx = json.load(f)
    with open('./data/id2word.json', 'r') as f:
        id2word = json.load(f)
    for i in range(1, cluster_num + 1):
        word_distribution = p[0].clusters[i].word_distribution
        sorted_distribution = np.argsort(-word_distribution)
        with open('./dhp/cluster_%d.txt' % i, 'w') as f:
            for j in range(100):
                word = id2word[str(ridx[str(sorted_distribution[j])])]
                num = word_distribution[sorted_distribution[j]]
                f.write(str(num) + '\t' + word + '\n')


def get_topic_word(p, ridx=False):
    with open('../data/word_reidx.json', 'r') as f:
        ridx = json.load(f)
    with open('../data/id2word.json', 'r') as f:
        id2word = json.load(f)
    for topic_id in range(1, p[0].topic_num_by_now+1):
        word_distribution = p[0].topic_word_distribution[topic_id]
        # sorted_distribution = np.argsort(-word_distribution)
        sorted_distribution = sorted(word_distribution.items(), key=lambda x:x[1], reverse=True)
        with open('./clusters/cluster_%d.txt' % topic_id, 'w') as f:
            for item in sorted_distribution[:100]:
                word = id2word[str(ridx[str(item[0])])]
                num = item[1]
                f.write( word + '\t' + str(num)+ '\n')

def topic_evolution(p):
    docs2ID = np.array(p[0].active_clusters)
    np.save('./result/evol.npy', docs2ID)

def plot_evolution(cluster_ID=2):
    '''
    with open('./data/all_the_news_2017.json', 'r') as f:
        data = json.load(f)

    m   at = np.load('./result/evol_mat.npy')
    arr = mat[0]
    '''
    d = np.load('./result/test.npy')
    d = [x * 3600 for x in d]
    d = [datetime.datetime.fromtimestamp(x) for x in d]
    d = [(x.year, x.month, x.day) for x in d]

def topic_change():
    pass

if __name__ == '__main__':
    with open('../result/particles_hdhp.pkl', 'rb') as f:
        p = pickle.load(f, encoding='latin1')
    #topic_evolution(p)
    get_topic_word(p)
