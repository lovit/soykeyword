from collections import namedtuple
from operator import itemgetter
from itertools import groupby
from sklearn.linear_model import LogisticRegression
from scipy.sparse import csr_matrix

KeywordScore = namedtuple('KeywordScore', 'word frequency coefficient')

class LassoKeywordExtractor:

    def __init__(self, min_tf=20, min_df=10, costs=None, verbose=True):
        self.min_tf = min_tf
        self.min_df = min_df
        self.costs = [500, 200, 100, 50, 10, 5, 1, 0.1] if costs == None else costs
        self.costs = sorted(self.costs)
        self.verbose = verbose
        self.index2word = None
        self.word2index = None
        self._tfs = {}
    
    def train(self, x, index2word=None):
        self.num_doc, self.num_term = x.shape
        self.index2word = index2word
        self.word2index = {w:i for i,w in enumerate(index2word)} if index2word else None
        
        rows, cols = x.nonzero()
        b = csr_matrix(([1] * len(rows), (rows, cols)))
        _df = dict(enumerate(b.sum(axis=0).tolist()[0]))
        _df = {word:df for word, df in _df.items() if df >= self.min_df}
        
        self._tfs = dict(enumerate(x.sum(axis=0).tolist()[0]))
        self._tfs = {word:freq for word, freq in self._tfs.items() if (freq >= self.min_df) and (word in _df)}
        
        rows_ = []
        cols_ = []
        data_ = []
        for r, c, d in zip(rows, cols, x.data):
            if not (c in self._tfs):
                continue
            rows_.append(r)
            cols_.append(c)
            data_.append(d)
        self.x = csr_matrix((data_, (rows_, cols_)))
        self._is_empty = [1 if float(d[0]) == 0 else 0 for d in self.x.sum(axis=1)]
    
    def extract_from_word(self, word, minimum_number_of_keywords=5):
        word = self._encoding(word)
        pos_idx = self.get_document_index(word)
        return self.extract_from_docs(pos_idx, minimum_number_of_keywords, except_words={word})
    
    def _encoding(self, word):
        if type(word) == str:
            if not self.word2index:
                raise ValueError('You should set index2word first')
            word = self.word2index.get(word, -1)
        return word
    
    def get_document_index(self, word):
        word = self._encoding(word)
        if not (0 <= word < self.num_term):
            return []
        return self.x[:,word].nonzero()[0].tolist()
        
    def extract_from_docs(self, docs_idx, minimum_number_of_keywords=5, except_words=None):
        pos_idx = set(docs_idx)
        y = [1 if (d in pos_idx and self._is_empty[d] == 0) else -1 for d in range(self.num_doc)]
        if (1 in set(y)) == False:
            if self.verbose:
                print('There is no corresponding documents')
            return []
        
        if except_words:
            data_ = []
            rows_ = []
            cols_ = []
            rows, cols = self.x.nonzero()
            for r, w, d in zip(rows, cols, self.x.data):
                if w in except_words:
                    continue
                data_.append(d)
                rows_.append(r)
                cols_.append(w)
            x_ = csr_matrix((data_, (rows_, cols_)))
        else:
            x_ = self.x
        
        for c in self.costs:
            logistic = LogisticRegression(penalty='l1', C=c)
            logistic.fit(x_, y)
            coefficients = logistic.coef_.reshape(-1)
            keywords = sorted(enumerate(coefficients), key=lambda x:x[1], reverse=True)
            keywords = [(word, self._tfs.get(word,0), coef) for word, coef in keywords if coef > 0]
            logistic = None
            if self.verbose:
                print('%d keywords extracted from %.3f cost' % (len(keywords), c))
            if len(keywords) >= minimum_number_of_keywords:
                break
        if self.index2word:
            keywords = [KeywordScore(self.index2word[word] if 0 <= word < self.num_term else 'Unk%d'%word, tf, coef) for word, tf, coef in keywords]
        else:
            keywords = [KeywordScore(word, tf, coef) for word, tf, coef in keywords]
        return keywords


class LassoClusteringLabeler(LassoKeywordExtractor):
    
    def __init__(self, min_tf=20, min_df=10, costs=None, verbose=True):
        super().__init__(min_tf, min_df, costs, verbose)
        
    def label_clusters(self, cluster_idx, minimum_number_of_keywords=5):
        groups = sorted(enumerate(cluster_idx), key=itemgetter(1))
        groups = {group:[d for d, g in docs] for group, docs in groupby(groups, key=itemgetter(1))}
        labels = {}
        for cluster, docs in groups.items():
            if self.verbose:
                print('labeling cluster = %d' % cluster)
            labels[cluster] = self.extract_from_docs(docs, minimum_number_of_keywords)
        return sorted(labels.items())