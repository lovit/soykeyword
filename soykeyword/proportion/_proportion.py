from collections import Counter
from collections import defaultdict
from collections import namedtuple
from math import log
import sys
import numpy as np
from scipy.sparse import csr_matrix
from soykeyword.utils import get_process_memory

KeywordScore = namedtuple('KeywordScore', 'word frequency score')

class CorpusbasedKeywordExtractor:

    def __init__(self, min_tf=20, min_df=2, tokenize=lambda x:x.strip().split(), verbose=True):
        self.min_tf = min_tf
        self.min_df = min_df
        self.tokenize = tokenize
        self.verbose = verbose
        
        self._d2t = None
        self._t2d = None
        self.num_doc = 0
        self.num_term = 0
        self._tfs = None

    def train(self, docs, temporal_pruning_points=100000, temporal_pruning_min_df=5):
        self._d2t = {}
        self._t2d = defaultdict(lambda: [])
        for d, doc in enumerate(docs):
            words = tuple(Counter(self.tokenize(doc)).items())
            if not words:
                continue
            self._d2t[d] = words
            for word, _ in words:
                self._t2d[word].append(d)
            if (d + 1) % temporal_pruning_points == 0:
                self._pruning_under_min_df(temporal_pruning_min_df)
            if self.verbose and ((d + 1) % 10000 == 0):
                args = (len(self._t2d), d+1, len(docs), get_process_memory())
                sys.stdout.write('\rtraining ... %d terms, %d in %d docs, memory = %.3f Gb' % args)
                
        self.num_doc = (d + 1)
        self._t2d = dict(self._t2d)
        
        self._pruning_under_min_df(self.min_df)
        self._df = {word:len(ds) for word, ds in self._t2d.items()}
        self.num_term = len(self._df)
        
        self._sort_by_tfidf()
        self._tfs = self._get_reference_sum()
        
        if self.verbose:
            args = (len(self._t2d), self.num_doc, get_process_memory())
            print('\rtraining was done %d terms, %d docs, memory = %.3f Gb' % args)
        
    def _pruning_under_min_df(self, min_df):
        under_min_df = {word for word, ds in self._t2d.items() if len(ds) < min_df}
        num_doc = len(self._d2t)
        empty_docs = []
        for d, tf in self._d2t.items():
            ts = [(word, freq) for word, freq in tf if not (word in under_min_df)]
            if not ts:
                empty_docs.append(d)
                continue
            self._d2t[d] = ts
        for d in empty_docs:
            del self._d2t[d]
        for word in under_min_df:
            del self._t2d[word]
    
    def _sort_by_tfidf(self):
        for d, tf in self._d2t.items():
            tf = sorted(tf, key=lambda x:x[1] * (1 / (1+log(1 + self._df.get(x[0], 0)))), reverse=True)
            self._d2t[d] = tf
            
    def _get_reference_sum(self):
        sum_ = defaultdict(lambda: 0)
        for d, ts in self._d2t.items():
            for word, freq in ts:
                sum_[word] += freq
        return dict(sum_)
    
    def frequency(self, word):
        return self._tfs.get(word, 0)
            
    def extract_from_word(self, word, min_count=20, min_score=0.75):
        pos_idx = self.get_document_index(word)
        if not pos_idx:
            return []
        return self.extract_from_docs(pos_idx, min_count, min_score)
    
    def get_document_index(self, word):
        return sorted(set(self._t2d.get(word, [])))
        
    def extract_from_docs(self, docs, min_count=20, min_score=0.75):
        ps = self._get_positive_sum(docs)
        ns = self._get_negative_sum(ps)
        pp = self._sum_to_proportion(ps)
        np = self._sum_to_proportion(ns)
        
        s = {word:(p/(p+np.get(word, 0))) for word, p in pp.items()}
        s = {word:score for word, score in s.items() if self.frequency(word) >= min_count and score >= min_score}
        s = sorted(s.items(), key=lambda x:x[1], reverse=True)
        s = [KeywordScore(word, self.frequency(word), score) for word, score in s]
        return s

    def _sum_to_proportion(self, sum_dict):
        sum_ = sum(sum_dict.values())
        return {word:(freq/sum_) for word, freq in sum_dict.items()}
    
    def _get_positive_sum(self, pos_idx):
        sum_ = defaultdict(lambda: 0)
        for d in pos_idx:
            for word, freq in self._d2t.get(d, []):
                sum_[word] += freq
        return sum_
    
    def _get_negative_sum(self, pos_sum):
        return {word:(freq - pos_sum.get(word, 0)) for word, freq in self._tfs.items()}

        
class MatrixbasedKeywordExtractor:
    
    def __init__(self, min_tf=20, min_df=2, verbose=True):
        self.x = None
        self.min_tf = min_tf
        self.min_df = min_df
        self.verbose = verbose
        self._tfs = None
        self.num_doc = 0
        self.num_term = 0
        self.index2word = None
        self.word2index = None
        
    def train(self, x, index2word=None):
        self.num_doc, self.num_term = x.shape
        self.index2word = index2word
        self.word2index = {word:index for index, word in enumerate(index2word)} if index2word is not None else None
        
        rows, cols = x.nonzero()
        b = csr_matrix(([1] * len(rows), (rows, cols)))
        self._df = dict(enumerate(b.sum(axis=0).tolist()[0]))
        self._df = {word:df for word, df in self._df.items() if df >= self.min_df}
        
        self._tfs = dict(enumerate(x.sum(axis=0).tolist()[0]))
        self._tfs = {word:freq for word, freq in self._tfs.items() if (freq >= self.min_df) and (word in self._df)}
        
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
        print('MatrixbasedKeywordExtractor trained')

    def extract_from_word(self, word, min_count=20, min_score=0.75):
        pos_idx = self.get_document_index(word)
        if not pos_idx:
            return []
        return self.extract_from_docs(pos_idx, min_count, min_score)

    def get_document_index(self, word):
        if type(word) == str:
            if not self.word2index:
                raise ValueError('If you want to insert str word, you should trained index2word first')
            word = self.word2index.get(word,-1)
        if 0 <= word < self.num_term:
            return self.x[:,word].nonzero()[0].tolist()
        return []
    
    def extract_from_docs(self, docs, min_count=20, min_score=0.75):
        ps = self._get_positive_sum(docs)
        ns = self._get_negative_sum(ps)
        pp = self._sum_to_proportion(ps)
        np = self._sum_to_proportion(ns)
        
        s = {word:(p/(p+np.get(word, 0))) for word, p in pp.items()}
        s = {word:score for word, score in s.items() if self._tfs.get(word,0) >= min_count and score >= min_score}
        if self.index2word:
            s = {self.index2word[w] if 0 <= w < self.num_term else 'Unk%d'%w:score for w, score in s.items()}
        s = sorted(s.items(), key=lambda x:x[1], reverse=True)
        s = [KeywordScore(word, self._tfs.get(word, 0), score) for word, score in s]        
        return s
    
    def _sum_to_proportion(self, sum_dict):
        sum_ = sum(sum_dict.values())
        return {word:(freq/sum_) for word, freq in sum_dict.items()}
    
    def _get_positive_sum(self, pos_idx):
        if type(pos_idx) != list:
            pos_idx = list(pos_idx)
        x_pos = self.x[pos_idx]
        return {word:freq for word, freq in enumerate(x_pos.sum(axis=0).tolist()[0]) if freq > 0}
        
    def _get_negative_sum(self, pos_sum):
        return {word:(freq - pos_sum.get(word, 0)) for word, freq in self._tfs.items()}