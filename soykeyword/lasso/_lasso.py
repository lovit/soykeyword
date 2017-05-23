from sklearn.linear_model import LogisticRegression

class LassoKeywordExtractor:

    def __init__(self, min_tf=20, min_df=10, costs=None):
        self.min_count = min_count
        self.costs = [500, 200, 100, 50, 10, 5, 1, 0.1] if costs == None else costs
    
    def train(self, x):
        self.num_doc, self.num_term = x.shape
        
        rows, cols = x.nonzero()
        b = csr_matrix(([1] * len(rows), (rows, cols)))
        _df = dict(enumerate(b.sum(axis=0).tolist()[0]))
        _df = {word:df for word, df in _df.items() if df >= self.min_df}
        
        _tfs = dict(enumerate(x.sum(axis=0).tolist()[0]))
        _tfs = {word:freq for word, freq in _tfs.items() if (freq >= self.min_df) and (word in _df)}
        
        rows_ = []
        cols_ = []
        data_ = []
        for r, c, d in zip(rows, cols, x.data):
            if not (c in _tfs):
                continue
            rows_.append(r)
            cols_.append(c)
            data_.append(d)
        self.x = csr_matrix((data_, (rows_, cols_)))
    
    def extract_from_word(self, term):
        if not (0 <= idx < self.num_term):
            return []
        pos_idx = self.x[:,word].nonzero()[0].tolist()
        return self.extract_from_docs(pos_idx)

    def extract_from_docs(self, docs):
        pos_idx = set(docs)
        y = [1 if d in pos_idx else -1 for d in range(self.num_doc)]
        
        
        raise NotImplemented

class LassoClusteringLabeler(LassoKeywordExtractor):

    def __init__(self, min_tf=20, min_df=10, costs=None):
        super().__init__(min_tf, min_df, costs)

    def label_clusters(self, cluster_labels):
        raise NotImplemented

