from sklearn.linear_model import LogisticRegression

class LassoKeywordExtractor:

    def __init__(self, min_count=20, costs=None, x):
        self.min_count = min_count
        self.costs = [500, 200, 100, 50, 10, 5, 1, 0.1] if costs == None else costs
        self.x = x

    def extract_from_positive_term(self, term):
        (n,m) = self.x.shape
        if not (0 <= idx < m):
            return []
        
        raise NotImplemented

    def extract_from_positive_docs(self, docs):
        
        raise NotImplemented

        
class LassoClusteringLabeler:
    
    def __init__(self, min_count=20, x):
        self.min_count = min_count
        self.x = x
        
    def label_clusters(self, cluster_idx, cost=10):
        raise NotImplemented