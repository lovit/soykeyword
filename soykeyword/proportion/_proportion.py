class RelatedProportionKeywordExtractor:

    def __init__(self, min_count=20):
        self.min_count = min_count
        self._d2t = None
        self._t2d = None

    def train(self, docs):
        raise NotImplemented

    def extract_from_word(self, word):
        raise NotImplemented

        
class RelatedProportionClusteringLabeler:
    
    def __init__(self, min_count=20, x):
        self.min_count = min_count
        self.x = x
        
    def label_clusters(self, cluster_idx, alpha=0.5):
        raise NotImplemented