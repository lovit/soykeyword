class LassoKeywordExtractor:

	def __init__(self, min_count):
		self.min_count = min_count

	def extract_from_docs(self, docs):
		raise NotImplemented

	def extract_from_sparse_matrix(self, x):
		raise NotImplemented

	def extract_from_clusters(self, x, c):
		raise NotImplemented		