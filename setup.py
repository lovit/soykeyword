from description import __version__, __author__
from setuptools import setup

setup(
   name="soykeyword",
   version=__version__,
   author=__author__,
   author_email='soy.lovit@gmail.com',
   url='https://github.com/lovit/soynlp',
   description="Unsupervised Korean Natural Language Processing Toolkits",
   long_description="""It contains two keyword extraction algorithms. First one uses Lasso logistic regression and the other uses relative proportion ratio
   """,
   install_requires=["numpy", "scikit-learn", "soynlp>=0.0.17"],
   keywords = ['keyword extractor'],
   packages=['soykeyword']
)