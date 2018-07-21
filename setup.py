''' ape package '''
from setuptools import setup


setup(name='nk_ape',
      version='1.0.0',
      description='Abstractive Prediction via Embeddings',
      packages=['nk_ape'],
      install_requires=[
          'numpy>=1.13.3',
          'gensim==3.2.0',
          'inflection>=0.3.1',
          'pandas>=0.19.2',
          'ontospy>=1.8.6'
      ],
      include_package_data=True
      )
