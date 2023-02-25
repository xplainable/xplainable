from setuptools import find_packages, setup
import Cython.Build
import numpy

exec(open('xplainable/_version.py').read())

setup(
    name='xplainable-core',
    packages=find_packages(),
    version=__version__,
    description='Explainable Machine Learning Algorithms',
    author='Tech Cowboys',
    license=None,
    url='https://github.com/xplainable/core-algorithms',
    install_requires=[
        'numpy>=1.24.1',
        'pandas>=1.5.2',
        'hyperopt>=0.2.7',
        'joblib>=1.1.0',
        'nltk>=3.6.2',
        'psutil>=5.9.0',
        'scikit-learn>=1.2.0',
        'spacy>=3.1.0'
    ],
    ext_modules = Cython.Build.cythonize("xplainable/core/xcython/*.pyx"),
    include_dirs=[numpy.get_include()]
)
