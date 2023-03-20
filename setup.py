from setuptools import find_packages, setup

exec(open('xplainable/_version.py').read())

setup(
    name='xplainable',
    packages=find_packages(),
    version=__version__,
    description='The ai engine for business optimisation, innovation, and decision making',
    author='xplainable pty ltd',
    license=None,
    install_requires=[
        'numpy>=1.24.1',
        'pandas>=1.5.2',
        'hyperopt>=0.2.7',
        'joblib>=1.1.0',
        'nltk>=3.6.2',
        'psutil>=5.9.0',
        'scikit-learn>=1.2.0',
        'spacy>=3.1.0'
    ]
)
