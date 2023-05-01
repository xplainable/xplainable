from setuptools import find_packages, setup

exec(open('xplainable/_version.py').read())

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='xplainable',
    packages=find_packages(),
    version=__version__,
    description='The ai engine for business optimisation, innovation, and decision making',
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='xplainable pty ltd',
    author_email="contact@xplainable.io",
    license=None,
    install_requires=[
        'numpy>=1.23.0',
        'pandas>=1.5.2',
        'hyperopt>=0.2.7',
        'joblib>=1.1.0',
        'nltk>=3.6.2',
        'psutil>=5.9.0',
        'spacy>=3.1.0',
        'dill==0.3.6',
        'ipython==8.8.0',
        'ipywidgets==8.0.3',
        'matplotlib==3.6.2',
        'requests==2.28.1',
        'scikit-learn>=1.1.3',
        'seaborn==0.12.1',
        'statsmodels==0.13.5',
        'traitlets==5.5.0',
        'urllib3==1.26.12',
        'altair==4.2.0',
        'scipy==1.6.2',
        'pyperclip>=1.8.2',
        'python-dotenv'
    ]
)
