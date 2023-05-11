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
        'drawsvg>=2.1.1',
        'hyperopt>=0.2.7',
        'ipywidgets>=8.0.5',
        'keyring>=23.2.1',
        'matplotlib>=3.4.2',
        'numba>=0.56.4',
        'numpy>=1.20.3,<=1.23.5',
        'pandas>=1.5.2,<=1.9.0',
        'pyperclip>=1.8.2',
        'requests',
        'scikit_learn>=1.2.2',
        'scipy>=1.6.2',
        'seaborn>=0.11.1',
        'traitlets>=5.5.0',
        'urllib3>=1.26.5',
        'psutil>=5.9.4',
        'joblib>=1.2.0',
    ]
)
