Installation
=========================

PyPI is the distribution channel for xplainable release versions. The best way
to install it is with pip::

    pip install xplainable


Environment
-------------------------------

Working in Jupyter
~~~~~~~~~~~~~~~~~~~~~

When using ``xplainable`` with Jupyter, it relies heavily on ``ipywidgets``.
Unfortunately, this requires a strict environment to ensure that
xplainable's embedded GUI works as expected. Ensuring that you have correctly
installed the required dependencies before importing xplainable is essential.

For now, **we recommend using Python 3.8**. This environment will yield the best
visible results until we can stabilise ``ipywidgets`` for later versions of Python.

Reproducible Installs
~~~~~~~~~~~~~~~~~~~~~

As libraries get updated, results from running your code can change, or your
code can break completely. It's essential to be able to reconstruct the set of
packages and versions you're using. Best practice is to:

 1. use a different environment per project you're working on,
 2. record package names and versions using your package installer; each has it's own metadata format for this:

   * **Conda:** conda environments and environment.yml
   * **Pip:** virtual environments and requirements.txt
   * **Poetry:** virtual environments and pyproject.toml
