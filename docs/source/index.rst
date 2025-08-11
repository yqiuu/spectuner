.. Spectuner documentation master file, created by
   sphinx-quickstart on Fri Jul 19 16:34:05 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Spectuner
=========

Spectuner is a tool for automated spectral line analysis of instellar molecules.
The code implements the one-dimensional LTE spectral line model and offers the
following applications:

* Robust broadband line identification.
* Efficient pixel-by-pixel spectral line fitting.

Installation
------------
The code requires Python>=3.10. If you do not have Python installed, we
recommend installing `Anaconda <https://www.anaconda.com/products/individual>`__.
Then, we can install the code from the repository.

.. code-block:: bash

   git clone https://github.com/yqiuu/spectuner.git
   cd spectuner
   pip install .

If you want to use the AI module, you need to install
`PyTorch <https://pytorch.org/>`__.

.. code-block:: bash

   pip install torch

In addition, the code also requires the Cologne Database for Molecular
Spectroscopy (`CDMS <https://cdms.astro.uni-koeln.de/>`__) as input. You may
download the database using the following command.

.. code-block:: bash

   wget https://cdms.astro.uni-koeln.de/static/cdms/download/official/cdms_sqlite__official-version__2024-01-01.db.gz

Prerequisites
-------------
This documentation aussmes that you are familiar with the following packages:

* `Numpy <https://numpy.org/>`__
* `Scipy <https://scipy.org/>`__
* `Matplotlib <https://matplotlib.org/>`__
* `Pandas <https://pandas.pydata.org/>`__

You may also be familiar with `jupyter notebook <https://jupyter.org/>`__ and YAML,
a human-readable data serialization format that is often used for configuration
files.

.. toctree::
   :maxdepth: 1
   :caption: Tutorials:

   notebooks/line_identification

.. toctree::
   :maxdepth: 1
   :caption: User guide

   guide/sl_model
   guide/identification_results

.. toctree::
   :maxdepth: 1
   :caption: API reference:

   api/config
   api/identify
   api/spectral_plot