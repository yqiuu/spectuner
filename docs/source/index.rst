.. Spectuner documentation master file, created by
   sphinx-quickstart on Fri Jul 19 16:34:05 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Spectuner
=========

Spectuner is an ease-of-use tool for automated spectral line identification of
interstellar molecules. The code integrates the following techniques:

* Spectral line model: XCLASS.
* Peak finder: Scipy.
* Spectral fitting: Particle swarm optimization & peak matching loss function.

Installation
------------
#. Install XCLASS according to this
   `link <https://xclass-pip.astro.uni-koeln.de/>`__.
#. Clone the repository and run ``setpy.py``:

   .. code-block:: bash

      git clone https://github.com/yqiuu/spectuner.git
      cd spectuner
      pip install .

.. toctree::
   :maxdepth: 1
   :caption: Contents:

   tutorial
