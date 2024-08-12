# Spectuner
[![Documentation Status](https://readthedocs.org/projects/spectuner/badge/?version=latest)](https://spectuner.readthedocs.io/en/latest/?badge=latest)

Spectuner is an ease-of-use tool for automated spectral line identification of
interstellar molecules. The code integrates the following techniques:

* Spectral line model: XCLASS.
* Peak finder: Scipy.
* Spectral fitting: Particle swarm optimization & peak matching loss function.


## Installation
1. Install XCLASS according to this [link](https://xclass-pip.astro.uni-koeln.de/).
2. Clone the repository and run ``setpy.py``:

```
git clone https://github.com/yqiuu/spectuner.git
cd spectuner
pip install .
```

## Documentation
Read the docs at this [link](https://spectuner.readthedocs.io/en/latest/index.html).
