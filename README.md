# Meteor
Meteor is an ease-of-use tool for automated spectral line identification of
interstellar molecules. The code integrates the following techniques:

* Spectral line model: XCLASS.
* Peak finder: Scipy.
* Spectral fitting: Particle swarm optimization & peak matching loss function.


# Installation
1. Install XCLASS according to this [link](https://xclass-pip.astro.uni-koeln.de/).
2. Clone the repository and run ``setpy.py``:

```
git clone https://github.com/yqiuu/meteor.git
cd meteor
pip install .
```