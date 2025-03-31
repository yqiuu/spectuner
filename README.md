# Spectuner
[![Documentation Status](https://readthedocs.org/projects/spectuner/badge/?version=latest)](https://spectuner.readthedocs.io/en/latest/?badge=latest)

Spectuner is an ease-of-use tool for automated spectral line identification of
interstellar molecules. The code integrates the following techniques:

* Spectral line model: XCLASS.
* Peak finder: Scipy.
* Spectral fitting: Particle swarm optimization & peak matching loss function.

Our methodology is described in [Qiu et al. 2024](https://arxiv.org/abs/2408.06004).

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

## Attribution
If you find this code useful in your research, please cite our work using the following BibTeX entry:
```
@ARTICLE{2025ApJS..277...21Q,
       author = {{Qiu}, Yisheng and {Zhang}, Tianwei and {M{\"o}ller}, Thomas and {Jiang}, Xue-Jian and {Song}, Zihao and {Chen}, Huaxi and {Quan}, Donghui},
        title = "{Spectuner: A Framework for Automated Line Identification of Interstellar Molecules}",
      journal = {\apjs},
     keywords = {Spectral line identification, Interstellar medium, 2073, 847, Astrophysics - Astrophysics of Galaxies, Astrophysics - Instrumentation and Methods for Astrophysics, Astrophysics - Solar and Stellar Astrophysics},
         year = 2025,
        month = mar,
       volume = {277},
       number = {1},
          eid = {21},
        pages = {21},
          doi = {10.3847/1538-4365/adaeba},
archivePrefix = {arXiv},
       eprint = {2408.06004},
 primaryClass = {astro-ph.GA},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2025ApJS..277...21Q},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```
