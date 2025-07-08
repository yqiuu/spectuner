# Spectuner
[![Documentation Status](https://readthedocs.org/projects/spectuner/badge/?version=latest)](https://spectuner.readthedocs.io/en/latest/?badge=latest)

Spectuner is a tool for automated spectral line analysis of instellar molecules.


## Notice (15/07/25)
This is a transitional version from 1.0 to 2.0. In a few months, we will release the pixel-by-pixel fitting functionality based on neural networks. Version 1.0 will no longer be maintained thereafter.


## Installation
The code requires Python>=3.10. If you do not have Python installed, we
recommend installing [Anaconda](https://www.anaconda.com/products/individual).
Then, we can install the code from the repository.

```
git clone https://github.com/yqiuu/spectuner.git
cd spectuner
pip install .
```

## Documentation
Read the docs at this [link](https://spectuner.readthedocs.io/en/latest/index.html).


## Attribution
If you find this code useful in your research, please cite our work in the acknowledgment section using the following BibTeX entry:
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
