# Meteor
Meteor is an ease of use tool for automated spectral line identification.

## Tutorial

### Preprocessing
The input the code should one or a list of text files with two columns. The first column should be frequency in a unit of MHz, and the second column should be temperature in a unit of K. The baseline line of the spectrum **must** be flat.
### Configuration
Run
```
meteor-config workspace
```
The code above will create the config files in a directory named as `workspace`. The config files are YAML files.

#### Compulsory settings
1. Set the path to the spectrum files in `workspace/config.yml`, e.g.
```
files:
  - XXX/spec_0.dat
  - XXX/spec_1.dat
```
2. Set telescope parameters in `workspace/config.yml`. For single dish telescopes, set `Inter_Flag: True` and `TelescopeSize` to the diameter of the telescope in a unit of meter. For interferometers, set `Inter_Flag: True` and provide `BMIN`, `BMAJ`, and `BPA`.
3. Set ``prominence`` in `workspace/config.yml`. The is the critical parameter to identify peaks. The code uses `find_peaks` from Scipy to find peaks. See the [document](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html) for its definition. We recommend setting ``prominence`` to a 4-fold RMS. If the spectra have different RMS, `prominence` can be a list, e.g.
```
peak_manager:
  prominence: [0.01, 0.02]
```
4. Set `tBack` in `workspace/config.yml` to the background temperature if it is not zero.
5. Set the range of the source size in `workspace/config_opt.yml` according to the telescope parameters. The numbers are in logarithmic scale by default. For example,
```
bounds:
  theta: [0.7, 2.3]
```
6. Set `n_process` in `workspace/config_opt.yml`. This is the number of processes and should be a multiple of `nswarm` and smaller than `nswarm`.

#### Optional setting
1. Set `n_trail` in `workspace/config_opt.yml`. The code runs the optimizer `n_trail` times. Larger `n_trail` may lead to better results but longer runtime.
2. Set `molecules` in `workspace/species.yml`. The code provides several commonly observed molecules by default. Users are allow to use their own molecule list. Ensure the given molecule names are consistent with those defined in the CDMS database. In addition, set `molecules: null` to explore all molecules in the database in the given frequency range.

### Running the pipeline
Run
```
meteor-run workspace workspace/results
```
The code above will save the results in ``workspace/results``.



