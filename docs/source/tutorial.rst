Tutorial
========

Preprocessing
-------------
The input of the code should be one or a list of text files with two columns.
The first column should be frequency in a unit of MHz, and the second column
should be temperature in a unit of K. The baseline of the spectrum **must**
be flat.

Configuration
-------------

   .. code-block:: bash

      spectuner-config workspace

The code above will create config files in a directory named as ``workspace/``.
All config files are YAML files.

Compulsory settings
^^^^^^^^^^^^^^^^^^^
#. Set the path to the spectrum files in ``workspace/config.yml``, e.g.

   .. code-block:: yaml

    files:
      - XXX/spec_0.dat
      - XXX/spec_1.dat

#. Set telescope parameters in ``workspace/config.yml``. For single dish
   telescopes, set ``Inter_Flag: False`` and ``TelescopeSize`` to the diameter
   of the telescope in a unit of meter. For interferometers, set
   ``Inter_Flag: True`` and provide ``BMIN``, ``BMAJ``, and ``BPA``.
#. Set ``prominence`` in ``workspace/config.yml``. The is the critical parameter
   to identify peaks. The code uses ``find_peaks`` from Scipy to find peaks.
   See the `link <https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html>`__
   for details. We recommend setting ``prominence`` to a 4-fold RMS. If the
   spectra have different RMSs, ``prominence`` can be a list, e.g.

   .. code-block:: yaml

    peak_manager:
      prominence: [0.01, 0.02]

#. Set ``tBack`` in ``workspace/config.yml`` to the background temperature if it
   is not zero.
#. Set ``vLSR`` in ``workspace/config.yml`` if the input spectra are not
   corrected to the rest frame.
#. Set the range of the source size in ``workspace/config.yml`` according to
   the telescope parameters. The numbers are in logarithmic scale by default.
   For example,

   .. code-block:: yaml

    bounds:
      theta: [0.7, 2.3]

   The scale of the parameters, i.e. linear or logarithmic, can be specified in
   ``workspace/config.yml``.

#. Set ``n_process`` in ``workspace/config.yml``. This is the number of
   processes and should be a multiple of ``nswarm`` and smaller than ``nswarm``.
   We recommend setting the ``nswarm`` to a value between 24 and 32.

Optional setting
^^^^^^^^^^^^^^^^
#. Set ``n_trail`` in ``workspace/config.yml``. The code runs the optimizer
   ``n_trail`` times. Larger ``n_trail`` may lead to better results but longer
   runtime.
#. Set ``molecules`` in ``workspace/species.yml``. The code provides several
   commonly observed molecules by default. Users are allowed to set their own
   molecule list. Ensure the given molecule names are consistent with those
   defined in the CDMS database. In addition, set ``molecules: null`` to explore
   all molecules in the database in the given frequency range.
#. Set ``iso_mode`` in ``workspace/species.yml``, which specifies the way to
   deal with isotoplogues

   - ``combine``: Collect all possible isotoplogues and fit them jointly.
   - ``separate``: Collect all possible isotoplogues and fit them separately.
   - ``manual``: Only collect isotoplogues given by ``molecules``.

Running the pipeline
--------------------

   .. code-block:: bash

    spectuner-run workspace workspace/results

The code above will save the results in ``workspace/results``.

Check the results
-----------------
If everything works correctly, the results will be saved in
``workspace/results``, e.g.

.. code-block:: text

   .
   └── results
      ├── combine
      │   ├── combine.pickle
      │   ├── identify_combine.pickle
      │   ├── identify.pickle
      │   ├── OCS;v=0;_2.pickle
      │   └── tmp_CH3COCH3;v=0;_0.pickle
      └── single
          ├── CH3COCH3;v=0;.pickle
          ├── CH3OCHO;v=0;.pickle
          ├── identify.pickle
          └── OCS;v=0;.pickle

The ``results/single/`` directory saves all individual fitting results. The
``results/combine/`` directory saves the combined results. Specifically,
``combine.pickle`` saves the combined spectrum, ``identify_combine.pickle``
saves the identification result of the combined spectrum, and
``identify.pickle`` saves the identification results of all candicates.

An example is given in ``examples/`` in the repository. Use the following code
to plot the fitting result (assume that you are in the ``examples/`` directory).

.. code-block:: python

   import pickle

   import spectuner
   import matplotlib.pyplot as plt


   obs_data = spectuner.load_preprocess(["mock_data/spec.dat"], T_back=0.)
   res = pickle.load(open("workspace/results/combine/identify_combine.pickle", "rb"))

   freq_data = spectuner.get_freq_data(obs_data)
   freq_per_row = 1000 # MHz
   y_min = -0.1
   y_max = 3.

   plot = spectuner.SpectralPlot(freq_data, freq_per_row)
   # Plot the mock spectrum
   plot.plot_spec(freq_data, spectuner.get_T_data(obs_data), color="k")
   # Plot the fitting spectrum
   plot.plot_T_pred(res, y_min, y_max, kwargs_spec={"color": "r", "linestyle": "--"}, fontsize=10)

   plot.set_ylim(y_min, y_max)
   for ax in plot.axes:
      ax.set_ylabel("Intensity [K]")
   plot.axes[-1].set_xlabel("Frequency [MHz]")

   plt.show()

The candicate results are saved in form ``dict``. The same method may be used
to plot a candicate result.

Manual review
-------------
Users are able to modify the combined result. Specifically, users can indicate
the species to be included or excluded in ``workspace/modify.yml``. In the
example above, users can include OCS;v=0; (``id=2``) by setting

   .. code-block:: yaml

      include_id_list: [2]

Then, run
   .. code-block:: bash

      spectuner-modify workspace workspace/results