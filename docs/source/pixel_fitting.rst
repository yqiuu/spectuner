.. include:: ./_aliases.rst

Pixel-by-pixel fitting
======================
Spectuner provides functionality for efficient pixel-by-pixel fitting using a
neural network. In this tutorial, we will learn how to use this functionality
step-by-step.

Preprocessing
-------------
Let's assume that we want to perform pixel-by-pixel fitting for the following
FITS files:

.. code-block:: bash

    XXX_spw1_line.fits
    XXX_spw1_continuum.fits
    XXX_spw2_line.fits
    XXX_spw2_continuum.fits

The first step is to perform preprocessing using the following code:

.. code-block:: python

    import spectuner

    file_list = [
        spectuner.CubeItem(
            line="XXX_spw1_line.fits",
            continuum="XXX_spw1_continuum.fits",
        ),
        spectuner.CubeItem(
            line="XXX_spw2_line.fits",
            continuum="XXX_spw2_continuum.fits",
        )
    ]
    pipline = spectuner.CubePipeline(
        noise_factor_local=6.,
        number_cut=3,
        v_LSR=0.
    )
    pipline.run(file_list, "XXX.h5")


This converts the FITS files into a compact HDF file. |CubePipeline| excludes
noisy pixels and estimates the RMS noise for each cube, which we use in the
subsequent fitting process. We can adjust the parameters
:code:`noise_factor_local` and :code:`number_cut` to determine which pixels to
exclude; if :code:`number_cut=0`, no pixels are excluded. Furthermore, it is
important to set the LSR velocity :code:`v_LSR` correctly, as our neural network
can only handle velocity offsets within the range of -12 km/s to +12 km/s.


Performing the fitting
----------------------
The code below demonstrates how to perform pixel-by-pixel fitting. We can employ
a neural network to provide initial guesses, which significantly improves the
fitting process. To use the neural network, please download the weights file
from `Hugging Face <https://huggingface.co/yqiuu/Spectuner-D1/tree/main>`__. We
highly recommend using a GPU for inference; otherwise, please set
:code:`device="cpu"`.

There are two types of loss functions available for pixel-by-pixel fitting:

#. :code:`"chi2"`: The :math:`\chi^2` loss function, which works well if line
   blending is not significant.
#. :code:`"pm"`: The peak matching loss function, which should be used if line
   blending is significant.

The example below fits the spectra of CH3OH. Update
:code:`species` to fit different species. Multiple species are supported.

.. code-block:: python

    import spectuner

    # Configuration
    config = spectuner.load_default_config()

    # Set the path to the spectroscopic database
    fname_db = "path/to/the/cdms/database"
    config.set_fname_db(fname_db)

    # Set information for inference
    config.set_inference_model(
        ckpt="path/to/the/network/weights/file"
        device="cuda:0",
        batch_size=64
    )

    # Set the species to fit
    species = ["CH3OH;v=0;"]
    config.set_pixel_by_pixel_fitting(
        species=species,
        loss_fn="chi2",
        need_spectra=True
    )

    # Run
    fname_cube = "XXX.h5" # HDF file created in preprocessing
    save_name = "XXX_results.h5" # Path to save the results
    spectuner.fit_pixel_by_pixel(config, fname_cube, save_name)

We can use :code:`spectuner.print_h5_structure` to check the structure of the
output file.

.. code-block:: python

    spectuner.print_h5_structure(save_name)

This may give the following output:

.. code-block:: text

    CH3OH;v=0;/
    N_tot (shape: (XXX,), type: float32)
    T_ex (shape: (XXX,), type: float32)
    T_pred/
        0 (shape: (XXX, XXXX), type: float32)
        1 (shape: (XXX, XXXX), type: float32)
    delta_v (shape: (XXX,), type: float32)
    theta (shape: (XXX,), type: float32)
    v_offset (shape: (XXX,), type: float32)
    score/
    fun (shape: (XXX,), type: float32)
    nfev (shape: (XXX,), type: int32)

We can access the fitting results using |HDFCubeManager|. Note that the
initialization of |HDFCubeManager| requires the HDF file of the observed data.

.. code-block:: python

    import spectuner

    # HDF file created in preprocessing
    fname_cube = "XXX.h5"

    # Path to the saved results
    fname_result = "XXX_result.h5"

    cube_mgr = spectuner.HDFCubeManager(fname_cube)
    # Load the predicted excitation temperature
    T_ex = cube_mgr.load_pred_data(fname, "CH3OH;v=0;/T_ex")
    # Load the predicted spectra of the first spectral window
    T_pred = cube_mgr.load_pred_data(fname, "CH3OH;v=0;/T_pred/0")


Converting the fitting results to FITS files
--------------------------------------------
This step is optional. We may use |HDFCubeManager| to convert the fitting
results to FITS files, which can then be visulaized using
`CARTA <https://cartavis.org/>`__. Note that the LSR velocity is corrected for
the predicted spectra. By setting :code:`add_v_LSR=True`, we add the LSR
velocity to the predicted spectra so that the results are comparable with the
observation.

.. code-block:: python

    # Directory to save the FITS files
    save_dir = "fits/"

    # Convert
    cube_mgr.pred_data_to_fits(fname_result, save_dir, add_v_LSR=True, overwrite=True)
