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
The code below demonstrates how to perform pixel-by-pixel fitting. It can employ
a neural network to provide initial guesses, which significantly improves the
fitting process. To use the neural network, please download the weights file
from `Hugging Face <https://huggingface.co/yqiuu/Spectuner-D1/tree/main>`__. We
highly recommend using a GPU for inference; otherwise, please set
:code:`device="cpu"`.

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
        need_spectra=True
    )

    # Run
    fname_cube = "XXX.h5" # HDF file created in preprocessing
    save_name = "XXX_results.h5" # Path to save the results
    spectuner.fit_pixel_by_pixel(config, fname_cube, save_name)


Converting the fitting results to FITS files
--------------------------------------------
This step is optional. The code allows us to convert the fitting results to FITS
files, which can then be visulaized using `CARTA <https://cartavis.org/>`__.

.. code-block:: python

    import spectuner

    # Directory to save the FITS files
    save_dir = "fits/"

    # HDF file created in preprocessing
    fname_cube = "XXX.h5"

    # Path to the saved results
    fname_result = "XXX_result.h5"

    # Convert
    cube_mgr = spectuner.HDFCubeManager(fname_cube)
    cube_mgr.obs_data_to_fits(save_dir, overwrite=True)
    cube_mgr.pred_data_to_fits(fname_result, save_dir, overwrite=True)
