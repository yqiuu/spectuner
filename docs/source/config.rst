Configuration
=============

The config dict is the starting point for all applications in Spectuner. It can
be created by

.. code-block:: python

   config = spectuner.load_default_config()

This creates an instance of the ``Config`` class, which inlcudes a coulpe of
user-friendly methods to update settings.

.. autoclass:: spectuner.Config
   :members: