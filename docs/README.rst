Install extra python requirements to be able to build html docs locally::

    pip install -r requirements.txt

Install `Pandoc <http://pandoc.org>`_ 2.x for notebook conversion.

To rebuild HTML docs, run ``make html``, then open
``_build/html/index.html`` file.
Note that html docs are not checked it,
rebuilding them is useful only to check how they will be rendered.

To sync tutorials with IPython notebooks run ``update-notebooks.sh`` script,
then rebuild the docs.
