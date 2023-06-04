Cloud Client
=========================

What is Xplainable Cloud?
-------------------------------
Xplainable Cloud is a hosted service that allows you to persist and load 
models and preprocessing pipelines and collaborate on them within teams 
and organisations. Persisted models are also able to be deployed as API 
endpoints in seconds. The cloud service is accessible via a web interface to 
manage organisations, teams, and users and provides an excellent interface for 
visualising model explainers and metrics. You can find more information about
Xplainable Cloud at https://www.xplainable.io.


What is the Cloud Client?
-------------------------------
The cloud client is built into the xplainable python package, allowing you to
connect to Xplainable Cloud and query the API, enabling you to manage
your account, models, and deployments within Python.


Initialising a session
-------------------------------
To initialise a session, you first must generate an API key at
https://app.xplainable.io.

.. automodule:: xplainable.client.init
   :members:
   :undoc-members:
   :show-inheritance:


Querying the API
-------------------------------
When you connect successfully to Xplainable Cloud, you can use the client
to query the API. The client is accessible by running::
   
      import xplainable as xp
      
      # Initialise your session
      xp.initialise()

      # Query the API
      xp.client.list_models()

.. autoclass:: xplainable.client.client.Client
    :members:
    :undoc-members:
    :show-inheritance: