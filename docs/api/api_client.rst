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
The cloud client is provided by the separate ``xplainable-client`` package,
which allows you to connect to Xplainable Cloud and query the API, enabling 
you to manage your account, models, and deployments within Python.

Installation
-------------------------------
To use the cloud client functionality, you need to install the external
client package::

   pip install xplainable-client

Initialising a session
-------------------------------
To initialise a session, you first must generate an API key at 
`xplainable cloud <https://beta.xplainable.io>`.

The client functionality is automatically imported when available::

   import xplainable as xp
   import os
   
   # Initialise your session
   xp.initialise(api_key=os.environ['XP_API_KEY'])

Querying the API
-------------------------------
When you connect successfully to Xplainable Cloud, you can use the client
to query the API. The client is accessible by running::
   
      import xplainable as xp
      import os
      
      # Initialise your session
      xp.initialise(api_key=os.environ['XP_API_KEY'])

      # Query the API
      xp.client.list_models()

For detailed API documentation, please refer to the ``xplainable-client`` 
package documentation.