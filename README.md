# Project Sisyphus

This is an ongoing long-term project to develop a concise and easy-to-use package for the modeling and analysis of neural network dynamics. 

Code is written and upkept by: @davidbrandfonbrener @dbehrlic @ABAtanasov 


## TODO

  ### @Dave Translate the structure of tasks into object oriented code:
    i.e. make a class "task.py" so that each task (e.g. romo, rdm) extends this class

  ### @Alex Make it possible to set up a simulator.py without needing to read in weights saved to a file
    It should be easy for a user to manually construct a 3-neuron network without relying on tensorflow
    
  ### @Daniel Add in any ipython notebooks that you've made over the past few months on this project
  
  ### @All add in more ipython notebooks to gain understanding of what analysis tools we need for compelling results
    Self-explanatory
    
  ### @Dave+Alex Clean up the model class
    So far we have been using a single "model" class for everything, and there are many redundant sub-methods

  ### @Alex Make a test directory
    Sync this package with travisCI so that after every new push, within an hour or so 
    we'll know that each network and analysis module still runs correctly
    This will also be more compelling for users to know that the code has coverage

