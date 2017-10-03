# Project Sisyphus

This is an ongoing long-term project to develop a concise and easy-to-use package for the modeling and analysis of neural network dynamics. 

Code is written and upkept by: @davidbrandfonbrener @dbehrlic @ABAtanasov 


## TODO

  ### Translate the structure of tasks into object oriented code:
    i.e. make a class "task.py" so that each task (e.g. romo, rdm) extends this class

  ### Make it possible to set up a simulator.py without needing to read in weights saved to a file
    It should be easy for a user to manually construct a 3-neuron network without relying on tensorflow
    
  ### Clean up the model class
    So far we have been using a single "model" class for everything, and there are many redundant sub-methods

