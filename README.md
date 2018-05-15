# Project Sisyphus

This is an ongoing long-term project to develop a concise and easy-to-use package for the modeling and analysis of neural network dynamics. 

Code is written and upkept by: @davidbrandfonbrener @dbehrlic @ABAtanasov 


## TODO

- Tasks. Move all tasks over to object oriented task class

- Throw errors on bad inputs. Try/except for params

- Test/expand simulator objects

- Demos

- Documentation

- Project names

- Does destruct destruct all models?

- Save params?

- Clean up dale ratio handling?



## Params

### Task:
- N_batch
- N_in
- N_out
- T
- dt
- tau
- stim_noise

  
(plus other params specific to the task)

implicit params:
  - alpha
  - N_steps


### Model:
- name
- N_rec
- N_in
- N_out
- N_steps
- dt
- tau
- dale_ratio
- rec_noise
- load_weights_path
- initializer 
-  trainability (many boolean variables)

implicit params:
  - alpha
  - N_batch


### Train:
- learning_rate
- training_iters
- loss_epoch
- verbosity
- save_weights_path
- save_training_weights_epoch
- training_weigts_path
- generator_function
- optimizer
- clip_grads