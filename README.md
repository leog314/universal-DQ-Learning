# universal-DQ-Learning
=======
**Requirements:** 
    Python and pip
    
to install all needed libraries run:

    pip install torch torchvision gymnasium matplotlib numpy
    
  Also you'll need some gymnasium-environments to test the algorithm on, for example the Box2d-env (https://gymnasium.farama.org/environments/box2d/).
  Because of this, you probably need to install additional packages like (for Box2d):
  
    pip install swig
    pip install "gymnasium[box2d]"

  Needed packages are listed at the specific websites of the environments (see also: https://gymnasium.farama.org/)

Notice that the installation of the packages may be different on Windows. It was so far only tested on Arch Linux.
Also the gymnasium environments might make some difficulties when installing them due to pip package errors. 

**Usage:**

To execute the program run the file ./general_training.py:

    python general_training.py

The implementation will automatically use the Lunar Lander task, with the preset action and state spaces.
When the program starts, it will detect if a gpu is available, otherwise it will ask the user how many cpu cores the program should be using.

To modify the environment, take a look on the appropriate website (https://gymnasium.farama.org/environments/) and modify the following variables:

    action_space = 2 # if the website states something like "Action Space: Discrete(2)"
    state_shape = 4 # use 4 for something like: Box([-4.8 -inf -0.41887903 -inf], [4.8 inf 0.41887903 inf], (4,), float32) (size 4)
    
    build = "CartPole-v1" # use whetever the website states (the latest version should be best)

    # you might also want to change the filename of torch.save()

The program might not run on all of the environments without modifications (due to some irregularities in the gymnasium api), however with the help of the website and the python interpreter, it should be easy to make some small adjustments if needed (e.g. adding extra paramters).
