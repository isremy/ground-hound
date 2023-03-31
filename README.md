# Ground-Hound
This repository includes a GridWorld class that provides an environment for training an agent that is tasked with
locating all the objects in the GridWorld environment, whether they are occluded by clutter or not.

## How to install Ground-Hound
After cloning this repository, you should first set up a local virtual environment in this repo so you can install all the necessary packages without them interfering with your other Python repos.

To create your virtual environments, do ```python3 -m venv /path/to/env```

For example, if I want to call my virtual environment "env", and want this environment in my current repo with path ```/home/my_repo```, I do ```python3 -m venv /home/my_repo/env```. This will create a new folder called ```env``` in my repository folder, which is what will contain all the libraries you install.  Once the environment is created, it needs to be activated. This is done with the command ```source [your_env_name]/bin/activate```. You can now install the packages within this repository and run the scripts. 

For more detailed documentation on Python virtual environments, see <a href=https://docs.python.org/3/library/venv.html>here</a>

With the environment activated, do

```pip install -r requirements.txt```

after activating the virtual environment.

## Using the test-harness
This repository uses PyTest for unit-testing many of the components of GroundHound. These tests all take the form of functions that are members of the ```TestHound``` class, located in ```test_hound.py```.  To run the whole harness for all the tests, do

```pytest test_hound.py```

PyTest also allows you to run individual tests, which takes the format of ```pytest test_hound.py::TestHound::[function name]```.  For example, if I wanted to test the basic grid-world class, I would do

```pytest test_hound.py::TestHound::test_basic_grid```