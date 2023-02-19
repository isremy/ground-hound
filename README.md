# Ground-Hound
This repository includes a GridWorld class that provides an environment for training an agent that is tasked with
locating all the objects in the GridWorld environment, whether they are occluded by clutter or not.

## How to install Ground-Hound
After cloning this repository, you should set up a local virtual environment in this repo and then do

```pip install -r requirements.txt```

after activating the virtual environment.

## Using the test-harness
This repository uses PyTest for unit-testing many of the components of GroundHound. These tests all take the form of functions that are members of the ```TestHound``` class, located in ```test_hound.py```.  To run the whole harness for all the tests, do

```pytest test_hound.py```

PyTest also allows you to run individual tests, which takes the format of ```pytest test_hound.py::TestHound::[function name]```.  For example, if I wanted to test the basic grid-world class, I would do

```pytest test_hound.py::TestHound::test_basic_grid```