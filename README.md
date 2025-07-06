# pyrpoc

Author: Ishaan Singh, Zhang Group (https://sites.google.com/view/zhangresearchgroup)

This software was written for the Zhang group's RPOC and SRS microscopy system, see https://www.nature.com/articles/s41467-022-32071-z. It is meant to serve as a general software for microscopy that seamlessly integrates RPOC, to enable more widespread adoption of the technique.  Functionalities include full flexibility of galvo mirror scanning, multi-channel imaging, multi-channel RPOC mask design and application, and more. 
With any feedback or suggestions, please reach out to sing1125@purdue.edu.

## Basic Installation

The software is available as a package - make sure python 3.12 is in use, with a virtual environment if necessary (I am unsure exactly why 3.13 doesn't work, but a virtual environment is a clean way to resolve the issue). To do this with venv, first ensure that python 3.12 is installed on the system by running:

``` 
py -0
```

If a version of python 3.12 is installed, then run the following commands.

```
py -3.12 -m venv your_env_name
your_env_name/scripts/activate
pip install pyrpoc
```

For development mode, ensure the code is downloaded and that you have navigated to the folder containing pyproject.toml, then run

```
pip install -e .
```

Then, run the command below to open the GUI. 

```
pyrpoc
```

If the command does not work, it is likely an issue with the PATH variable - make sure that Python scripts are available to PATH.
