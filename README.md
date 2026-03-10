# pyrpoc v3

Author: Ishaan Singh, Zhang Group (https://sites.google.com/view/zhangresearchgroup)

This software was written for the Zhang group's RPOC and SRS microscopy system, see https://www.nature.com/articles/s41467-022-32071-z. It is meant to serve as a general software for microscopy that seamlessly integrates RPOC, to enable more widespread adoption of the technique. 
With any feedback or suggestions, please reach out to sing1125@purdue.edu.

## Basic Installation

The software is available on PyPI - make sure python 3.13 is in use with a virtual environment. Install using:
```
py -m venv your_env_name
your_env_name/scripts/activate
pip install pyrpoc
```

For development mode, ensure the code is downloaded/cloned and that you have navigated to root directory of this project (the folder on your system that contains this README.md, the pyproject.toml, and the pyrpoc/ folder), then run

```
pip install -e .
```

Once the virtual environment is active, the GUI can be opened with the simple command ```pyrpoc```. If the command does not work, please check your environment variables and ensure that you have set up the virtual environment correctly.
