# pyrpoc v3.1
With any feedback or suggestions, please reach out to sing1125@purdue.edu.

Author: Ishaan K. Singh
Contributors: Callum Appel, Karsten J. Mohn
Zhang Group (https://sites.google.com/view/zhangresearchgroup). 



## Installation
The instructions here are for windows. This project uses `uv`: to get set up, follow  [Astral's instructions for installing uv](https://docs.astral.sh/uv/getting-started/installation/#__tabbed_1_2). Then, with `uv` installed:

1. Open PowerShell (Windows start button, search PowerShell, click it). You do not need administrator mode. 
2. Enter the command `uv tool install pyrpoc`
3. The first install can take several minutes. Let it run, then you are good to go! To upgrade later, use `uv tool upgrade pyrpoc`

Then, any time you want to open pyrpoc, open the command line and type the command `pyrpoc`. 

## Developer installation

1. Clone the repository and navigate to the project root.
2. Create the virtual environment and install the project in editable mode along with the development dependencies with `uv sync`
3. Run the software from the clone with `uv run pyrpoc`. Use `uv run pyrpoc --simulated` when running on a personal computer to have simulated programs/acquisitions/etc. . 
