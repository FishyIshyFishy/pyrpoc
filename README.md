# pyrpoc v3.1

Author: Ishaan Singh, Zhang Group (https://sites.google.com/view/zhangresearchgroup). With any feedback or suggestions, please reach out to sing1125@purdue.edu.

The instructions for install below are only for windows, as your instrument control system likely is a windows computer. If you would like instructions for a linux or mac system, please email.

## Installation

This project uses a free tool called `uv` that handles everything for you. To get set up with it on windows, install the instuctions below. 

1. Click the Windows start button and search for powershell. 
2. In the list that appears, right-click on powershell and choose **run as administrator**. If a popup comes up to ask *"Do you want to allow this app to make changes?"*, click yes.
3. Type the word **PowerShell**.
4. Copy the line below, paste it into that window, and press enter to run the command. This command is straight from [Astral's instructions for installing uv](https://docs.astral.sh/uv/getting-started/installation/#__tabbed_1_2).

   ```
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```
5. After install is complete, fully close powershell. 

Now that `uv` is installed, you can install the pyrpoc software itself.

1. Open PowerShell again (Windows start button → search **PowerShell** → click it). You do **not** need administrator mode. 
2. Copy the line below, paste it into the window (right-click inside the window to paste), and press **Enter**:

   ```
   uv tool install pyrpoc
   ```

3. Wait until the text stops scrolling. This step downloads the correct version of Python and every component the software needs, so the first install can take several minutes.
## Running the software

Any time you want to open pyrpoc:

1. Open PowerShell 
2. Type the command below

   ```
   pyrpoc
   ```
That's all.

## Updating to a newer version

When a new version is released, open PowerShell and run:

```
uv tool upgrade pyrpoc
```