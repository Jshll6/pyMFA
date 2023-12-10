# pyMFA

#Read Me

## Installation and Execution


### Cloning repository and installing dependencies

#### 1. Clone the repository into your desired local directory

```git clone https://github.com/Jshll6/pyMFA.git```

#### 2. Navigate to local directory with the cloned repository

```cd pyMFA```


#### 3. Inside the code directory, create a fresh virtual environment using at least python version 3.5.

```python -m venv venv```

*If the python command fails on Windows:* Make sure you have python configured in your environment variables (see also: [Python Docs - Setting Environment Variable](https://docs.python.org/3/using/windows.html#excursus-setting-environment-variables) )

#### 4. Activate your virtual environment - use instructions based on your operating system.

- If on **Windows** using **powershell**:**

```Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process -Force```

```./venv/Scripts/activate.ps1```

- If on **Windows** using **command prompt (cmd)** run:

```.\venv\Scripts\activate.bat```

- If on **Mac** or **Linux** run:

```source venv/bin/activate```

You should now see your virtual environment is activated, which is indicated by `(venv)` in the start of your current prompt.

#### 5. Lastly, you install the correct versions of the package dependencies into your virtual environment. Make sure you are still in the **pyMFA** directory.

```pip install -r requirements.txt```


### Running the code

#### 1. Optional: Update configuration file

Ensure to update the configuration file **`config.yaml`** to use the input filenames and output filenames that are applicable for your case.

If you do not edit the file, the script will run with the prepared default scenario and generate some example sankey charts for that scenario.

#### 2. Execute code from virtual environment
Navigate to the **pyMFA** directory on your system.

If you have not yet activated your virtual environment yet, follow **step 5** in [Cloning repository and installing dependencies](#cloning-repository-and-installing-dependencies).
Once you have activated the virtual environment (your prompt shows `(venv)` in the start), run:

```python -W ignore pyMFA.py```
