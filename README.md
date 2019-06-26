# Reinforcement Learning Labs
Exercises in reinforcement learning.


## Setup

### Get the Code
1.  Clone Repository.
    ```sh
    git clone https://github.com/djjh/reinforcement-learning-labs
    ```
2.  Go to working directory. All steps assume you are working from `/path/to/repo-directory`.
    ```sh
    cd reinforcement-learning-labs
    ```

### Install Pyenv
```sh
brew install pyenv
```

### Install Python Version

1.  Install required headers on Mac OSX Mojave. Without this step, you may see this error while installing python versions: `'zipimport.ZipImportError: can't decompress data; zlib not available`.
    ```sh
    open /Library/Developer/CommandLineTools/Packages/ macOS_SDK_headers_for_macOS_10.14.pkg
    ```
    Sources:
    *   <https://github.com/pyenv/pyenv/issues/1219#issuecomment-487206619>
2.  Install python 3.6.3. On Mac OSX you must also enable a framework build to use matplotlib.
    ```sh
    env PYTHON_CONFIGURE_OPTS="--enable-framework" pyenv install 3.6.3
    ```
    Sources:
    *   <https://matplotlib.org/faq/osx_framework.html>
    *   <https://github.com/pyenv/pyenv/wiki#how-to-build-cpython-with-framework-support-on-os-x>


### Setup Virtual Python Environment
1.  Enable the new version of python you just installed.

    ```sh
    pyenv local 3.6.3
    ```
    If you want to use this new version by default, then:
    ```sh
    pyenv global 3.6.3
    ```
2.  Create the virtual environment. Note that you must use `venv`
    to be able to use the framework build of python you just installed.
    ```sh
    cd /path/to/repo-directory
    python -m venv .
    ```
3.  Enable the virtual environment.
    ```sh
    source bin/activate
    ```
4.  Install dependencies using pip.
    ```sh
    pip install -e .
    ```

## Usage

### Go to working directory
All steps assume you are in working directory `/path/to/repo-directory`.
```sh
cd reinforcement-learning-labs
```

### Enable Python Version
```sh
pyenv local 3.6.3
```
### Enable Virtual Environment
```sh
source bin/activate
```
### Run Algorithms
Each practice algorithm implementation consists of a single file.

To run an algorithm:

```sh
python practice/<XX_short_algorithm_description>.py
```


## Development

### Notes
python import traps: <http://python-notes.curiousefficiency.org/en/latest/python_concepts/import_traps.html>
