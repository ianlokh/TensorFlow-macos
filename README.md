# TensorFlow-macos
TensorFlow macos

Setting up environment
1. Install XCode Command Line Tools.  Note that python3 is installed by default with CLT.
2. Install Homebrew
3. Install Anaconda
4. Install miniforge through Homebrew
```commandline=bash
brew install miniforge
```
5. Setup zshrc.anaconda3 and zshrc.miniforge3 scripts
```commandline=bash
# Make a copy of the .zshrc file (from Anaconda install)
cp .zshrc zshrc.anaconda3
cp .zshrc zshrc.miniconda3
```
Edit zshrc.miniconda3 to replace
- /Users/<user's name>/anaconda3/bin/conda >> /opt/homebrew/Caskroom/miniforge/base/condabin/conda
- /Users/<user's name>/anaconda3/etc/profile.d/conda.sh >> /opt/homebrew/Caskroom/miniforge/base/etc/profile.d/conda.sh
- /Users/<user's name>/anaconda3/bin:$PATH >> /opt/homebrew/Caskroom/miniforge/base/bin:$PATH

This step creates 2 different scripts where you can toggle across different conda environments.
To switch environments in the command line:
```commandline=bash
source zshrc.anaconda3
source zshrc.miniconda3
```
6. Setup environment for Tensorflow Metal
```commandline=bash
source zshrc.miniconda3
conda create -n <environment name> python==<python version>
conda activate <environment name>
SYSTEM_VERSION_COMPAT=0 pip install tensorflow tensorflow-metal
```
7. Other optimisations
To install other Apple Silicon optimised packages, use the following command:
```commandline=bash
conda install -c conda-forge numpy "libblas=*=*accelerate"

```