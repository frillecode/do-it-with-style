#!/usr/bin/env bash

VENVNAME=dsexamvenv

python3 -m venv $VENVNAME
source $VENVNAME/bin/activate

pip --version
pip install --upgrade pip
pip --version

# problems when installing from requirements.txt
pip install ipython
pip install jupyter
pip install matplotlib
pip install opencv-python

python -m ipykernel install --user --name=$VENVNAME

test -f requirements.txt && pip install -r requirements.txt

deactivate
echo "build $VENVNAME"

