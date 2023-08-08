#!/user/bin/env bash
# exit on error
set -0 errexit

pip install --upgrade pip
pip install --user -U nltk
pip install -r requirements.txt
