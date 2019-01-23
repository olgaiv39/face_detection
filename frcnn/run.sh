#!/usr/bin/env bash

pip install -r requirements.txt 

cd lib 

python setup.py build develop

chmod +x make.sh

./make.sh 

cd ..

python trainval.py 

python test.py
