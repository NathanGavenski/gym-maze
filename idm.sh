#!/bin/bash
if [[ $1 != '' ]]
then
    echo "Running train on virtual screen $1"
    xvfb-run -a -s "-screen $1 1400x900x24" python IDM.py
else
    echo "Running train on virtual screen 0" 
    xvfb-run -a -s "-screen 0 1400x900x24" python IDM.py
fi
