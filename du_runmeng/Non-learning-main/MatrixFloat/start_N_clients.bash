#!/bin/bash

# The first three arguments are the number of times to run the script, the IP address, and the port
N=$1
IP=$2
PORT=$3

# Source conda.sh
source ~/anaconda3/etc/profile.d/conda.sh

# Run 'python ES_float.py' N times with the IP address and port
for ((i=1; i<=$N; i++))
do
    gnome-terminal -- bash -c "python ES_float.py $IP $PORT; exec bash"
done