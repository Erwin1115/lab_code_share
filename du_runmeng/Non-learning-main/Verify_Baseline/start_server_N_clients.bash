#!/bin/bash

# The first three arguments are the number of times to run the script, the IP address, and the port
N=$1
IP=$2
PORT=$3

# Run 'python CSP_base.py' with the IP address and port
gnome-terminal -- bash -c "python CSP_base.py $IP $PORT; exec bash"

# Wait for 7 seconds
sleep 7

# Run 'python ES_base.py' N times with the IP address and port
for ((i=1; i<=$N; i++))
do
    gnome-terminal -- bash -c "python RE_base.py $IP $PORT; exec bash"
done