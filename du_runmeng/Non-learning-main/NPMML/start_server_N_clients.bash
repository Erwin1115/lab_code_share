#!/bin/bash

# The first three arguments are the number of times to run the script, the IP address, and the port
N=$1
IP=$2
PORT=$3

# Run 'python CSP_npmml.py' with the IP address and port
gnome-terminal -- bash -c "python CSP_npmml.py $IP $PORT; exec bash"

# Wait for 7 seconds
sleep 7

# Run 'python ES_npmml.py' N times with the IP address and port
for ((i=1; i<=$N; i++))
do
    gnome-terminal -- bash -c "python ES_npmml.py $IP $PORT; exec bash"
done