#!/bin/bash

home=`dirname $(readlink -e $0)`

start=$1
end=$2

device=$3

for i in $(seq $start $end); do 
    screen -S "fed_$device_$i" -d -m sh -c "python3.6 $home/script.py --index $i --device $device; exec bash"
done

