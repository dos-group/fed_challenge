## Run the script
Python 3.6 is required to run the script.
To run the script simply do:
`python run.py`

All 10000 series will be predicted. This might take a while 
(~40 hours on one Nvidia Titan GPU, will run forever on CPU). 

`python3 run.py --start 0 --end 10`
Alternatively it is possible to predict a subset of series.
This can be used for testing or for parallelization by running
this script several times and defining respective start and
end indices.
<br>

Example:
* `python3 run.py --start 0 --end 2500`
* `python3 run.py --start 2500 --end 5000`
* `python3 run.py --start 5000 --end 7500`
* `python3 run.py --start 7500 --end 10000`
<br>

This will produce 4 submission files in data folder.
<br>