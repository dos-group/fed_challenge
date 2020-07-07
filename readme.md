This repository contains the code for the paper <link>.

## Run the script
Python 3.6 is required to run the script.
To run the script simply do:
<br>
<br>
`python code/run.py`
<br>
<br>
All 10000 series will be predicted. This might take a while 
(~40 hours on one Nvidia Titan GPU, will run forever on CPU). 
<br>
<br>
Alternatively it is possible to predict a subset of series.
<br>
<br>
`python code/run.py --start 0 --end 10`
<br>
<br>
This can be used for testing or for parallelization by running
this script several times and defining respective start and
end indices.
<br>

Example:
* `python code/run.py --start 0 --end 2500`
* `python code/run.py --start 2500 --end 5000`
* `python code/run.py --start 5000 --end 7500`
* `python code/run.py --start 7500 --end 10000`
<br>
<br>

This will produce 4 submission files in data folder.
<br>
