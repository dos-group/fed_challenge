## Requirements:
* needs access to: ../data/solution_template.csv
* needs access to: ../data/series.db
<br>

## To run the script use:
`python3 script_thorsten.py --start 0 --end 10000 --epochs 6 --device "cuda:0"`
All 10000 submission series will be calculated in one file. 
-> This will produce 1 submission file in the folder of the script
<br>

To distribute the tasks among multiple machines use:
<br>
##for example 4
* `python3 script_thorsten.py --start 0 --end 2500 --epochs 6 --device "cuda:0"`
* `python3 script_thorsten.py --start 2500 --end 5000 --epochs 6 --device "cuda:0"`
* `python3 script_thorsten.py --start 5000 --end 7500 --epochs 6 --device "cuda:0"`
* `python3 script_thorsten.py --start 7500 --end 10000 --epochs 6 --device "cuda:0"`
<br>
-> This will produce 4 submission files in the folder of the scipt. These need to be gathered after.
Output format = submission_results.to_csv("submission_thorsten_from{}_to{}.csv".format(start, end), header=False, index=False)
<br>
6 epochs needs about 15 seconds (on trpe)... so 8 epochs are maybe also ok
