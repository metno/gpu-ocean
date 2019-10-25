#!/bin/bash
#bash run_benchmarks.sh met_p100/*.csv > run_log.log 2>&1

FILES=`ls $@`

echo "Will run the following: "
for FILE in $FILES; do 
    SIM=$(echo $FILE | sed 's/met_p100\///' | cut -d '_' -f 1 | tr '[:lower:]' '[:upper:]')
    echo python3 benchmark_git_commits.py $FILE --run_benchmark_opts="--simulator=$SIM --steps_per_download 1000 --iterations 5" 
    echo python3 power_benchmark_git_commits.py $FILE --run_benchmark_opts="--simulator=$SIM --steps_per_download 1000 --iterations 5" 
done



for FILE in $FILES; do 
    SIM=$(echo $FILE | sed 's/met_p100\///' | cut -d '_' -f 1 | tr '[:lower:]' '[:upper:]')
    python3 benchmark_git_commits.py $FILE --run_benchmark_opts="--simulator=$SIM --steps_per_download 1000 --iterations 5" 
done

for FILE in $FILES; do 
    SIM=$(echo $FILE | sed 's/met_p100\///' | cut -d '_' -f 1 | tr '[:lower:]' '[:upper:]')
    python3 power_benchmark_git_commits.py $FILE --run_benchmark_opts="--simulator=$SIM --steps_per_download 1000 --iterations 5" 
done
