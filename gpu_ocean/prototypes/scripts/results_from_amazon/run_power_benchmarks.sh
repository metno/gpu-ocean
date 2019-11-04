#!/bin/bash


set -x

steps=10000
iters=10
args=(
	CDKLM
	CTCS
	FBL
)

logfile="run_power_benchmarks.log"
statusfile="run_power_benchmarks_status.log"

echo "" > $logfile
echo "" > $statusfile

for arch in "opencl" "cuda"; do
	for arg in "${args[@]}"; do
		ARG=$(echo "$arg" | tr '[:lower:]' '[:upper:]')
		arg=$(echo "$arg" | tr '[:upper:]' '[:lower:]')

		echo "===========================================" >> $logfile
		echo "===========================================" >> $logfile
		echo "===========================================" >> $logfile
		echo "===========================================" >> $logfile
		echo "===========================================" >> $logfile
		echo `date` >> $logfile
		echo $arg - $arch >> $logfile
		echo `date` >> $statusfile
		echo $arg - $arch >> $statusfile
		echo "===========================================" >> $logfile
		echo "===========================================" >> $logfile
		echo "===========================================" >> $logfile
		echo "===========================================" >> $logfile
		echo "===========================================" >> $logfile
		python3 "power_benchmark_git_commits.py" \
			"amazon_v100/${arg}_${arch}_git_versions_amazon_v100.csv" \
			--python python3 \
			--run_benchmark_opt "--simulator ${ARG} --steps_per_download $steps --iterations $iters" \
			>> $logfile 2>&1
	done
done

tar zcvf "run_power_benchmarks_out_$(date --iso-8601=minutes).tgz" $(git ls-files --others --exclude-standard)

echo "######################################"
echo "######################################"
echo "################ DONE ################"
echo "######################################"
echo "######################################"
