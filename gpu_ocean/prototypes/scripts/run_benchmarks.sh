FILES=`ls $@`

echo "Will run the following: "
for FILE in $FILES; do 
    echo python3 benchmark_git_commits.py $FILE --run_benchmark_opts="--simulator=FBL --steps_per_download 1000 --iterations 5" 
    echo python3 power_benchmark_git_commits.py $FILE --run_benchmark_opts="--simulator=FBL --steps_per_download 1000 --iterations 5" 
done


for FILE in $FILES; do 
    python3 benchmark_git_commits.py $FILE --run_benchmark_opts="--simulator=FBL --steps_per_download 1000 --iterations 5" 
done

for FILE in $FILES; do 
    python3 power_benchmark_git_commits.py $FILE --run_benchmark_opts="--simulator=FBL --steps_per_download 1000 --iterations 5" 
done
