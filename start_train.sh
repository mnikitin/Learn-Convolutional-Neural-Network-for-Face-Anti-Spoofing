folds_num=5
declare -a scales=("1.0" "1.4" "1.8" "2.2" "2.6")

for scale in "${scales[@]}"
do
    for k in $(seq 1 $folds_num)
    do
        python start_train.py $scale $k
    done
done

