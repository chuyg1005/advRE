# for dataset in 'semeval' 'wiki80' 'tacred' 'retacred' 'tacrand'
for dataset in 'tacrand'
do
    echo "start process ${dataset}."
    python constants.py --data_dir data/${dataset}
    echo "finish process ${dataset}."
done