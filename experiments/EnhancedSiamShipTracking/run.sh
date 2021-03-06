ROOT=`git rev-parse --show-toplevel`
export PYTHONPATH=$ROOT:$PYTHONPATH

mkdir -p logs

python -u $ROOT/tools/train_siammask.py \
    --config=config.json -b 16 \
    -j 20 \
    --epochs 20 \
    --log logs/log.txt \
    2>&1 | tee logs/train.log

# bash test_all.sh -s 1 -e 20 -d VOT2018 -g 4
