HF_MODEL=EleutherAI/gpt-neox-20b
HF_DATASET=Salesforce/fineweb_deduplicated
VERSION_NAME=test
TRAIN_DATA_FILES="data/part-00000-e9c46804-8d86-45cf-8ebc-4744ff73914d-c000.parquet"
for i in $(seq -f "%05g" 1 8)
do
    TRAIN_DATA_FILES=${TRAIN_DATA_FILES},data/part-${i}-e9c46804-8d86-45cf-8ebc-4744ff73914d-c000.parquet
done
VALID_DATA_FILES=data/part-01024-e9c46804-8d86-45cf-8ebc-4744ff73914d-c000.parquet

export ROOT_DIR="$(dirname "$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )")"

export PYTHONPATH="$ROOT_DIR:$PYTHONPATH"

python $ROOT_DIR/open_lm/misfitting/scripts/data_download.py \
    --output-dir $ROOT_DIR/misfitting/data/raw \
    --dataset $HF_DATASET \
    --train-data-files $TRAIN_DATA_FILES \
    --valid-data-files $VALID_DATA_FILES \
    --version-name $VERSION_NAME

python $ROOT_DIR/open_lm/misfitting/scripts/make_chunks.py \
    --input-files $ROOT_DIR/misfitting/data/raw/$HF_DATASET/$VERSION_NAME/valid.jsonl \
    --output-dir $ROOT_DIR/misfitting/data/preprocessed/$HF_MODEL/$HF_DATASET \
    --version-name valid_${VERSION_NAME} \
    --tokenizer $HF_MODEL \
    --num-workers 1 \
    --num-consumers 1 

python $ROOT_DIR/open_lm/misfitting/scripts/make_chunks.py \
    --input-files $ROOT_DIR/misfitting/data/raw/$HF_DATASET/$VERSION_NAME/train.jsonl \
    --version-name train_${VERSION_NAME} \
    --output-dir $ROOT_DIR/misfitting/data/preprocessed/$HF_MODEL/$HF_DATASET \
    --tokenizer $HF_MODEL \
    --num-workers 4 \
    --num-consumers 4

python -m open_lm.utils.make_wds_manifest --data-dir $ROOT_DIR/misfitting/data/preprocessed/$HF_MODEL/$HF_DATASET/chunk-size_2048-v_train_${VERSION_NAME}/

bash $ROOT_DIR/misfitting/scripts/train.sh 

