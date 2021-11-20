#!/bin/bash

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=4,5

#Requires an already trained model with the foll. parameters such that only testing takes place, from the stored model
DATASET=W2_Dataset
LABEL_NAME_FILE=label_names_25_multiple.txt
TRAIN_CORPUS=train.txt
#TEST_LABEL=test_labels.txt not available
MAX_LEN=200
TRAIN_BATCH=32 #for mBert cased 16
ACCUM_STEP=2
EVAL_BATCH=128
GPUS=2
MCP_EPOCH=3
SELF_TRAIN_EPOCH=1

#fist time: not enough cat. indivative words found on W2, SOL: restart training again in line 32 from stored checkpoint
out_file = en_out.txt
python3 src/train.py --dataset_dir datasets/${DATASET}/ \
                        --label_names_file ${LABEL_NAME_FILE} \
                        --train_file ${TRAIN_CORPUS} \
                        --test_file ${TEST_CORPUS}  \
                        --out_file ${OUT_FILE}  \
                        --doc_topic_distribution ${TEST_DISTRIBUTIONS} \
                        --max_len ${MAX_LEN} \
                        --train_batch_size ${TRAIN_BATCH} --accum_steps ${ACCUM_STEP} --eval_batch_size ${EVAL_BATCH} \
                        --gpus ${GPUS} \
                        --mcp_epochs ${MCP_EPOCH} --self_train_epochs ${SELF_TRAIN_EPOCH} 

for value in en fr it port ger translated_test_ger_to_en
do
    echo $value
    TEST_CORPUS=test_${value}.txt
    OUT_FILE=${value}_out.txt # model predictions on the test corpus
    TEST_DISTRIBUTIONS=${value}_test_topic_dist.txt # name of the file containing the predicted distributions of topics on the test doc
    python3 src/train.py --dataset_dir datasets/${DATASET}/ \
                        --label_names_file ${LABEL_NAME_FILE} \
                        --train_file ${TRAIN_CORPUS} \
                        --test_file ${TEST_CORPUS}  \
                        --out_file ${OUT_FILE}  \
                        --doc_topic_distribution ${TEST_DISTRIBUTIONS} \
                        --max_len ${MAX_LEN} \
                        --train_batch_size ${TRAIN_BATCH} --accum_steps ${ACCUM_STEP} --eval_batch_size ${EVAL_BATCH} \
                        --gpus ${GPUS} \
                        --mcp_epochs ${MCP_EPOCH} --self_train_epochs ${SELF_TRAIN_EPOCH} 
done
