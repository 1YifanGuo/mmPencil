NPROC_PER_NODE=8 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
swift infer \
    --adapters PATH_TO_ADAPTER_CHECKPOINT \
    --val_dataset 'mmPencil_dataset/text/test.jsonl' \
    --infer_backend pt \
    --max_batch_size 8