#!/usr/bin/env bash

set -o xtrace

split="${1:-}"
num_gpus="${num_gpus:-$(nvidia-smi -L | awk 'END {print NR}')}"

case "$split" in
    1 | 2 | 3) ;;
    *)
        echo "usage: $0 {1|2|3}" >&2
        exit 2
        ;;
esac

[[ "$num_gpus" =~ ^[1-9][0-9]*$ ]] || {
    echo "error: num_gpus must be a positive integer" >&2
    exit 2
}

python3 tools/train_net.py \
    --num-gpus "$num_gpus" \
    --config-file "configs/VOC_RPN/faster_rcnn_R_50_C4.few_shot_s${split}.yaml" \
    OUTPUT_DIR "output/pascal_voc_mrcnn_rpn_s${split}"
