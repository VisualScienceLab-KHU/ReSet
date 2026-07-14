#!/usr/bin/env bash

mode="${1:-}"
num_gpus="${num_gpus:-$(nvidia-smi -L | awk 'END {print NR}')}"

case "$mode" in
    os1 | os2 | os3 | os4)
        split="${mode#os}"
        config_file="configs/RPN/mask_rcnn_R_50_C4_1x_oneshot_s${split}.yaml"
        output_dir="output/mrcnn_rpn_oneshot_s${split}"
        ;;
    fs14)
        config_file="configs/RPN/mask_rcnn_R_50_C4_1x_fewshot_14.yaml"
        output_dir="output/mrcnn_rpn_fewshot14"
        ;;
    *)
        echo "usage: $0 {os1|os2|os3|os4|fs14}" >&2
        exit 2
        ;;
esac

[[ "$num_gpus" =~ ^[1-9][0-9]*$ ]] || {
    echo "error: num_gpus must be a positive integer" >&2
    exit 2
}

python3 tools/train_net.py \
    --num-gpus "$num_gpus" \
    --config-file "$config_file" \
    OUTPUT_DIR "$output_dir"
