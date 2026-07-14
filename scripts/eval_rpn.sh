#!/usr/bin/env bash

set -o xtrace

mode="${1:-}"
scope="${2:-all}"
weights_root="${weights_root:-/your_path/weights}" # enter your path
num_gpus="${num_gpus:-$(nvidia-smi -L | awk 'END {print NR}')}"

case "$scope" in
    base | all) ;;
    *)
        echo "error: evaluation scope must be base or all" >&2
        exit 2
        ;;
esac

case "$mode" in
    os1 | os2 | os3 | os4)
        split="${mode#os}"
        weights="${weights_root}/initial/rpn/one-shot-split${split}/model_final.pth"
        output_dir="output/rpn/one-shot-split${split}"
        if [[ "$scope" == "base" ]]; then
            output_dir="${output_dir}/base"
            dataset_test="(\"coco_2017_val_oneshot_s${split}\",)"
        else
            dataset_test='("coco_2017_val",)'
        fi
        ;;
    fs14)
        weights="${weights_root}/initial/rpn/few-shot-coco14/model_final.pth"
        output_dir="output/rpn/few-shot-coco14"
        if [[ "$scope" == "base" ]]; then
            output_dir="${output_dir}/base"
            dataset_test='("fs_coco17_base_val",)'
        else
            dataset_test='("fs_coco_test_all",)'
        fi
        ;;
    *)
        echo "usage: $0 {os1|os2|os3|os4|fs14} [base|all]" >&2
        exit 2
        ;;
esac

[[ "$num_gpus" =~ ^[1-9][0-9]*$ ]] || {
    echo "error: num_gpus must be a positive integer" >&2
    exit 2
}

python3 tools/train_net.py \
    --eval-only \
    --num-gpus "$num_gpus" \
    --config-file configs/RPN/rpn_R_50_C4_1x.yaml \
    OUTPUT_DIR "$output_dir" \
    MODEL.WEIGHTS "$weights" \
    DATASETS.TEST "$dataset_test"
