#!/usr/bin/env bash

task="${task:-fsod}" # fsod, osod
vit="${vit:-l}" # s, b, l; osod supports l only
dataset="${dataset:-coco}" # coco, voc
shot="${shot:-30}"
split="${split:-1}"
weights_root="${weights_root:-/your_path/weights}" # enter your path
num_gpus="${num_gpus:-$(nvidia-smi -L | awk 'END {print NR}')}"

fail() {
    echo "error: $*" >&2
    exit 2
}

case "$task" in
    fsod | osod) ;;
    *) fail "task must be fsod or osod" ;;
esac

case "$vit" in
    s | b | l) ;;
    *) fail "vit must be s, b, or l" ;;
esac

case "$dataset" in
    coco | voc) ;;
    *) fail "dataset must be coco or voc" ;;
esac

[[ "$num_gpus" =~ ^[1-9][0-9]*$ ]] || fail "num_gpus must be a positive integer"

if [[ "$task" == "fsod" && "$dataset" == "coco" ]]; then
    case "$shot" in
        5 | 10 | 30) ;;
        *) fail "COCO few-shot supports shot=5, 10, or 30" ;;
    esac
elif [[ "$task" == "fsod" ]]; then
    case "$shot" in
        1 | 2 | 3 | 5 | 10) ;;
        *) fail "VOC few-shot supports shot=1, 2, 3, 5, or 10" ;;
    esac
    case "$split" in
        1 | 2 | 3) ;;
        *) fail "VOC few-shot supports split=1, 2, or 3" ;;
    esac
else
    [[ "$vit" == "l" ]] || fail "one-shot supports vit=l only"
    case "$split" in
        1 | 2 | 3 | 4) ;;
        *) fail "one-shot supports split=1, 2, 3, or 4" ;;
    esac
fi

echo "task=$task, vit=$vit, dataset=$dataset, shot=$shot, split=$split, num_gpus=$num_gpus"

case "$task" in
    fsod)
        if [[ "$dataset" == "coco" ]]; then
            python3 tools/train_net.py \
                --num-gpus "$num_gpus" \
                --eval-only \
                --config-file "configs/few-shot/vit${vit}_shot${shot}.yaml" \
                MODEL.WEIGHTS `ls ${weights_root}/trained/few-shot/vit${vit}_*.pth | head -n 1` \
                DE.OFFLINE_RPN_CONFIG configs/RPN/mask_rcnn_R_50_C4_1x_fewshot_14.yaml \
                OUTPUT_DIR "output/eval/few-shot/shot-${shot}/vit${vit}/" \
                "$@"
        else
            python3 tools/train_net.py \
                --num-gpus "$num_gpus" \
                --eval-only \
                --config-file "configs/few-shot-voc/${shot}shot/vit${vit}_${split}s.yaml" \
                MODEL.WEIGHTS `ls ${weights_root}/trained/few-shot-voc/${split}/vit${vit}_*.pth | head -n 1` \
                DE.OFFLINE_RPN_CONFIG "configs/VOC_RPN/faster_rcnn_R_50_C4.few_shot_s${split}.yaml" \
                OUTPUT_DIR "output/eval/few-shot-voc/${shot}shot/${split}/vit${vit}/" \
                "$@"
        fi
        ;;
    osod)
        python3 tools/train_net.py \
            --num-gpus "$num_gpus" \
            --eval-only \
            --config-file "configs/one-shot/split${split}_vit${vit}.yaml" \
            MODEL.WEIGHTS `ls ${weights_root}/trained/one-shot/vit${vit}_*.split${split}.pth | head -n 1` \
            DE.OFFLINE_RPN_CONFIG "configs/RPN/mask_rcnn_R_50_C4_1x_oneshot_s${split}.yaml" \
            OUTPUT_DIR "output/eval/one-shot/split${split}/vit${vit}/" \
            DE.ONE_SHOT_MODE True \
            "$@"
        ;;
esac
