from detectron2.data import MetadataCatalog

coco17_all_classes = MetadataCatalog.get('coco_2017_val').thing_classes

contiguous_id_to_thing_dataset_id = {countId: catId for catId, countId in MetadataCatalog.get('coco_2017_val').thing_dataset_id_to_contiguous_id.items()}

# Dataset Names

COCO_2017_SPLIT_1 = 'coco_2017_train_oneshot_s1'
COCO_2017_SPLIT_2 = 'coco_2017_train_oneshot_s2'
COCO_2017_SPLIT_3 = 'coco_2017_train_oneshot_s3'
COCO_2017_SPLIT_4 = 'coco_2017_train_oneshot_s4'

COCO_2014_FEW_SHOT = 'fs_coco14_base_train'
COCO_2017_FEW_SHOT = 'fs_coco17_base_train' # not used

PASCAL_VOC_SPLIT_1 = 'pascal_voc_train_split_1'
PASCAL_VOC_SPLIT_2 = 'pascal_voc_train_split_2'
PASCAL_VOC_SPLIT_3 = 'pascal_voc_train_split_3'

voc_all_classes_1 = [ "aeroplane", "bicycle", "boat", "bottle", "car", "cat", "chair", "diningtable", "dog", "horse", "person", "pottedplant", "sheep", "train", "tvmonitor", "bird", "bus", "cow", "motorbike", "sofa"]
voc_all_classes_2 = ['bicycle', 'bird', 'boat', 'bus', 'car', 'cat', 'chair', 'diningtable', 'dog', 'motorbike', 'person', 'pottedplant', 'sheep', 'train', 'tvmonitor', 'aeroplane', 'bottle', 'cow', 'horse', 'sofa']
voc_all_classes_3 = ['aeroplane', 'bicycle', 'bird', 'bottle', 'bus', 'car', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'person', 'pottedplant', 'train', 'tvmonitor', 'boat', 'cat', 'motorbike', 'sheep', 'sofa']

voc_split_1_seen_classes = [ "aeroplane", "bicycle", "boat", "bottle", "car", "cat", "chair", "diningtable", "dog", "horse", "person", "pottedplant", "sheep", "train", "tvmonitor"]
voc_split_2_seen_classes = ["bicycle","bird","boat","bus","car","cat","chair","diningtable","dog","motorbike","person","pottedplant","sheep","train","tvmonitor"]
voc_split_3_seen_classes = ["aeroplane","bicycle","bird","bottle","bus","car","chair","cow","diningtable","dog","horse","person","pottedplant","train","tvmonitor"]



fs_coco_2014_seen_classes = ['truck', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'bed', 'toilet', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

# from fs_coco_test_all
fs_coco_2014_all_classes = MetadataCatalog.get('coco_2014_val').thing_classes

def get_oneshot_split(split):
    return [coco17_all_classes[cid] for cid in range(80) if contiguous_id_to_thing_dataset_id[cid] % 4 != split]



SEEN_CLS_DICT = {

    COCO_2017_SPLIT_1: get_oneshot_split(1),
    COCO_2017_SPLIT_2: get_oneshot_split(2),
    COCO_2017_SPLIT_3: get_oneshot_split(3),
    COCO_2017_SPLIT_4: get_oneshot_split(0),

    COCO_2014_FEW_SHOT: fs_coco_2014_seen_classes,
    COCO_2017_FEW_SHOT: fs_coco_2014_seen_classes,
    
    PASCAL_VOC_SPLIT_1: voc_split_1_seen_classes,
    PASCAL_VOC_SPLIT_2: voc_split_2_seen_classes,
    PASCAL_VOC_SPLIT_3: voc_split_3_seen_classes,
}


ALL_CLS_DICT = {

    COCO_2017_SPLIT_1: coco17_all_classes,
    COCO_2017_SPLIT_2: coco17_all_classes,
    COCO_2017_SPLIT_3: coco17_all_classes,
    COCO_2017_SPLIT_4: coco17_all_classes,
    
    COCO_2014_FEW_SHOT: fs_coco_2014_all_classes,
    COCO_2017_FEW_SHOT: fs_coco_2014_all_classes,
    
    PASCAL_VOC_SPLIT_1: voc_all_classes_1,
    PASCAL_VOC_SPLIT_2: voc_all_classes_2,
    PASCAL_VOC_SPLIT_3: voc_all_classes_3
}



if __name__ == "__main__":
    print("COCO_2017_SPLIT_1 (seen)", SEEN_CLS_DICT[COCO_2017_SPLIT_1])
    print("COCO_2017_SPLIT_1 (unseen)", [c for c in ALL_CLS_DICT[COCO_2017_SPLIT_1] if c not in SEEN_CLS_DICT[COCO_2017_SPLIT_1]])