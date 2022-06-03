

# CP-Cluster
Confidence Propagation Cluster aims to replace NMS-based methods as a better box fusion framework in 2D/3D Object detection, Instance Segmentation:
> [**Confidence Propagation Cluster: Unleash the Full Potential of Object Detectors**](https://arxiv.org/abs/2112.00342),
> Yichun Shen, Wanli Jiang, Zhen Xu, Rundong Li, Junghyun Kwon, Siyi Li,

## Updates
- **June 3rd, 2022:** Applied CP-Cluster to YoloX and achieved 0.45 mAP improvement on average(MS COCO-VAL).
- **May 31th, 2022:** Fix the "time_limit" bug for CP in Yolov5 repo. Update KPIs with yolov5 v6.1 models, then "yolov5x6+TTA" could reach 56.2 mAP on COCO val with CP.
- **May 29th, 2022:** Rebase mmcv, mmdetection, yolov5 repo to codebase of May 29th. Activate box coordinates in CP by default.
- **Mar 3rd, 2022:** Accepted by CVPR 2022

## Abstract

It’s been a long history that most object detection methods obtain objects by using the non-maximum suppression(NMS) and its improved versions like Soft-NMS to remove redundant bounding boxes. We challenge those NMS-based methods from three aspects: 1) The bounding box with highest confidence value may not be the true positive having the biggest overlap with the ground-truth box. 2) Not only suppression is required for redundant boxes, but also confidence enhancement is needed for those true positives. 3) Sorting candidate boxes by confidence values is not necessary so that full parallelism is achievable.

Inspired by belief propagation (BP), we propose the Confidence Propagation Cluster (CP-Cluster) to replace NMS-based methods, which is fully parallelizable as well as better in accuracy. In CP-Cluster, we borrow the message passing mechanism from BP to penalize redundant boxes and enhance true positives simultaneously in an iterative way until convergence. We verified the effectiveness of CP-Cluster by applying it to various mainstream detectors such as FasterRCNN, SSD, FCOS, YOLOv3, YOLOv5, Centernet etc. Experiments on MS COCO show that our plug and play method, without retraining detectors, is able to steadily improve average mAP of all those state-of-the-art models with a clear margin from 0.2 to 1.9 respectively when compared with NMS-based methods.

## Highlights

- **Better accuracy:** Compared with all previous NMS-based methods, CP-Cluster manages to achieve better accuracy

- **Fully parallelizable:** No box sorting is required, and each candidate box can be handled separately when propagating confidence messages

## Main results

### Detectors from MMDetection on COCO val/test-dev

| Method         |    NMS       |     Soft-NMS    |         CP-Cluster       |
|----------------|--------------|-----------------|--------------------------|
|FRcnn-fpn50     |  38.4 / 38.7 | 39.0 / 39.2     |    39.2 / 39.4           |
|Yolov3          |  33.5 / 33.5 | 33.6 / 33.6     |    34.1 / 34.1           |
|Retina-fpn50    |  37.4 / 37.7 | 37.5 / 37.9     |    38.1 / 38.4           |
|FCOS-X101       |  42.7 / 42.8 | 42.7 / 42.8     |    42.9 / 43.1           |
|AutoAssign-fpn50|  40.4 / 40.6 | 40.5 / 40.7     |    41.0 / 41.2           |

### Yolov5(v6 model) on COCO val

| Model               |     NMS    |    Soft-NMS      |    CP-Cluster   |
|---------------------|------------|------------------|-----------------|
|Yolov5n              |    28.0    |     28.3         |      28.5       |
|Yolov5s              |    37.4    |     37.6         |      38.0       |
|Yolov5m              |    45.4    |     45.6         |      45.8       |
|Yolov5l              |    49.0    |     49.1         |      49.4       |
|Yolov5x              |    50.7    |     50.8         |      51.1       |
|Yolov5s6_1280        |    44.9    |     45.0         |      45.2       |
|Yolov5m6_1280        |    51.3    |     51.5         |      51.7       |
|Yolov5l6_1280        |    53.7    |     53.8         |      54.0       |
|Yolov5x6_1280        |    55.0    |     55.1         |      55.4       |
|Yolov5x6_1280_tta    |    55.8    |     55.8         |      56.2       |

### YoloX on COCO val
| Method/mAP   | YoloX-Nano | YoloX-Tiny |  YoloX-S   |  YoloX-M   |  YoloX-L   |  YoloX-X   |
|--------------|------------|------------|------------|------------|------------|------------|
|NMS           |    25.8    |     32.8   |      40.5  |      46.9  |      49.7  |      51.1  |
|CP-Cluster    |    26.4    |     33.4   |      41.0  |      47.3  |      50.1  |      51.4  |


### Replace maxpooling with CP-Cluster for Centernet(Evaluated on COCO test-dev), where "flip_scale" means flip and multi-scale augmentations

| Model           |   maxpool  |    Soft-NMS      |    CP-Cluster   |
|-----------------|------------|------------------|-----------------|
|dla34            |    37.3    |     38.1         |      39.2       |
|dla34_flip_scale |    41.7    |     40.6         |      43.3       |
|hg_104           |    40.2    |     40.6         |      41.1       |
|hg_104_flip_scale|    45.2    |     44.3         |      46.6       |

### Instance Segmentation(MASK-RCNN, 3X models) from MMDetection on COCO test-dev

| Box/Mask AP   |   NMS      |    Soft-NMS      |    CP-Cluster   |
|-----------------|------------|------------------|-----------------|
|MRCNN_R50        |  41.5/37.7 |   42.0/37.8      |    42.2/38.1    |
|MRCNN_R101       |  43.1/38.8 |   43.6/39.0      |    43.7/39.2    |
|MRCNN_X101       |  44.6/40.0 |   45.2/40.2      |    45.2/40.2    |


## Integrate into MMCV
Clone the mmcv repo from https://github.com/shenyi0220/mmcv (Cut down by 5/29/2022 from main branch with no extra modifications)


Copy the implementation of "cp_cluster_cpu" in "mmcv/ops/csrc/pytorch/cpu/nms.cpp" to the mmcv nms code("mmcv/ops/csrc/pytorch/cpu/nms.cpp")

Borrow the "soft_nms_cpu" API by calling "cp_cluster_cpu" rather than orignal Soft-NMS implementations, so that modify "mmcv/ops/csrc/pytorch/nms.cpp" like below:
~~~
+Tensor cp_cluster_impl(Tensor boxes, Tensor scores, Tensor dets,
+                       float iou_threshold, float min_score,
+                       int offset, float wfa_thresh, int tune_coords, int opt_id) {
+  return DISPATCH_DEVICE_IMPL(cp_cluster_impl, boxes, scores, dets, iou_threshold,
+                              min_score, offset, wfa_thresh, tune_coords, opt_id);
+}

 Tensor softnms(Tensor boxes, Tensor scores, Tensor dets, float iou_threshold,
                float sigma, float min_score, int method, int offset) {
-  return softnms_impl(boxes, scores, dets, iou_threshold, sigma, min_score,
-                      method, offset);
+  //return softnms_impl(boxes, scores, dets, iou_threshold, sigma, min_score,
+  //                    method, offset);
+  return cp_cluster_impl(boxes, scores, dets, iou_threshold, min_score,
+                         offset, 0.8f, 0, 3);
}

~~~


Compile mmcv with source code
~~~
MMCV_WITH_OPS=1 pip install -e .
~~~


## Reproduce CP-Cluster Object Detection and Instance Segmentation in MMDetection

Make sure that the MMCV with CP-Cluster has been successfully installed.

Download code from https://github.com/shenyi0220/mmdetection (Cut down by 5/29/2022 from main branch with some config file modifications to call Soft-NMS/CP-Cluster), and install all the dependancies accordingly.

Download models from model zoo

Run below command to reproduce Faster-RCNN-r50-fpn-2x:
~~~
python tools/test.py ./configs/faster_rcnn/faster_rcnn_r50_fpn_2x_coco.py ./checkpoints/faster_rcnn_r50_fpn_2x_coco_bbox_mAP-0.384_20200504_210434-a5d8aa15.pth --eval bbox
~~~

To check original metrics with NMS, you can switch the model config back to use default NMS.

To check Soft-NMS metrics, just re-compile with mmcv without CP-Cluster modifications.

## Reproduce CP-Cluster exps with yolov5

Make sure that the MMCV with CP-Cluster has been successfully installed.

Download code from https://github.com/shenyi0220/yolov5 (Cut down by 5/30/2022 from main branch, replacing the default torchvision.nms with CP-Cluster from mmcv), and install all the dependancies accordingly.

Run below command to reproduce the CP-Cluster exp with yolov5s-v6
~~~
python val.py --data coco.yaml --iou 0.6 --weights yolov5s.pt --batch-size 32
~~~

Run below command to reproduce the CP-Cluster exp with yolov5x6
~~~
python val.py --data coco.yaml --iou 0.6 --weights yolov5x6.pt --img 1280 --batch-size 16
~~~

Run below command to reproduce the CP-Cluster exp with yolov5x6+TTA
~~~
python val.py --data coco.yaml --iou 0.6 --weights yolov5x6.pt --img 1536 --batch-size 8 --augment
~~~

## Reproduce CP-Cluster exps with YoloX
Make sure that the MMCV with CP-Cluster has been successfully installed.

Download code from https://github.com/shenyi0220/YOLOX (Cut down by 6/3/2022 from main branch, replacing the default torchvision.nms with CP-Cluster from mmcv), and install all the dependancies accordingly.

Run below command to reproduce the CP-Cluster exp with YoloX-m
~~~
python -m yolox.tools.eval -n  yolox-m -c yolox_m.pth -b 16 -d 1 --conf 0.001
~~~

## Reproduce CP-Cluster exps with Centernet
Clone the Centernet repo from https://github.com/shenyi0220/centernet-cp-cluster (Added CP-Cluster compatible utilities)

Prepare and configure the env according to https://github.com/shenyi0220/centernet-cp-cluster/blob/main/readme/INSTALL.md (Similar to original repo), suggesting Pytorch 1.7

Copy the CP-Cluster implementation("def cp_cluster") from "src/centernet/nms.pyx" to the centernet nms source file("src/lib/external/nms.pyx"), replacing the below APIs:
~~~
def cp_cluster(np.ndarray[float, ndim=2] boxes, float Nt=0.5, float threshold=0.01,
               int opt_sna=0, float wfa_threshold=0.8, int opt_sai=0):
    return soft_nms(boxes, 0.5, Nt, threshold, 1)
~~~

Compile the nms lib with below command:
~~~
cd src/lib/external
make
~~~

### Hourglass model

python test.py ctdet --exp_id coco_hourglass_bp --arch hourglass --keep_res --nms --pre_cluster_method empty --filter_threshold 0.05 --nms_opt_sna 1 --nms_sna_threshold 0.8 --load_model ../models/ctdet_coco_hg.pth

### Hourglass model with flip and multi-scale

python test.py ctdet --exp_id coco_hourglass_bp --arch hourglass --keep_res --nms --pre_cluster_method empty --filter_threshold 0.05 --nms_opt_sna 1 --nms_sna_threshold 0.8 --load_model ../models/ctdet_coco_hg.pth --flip_test --test_scales 0.5,0.75,1,1.25,1.5

### DLA-34 model
python test.py ctdet --exp_id coco_dla_exp1 --arch hourglass --keep_res --nms --pre_cluster_method empty --filter_threshold 0.05 --nms_opt_sna 1 --nms_sna_threshold 0.8 --load_model ../models/ctdet_coco_dla_2x.pth

### DLA-34 model with flip and multi-scale
python test.py ctdet --exp_id coco_dla_exp1 --arch hourglass --keep_res --nms --pre_cluster_method empty --filter_threshold 0.05 --nms_opt_sna 1 --nms_sna_threshold 0.8 --load_model ../models/ctdet_coco_dla_2x.pth --flip_test --test_scales 0.5,0.75,1,1.25,1.5

## Open Source Limitation

Due to proprietary and patent limitations, for the time being, only CPU implementation of CP-Cluster is open sourced. Full GPU-implementation and looser open source license are in application process.

## License

For the time being, this implementation is published with NVIDIA proprietary license, and the only usage of the source code is to reproduce the experiments of CP-Cluster. For any possible commercial use and redistribution of the code, pls contact ashen@nvidia.com