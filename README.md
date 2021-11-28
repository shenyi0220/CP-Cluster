# CP-Cluster

Confidence Propagation Cluster aims to replace NMS-based methods as a better box fusion framework in 2D/3D Object detection, Instance Segmentation:
> [**Confidence Propagation Cluster: Unleash the Full Potential of Object Detectors**](arxivlink to do),            
> Yichun Shen*, Wanli Jiang*, Zhen Xu, Rundong Li, Junghyun Kwon, Siyi Li,        


Contact: [ashen@nvidia.com](mailto:ashen@nvidia.com). Welcome for any questions and comments!

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
|FRcnn-fpn50     |  38.4 / 38.7 | 39.0 / 39.2     |    39.2 / 39.4            |
|Yolov3          |  33.5 / 33.5 | 33.6 / 33.6     |    34.1 / 34.1           |
|Retina-fpn50    |  37.4 / 37.7 | 37.5 / 37.9     |    38.1 / 38.4           |
|FCOS-X101       |  42.7 / 42.8 | 42.7 / 42.8     |    42.9 / 43.1           |
|AutoAssign-fpn50|  40.4 / 40.6 | 40.5 / 40.7     |    41.0 / 41.2           |

### Yolov5(v6 model) on COCO val

| Model       |     NMS    |    Soft-NMS      |    CP-Cluster   |
|-------------|------------|------------------|-----------------|
|Yolov5s      |    37.2    |     37.4         |      37.5       |
|Yolov5m      |    45.2    |     45.3         |      45.5       |
|Yolov5l      |    48.8    |     48.8         |      49.1       |
|Yolov5x      |    50.7    |     50.8         |      51.0       |
|Yolov5s_1280 |    44.5    |     50.8         |      44.8       |
|Yolov5m_1280 |    51.1    |     51.1         |      51.3       |
|Yolov5l_1280 |    53.6    |     53.7         |      53.8       |
|Yolov5x_1280 |    54.7    |     54.8         |      55.0       |

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
|MRCNN_R50        |  41.5/37.7 |   42.0/37.8      |    42.1/38.0    |
|MRCNN_R101       |  43.1/38.8 |   43.6/39.0      |    43.6/39.1    |
|MRCNN_X101       |  44.6/40.0 |   45.2/40.2      |    45.2/40.2    |


## Integrate into MMCV
Clone the mmcv repo from https://github.com/shenyi0220/mmcv (Cut down by 9/28/2021 from main branch with no extra modifications)


Copy the implementation of "cp_cluster_cpu" in src/nms.cpp to the mmcv nms code("mmcv/ops/csrc/pytorch/nms.cpp")

Borrow the "soft_nms_cpu" API by calling "cp_cluster_cpu" rather than orignal Soft-NMS implementations, so that modify the code like below:
~~~
@@ -186,8 +186,8 @@ Tensor softnms(Tensor boxes, Tensor scores, Tensor dets, float iou_threshold,
   if (boxes.device().is_cuda()) {
     AT_ERROR("softnms is not implemented on GPU");
   } else {
-    return softnms_cpu(boxes, scores, dets, iou_threshold, sigma, min_score,
-                       method, offset);
+    return cp_cluster_cpu(boxes, scores, dets, iou_threshold, min_score,
+                          offset, 0.8, 3);
   }
 }
~~~


Compile mmcv with source code
~~~
MMCV_WITH_OPS=1 pip install -e .
~~~


## Reproduce Object Detection and Instance Segmentation in MMDetection

Make sure that the MMCV with CP-Cluster has been successfully installed.

Download code from https://github.com/shenyi0220/mmdetection (Cut down by 9/26/2021 from main branch with some config file modifications to call Soft-NMS/CP-Cluster)

Download models from model zoo

Run below command to reproduce Faster-RCNN-r50-fpn-2x:
~~~
python tools/test.py ./configs/faster_rcnn/faster_rcnn_r50_fpn_2x_coco.py ./checkpoints/faster_rcnn_r50_fpn_2x_coco_bbox_mAP-0.384_20200504_210434-a5d8aa15.pth --eval bbox
~~~

To check original metrics with NMS, you can switch the model config back to use default NMS.

To check Soft-NMS metrics, just re-compile with mmcv without CP-Cluster modifications.
## License

For the time being, this implementation is published with NVIDIA proprietary license, and the only usage of the source code is to reproduce the result of CP-Cluster. For any possible commercial use and redistribution of the code, ps contact ashen@nvidia.com

## Citation

If you find this project useful for your research, please use the following BibTeX entry.

