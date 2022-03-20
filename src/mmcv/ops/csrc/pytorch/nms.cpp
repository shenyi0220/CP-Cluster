 /*
 * SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related 
 * documentation and any modifications thereto. Any use, reproduction, 
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or 
 * its affiliates is strictly prohibited.
 * This source code file is only allowed to repro exps of cp-cluster.
*/
#include "pytorch_cpp_helper.hpp"
#include "pytorch_device_registry.hpp"

Tensor cp_cluster_impl(Tensor boxes, Tensor scores, Tensor dets,
                       float iou_threshold, float min_score,
                       int offset, float wfa_thresh, int tune_coords, int opt_id) {
  return DISPATCH_DEVICE_IMPL(cp_cluster_impl, boxes, scores, dets, iou_threshold,
                              min_score, offset, wfa_thresh, tune_coords, opt_id);
}

Tensor softnms(Tensor boxes, Tensor scores, Tensor dets, float iou_threshold,
               float sigma, float min_score, int method, int offset) {
  //return softnms_impl(boxes, scores, dets, iou_threshold, sigma, min_score,
  //                    method, offset);
  return cp_cluster_impl(boxes, scores, dets, iou_threshold, min_score,
                         offset, 0.8f, 0, 3);
}