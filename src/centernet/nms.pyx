# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related 
# documentation and any modifications thereto. Any use, reproduction, 
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or 
# its affiliates is strictly prohibited.
# This source code file is only allowed to repro exps of cp-cluster.

import numpy as np
cimport numpy as np

cdef inline np.float32_t max(np.float32_t a, np.float32_t b):
    return a if a >= b else b

cdef inline np.float32_t min(np.float32_t a, np.float32_t b):
    return a if a <= b else b

cdef np.float32_t getSizeAwareIOUThresh(np.float32_t w, np.float32_t h, np.float32_t lowerBound=20.0,
                                        np.float32_t upperBound=120.0, np.float32_t minThresh=0.4, np.float32_t maxThresh=0.7):
    cdef np.float32_t boxSize
    boxSize = (w + h) / 2.0
    if boxSize <= lowerBound:
        return minThresh
    elif boxSize >= upperBound:
        return maxThresh
    return minThresh + (maxThresh - minThresh) * (boxSize - lowerBound) / (upperBound - lowerBound)

def swap_array_val(np.ndarray[float, ndim=2] arrs, unsigned int idx1, unsigned int idx2):
    cdef unsigned int D = arrs.shape[1]
    cdef float tmpVal
    if idx1 == idx2:
        return
    for i in range(D):
        tmpVal = arrs[idx1, i]
        arrs[idx1, i] = arrs[idx2, i]
        arrs[idx2, i] = tmpVal


def cp_cluster(np.ndarray[float, ndim=2] boxes, float Nt=0.5, float threshold=0.01,
               int opt_sna=0, float wfa_threshold=0.8, int opt_sai=0):
    cdef unsigned int N = boxes.shape[0]

    # Pre-calculate area sizes.
    cdef np.ndarray[np.float32_t, ndim=1] areas

    cdef float iw, ih, box_area
    cdef float ua, inter
    cdef int pos = 0
    cdef int maxiter = 2
    cdef float x1,x2,y1,y2,tx1,tx2,ty1,ty2,ts,area,weight,ov,auxProposalNumber,auxMaxConf, auxMaxX1, auxMaxX2, auxMaxY1, auxMaxY2
    cdef np.ndarray[np.float32_t, ndim=2] posTerms = np.zeros([N, 6], dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=2] negTerms = np.zeros([N, 3], dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=1] iou_thresholds = np.zeros(10, dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=1] alphas = np.ones(10, dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=1] betas = np.ones(10, dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=1] m_w1 = np.ones(10, dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=1] m_w2 = np.ones(10, dtype=np.float32)
    # Suppress mat: [i, j]=m indicates that Det[i] suppressed Det[j] for m times.
    cdef np.ndarray[np.int_t, ndim=2] suppressRecodMat = np.zeros([N, N], dtype=np.int)
    cdef int maxSuppressTime = 1
    cdef float momentum = 0.0
    cdef int suppressIdx
    iou_thresholds[0] = Nt
    iou_thresholds[1] = 0.75
    iou_thresholds[2] = 0.8
    alphas[0] = 1.0
    betas[0] = 1.0
    alphas[1] = 1.0
    betas[1] = 1.0
    alphas[2] = 1.0
    betas[2] = 1.0
    m_w1[0] = 1.0
    m_w2[0] = 0.0
    m_w1[1] = 0.0
    m_w2[1] = 1.0

    for iter in range(maxiter):
        posTerms = np.zeros([N, 6], dtype=np.float32)
        negTerms = np.zeros([N, 6], dtype=np.float32)
        areas = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1)
        for i in range(N):
            tx1 = boxes[i,0]
            ty1 = boxes[i,1]
            tx2 = boxes[i,2]
            ty2 = boxes[i,3]
            ts = boxes[i,4]
            tarea = areas[i]
            if ts <= threshold:
                continue

            # apply sai
            if opt_sai:
                Nt = getSizeAwareIOUThresh(tx2 - tx1, ty2 - ty1, 30.0, 100.0, 0.45, 0.6)
                wfa_threshold = getSizeAwareIOUThresh(tx2 - tx1, ty2 - ty1, 50.0, 80.0, 0.8, 0.9)

            # vars for sna
            auxMaxConf = 0.0
            auxProposalNumber = 0.0
            auxMaxX1 = tx1
            auxMaxX2 = tx2
            auxMaxY1 = ty1
            auxMaxY2 = ty2

            pos = 0
            # NMS iterations, note that N changes if detection boxes fall below threshold
            for pos in range(N):
                x1 = boxes[pos, 0]
                y1 = boxes[pos, 1]
                x2 = boxes[pos, 2]
                y2 = boxes[pos, 3]
                s = boxes[pos, 4]
                area = areas[pos]


                if pos == i or s <= threshold or s > ts:
                    continue

                # area = (x2 - x1 + 1) * (y2 - y1 + 1)
                iw = (min(tx2, x2) - max(tx1, x1) + 1)
                if iw > 0:
                    ih = (min(ty2, y2) - max(ty1, y1) + 1)
                    if ih > 0:
                        inter = iw * ih
                        ua = float(tarea + area - inter)
                        ov = inter / ua #iou between max box and detection box

                        if ov > iou_thresholds[iter] and suppressRecodMat[i, pos] < maxSuppressTime:
                            #if ov > negTerms[pos, 0]:
                            #    negTerms[pos, 0] = ov
                            #    negTerms[pos, 2] = (float)(i)
                            momentum = m_w1[iter] * (ts / s) + m_w2[iter] * (ov / iou_thresholds[iter])
                            if momentum > negTerms[pos, 1]:
                                negTerms[pos, 0] = ov
                                negTerms[pos, 1] = momentum
                                negTerms[pos, 2] = (float)(i)

                        if ov >= wfa_threshold:
                            auxProposalNumber = auxProposalNumber + 1.0
                            if boxes[pos, 4] > auxMaxConf:
                                auxMaxConf = boxes[pos, 4]
                                auxMaxX1 = boxes[pos, 0]
                                auxMaxY1 = boxes[pos, 1]
                                auxMaxX2 = boxes[pos, 2]
                                auxMaxY2 = boxes[pos, 3]

            if opt_sna == 1:
                posTerms[i, 0] = auxProposalNumber
                posTerms[i, 1] = auxMaxConf
                posTerms[i, 2] = auxMaxX1
                posTerms[i, 3] = auxMaxY1
                posTerms[i, 4] = auxMaxX2
                posTerms[i, 5] = auxMaxY2

        for i in range(N):
            if boxes[i, 4] <= threshold:
                continue
            boxes[i, 4] = min(1.0, boxes[i, 4] + alphas[iter] * (1.0 - boxes[i,4]) * (posTerms[i, 0] / (posTerms[i, 0] + 1.0)) * posTerms[i, 1])
            boxes[i, 0] = (boxes[i, 4] * boxes[i, 0] + posTerms[i, 1] * posTerms[i, 2]) / (boxes[i, 4] + posTerms[i, 1])
            boxes[i, 1] = (boxes[i, 4] * boxes[i, 1] + posTerms[i, 1] * posTerms[i, 3]) / (boxes[i, 4] + posTerms[i, 1])
            boxes[i, 2] = (boxes[i, 4] * boxes[i, 2] + posTerms[i, 1] * posTerms[i, 4]) / (boxes[i, 4] + posTerms[i, 1])
            boxes[i, 3] = (boxes[i, 4] * boxes[i, 3] + posTerms[i, 1] * posTerms[i, 5]) / (boxes[i, 4] + posTerms[i, 1])
            if negTerms[i, 0] > 0.01:
                boxes[i, 4] = boxes[i, 4] - betas[iter] * boxes[i, 4] * negTerms[i, 0]
                suppressIdx = (int)(negTerms[i, 2])
                suppressRecodMat[suppressIdx, i] = suppressRecodMat[suppressIdx, i] + 1

    for i in range(N):
        if boxes[i, 4] < threshold:
            swap_array_val(boxes, i, N-1)
            swap_array_val(negTerms, i, N-1)
            swap_array_val(posTerms, i, N-1)
            N = N - 1

    keep = [i for i in range(N)]
    return keep
