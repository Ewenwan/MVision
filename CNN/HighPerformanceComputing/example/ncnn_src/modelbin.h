// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2017 THL A29 Limited, a Tencent company. All rights reserved.
// 修改：万有文
// 从io文件、内存、数组 中载入模型权重数据==========

#ifndef NCNN_MODELBIN_H
#define NCNN_MODELBIN_H

#include <stdio.h>
#include "mat.h"
#include "platform.h"

namespace ncnn {

class Net;

// 基类 模型===================
class ModelBin
{
public:
    // element type
    // 0 = auto
    // 1 = float32
    // 2 = float16
    // 3 = int8
    // load vec
    virtual Mat load(int w, int type) const = 0;
    // load image
    virtual Mat load(int w, int h, int type) const;
    // load dim
    virtual Mat load(int w, int h, int c, int type) const;
};

// 从io文件中载入模型=============
#if NCNN_STDIO
class ModelBinFromStdio : public ModelBin
{
public:
    // construct from file
    ModelBinFromStdio(FILE* binfp);

    virtual Mat load(int w, int type) const;

protected:
    FILE* binfp;
};
#endif // NCNN_STDIO

// 从内存中载入模型=========================
class ModelBinFromMemory : public ModelBin
{
public:
    // construct from external memory
    ModelBinFromMemory(const unsigned char*& mem);

    virtual Mat load(int w, int type) const;

protected:
    const unsigned char*& mem;
};

// 从数组中载入模型=========================
class ModelBinFromMatArray : public ModelBin
{
public:
    // construct from weight blob array
    ModelBinFromMatArray(const Mat* weights);

    virtual Mat load(int w, int type) const;

protected:
    mutable const Mat* weights;
};

} // namespace ncnn

#endif // NCNN_MODELBIN_H
