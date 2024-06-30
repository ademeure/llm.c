/*
cuDNN (flash) attention
*/
#ifndef CUDNN_ATT_H
#define CUDNN_ATT_H

#include "cuda_common.h"

// forward declarations of functions defined in cudnn_att.cpp
void create_cudnn();
void destroy_cudnn();
void attention_forward_cudnn(floatX16* out,  // output: (B, T, NH, HS)
                             float* stats,   // output for backward pass: (B, NH, T)
                             floatX16* inp,  // input: (B, T, 3, NH, HS) QKV
                             int B, int T, int NH, int C, cudaStream_t stream);

void attention_backward_cudnn(floatX16* dqkvr,                                           // output
                              floatX16* dout, floatX16* qkvr, floatX16* o, float* stats, // inputs
                              int B, int T, int NH, int C, cudaStream_t stream);

#endif // CUDNN_ATT_H