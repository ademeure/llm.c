// all cudnn-related functions are in this file, so that they don't need to be recompiled everytime
// we change some unrelated piece of the code.
// TODO this currently duplicates some of the utilities from the main file

#define CUDNN_CPP
#define NOMINMAX
#include "cudnn_att.h"
#include <cudnn_frontend.h>

namespace fe = cudnn_frontend;

#ifdef ENABLE_FP8 // FP8 mode is not ready yet, and only supports head size of 128, sigh...
//#define CUDNN_FP8_MODE
#endif

// Specific configurations based on the enabled precision
#if defined(ENABLE_FP32)
static_assert(false, "cuDNN is not supported in FP32 mode.")
// use fp16 (note: this may require gradient scaler, currently not implemented!)
#elif defined(ENABLE_FP16)
#define CUDNN_TYPE fe::DataType_t::HALF
#elif defined(CUDNN_FP8_MODE)
#define CUDNN_TYPE fe::DataType_t::FP8_E4M3
#else // Default to bfloat16
#define CUDNN_TYPE fe::DataType_t::BFLOAT16
#endif

static cudnnHandle_t cudnn_handle;
static size_t cudnn_workspace_size = 0; // dynamically allocated as needed (up to 256MiB!)
static void* cudnn_workspace = NULL;

static void cuDNNCheck(cudnnStatus_t error, const char *file, int line) {
    if (error != CUDNN_STATUS_SUCCESS) {
        printf("[CUDNN ERROR] at file %s:%d:\n%s\n", file, line, cudnnGetErrorString(error));
        exit(EXIT_FAILURE);
    }
};
#define cuDNNCheck(err) (cuDNNCheck(err, __FILE__, __LINE__))

static void checkCudnnFE(const fe::error_object& e, const char *file, int line) {
    if(!e.is_good()) {
        printf("[CUDNN ERROR] at file %s:%d:\n%s\n", file, line, e.err_msg.c_str());
        exit(EXIT_FAILURE);
    }
}
#define checkCudnnFE(err) checkCudnnFE(err, __FILE__, __LINE__)

enum UIDs {
    Q_UID,
    K_UID,
    V_UID,
    Attn_scale_UID,
    O_UID,
    Stats_UID,
    dO_UID,
    dQ_UID,
    dK_UID,
    dV_UID,
    descale_q_UID, descale_k_UID, descale_v_UID,
    descale_s_UID, descale_o_UID, descale_dO_UID, descale_dP_UID,
    scale_s_UID, scale_o_UID, scale_dP_UID, scale_dQ_UID, scale_dK_UID, scale_dV_UID,
    amax_s_UID, amax_o_UID, amax_dQ_UID, amax_dK_UID, amax_dV_UID, amax_dP_UID
};

// Need a cache because graph->build_operation_graph() is slow but everything else seems fast
using cache_type_fwd = std::map<std::tuple<int,int,int,int, int>, std::shared_ptr<fe::graph::Graph>>;
using cache_type_bwd = std::map<std::tuple<int,int,int,int>, std::shared_ptr<fe::graph::Graph>>;

// Loosely based on cuDNN frontend samples functions and massively simplified
auto lookup_cache_or_build_graph_fwd(int B,int H,int T,int HS, int is_inference_only) {

    static cache_type_fwd user_maintained_cache_fwd;

    auto key = std::make_tuple(B, H, T, HS, is_inference_only);

    auto it = user_maintained_cache_fwd.find(key);
    if (it != user_maintained_cache_fwd.end()) {
        return it->second;
    }

    auto graph = std::make_shared<fe::graph::Graph>();
    graph->set_io_data_type(CUDNN_TYPE)
          .set_intermediate_data_type(fe::DataType_t::FLOAT)
          .set_compute_data_type(fe::DataType_t::FLOAT);

    // QKV is (B, T, 3, NH, HS) which cuDNN can handle directly without an external permute
    auto Q = graph->tensor(fe::graph::Tensor_attributes().set_name("Q")
                               .set_dim({B, H, T, HS})
                               .set_uid(Q_UID)
                               .set_stride({3 * H * HS * T,  HS, 3 * H * HS, 1}));
    auto K = graph->tensor(fe::graph::Tensor_attributes().set_name("K")
                               .set_dim({B, H, T, HS})
                               .set_uid(K_UID)
                               .set_stride({3 * H * HS * T, HS, 3 * H * HS, 1}));
    auto V = graph->tensor(fe::graph::Tensor_attributes().set_name("V")
                               .set_dim({B, H, T, HS})
                               .set_uid(V_UID)
                               .set_stride({3 * H * HS * T, HS, 3 * H * HS, 1}));
    auto attn_scale = graph->tensor(fe::graph::Tensor_attributes().set_name("attn_scale")
                               .set_dim({1, 1, 1, 1})
                               .set_stride({1, 1, 1, 1})
                               .set_uid(Attn_scale_UID)
                               .set_is_pass_by_value(true)
                               .set_data_type(fe::DataType_t::FLOAT));

#if defined(CUDNN_FP8_MODE)
    auto sdpa_fp8_options = fe::graph::SDPA_fp8_attributes()
                                .set_name("flash_attention_fp8")
                                .set_is_inference(is_inference_only)
                                .set_causal_mask(true)
                                .set_attn_scale(attn_scale);

    assert((cudnnGetVersion() >= 90100) && deviceProp.major >= 9);

    auto descale_q = graph->tensor(fe::graph::Tensor_attributes()
                                          .set_name("Descale_Q")
                                          .set_uid(descale_q_UID)
                                          .set_dim({1, 1, 1, 1})
                                          .set_stride({1, 1, 1, 1})
                                          .set_data_type(fe::DataType_t::FLOAT));
    auto descale_k = graph->tensor_like(descale_q, "Descale_K"); descale_k->set_uid(descale_k_UID);
    auto descale_v = graph->tensor_like(descale_q, "Descale_V"); descale_v->set_uid(descale_v_UID);
    auto descale_s = graph->tensor_like(descale_q, "Descale_S"); descale_s->set_uid(descale_s_UID);
    auto scale_s   = graph->tensor_like(descale_q, "Scale_S"); scale_s->set_uid(scale_s_UID);
    auto scale_o   = graph->tensor_like(descale_q, "Scale_O"); scale_o->set_uid(scale_o_UID);

    auto [O, stats, amax_s, amax_o] =
        graph->sdpa_fp8(Q, K, V, descale_q, descale_k, descale_v, descale_s, scale_s, scale_o, sdpa_fp8_options);

    amax_o->set_output(true).set_dim({1, 1, 1, 1}).set_data_type(fe::DataType_t::FLOAT).set_uid(amax_o_UID);
    amax_s->set_output(true).set_dim({1, 1, 1, 1}).set_data_type(fe::DataType_t::FLOAT).set_uid(amax_s_UID);
#else
    auto sdpa_options = fe::graph::SDPA_attributes().set_name("flash_attention");
    sdpa_options.set_is_inference(is_inference_only);
    sdpa_options.set_attn_scale(attn_scale);
    sdpa_options.set_causal_mask(true);

    // Create the graph operation and get the output tensors back
    auto [O, stats] = graph->sdpa(Q, K, V, sdpa_options);
#endif

    // Output is (B, T, NH, HS) BF16/FP16 and stats for backward pass is (B, NH, T) FP32
    O->set_output(true).set_dim({B, H, T, HS}).set_stride({H * HS * T, HS, H * HS, 1}).set_uid(O_UID);

    assert(stats == nullptr || is_inference_only == false);
    if (is_inference_only == false) {
        stats->set_output(true).set_data_type(fe::DataType_t::FLOAT)
                               .set_dim({B, H, T, 1})
                               .set_stride({H * T, T, 1, 1})
                               .set_uid(Stats_UID);
    }

    checkCudnnFE(graph->validate());

    // Build the operation graph and execution part (this is the VERY SLOW PART)
    checkCudnnFE(graph->build_operation_graph(cudnn_handle));
    auto plans = graph->create_execution_plans({fe::HeurMode_t::A});
    checkCudnnFE(graph->check_support(cudnn_handle));
    checkCudnnFE(graph->build_plans(cudnn_handle));
    // Reallocate the workspace if the required size is greater than the current workspace
    // In H100 this may be around 16B
    if (graph->get_workspace_size() > cudnn_workspace_size) {
        if (cudnn_workspace_size > 0) {
            cudaCheck(cudaFree(cudnn_workspace));
        }
        cudnn_workspace_size = graph->get_workspace_size();
        cudaCheck(cudaMalloc(&cudnn_workspace, cudnn_workspace_size));
    }

    user_maintained_cache_fwd.insert({key, graph});

    return graph;
}

auto lookup_cache_or_build_graph_bwd(int B, int NH, int T, int HS) {
    static cache_type_bwd user_maintained_cache_bwd;

    auto key = std::make_tuple(B, NH, T, HS);

    auto it = user_maintained_cache_bwd.find(key);
    if (it != user_maintained_cache_bwd.end()) {
        return it->second;
    }

    auto graph = std::make_shared<fe::graph::Graph>();
    graph->set_io_data_type(CUDNN_TYPE)
          .set_intermediate_data_type(fe::DataType_t::FLOAT)
          .set_compute_data_type(fe::DataType_t::FLOAT);

    // (B, N, 3, NH, HS)
    // must come from inp (which means we also need to convert THAT to FP16)
    auto Q = graph->tensor(fe::graph::Tensor_attributes().set_name("Q")
                            .set_dim({B, NH, T, HS})
                            .set_uid(Q_UID)
                            .set_stride({3 * NH * HS * T, HS, 3 * NH * HS, 1}));
    auto K = graph->tensor(fe::graph::Tensor_attributes().set_name("K")
                            .set_dim({B, NH, T, HS})
                            .set_uid(K_UID)
                            .set_stride({3 * NH * HS * T, HS, 3 * NH * HS, 1}));
    auto V = graph->tensor(fe::graph::Tensor_attributes().set_name("V")
                            .set_dim({B, NH, T, HS})
                            .set_uid(V_UID)
                            .set_stride({3 * NH * HS * T, HS, 3 * NH * HS, 1}));
    auto O = graph->tensor(fe::graph::Tensor_attributes().set_name("O")
                            .set_dim({B, NH, T, HS})
                            .set_uid(O_UID)
                            .set_stride({NH * HS * T, HS, NH * HS, 1}));
    auto dO = graph->tensor(fe::graph::Tensor_attributes().set_name("dO")
                            .set_dim({B, NH, T, HS})
                            .set_uid(dO_UID)
                            .set_stride({NH * HS * T, HS, NH * HS, 1}));

    auto stats = graph->tensor(fe::graph::Tensor_attributes().set_name("stats")
                            .set_dim({B, NH, T, 1})
                            .set_uid(Stats_UID)
                            .set_stride({NH * T, T, 1, 1})
                            .set_data_type(fe::DataType_t::FLOAT));
    auto attn_scale = graph->tensor(fe::graph::Tensor_attributes().set_name("attn_scale")
                            .set_dim({1, 1, 1, 1})
                            .set_stride({1, 1, 1, 1})
                            .set_is_pass_by_value(true)
                            .set_uid(Attn_scale_UID)
                            .set_data_type(fe::DataType_t::FLOAT));

#if defined(CUDNN_FP8_MODE)
    assert((cudnnGetVersion() >= 90100) && deviceProp.major >= 9);

    auto descale_q  = graph->tensor(fe::graph::Tensor_attributes()
                                          .set_name("Descale_Q")
                                          .set_uid(descale_q_UID)
                                          .set_dim({1, 1, 1, 1})
                                          .set_stride({1, 1, 1, 1})
                                          .set_data_type(fe::DataType_t::FLOAT));
    auto descale_k  = graph->tensor_like(descale_q, "Descale_K"); descale_k->set_uid(descale_k_UID);
    auto descale_v  = graph->tensor_like(descale_q, "Descale_V"); descale_v->set_uid(descale_v_UID);
    auto descale_s  = graph->tensor_like(descale_q, "Descale_S"); descale_s->set_uid(descale_s_UID);
    auto descale_o  = graph->tensor_like(descale_q, "Descale_O"); descale_o->set_uid(descale_o_UID);
    auto descale_dO = graph->tensor_like(descale_q, "Descale_dO"); descale_dO->set_uid(descale_dO_UID);
    auto descale_dP = graph->tensor_like(descale_q, "Descale_dP"); descale_dP->set_uid(descale_dP_UID);

    auto scale_s  = graph->tensor_like(descale_q, "Scale_S"); scale_s->set_uid(scale_s_UID);
    auto scale_dP = graph->tensor_like(descale_q, "Scale_dP"); scale_dP->set_uid(scale_dP_UID);
    auto scale_dQ = graph->tensor_like(descale_q, "Scale_dQ"); scale_dQ->set_uid(scale_dQ_UID);
    auto scale_dK = graph->tensor_like(descale_q, "Scale_dK"); scale_dK->set_uid(scale_dK_UID);
    auto scale_dV = graph->tensor_like(descale_q, "Scale_dV"); scale_dV->set_uid(scale_dV_UID);

    // todo - there is no equivalent to set_deterministic_algorithm() for FP8
    // does that mean it is not deterministic? :(
    auto sdpa_fp8_backwards_options = fe::graph::SDPA_fp8_backward_attributes()
                                    .set_name("sdpa_fp8_backward")
                                    .set_causal_mask(true)
                                    .set_attn_scale(attn_scale);

    auto [dQ, dK, dV, amax_dQ, amax_dK, amax_dV, amax_dP] = graph->sdpa_fp8_backward(
          Q, K, V, O, dO, stats,
          descale_q, descale_k, descale_v, descale_o, descale_dO, descale_s, descale_dP,
          scale_s, scale_dQ, scale_dK, scale_dV, scale_dP, sdpa_fp8_backwards_options);

    amax_dQ->set_output(true).set_dim({1, 1, 1, 1}).set_data_type(fe::DataType_t::FLOAT).set_uid(amax_dQ_UID);
    amax_dK->set_output(true).set_dim({1, 1, 1, 1}).set_data_type(fe::DataType_t::FLOAT).set_uid(amax_dK_UID);
    amax_dV->set_output(true).set_dim({1, 1, 1, 1}).set_data_type(fe::DataType_t::FLOAT).set_uid(amax_dV_UID);
    amax_dP->set_output(true).set_dim({1, 1, 1, 1}).set_data_type(fe::DataType_t::FLOAT).set_uid(amax_dP_UID);
#else
    auto sdpa_backward_options = fe::graph::SDPA_backward_attributes()
                                .set_name("flash_attention_backward")
#if CUDNN_FRONTEND_MAJOR_VERSION > 1 || CUDNN_FRONTEND_MINOR_VERSION >= 5
                                .set_deterministic_algorithm(true) // 1.5+ needs this for determinism
#endif
                                .set_causal_mask(true)
                                .set_attn_scale(attn_scale);

    // Create the graph operation and get the output tensors back
    auto [dQ, dK, dV] = graph->sdpa_backward(Q, K, V, O, dO, stats, sdpa_backward_options);
#endif

    dQ->set_output(true).set_dim({B, NH, T, HS}).set_stride({3 * NH * HS * T, HS, 3 * NH * HS, 1}).set_uid(dQ_UID);
    dK->set_output(true).set_dim({B, NH, T, HS}).set_stride({3 * NH * HS * T, HS, 3 * NH * HS, 1}).set_uid(dK_UID);
    dV->set_output(true).set_dim({B, NH, T, HS}).set_stride({3 * NH * HS * T, HS, 3 * NH * HS, 1}).set_uid(dV_UID);

    checkCudnnFE(graph->validate());

    // Build the operation graph and execution part (this is the VERY SLOW PART)
    checkCudnnFE(graph->build_operation_graph(cudnn_handle));
    auto plans = graph->create_execution_plans({fe::HeurMode_t::A});
    checkCudnnFE(graph->check_support(cudnn_handle));
    checkCudnnFE(graph->build_plans(cudnn_handle));

    // Reallocate the workspace if the required size is greater than the current workspace
    // By default, cuDNN uses up to 256MiB of workspace, so we don't want to just allocate the maximum
    if (graph->get_workspace_size() > cudnn_workspace_size) {
        if (cudnn_workspace_size > 0) {
            cudaCheck(cudaFree(cudnn_workspace));
        }
        cudnn_workspace_size = graph->get_workspace_size();
        cudaCheck(cudaMalloc(&cudnn_workspace, cudnn_workspace_size));
    }

    user_maintained_cache_bwd.insert({key, graph});
    return graph;
}

void attention_forward_cudnn(floatX* out,  // output: (B, T, NH, HS)
                             float* stats, // output for backward pass: (B, NH, T)
                             floatX* inp,  // input: (B, T, 3, NH, HS) QKV
                             int B, int T, int NH, int C, cudaStream_t stream) {
    NVTX_RANGE_FN();
    int HS = C / NH; // number of features per head
    bool is_inference_only = (stats == nullptr);

    #if defined(ENABLE_FP8) && !defined(CUDNN_FP8_MODE)
    NH /= 2; // todo - hack: half heads to avoid out of bounds memory reads, very unrealistic
    #endif

    cuDNNCheck(cudnnSetStream(cudnn_handle, stream));

    // Get graph and tensors from cache (or generate it on first use)
    auto graph = lookup_cache_or_build_graph_fwd(B, NH, T, HS, is_inference_only);

    // Prepare all the tensor pointers for executing the graph
    void* devPtrQ = inp;
    void* devPtrK = (inp + C);
    void* devPtrV = (inp + 2 * C);
    float attn_scale_cpu = 1.0 / sqrtf(HS);
    void* devPtrO = out;

    // Build variant pack
    std::unordered_map<int64_t , void*> variant_pack = {
        {Q_UID, devPtrQ}, {K_UID, devPtrK}, {V_UID, devPtrV}, {Attn_scale_UID, &attn_scale_cpu}, {O_UID, devPtrO}};

    #if defined(CUDNN_FP8_MODE)
    auto base_stats = stats - descale_q_UID;
    variant_pack[descale_q_UID] = base_stats + descale_q_UID;
    variant_pack[descale_k_UID] = base_stats + descale_k_UID;
    variant_pack[descale_v_UID] = base_stats + descale_v_UID;
    variant_pack[descale_s_UID] = base_stats + descale_s_UID;
    variant_pack[scale_s_UID]   = base_stats + scale_s_UID;
    variant_pack[scale_o_UID]    = base_stats + scale_o_UID;
    variant_pack[amax_s_UID]    = base_stats + amax_s_UID;
    variant_pack[amax_o_UID]    = base_stats + amax_o_UID;
    #endif

    // Add the stats tensor unless we are only doing inference (only needed for backward pass)
    if (is_inference_only == false) {
        variant_pack[Stats_UID] = stats;
        #if defined(CUDNN_FP8_MODE)
        variant_pack[Stats_UID] += 32; // stats are after the scale/descale tensors
        #endif
    }

    // Execute graph
    checkCudnnFE(graph->execute(cudnn_handle, variant_pack, cudnn_workspace));
    cudaCheck(cudaGetLastError());
}

void attention_backward_cudnn(floatX* dqkvr,                                       // output
                              floatX* dout, floatX* qkvr, floatX* o, float* stats, // inputs
                              int B, int T, int NH, int C, cudaStream_t stream) {
    NVTX_RANGE_FN();
    int HS = C / NH; // number of features per head

    #if defined(ENABLE_FP8) && !defined(CUDNN_FP8_MODE)
    NH /= 2; // todo - hack: half heads to avoid out of bounds memory reads, very unrealistic
    #endif

    // Get graph and tensors from cache (or generate it on first use)
    auto graph = lookup_cache_or_build_graph_bwd(B, NH, T, HS);

    // Prepare all the tensor pointers for executing the graph
    void* devPtrQ = qkvr;
    void* devPtrK = (qkvr + NH * HS);
    void* devPtrV = (qkvr + 2 * NH * HS);
    void* devPtrO = o;
    void* devPtrdO = dout;
    void* devPtrStats = stats;
    float attn_scale_cpu = 1.0 / sqrtf(HS);

    void* devPtrdQ = dqkvr;
    void* devPtrdK = (dqkvr + NH * HS);
    void* devPtrdV = (dqkvr + 2 * NH * HS);

    // Build variant pack that links each tensor to its data pointer
    std::unordered_map<int64_t, void*> variant_pack = {
        {Q_UID, devPtrQ}, {K_UID, devPtrK}, {V_UID, devPtrV}, {O_UID, devPtrO}, {dO_UID, devPtrdO}, {Stats_UID, devPtrStats},
        {dQ_UID, devPtrdQ}, {dK_UID, devPtrdK}, {dV_UID, devPtrdV},
        {Attn_scale_UID, &attn_scale_cpu}};

    #if defined(CUDNN_FP8_MODE)
    auto base_stats = stats - descale_q_UID;
    variant_pack[descale_q_UID] = base_stats + descale_q_UID;
    variant_pack[descale_k_UID] = base_stats + descale_k_UID;
    variant_pack[descale_v_UID] = base_stats + descale_v_UID;
    variant_pack[descale_o_UID] = base_stats + descale_o_UID;
    variant_pack[descale_s_UID] = base_stats + descale_s_UID;
    variant_pack[descale_dP_UID] = base_stats + descale_dP_UID;
    variant_pack[descale_dO_UID] = base_stats + descale_dO_UID;
    variant_pack[scale_s_UID] = base_stats + scale_s_UID;
    variant_pack[scale_dQ_UID] = base_stats + scale_dQ_UID;
    variant_pack[scale_dK_UID] = base_stats + scale_dK_UID;
    variant_pack[scale_dV_UID] = base_stats + scale_dV_UID;
    variant_pack[scale_dP_UID] = base_stats + scale_dP_UID;
    variant_pack[amax_dQ_UID] = base_stats + amax_dQ_UID;
    variant_pack[amax_dK_UID] = base_stats + amax_dK_UID;
    variant_pack[amax_dV_UID] = base_stats + amax_dV_UID;
    variant_pack[amax_dP_UID] = base_stats + amax_dP_UID;
    variant_pack[Stats_UID] += 32;
    #endif

    // Execute graph
    cuDNNCheck(cudnnSetStream(cudnn_handle, stream));
    checkCudnnFE(graph->execute(cudnn_handle, variant_pack, cudnn_workspace));
    cudaCheck(cudaGetLastError());
}

void create_cudnn() {
    cuDNNCheck(cudnnCreate(&cudnn_handle));
}

void destroy_cudnn() {
    if (cudnn_workspace != NULL) { cudaCheck(cudaFree(cudnn_workspace)); }
    cuDNNCheck(cudnnDestroy(cudnn_handle));
}