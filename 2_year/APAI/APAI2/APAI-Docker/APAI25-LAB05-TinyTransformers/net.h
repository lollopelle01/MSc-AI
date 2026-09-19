#include "pmsis.h"
#include "stats.h"
#include "pulp_rmsnorm_fp16.h"
#include "conf_and_weights_fp16.h"
#include "Tokenizer.h"
#include "Sampler.h"
#include "MHSA.h"

typedef struct {
    int dim; // transformer dimension
    int hidden_dim; // for ffn layers
    int n_layers; // number of layers
    int n_heads; // number of query heads
    int n_kv_heads; // number of key/value heads (can be < query heads because of multiquery)
    int vocab_size; // vocabulary size, usually 256 (byte-level)
    int seq_len; // max sequence length
} Config;

typedef struct {
    // token embedding table
    fp16* token_embedding_table;    // (vocab_size, dim)
    // weights for rmsnorms
    fp16* rms_att_weight; // (layer, dim) rmsnorm weights
    fp16* rms_ffn_weight; // (layer, dim)
    // weights for matmuls. note dim == n_heads * head_size
    fp16* wq; // weight for the calculating the Q tensor (layer, dim, n_heads * head_size) (as you can imagine weight are stored contiguously per layer!!!)
    fp16* wk; // weight for the calculating the K tensor (layer, dim, n_kv_heads * head_size)
    fp16* wv; // weight for the calculating the V tensor(layer, dim, n_kv_heads * head_size)
    fp16* wo; // (layer, n_heads * head_size, dim)
    // weights for ffn
    fp16* w1; // (layer, hidden_dim, dim)
    fp16* w2; // (layer, dim, hidden_dim)
    fp16* w3; // (layer, hidden_dim, dim)
    // final rmsnorm
    fp16* rms_final_weight; // (dim,)
    // (optional) classifier weights for the logits, on the last layer
    fp16* wcls;
} TransformerWeights;

typedef struct {
    // current wave of activations
    fp16 *x; // activation at current time stamp (dim,)
    fp16 *xb; // same, but inside a residual branch (dim,)
    fp16 *xb2; // an additional buffer just for convenience (dim,)
    fp16 *hb; // buffer for hidden dimension in the ffn (hidden_dim,)
    fp16 *hb2; // buffer for hidden dimension in the ffn (hidden_dim,)
    fp16 *q; // query (dim,)
    fp16 *k; // key (dim,)
    fp16 *v; // value (dim,)
    fp16 *att; // buffer for scores/attention values (n_heads, seq_len)
    fp16 *logits; // output logits (vocab_size, )
    // kv cache
    fp16* key_cache;   // (layer, seq_len, dim)
    fp16* value_cache; // (layer, seq_len, dim)
} RunState;

typedef struct {
    Config config; // the hyperparameters of the architecture (the blueprint)
    TransformerWeights weights; // the weights of the model
    RunState state; // buffers for the "wave" of activations in the forward pass
    // some more state needed to properly clean up the memory mapping (sigh)
    int fd; // file descriptor for memory mapping
    fp16* data; // memory mapped data pointer
    ssize_t file_size; // size of the checkpoint file in bytes
} Transformer;

struct rope_args_fp16{
    fp16* q;
    fp16* k;
    int pos;
    int dim;
    int head_size;
    int kv_dim;
};


void net_step();


void memory_map_weights(TransformerWeights *w, Config* p, fp16* ptr, int shared_weights);

void read_checkpoint(Config* config, TransformerWeights* weights, int* fd, fp16** data, ssize_t* file_size) ;

void malloc_run_state(RunState* s, Config* p) ;

void build_transformer(Transformer *t) ;


void rope_parallelized_fp16_cl(void* void_args);


void memcopy_setup(fp16 * ext,fp16* loc, size_t size, int dir,pi_cl_dma_copy_t* dma_cmd);
