
#include "net.h"


void memory_map_weights(TransformerWeights *w, Config* p, fp16* ptr, int shared_weights) {
    int head_size = p->dim / p->n_heads;
    // make sure the multiplications below are done in 64bit to fit the parameter counts of 13B+ models
    unsigned long long n_layers = p->n_layers;
    w->token_embedding_table = ptr;
    ptr += p->vocab_size * p->dim;
    w->rms_att_weight = ptr;
    ptr += n_layers * p->dim;
    w->wq = ptr;
    ptr += n_layers * p->dim * (p->n_heads * head_size);
    w->wk = ptr;
    ptr += n_layers * p->dim * (p->n_kv_heads * head_size);
    w->wv = ptr;
    ptr += n_layers * p->dim * (p->n_kv_heads * head_size);
    w->wo = ptr;
    ptr += n_layers * (p->n_heads * head_size) * p->dim;
    w->rms_ffn_weight = ptr;
    ptr += n_layers * p->dim;
    w->w1 = ptr;
    ptr += n_layers * p->dim * p->hidden_dim;
    w->w2 = ptr;
    ptr += n_layers * p->hidden_dim * p->dim;
    w->w3 = ptr;
    ptr += n_layers * p->dim * p->hidden_dim;
    w->rms_final_weight = ptr;
    ptr += p->dim;
    ptr += p->seq_len * head_size / 2; // skip what used to be freq_cis_real (for RoPE)
    ptr += p->seq_len * head_size / 2; // skip what used to be freq_cis_imag (for RoPE)
    w->wcls = shared_weights ? w->token_embedding_table : ptr;
}

void read_checkpoint(Config* config, TransformerWeights* weights, int* fd, fp16** data, ssize_t* file_size) {
    config->dim = DIM;
    config->hidden_dim = HIDDEN_DIM;
    config->n_heads = N_HEADS;
    config->n_kv_heads = N_KV_HEADS;
    config->n_layers = N_LAYERS;
    config->seq_len = SEQ_LEN;
    config->vocab_size = VOCAB_SIZE;

    int shared_weights;
    if(config->vocab_size > 0)
        shared_weights = 1;
    else{
        shared_weights = 0;
        config->vocab_size = - config->vocab_size;
    }
    fp16* weights_ptr = weights_list;
    memory_map_weights(weights, config, weights_ptr, shared_weights);
}

void malloc_run_state(RunState* s, Config* p) {
    s->key_cache = KEY_CACHE;
    s->value_cache = VALUE_CACHE;
}

void build_transformer(Transformer *t) {
    // read in the Config and the Weights from the checkpoint
    read_checkpoint(&t->config, &t->weights, &t->fd, &t->data, &t->file_size);
    // allocate the RunState buffers
    malloc_run_state(&t->state, &t->config);
}

void rope_parallelized_fp16_cl(void* void_args){

    // Works only width head_size = NUM_CORES. TODO: implement a more general version

    struct rope_args_fp16* args = (struct rope_args_fp16* ) void_args;
    int head_size = args->head_size;
    int dim = args->dim;
    int kv_dim = args->kv_dim;
    int pos = args->pos;

    int id = pi_core_id();

    int head_dim = (id*2) % head_size;
    fp16 freq = 1.0f / fastexp_gist_fp16(9.21034037198 * head_dim / (float)head_size);
    // fp16 freq = 1.0f / powf(10000.0f, head_dim/ (float)head_size);
    
    fp16 val = pos*freq;
    fp16 fcr, fci;

    if(pos <= 200){
        fcr = (fp16) cosf((float) val);
        fci = (fp16) sinf((float) val);
    } else
       cordic_cos_sin_fp16(val, &fcr, &fci);

    for(int i=id*2; i < dim ; i+=2*NUM_CORES){
        int rotn = i < kv_dim ? 2 : 1; // how many vectors? 2 = q & k, 1 = q only
            for (int v = 0; v < rotn; v++) {
                fp16* vec = v == 0 ? args->q : args->k; // the vector to rotate (query or key)
                fp16 v0 = vec[i];
                fp16 v1 = vec[i+1];
                vec[i]   = v0 * fcr - v1 * fci;
                vec[i+1] = v0 * fci + v1 * fcr;
            }
    }
}

