#include "conf_and_weights_fp16.h"
#include <math.h>
struct llama2_mhsa_args_fp16{
    fp16* q;
    fp16* att;
    fp16* key_cache;
    fp16* value_cache;
    fp16* xb;
    int pos;
    int kv_dim;
    int kv_mul;
    int head_size;
    int n_heads;
    int steps;
};

void llama2_mhsa_fp16_cl(void *llama2_mhsa_args);