// ----------------------------------------------------------------------------
// The Sampler, which takes logits and returns a sampled token
// sampling can be done in a few ways: greedy argmax, sampling, top-p sampling

#include "conf_and_weights_fp16.h"
#ifndef _ProbIndex_
#define _ProbIndex_
typedef struct {
    float prob;
    int index;
} ProbIndex; // struct used when sorting probabilities during top-p sampling
#endif

typedef struct {
    int vocab_size;
    ProbIndex* probindex; // buffer used in top-p sampling
    float temperature;
    float topp;
    unsigned long long rng_state;
} Sampler;

int sample_argmax(fp16* probabilities, int n);

int sample_mult(fp16* probabilities, int n, float coin);

int part_probIndex(ProbIndex* a, int l, int h);

void quickSort_probIndex(ProbIndex* a, int l, int h);

int sample_topp(fp16* probabilities, int n, float topp, ProbIndex* probindex, float coin) ;

void build_sampler(Sampler* sampler, int vocab_size, float temperature, float topp, unsigned long long rng_seed);

unsigned int random_u32(unsigned long long *state);

float random_f32(unsigned long long *state);

int sample(Sampler* sampler, fp16* logits, char isLastPos);
