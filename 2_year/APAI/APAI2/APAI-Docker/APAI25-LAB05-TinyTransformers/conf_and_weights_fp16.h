// Config Definition
#pragma once
#include "pulp_train.h"
#define DIM 64
#define HIDDEN_DIM 172
#define N_LAYERS 5
#define N_HEADS 8
#define N_KV_HEADS 4
#define VOCAB_SIZE 512
#define SEQ_LEN 256
#define KV_DIM 32
#define STEPS 256
#define TEMPERATURE 1.000f
#define RND_SEED 42

PI_L2 extern char* PROMPT;
PI_L2 extern int PROMPT_TOKENS[19];

extern PI_L1 fp16 BUFF1[64];
extern PI_L1 fp16 buffer_n_cores[NUM_CORES];

// RunState allocation
PI_L2 extern fp16 KEY_CACHE [N_LAYERS*STEPS*KV_DIM];
PI_L2 extern fp16 VALUE_CACHE [N_LAYERS*STEPS*KV_DIM];//KV_DIM = DIM * N_KV_HEADS/ N_HEADS

PI_L2 extern char PROB_INDEX [VOCAB_SIZE*8];

// Tokenizer allocation
#ifndef _TokenIndex_
#define _TokenIndex_
typedef struct {
char *str;
int id;
} TokenIndex;
#endif
#define MAX_TOKEN_LENGTH 7
PI_L2 extern fp16 VOCAB_SCORES [VOCAB_SIZE];

PI_L2 extern char* VOCAB[VOCAB_SIZE];
PI_L2 extern unsigned char VOCAB_DATA [2639];

PI_L2 extern TokenIndex SORTED_VOCAB [VOCAB_SIZE];

// Model weights
PI_L2 extern fp16 weights_list[260032] ;