// ----------------------------------------------------------------------------
// The Byte Pair Encoding (BPE) Tokenizer that translates strings <-> tokens
#include "conf_and_weights_fp16.h"
#include <string.h>
#include <ctype.h>
#include "stdlib.h"
#include "stdio.h"

#ifndef _TokenIndex_
#define _TokenIndex_
typedef struct {
char *str;
int id;
} TokenIndex;
#endif


typedef struct {
    char** vocab;
    fp16* vocab_scores;
    TokenIndex *sorted_vocab;
    int vocab_size;
    unsigned int max_token_length;
    unsigned char byte_pieces[512]; // stores all single-byte strings
} Tokenizer;



void build_tokenizer(Tokenizer* t, int vocab_size);

char* decode(Tokenizer* t, int prev_token, int token);

void safe_printf(char *piece) ;

int compare_tokens(const void *a, const void *b) ;

void *bsearch (const void *key, const void *base0, size_t nmemb, size_t size, int (*compar)(const void *, const void *));

int str_lookup(char *str, TokenIndex *sorted_vocab, int vocab_size);

void encode(Tokenizer* t, char *text, int8_t bos, int8_t eos, int *tokens, int *n_tokens);