#include "net.h"

PI_L1 fp16 buffer_n_cores[NUM_CORES];   // for parallelized RMSNorm and softmax scratch

// ----------------------------------------------------------------------------
// Transformer model

// L1 buffer allocation
PI_L1  fp16 BUFF1[64];
PI_L1  fp16 BUFF2[64];
PI_L1  fp16 BUFF3[172];
PI_L1  fp16 BUFF4[2048];

PI_L1  fp16 BUFF_W_1[11008];
PI_L1  fp16 BUFF_W_2[11008];




void matmul(fp16* xout, fp16* x, fp16* w, int n, int d) {
/*
    original code: 
    int i;
    for (i = 0; i < d; i++) {
        fp16 val = 0.0f;
        for (int j = 0; j < n; j++) {
            val += w[i * n + j] * x[j];
        }
        xout[i] = val;
    }
*/
    struct matMul_args_fp16 mm_args;
    mm_args.A = w;
    mm_args.B = x;
    mm_args.C = xout; 
    mm_args.N = d;
    mm_args.K = n;
    mm_args.M = 1;
    mm_args.trans_B = 0;

    pi_cl_team_fork(NUM_CORES, mv_fp16_SIMD_4x1, &mm_args);
}


void memcopy_setup(fp16 * ext,fp16* loc, size_t size, int dir,pi_cl_dma_copy_t* dma_cmd){

    // ext: indicates the location in the external memory FROM which data are transferred (it depends from dfir actually but in your cases you can assume this)
    // loc: indicates the location in the external memory TO which data are transferred (same caveat as up)
    // size: the size in number of bytes of data transferred 
    // dir: direction of the transfer
    // dma_cmd: reference struct for passing command to the data mover (do not care at the moment you will find it filled)

    dma_cmd->ext = (uint32_t)ext;
    dma_cmd->loc = (uint32_t)loc;
    dma_cmd->size = size;
    dma_cmd->dir = dir;

    pi_cl_dma_memcpy(dma_cmd);

}


fp16* forward(Transformer* transformer, int token, int pos) {

    // a few convenience variables
    Config* p = &transformer->config;
    TransformerWeights* w = &transformer->weights;
    RunState* s = &transformer->state;
    
    int dim = p->dim;
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;

    int kv_mul = p->n_heads / p->n_kv_heads; // integer multiplier of the kv sharing in multiquery
    int hidden_dim =  p->hidden_dim;
    int head_size = dim / p->n_heads;
  
    // copy the token embedding into x
    fp16* content_row = w->token_embedding_table + token * dim;

    fp16* x = BUFF1;
    
    // memory transfer from the token embedding table to the x vector (BUFF1)
    pi_cl_dma_copy_t token_emb_table_to_x;
    memcopy_setup(content_row,x, dim*sizeof(*x), PI_CL_DMA_DIR_EXT2LOC,&token_emb_table_to_x);

    // transfer the rmsnorm weights
    pi_cl_dma_copy_t rms_weight;  
    memcopy_setup(w->rms_att_weight,BUFF4, dim* sizeof(*w->rms_att_weight), PI_CL_DMA_DIR_EXT2LOC,&rms_weight);


    // forward all the layers
    for(unsigned long long l = 0; l < p->n_layers; l++) {
        // key and value point to the k cache
        int loff = l * STEPS * kv_dim; // kv cache layer offset for convenience
        // s->k = s->key_cache + loff + pos * kv_dim;
        s->v = s->value_cache + loff + pos * kv_dim;
        s->xb = BUFF2;
        s->q = BUFF3;

        // transfer the weights for the matmul calculating the V tensor 
        pi_cl_dma_copy_t kv_weight;
        /**
         * TASK 1:
         * 
         * Complete this function. Be aware: the execution of the matmul is down in the code, not here.
         * memcopy_setup is define in this file: look inside the function to understand the arguments. 
         */
        memcopy_setup((w->wv + l*dim*kv_dim), BUFF_W_2, dim*kv_dim*sizeof(*w->wv), PI_CL_DMA_DIR_EXT2LOC, &kv_weight);


        pi_cl_dma_wait(&token_emb_table_to_x);
        pi_cl_dma_wait(&rms_weight);

        rmsnorm_parallelized_fp16(s->xb, x, BUFF4, buffer_n_cores, dim);

        // qkv matmuls for this position

        // transfer the weights for the matmul calculating the Q tensor 
        pi_cl_dma_copy_t q_weight;
        /**
         * TASK 1:
         * 
         * Complete this function. Be aware: the execution of the matmul is down in the code, not here.
         * memcopy_setup is define in this file: look inside the function to understand the arguments. 
         */
        memcopy_setup((w->wq + l*dim*dim),BUFF_W_1,dim*dim*sizeof(*w->wq), PI_CL_DMA_DIR_EXT2LOC,&q_weight);

        //wait the completion of the memory transfer for the V weights 
        pi_cl_dma_wait(&kv_weight);

        #ifdef STATS
        if (l==0)
        {printf("\nV matmul: \n");
        RESET_STATS();
        START_STATS();}
        #endif
        //MATMUL for calculating the V TENSOR
        //the third argument is the WEIGHT buffer!
        matmul(BUFF4, s->xb, BUFF_W_2, dim, kv_dim);
        #ifdef STATS
        if (l==0)
        {STOP_STATS();}
        #endif

        // transfer the weights for the matmul calculating the K tensor 
        //complete the value in the brakets (it reuses part of the information of the V transer 
        //so you need to change only one parameter )
        /**
         * TASK 1:
         * 
         * Set the size of the transfer. The dimensions of the KV weights are l, dim, kv_dim.
         */
        kv_weight.ext = (uint32_t) (w->wk + l*dim*kv_dim);
        pi_cl_dma_memcpy(&kv_weight);

        // transfer the v vector to the value cache
        pi_cl_dma_copy_t kv_to_L2;
        memcopy_setup((s->v),BUFF4,kv_dim*sizeof(*s->v), PI_CL_DMA_DIR_LOC2EXT,&kv_to_L2);

        pi_cl_dma_wait(&q_weight);

        #ifdef STATS
        if (l==0)
        {printf("\nQ matmul: ");
        RESET_STATS();
        START_STATS();}
        #endif
        //MATMUL for caluclating the Q Tensor
        matmul(s->q, s->xb, BUFF_W_1, dim, dim);
        #ifdef STATS
        if (l==0)
        {STOP_STATS();}
        #endif
        
        // transfer the key cache to BUFF_W_1 (except for the current position)
        pi_cl_dma_copy_t k_cache_to_L1;
        memcopy_setup((s->key_cache + loff),BUFF_W_1,kv_dim * pos * sizeof(*s->key_cache), PI_CL_DMA_DIR_EXT2LOC,&k_cache_to_L1);


        s->k = BUFF_W_1 + kv_dim*pos;
        pi_cl_dma_wait(&kv_weight);

        #ifdef STATS
        if (l==0)
        {printf("\n K matmul: ");
        RESET_STATS();
        START_STATS();}
        #endif
        //MATMUL for caluclating the K Tensor
        matmul(s->k, s->xb, BUFF_W_2, dim, kv_dim);
        #ifdef STATS
        if (l==0)
        {STOP_STATS();}
        #endif

        // transfer the value cache to BUFF_W_2
        pi_cl_dma_wait(&kv_to_L2);
        pi_cl_dma_copy_t v_cache_to_L1;
        memcopy_setup((s->value_cache + loff),BUFF_W_2,kv_dim * (pos+1) * sizeof(*s->value_cache), PI_CL_DMA_DIR_EXT2LOC,&v_cache_to_L1);

        

        // RoPE relative positional encoding: complex-valued rotate q and k in each head

        if( head_size == NUM_CORES ){
            // current version of rope_parallelized_fp16_cl work only if for head_size == N_CORES
            // TODO: implement a more general version of rope_parallelized_fp16_cl
            struct rope_args_fp16 ra;
            ra.q = s->q;
            ra.k = s->k;
            ra.dim = dim;
            ra.head_size = head_size;
            ra.pos = pos;
            ra.kv_dim = kv_dim;

            pi_cl_team_fork(NUM_CORES, rope_parallelized_fp16_cl, &ra);

        } else {
            for (int i = 0; i < dim; i+=2) {
                int head_dim = i % head_size;
                float freq = 1.0f / powf(10000.0f, head_dim / (float)head_size);
                float val = pos * freq;
                float fcr = cosf(val);
                float fci = sinf(val);
                int rotn = i < kv_dim ? 2 : 1; // how many vectors? 2 = q & k, 1 = q only
                for (int v = 0; v < rotn; v++) {
                    fp16* vec = v == 0 ? s->q : s->k; // the vector to rotate (query or key)
                    fp16 v0 = vec[i];
                    fp16 v1 = vec[i+1];
                    vec[i]   = v0 * fcr - v1 * fci;
                    vec[i+1] = v0 * fci + v1 * fcr;
                }
            }
        }

        // transfer the k vector to the key cache
        kv_to_L2.loc = (uint32_t) s->k;
        kv_to_L2.ext = (uint32_t) (s->key_cache + loff + pos * kv_dim);
        pi_cl_dma_memcpy(&kv_to_L2);

        // multihead attention

        struct llama2_mhsa_args_fp16 mhsa_args;

        mhsa_args.q = s->q;         // BUFF3
        mhsa_args.att = BUFF4;
        mhsa_args.key_cache = BUFF_W_1;
        mhsa_args.value_cache = BUFF_W_2;
        mhsa_args.xb = s->xb;       // BUFF2
        mhsa_args.pos = pos;
        mhsa_args.kv_dim = kv_dim;
        mhsa_args.kv_mul = kv_mul;
        mhsa_args.head_size = head_size;
        mhsa_args.n_heads = p->n_heads;
        mhsa_args.steps = STEPS;

        pi_cl_dma_wait(&k_cache_to_L1);
        pi_cl_dma_wait(&v_cache_to_L1);

        #ifdef STATS
        if (l==0)
        {printf("\n ATTENTION: ");
        RESET_STATS();
        START_STATS();}
        #endif
        pi_cl_team_fork(NUM_CORES, llama2_mhsa_fp16_cl, &mhsa_args);
        #ifdef STATS
        if (l==0)
        {STOP_STATS();}
        #endif

        pi_cl_dma_wait(&kv_to_L2);
        
        // tranfers the weights for the wo matmul
        pi_cl_dma_copy_t wo_to_L1;
        memcopy_setup((w->wo + l*dim*dim),BUFF_W_1,dim * dim * sizeof(*w->wo), PI_CL_DMA_DIR_EXT2LOC,&wo_to_L1);

        
        s->xb2 = BUFF3;

        // transfer the weights for the ffn rmsnorm
        pi_cl_dma_copy_t rms_ffn_weight_to_L1;
        memcopy_setup((w->rms_ffn_weight + l*dim),BUFF4,dim * sizeof(*w->rms_ffn_weight), PI_CL_DMA_DIR_EXT2LOC,&rms_ffn_weight_to_L1);


        // tranfers the weights for the first matmul in ffn
        pi_cl_dma_copy_t mm1_ffn_weight_to_L1;
        memcopy_setup((w->w1 + l*dim*hidden_dim),BUFF_W_2,dim * hidden_dim * sizeof(*w->w1), PI_CL_DMA_DIR_EXT2LOC,&mm1_ffn_weight_to_L1);


        // final matmul to get the output of the attention
        pi_cl_dma_wait(&wo_to_L1);

        matmul(s->xb2, s->xb, BUFF_W_1, dim, dim);

        // residual connection back into x
        struct vect_sum_args_fp16 vsa;
        vsa.op_1 = s->xb2;          // BUFF3
        vsa.op_2 = x;               // BUFF1
        vsa.dest = x;               // BUFF1
        vsa.size = dim;

        pi_cl_team_fork(NUM_CORES, vect_sum_fp16, &vsa);
        
        // ffn rmsnorm
        pi_cl_dma_wait(&rms_ffn_weight_to_L1);
        rmsnorm_parallelized_fp16(s->xb, x, BUFF4, buffer_n_cores, dim);

        // original code for the FFN matmul:
        // matmul(s->hb, s->xb, w->w1 + l*dim*hidden_dim, dim, hidden_dim);
        // matmul(s->hb2, s->xb, w->w3 + l*dim*hidden_dim, dim, hidden_dim);
        
        s->hb = BUFF3;
        s->hb2 = BUFF4;

        // tranfers the weights for the second matmul in ffn
        pi_cl_dma_copy_t mm2_ffn_weight_to_L1;
        memcopy_setup((w->w3 + l*dim*hidden_dim),BUFF_W_1,dim * hidden_dim * sizeof(*w->w3), PI_CL_DMA_DIR_EXT2LOC,&mm2_ffn_weight_to_L1);


        // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
        // first calculate self.w1(x) and self.w3(x)
        pi_cl_dma_wait(&mm1_ffn_weight_to_L1);

        matmul(s->hb, s->xb, BUFF_W_2, dim, hidden_dim);

        // transfer the weights for the third matmul in ffn
        mm1_ffn_weight_to_L1.ext = (uint32_t) (w->w2 + l*dim*hidden_dim);
        pi_cl_dma_memcpy(&mm1_ffn_weight_to_L1);

        pi_cl_dma_wait(&mm2_ffn_weight_to_L1);

        matmul(s->hb2, s->xb, BUFF_W_1, dim, hidden_dim);
        
        // SwiGLU non-linearity
        struct swiglu_args_fp16 sa;
        sa.in1 = s->hb;             // BUFF3
        sa.in2 = s->hb2;            // BUFF4
        sa.out = s->hb;             // BUFF3
        sa.dim = hidden_dim;

        pi_cl_team_fork(NUM_CORES, pulp_swiglu_fp16_cl, &sa);
        
        // transfer weights for the next layer RMSNorm or for final RMSNorm
        if(l < p->n_layers - 1)
            rms_weight.ext = (uint32_t) (w->rms_att_weight + (l+1)*dim);
        else
            rms_weight.ext = (uint32_t) (w->rms_final_weight);
        pi_cl_dma_memcpy(&rms_weight);

        // final matmul to get the output of the ffn
        pi_cl_dma_wait(&mm1_ffn_weight_to_L1);

        matmul(s->xb, s->hb, BUFF_W_2, hidden_dim, dim);
        // matmul(s->xb, s->hb, w->w2 + l*dim*hidden_dim, hidden_dim, dim);

        // residual connection
        vsa.op_1 = s->xb;         // BUFF2
        vsa.op_2 = x;             // BUFF1
        vsa.dest = x;             // BUFF1
        vsa.size = dim;

        pi_cl_team_fork(NUM_CORES, vect_sum_fp16, &vsa);
    }
    
    int mm_div = 4;   // split matmul in mm_div part because it's too big. Must be a divider of vocab_size
    int part = p->vocab_size / mm_div; 
    s->logits = BUFF4;
    
    pi_cl_dma_copy_t mm_weights_to_BUFF_W_1, mm_weights_to_BUFF_W_2;
    memcopy_setup((w->wcls),BUFF_W_1,dim * part * sizeof(*w->wcls), PI_CL_DMA_DIR_EXT2LOC,&mm_weights_to_BUFF_W_1);



    mm_weights_to_BUFF_W_2.loc = (uint32_t) BUFF_W_2;
    mm_weights_to_BUFF_W_2.size = dim * part * sizeof(*w->wcls);
    mm_weights_to_BUFF_W_2.dir = PI_CL_DMA_DIR_EXT2LOC;
    
    // final rmsnorm
    pi_cl_dma_wait(&rms_weight);

    rmsnorm_parallelized_fp16(s->xb, x, BUFF4, buffer_n_cores, dim);
    
    // classifier into logits. Orignal implementation: 
    // matmul(s->logits, s->xb, w->wcls, p->dim, p->vocab_size);

    for(int i=0; i<mm_div; i+=2){
        mm_weights_to_BUFF_W_2.ext = (uint32_t) (w->wcls + (i+1)*part*dim);
        pi_cl_dma_memcpy(&mm_weights_to_BUFF_W_2);

        pi_cl_dma_wait(&mm_weights_to_BUFF_W_1);
        matmul(s->logits+i*part, s->xb, BUFF_W_1, p->dim, part);

        if(i < mm_div - 2){
            mm_weights_to_BUFF_W_1.ext = (uint32_t) (w->wcls + (i+2)*part*dim);
            pi_cl_dma_memcpy(&mm_weights_to_BUFF_W_1);
        }
        
        pi_cl_dma_wait(&mm_weights_to_BUFF_W_2);
        matmul(s->logits+(i+1)*part, s->xb, BUFF_W_2, p->dim, part);
    }

    return s->logits;
}


// ----------------------------------------------------------------------------



void net_step(){
    INIT_STATS();
    PRE_START_STATS();
   

    int steps = STEPS;
    float temperature = TEMPERATURE;
    float topp = 0.9f;
    unsigned long long rng_seed = RND_SEED;

    Transformer transformer;
    build_transformer(&transformer);

    Tokenizer tokenizer;
    build_tokenizer(&tokenizer, transformer.config.vocab_size);

    Sampler sampler;
    build_sampler(&sampler, transformer.config.vocab_size, temperature, topp, rng_seed);

    fp16* log;
    int token, next, tok_diversi=0;
    int num_prompt_tokens=0;
    int* prompt_tokens = PROMPT_TOKENS;

    /**
     * TASK 2: 
     * 
     * Complete the code of the autoregressive phase with the missing steps.
     * UNCOMMENT THE CODE BELOW, FIRST!!
     */

    encode(&tokenizer, PROMPT, 1, 0, prompt_tokens, &num_prompt_tokens);    // from Tokenizer.c

    token = prompt_tokens[0];
    for(int pos = 0; pos < steps; pos++ ) {

        log = forward(&transformer, token, pos);    // from net.c
        
        if(pos < num_prompt_tokens -1)
            next = prompt_tokens[pos+1];
        else{
            next = sample(&sampler, log, pos==STEPS-1);  // from Sampler.c
        }
        
        if(next==1)
            break; 
        
        char* piece = decode(&tokenizer, token, next);  // from Tokenizer.c
        
        safe_printf(piece);

        token = next;
    }

    
    return;
}