#include "MHSA.h"


void llama2_mhsa_fp16_cl(void *llama2_mhsa_args){
    struct llama2_mhsa_args_fp16* args = (struct llama2_mhsa_args_fp16*) llama2_mhsa_args;

    int pos = args->pos;
    int kv_dim = args->kv_dim;
    int kv_mul = args->kv_mul;
    int head_size = args->head_size;
    int n_heads = args->n_heads;
    const fp16 sqrt_head_size = (fp16) sqrtf(head_size);

    int id = pi_core_id();

    const uint32_t blockSize = (n_heads + NUM_CORES-1) / NUM_CORES;
    const uint32_t start = pi_core_id()*blockSize;
    const uint32_t stop = start+blockSize > n_heads ? n_heads : start+blockSize;

    for (int h = start; h < stop; h++) {
            // get the query vector for this head
            fp16* q = args->q + h * head_size;
            // attention scores for this head
            fp16* att = args->att + h * (STEPS+1);
            // iterate over all timesteps, including the current one

            fp16 max_val = -100000;
            int t;
            for(t=0; t <= pos-3; t+=4) {
                // get the key vector for this head and at this timestep
                fp16* k = args->key_cache + t * kv_dim + (h / kv_mul) * head_size;
                // calculate the attention score as the dot product of q and k
                
                v2f16 temp1 = (v2f16) {0, 0};
                v2f16 temp2 = (v2f16) {0, 0};
                v2f16 temp3 = (v2f16) {0, 0};
                v2f16 temp4 = (v2f16) {0, 0};
                v2f16 A, B1, B2, B3, B4;
                for (int i = 0; i < head_size; i+=2) { //+=2 because each vector is composed of 2 elmenets
                    A = *((v2f16*) &q[i]);
                    B1 = *((v2f16*) &k[i]);
                    B2 = *((v2f16*) &k[i + kv_dim]);
                    B3 = *((v2f16*) &k[i + 2*kv_dim]);
                    B4 = *((v2f16*) &k[i + 3*kv_dim]);
                    /**
                     * TASK 1:
                     * 
                     * Complete the code with the scalar product between the Query Q and the Key K, among the embedding dimensions. 
                     * We already extracted the values from the buffers: what do you need to do here?
                     * Remember you have multiple keys for each query.  
                     */
                    temp1 += A * B1;
                    temp2 += A * B2;
                    temp3 += A * B3;
                    temp4 += A * B4;
                }

                // save the score to the attention buffer
                att[t] = (temp1[0] + temp1[1]) / sqrt_head_size;
                if(att[t] > max_val) 
                    max_val = att[t];
                
                att[t+1] = (temp2[0] + temp2[1]) / sqrt_head_size;
                if(att[t+1] > max_val)
                    max_val = att[t+1];
                
                att[t+2] = (temp3[0] + temp3[1]) / sqrt_head_size;
                if(att[t+2] > max_val)
                    max_val = att[t+2];
                
                att[t+3] = (temp4[0] + temp4[1]) / sqrt_head_size;
                if(att[t+3] > max_val)
                    max_val = att[t+3];
            }
            
            // leftover
            while(t <= pos) {
                // get the key vector for this head and at this timestep
                fp16* k = args->key_cache + t * kv_dim + (h / kv_mul) * head_size;
                // calculate the attention score as the dot product of q and k
                
                v2f16 temp = (v2f16) {0, 0};
                v2f16 A,B;
                for (int i = 0; i < head_size; i+=2) {
                    A = *((v2f16*) &q[i]);
                    B = *((v2f16*) &k[i]);
                    /**
                     * TASK 1:
                     * 
                     * Complete the code with the scalar product along the embedding dimensions. 
                     * We already extracted the values from the buffers: what do you need to do here?
                     */
                    temp += A * B;
                }
                // save the score to the attention buffer
                att[t] = ( temp[0] + temp[1] ) / sqrt_head_size;
                if(att[t] > max_val)
                    max_val = att[t];
                t++;
            }

            // softmax the scores to get attention weights, from 0..pos inclusively
            // softmax_original_fp16(att, pos + 1);
            fp16 sum = 0.0f;
            for (int t = 0; t < pos+1; t++) {
                // FastExp
                float x = (float) (att[t] - max_val);
                x = GIST_A * x + GIST_B;
                if (x < GIST_C )    // no need to check if x > GIST_D, because x <= 0
                    x = 0.0f;

                uint32_t n = (uint32_t) (x);
                att[t] = (fp16) *(float*) &n;

                sum += att[t];
            }

            // weighted sum of the values, store back into xb
            fp16* xb = args->xb + h * head_size;
            fp16* v = args->value_cache + (h / kv_mul) * head_size;

            // for each t:  xb += v[t] * att[t];
            for(int i=0 ; i < head_size ; i+=2){           
                v2f16 temp = (v2f16) {0, 0};
                for(int t = 0; t <= pos; t++){
                    temp += *((v2f16*)&v[i + t*kv_dim]) * (v2f16) {att[t], att[t]};
                }
                xb[i] = temp[0] / sum;
                xb[i+1] = temp[1] / sum;
            }
    }
}