# Code Structure
>Clang Static analyzer plugins see https://github.com/Yunlongs/NLP-CSA
## Directories
- `data_process:` The scripts for pre-processing, parsing and normalizing the function prototypes.
- `model:` Pre-trained Siamese network, can be directly used to classify functions.
- `subword_dataset:`  The learned subword vocabulary and embedding for function prototype segmentation, and official MM function list.

## Main Scripts
- `run.py:` The entry point of Goshawk, performs each steps of Goshawk.
- `train.py:` Train the Siamese network.
- `cal_metric.py:` Evaluate the accuracy of the trained model.
- `similarity_inference.py:` Utilize the trained Siamese network to generate similarity scores for each function prototype.
- `mysegment.py:` The ULM based function prototype segmentation algorithm.
- `frontend_checker.py:` Validate the MM functions according the function prototype and data flow.
 
 
## Ⅰ. Environment Setup
```buildoutcfg
tensorflow = 2.1
```

Download the [subword embeddings](https://yunlongs-1253041399.cos.ap-chengdu.myqcloud.com/word_embedding) to the directory `subword_dataset/word_embedding`

## Ⅱ. Function Proatotype Segmentation
Function `normalize_prototype_file(in_file, out_file)` in `normalize.py` can be used to segment function prototypes.

It Segments and normalizes the function prototypes in the `in_file`, and the results are saved at `out_file.`

```buildoutcfg
For example,
    before: void * kmalloc_array(size_t n, size_t size, gfp_t flags)
    after:  <cls> <ptr> kmalloc array ( <noptr> n <dot> <noptr> size <dot> <noptr> flags )
```


## Ⅲ. Re-train Simaese network for your customized target function identification task (e.g.,MM functions, crypto functions,...).
### 1. Prepare your training function prototype dataset.

Take crypto function as example, the dataset should be the prototypes of your collected crypto functions.
Each line is a function prototype.

```buildoutcfg
crypto.txt
-------------
int crypto_aead_encrypt(struct aead_request *req)
int crypto_aead_decrypt(struct aead_request *req)
static int crypto_aegis128_encrypt_generic(struct aead_request *req)
static int crypto_aegis128_decrypt_simd(struct aead_request *req)
void crypto_aegis128_encrypt_chunk_neon(void *state, void *dst, const void *src,unsigned int size)
void crypto_aegis128_decrypt_chunk_neon(void *state, void *dst, const void *src,unsigned int size)
static int crypto_authenc_esn_encrypt(struct aead_request *req)
static int crypto_authenc_esn_decrypt(struct aead_request *req)
...
``` 

### 2. Train the Siamese network.

We have implemented the re-train of Siamese network in the script `Re-train.py`.
 It takes two arguments:
 - training corpus
 - your trained model name
 
 For example:
```buildoutcfg
python Re-train.py crypto.txt crypto
```

After the training finished, your trained model which names "crypto" is saved at directory "model/crypto".


### 3. Infer similarities.

The already trained model were saved in the directory "model", 
you can use them to infer similarity for other functions directly.

We have implemented these functions in the script `similarity_inference.py`.

You can call the function `working_on_raw_function_prototype(model_name, filename)` to infer similarity
for the functions whose prototypes saved in the argument `filename`.

Here, `model_name` should be the name of model that save in the directory `model`. 

For example, there is a file names `test.func` which contains the follow functions:
```buildoutcfg
test.func
---------
void * mem_malloc(unsigned long size)
void mem_free(void *ptr)
void CAST_set_key(CAST_KEY *key, int len, const unsigned char *data)
```

We can call the function `working_on_raw_function_prototype` to infer similarities for them.
```buildoutcfg
from similarity_inference import working_on_raw_function_prototype
working_on_raw_function_prototype("alloc", "test.func") # Infer the similarity for allocation functions.
```
The result are saved at "temp/func_name_similarity"
```buildoutcfg
temp/func_name_similarity
----
mem_malloc 0.938829920833657
mem_free -0.9019584597976495
cast_set_key -0.9085114460471964
```