
# subword segment config
vocab_path = "subword_dataset/vocab"
multi = None

# Transformer hyper-parameters config
n_layers = 6
d_model = 512
n_heads = 8
diff = 512
embedding_size = 128
pooling = "CLS" # Mean or CLS
warmup_steps = 80000


# Function prototype normalization config
embedding_path = "subword_dataset/word_embedding"
train_data_prefix = "subword_dataset/training"
conserved_words = ["<cls>","<ptr>","<dptr>","<noptr>","<dot>","(",")"]
max_seq_length = 60

#Training sample strategy config
k = 40
test_k = 20
test_ratio = 0.1
valid_ratio = 0.1
inference_threshold = 0.5   #alloc
#inference_threshold = 0.99 #free
inference_recall_threshold = -0.9
inference_accuracy_threshold = 0.95
semi_hard_positive_threshold = 0.5
semi_hard_negative_threshold = -0.8
hard_alloc_path = "../Dataset/labeled_dataset/hard_alloc_prototype.json"
hard_free_path = "../Dataset/labeled_dataset/hard_free_prototype.json"
hardest_threshold = 0.98



# triplet loss margin
#margin = 1.4

# constrastive loss margin
#constrastive_margin = 1

# Training config
Buffer_size = 10000
mini_batch = 100
feature_size = 128
epochs = 30
step_per_epoch = 1500
inference_batch = 3000
loss = "mse"   ## mse or constrastive
drop_rate = 0.2
banalace_split = False


# Similarity Inference config
model_dir = "model"
func_sim_path="temp/func_similarity"
func_name_sim_path = "temp/func_name_similarity"


# Frontend Checker config
strong_belief_threshold = 0.95
#min_call = 10
min_reset = 50 # kernel for 20, others for 2


import os
cur_work_dir = os.getcwd()
plugin_dir = cur_work_dir + "/plugins"
temp_dir = cur_work_dir + "/temp"

#Call Graph
call_graph_path= temp_dir + os.sep + "call_graph.json"

# MOS Free Plugin
candidate_free_path = cur_work_dir + "/temp/candidate_free.txt"
seed_free_path = cur_work_dir + "/temp/seed_free.txt"
mos_seed_path = cur_work_dir + "/temp/last_step_mos.json"
mos_free_outpath = cur_work_dir + "/temp/memory_flow_free.json"
visited_file_path = cur_work_dir + "/temp/visited.txt"

# MOS Alloc Plugin
candidate_alloc_path = cur_work_dir + "/temp/candidate_alloc.txt"
seed_alloc_path = cur_work_dir + "/temp/seed_alloc.txt"
mos_alloc_outpath = cur_work_dir + "/temp/memory_flow_alloc.json"

max_iteration = 15

# temporary files
free_check_file = "temp/free_check.txt"
time_record_file = "temp/time_record.txt"

# Error compilation flags
comp_err_flags = ["-D__GENKSYMS__"]
