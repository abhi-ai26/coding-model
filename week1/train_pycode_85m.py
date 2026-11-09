# nanoGPT config, Week 1 Day 2: ~85M params on ~520M tokens. Predict the time with 6*N*D first.
out_dir = 'out-pycode-85m'
dataset = 'pycode'
n_layer = 12; n_head = 12; n_embd = 768; block_size = 1024
batch_size = 16; gradient_accumulation_steps = 8
learning_rate = 6e-4; max_iters = 4000; lr_decay_iters = 4000; warmup_iters = 300
eval_interval = 500; eval_iters = 50
dtype = 'bfloat16'; compile = True
wandb_log = True; wandb_project = 'pycode'; wandb_run_name = 'd2-85m'
