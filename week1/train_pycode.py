# nanoGPT config, Week 1 Day 1: ~30M params, ~1 hour on an RTX 4090.
# Run from the nanoGPT folder:  python train.py config/train_pycode.py
out_dir = 'out-pycode'
dataset = 'pycode'
n_layer = 8; n_head = 8; n_embd = 512; block_size = 512
batch_size = 24; gradient_accumulation_steps = 4
learning_rate = 6e-4; max_iters = 3000; lr_decay_iters = 3000; warmup_iters = 200
eval_interval = 250; eval_iters = 50
dtype = 'bfloat16'; compile = True
wandb_log = False; wandb_project = 'pycode'; wandb_run_name = 'd1-30m'
