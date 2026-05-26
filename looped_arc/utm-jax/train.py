import os
import argparse
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from flax import nnx
import optax
import orbax.checkpoint as ocp
import torch
import wandb
import numpy as np
import tqdm

from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig
from models.ut import UniversalTransformer

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_paths", type=str, nargs="+", required=True)
    parser.add_argument("--test_data_paths", type=str, nargs="+", default=[])
    parser.add_argument("--global_batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--eval_interval", type=int, default=1000)
    parser.add_argument("--eval_steps", type=int, default=50)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--clip_grad_norm", type=float, default=1.0)
    parser.add_argument("--use_ema", action="store_true", help="Use Exponential Moving Average for weights")
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--project_name", type=str, default="tpu-sprint-utm")
    parser.add_argument("--run_name", type=str, default="universal-transformer")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adamw", "muon", "adamw_poprisk"], help="Swappable optimizer")
    parser.add_argument("--poprisk_gate", type=str, default="soft", choices=["hard", "soft", "snr"], help="Population-risk gate variant (Litman & Guo 2026; ADR_016)")
    parser.add_argument("--poprisk_rho", type=float, default=0.99, help="Gradient-variance EMA decay for poprisk gate")
    parser.add_argument("--poprisk_alpha", type=float, default=0.0, help="LOO coefficient. Paper §F.4 / eq. 245: 1.0 for fresh-batch / online streaming, b/(n-b) for finite-dataset training. UTM-Sudoku (n~3.8M, b=256) is finite-dataset, so α≈1e-4. Default 0.0 is safe (gate ≈ identity). Setting α=1.0 in finite-dataset regime over-suppresses updates and destabilizes training — see ADR 016 update 2026-05-12.")
    parser.add_argument("--poprisk_lambda_pop", type=float, default=0.0, help="Population-risk gate sharpness (paper: typically 0 at scale)")
    parser.add_argument("--poprisk_eps", type=float, default=1e-12, help="Numerical stabiliser in poprisk gate denominator")
    parser.add_argument("--poprisk_skip_router", action="store_true", help="Hybrid optimizer: route ACT-router params through plain AdamW (no gate), poprisk on everything else. Tests whether the gate's pathology is router-specific (see FINDING_Poprisk_ACT_Incompatibility.md).")
    
    # Model Config
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--num_memory_tokens", type=int, default=16)
    parser.add_argument("--max_ponder_steps", type=int, default=15)
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--ponder_lambda", type=float, default=0.01)
    parser.add_argument("--disable_act", action="store_true", help="Disable ACT halting logic to force fixed ponder steps")
    parser.add_argument("--router_init_bias", type=float, default=-3.0, help="Initial bias for ACT router (-3.0 = deep-start default, 0.0 = legacy shallow-start)")
    parser.add_argument("--lambda_warmup_steps", type=int, default=0, help="Linear warmup steps for ponder_lambda (0 = no warmup)")
    parser.add_argument("--checkpoint_interval", type=int, default=2500, help="Save checkpoint every N steps")
    parser.add_argument("--use_rmsnorm", action="store_true", help="Use RMSNorm instead of DerfNorm (for ablation)")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint in checkpoint_dir/run_name")
    parser.add_argument("--lr_schedule", type=str, default="cosine", choices=["cosine", "constant_then_cosine"], help="LR schedule type")
    parser.add_argument("--lr_decay_fraction", type=float, default=0.2, help="For constant_then_cosine: fraction of training spent decaying (default 0.2 = decay in last 20%%)")
    return parser.parse_args()


@nnx.jit(static_argnames=['num_memory_tokens', 'pad_id'])
def train_step(model, optimizer, batch, ponder_lambda: float, num_memory_tokens: int, pad_id: int, rngs):
    def loss_fn(model):
        inputs = batch["inputs"]
        labels = batch["labels"]
        # Attention mask reflects valid INPUT positions; loss mask reflects valid LABELS.
        input_mask = (inputs != pad_id)
        loss_mask = (labels != -100)

        logits, ponder_loss, halt_steps, diagnostics = model(inputs, input_mask)

        # Slice out memory token outputs to match sequence length
        logits_seq = logits[:, num_memory_tokens:, :]

        # Cross Entropy Logic
        loss_per_token = optax.softmax_cross_entropy_with_integer_labels(logits_seq, jnp.maximum(labels, 0))
        loss_per_token = jnp.where(loss_mask, loss_per_token, 0.0)
        lm_loss_sum = jnp.sum(loss_per_token)
        total_valid = jnp.maximum(jnp.sum(loss_mask), 1.0)
        lm_loss = lm_loss_sum / total_valid

        total_loss = lm_loss + ponder_lambda * ponder_loss

        # Calculate accuracy strictly on target tokens
        preds = jnp.argmax(logits_seq, axis=-1)
        is_correct = (preds == labels) & loss_mask
        accuracy = jnp.sum(is_correct) / total_valid

        # Exact match logic (if all valid tokens in the sequence match)
        seq_lens = jnp.sum(loss_mask, axis=-1)
        seq_correct = jnp.sum(is_correct, axis=-1)
        exact_match = jnp.mean((seq_correct == seq_lens) & (seq_lens > 0))

        metrics = {
            "loss": total_loss,
            "lm_loss": lm_loss,
            "ponder_loss": ponder_loss,
            "accuracy": accuracy,
            "exact_match": exact_match,
            "mean_halt_steps": jnp.mean(halt_steps)
        }
        metrics.update(diagnostics)
        return total_loss, metrics

    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    (_, metrics), grads = grad_fn(model)

    grad_norm = jnp.sqrt(sum([jnp.sum(jnp.square(x)) for x in jax.tree_util.tree_leaves(grads)]))
    metrics["grad_norm"] = grad_norm

    # Router-specific gradient norm
    router_grads = grads.router
    router_grad_norm = jnp.sqrt(sum([jnp.sum(jnp.square(x)) for x in jax.tree_util.tree_leaves(router_grads)]))
    metrics["diag/router_grad_norm"] = router_grad_norm

    optimizer.update(grads)
    return metrics

@nnx.jit(static_argnames=['num_memory_tokens', 'pad_id'])
def eval_step(model, batch, ponder_lambda: float, num_memory_tokens: int, pad_id: int):
    # Pure evaluation hook mimicking train_step loss_fn structure
    inputs = batch["inputs"]
    labels = batch["labels"]
    input_mask = (inputs != pad_id)
    loss_mask = (labels != -100)

    logits, ponder_loss, halt_steps, _diagnostics = model(inputs, input_mask)
    logits_seq = logits[:, num_memory_tokens:, :]

    loss_per_token = optax.softmax_cross_entropy_with_integer_labels(logits_seq, jnp.maximum(labels, 0))
    loss_per_token = jnp.where(loss_mask, loss_per_token, 0.0)
    lm_loss_sum = jnp.sum(loss_per_token)
    total_valid = jnp.maximum(jnp.sum(loss_mask), 1.0)
    lm_loss = lm_loss_sum / total_valid

    total_loss = lm_loss + ponder_lambda * ponder_loss

    preds = jnp.argmax(logits_seq, axis=-1)
    is_correct = (preds == labels) & loss_mask
    accuracy = jnp.sum(is_correct) / total_valid

    seq_lens = jnp.sum(loss_mask, axis=-1)
    seq_correct = jnp.sum(is_correct, axis=-1)
    exact_match = jnp.mean((seq_correct == jnp.maximum(seq_lens, 1)) & (seq_lens > 0))
    
    metrics = {
        "loss": total_loss,
        "lm_loss": lm_loss,
        "ponder_loss": ponder_loss,
        "accuracy": accuracy,
        "exact_match": exact_match,
        "mean_halt_steps": jnp.mean(halt_steps)
    }
    return metrics

@nnx.jit
def update_ema_step(ema_model, model, decay=0.999):
    def update_ema(ema_param, param):
        return ema_param * decay + param * (1.0 - decay)
    
    # We map over the states to update EMA parameters
    ema_state = nnx.state(ema_model)
    model_state = nnx.state(model)
    
    new_ema_state = jax.tree_util.tree_map(update_ema, ema_state, model_state)
    nnx.update(ema_model, new_ema_state)


def main():
    args = parse_args()

    # Wandb resume: if resuming and a previous run ID exists, continue that run
    run_ckpt_dir = os.path.join(os.path.abspath(args.checkpoint_dir), args.run_name)
    wandb_id_file = os.path.join(run_ckpt_dir, "wandb_run_id.txt")
    wandb_kwargs = dict(project=args.project_name, name=args.run_name, config=vars(args))
    if args.resume and os.path.exists(wandb_id_file):
        with open(wandb_id_file) as f:
            saved_id = f.read().strip()
        print(f"Resuming wandb run {saved_id}...")
        wandb_kwargs["id"] = saved_id
        wandb_kwargs["resume"] = "must"
    wandb.init(**wandb_kwargs)
    # Save wandb run ID for future resume
    os.makedirs(run_ckpt_dir, exist_ok=True)
    with open(wandb_id_file, "w") as f:
        f.write(wandb.run.id)
    
    # Init Data train setup
    train_dataset = PuzzleDataset(PuzzleDatasetConfig(
        seed=args.seed,
        dataset_paths=args.data_paths,
        global_batch_size=args.global_batch_size,
        test_set_mode=False,
        epochs_per_iter=args.epochs,
        rank=0,
        num_replicas=1
    ), split="train")
    
    metadata = train_dataset.metadata
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=None, num_workers=0)
    
    # Init Data eval setup
    test_dataset_paths = args.test_data_paths if args.test_data_paths else args.data_paths
    test_dataset = PuzzleDataset(PuzzleDatasetConfig(
        seed=args.seed,
        dataset_paths=test_dataset_paths,
        global_batch_size=args.global_batch_size,
        test_set_mode=True,
        epochs_per_iter=1,
        rank=0,
        num_replicas=1
    ), split="test")
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=None, num_workers=0)

    # Device Mesh setup
    mesh = Mesh(jax.devices(), ('batch',))
    sharding = NamedSharding(mesh, PartitionSpec('batch'))
    print(f"Running on {len(jax.devices())} devices. Devices: {jax.devices()}")

    def shard_batch(batch):
        # Convert dictionary values to jax arrays and shard
        return jax.tree_util.tree_map(lambda x: jax.device_put(jnp.array(x.numpy() if isinstance(x, torch.Tensor) else x), sharding), batch)

    import threading
    import queue
    def prefetch_generator(loader, qsize=4, cyclical=False):
        q = queue.Queue(maxsize=qsize)
        def producer():
            while True:
                for b in loader:
                    if isinstance(b, (tuple, list)):
                        _, batch_data, _ = b
                    else:
                        batch_data = b
                    q.put(shard_batch(batch_data))
                if not cyclical:
                    break
            q.put(None)
        t = threading.Thread(target=producer, daemon=True)
        t.start()
        while True:
            item = q.get()
            if item is None:
                break
            yield item
            
    # Cyclic iterator for evaluations
    test_iterator = iter(prefetch_generator(test_loader, cyclical=True))

    # Init Model
    rngs = nnx.Rngs(args.seed)
    model = UniversalTransformer(
        vocab_size=metadata.vocab_size,
        hidden_size=args.hidden_size,
        num_heads=args.num_heads,
        max_len=metadata.seq_len,
        num_memory_tokens=args.num_memory_tokens,
        max_ponder_steps=args.max_ponder_steps,
        epsilon=args.epsilon,
        rngs=rngs,
        disable_act=args.disable_act,
        router_init_bias=args.router_init_bias,
        use_rmsnorm=args.use_rmsnorm
    )
    
    # Compute dynamic training steps
    steps_per_epoch = metadata.total_puzzles // args.global_batch_size
    total_steps = steps_per_epoch * args.epochs
    
    # Print configuration summary
    print("=" * 60)
    print("UT MODEL SCALING & HYPERPARAMETER CONFIGURATION")
    print("=" * 60)
    print(f"Dataset      : {args.data_paths}")
    print(f"Puzzles      : {metadata.total_puzzles} (Train) => Steps/Epoch: {steps_per_epoch}")
    print(f"Batch Size   : {args.global_batch_size}")
    print(f"Epochs       : {args.epochs}")
    print(f"Total Steps  : {total_steps}")
    print(f"Optimizer    : {args.optimizer.upper()} (LR: {args.lr}, Warmup: {args.warmup_steps}, Clip: {args.clip_grad_norm})")
    print(f"EMA Active   : {args.use_ema}")
    print("-" * 60)
    print(f"Hidden Size  : {args.hidden_size}")
    print(f"Heads        : {args.num_heads}")
    print(f"Mem Tokens   : {args.num_memory_tokens}")
    print(f"Ponder Steps : {args.max_ponder_steps}")
    print(f"Ponder Lambda: {args.ponder_lambda}")
    print("=" * 60)

    if args.lr_schedule == "constant_then_cosine":
        # Hold peak LR for most of training, cosine decay only at the end.
        decay_steps = max(1, int(total_steps * args.lr_decay_fraction))
        constant_steps = total_steps - decay_steps - args.warmup_steps
        scheduler = optax.join_schedules(
            [optax.linear_schedule(0.0, args.lr, args.warmup_steps),
             optax.constant_schedule(args.lr),
             optax.cosine_decay_schedule(args.lr, decay_steps, 1e-5)],
            boundaries=[args.warmup_steps, args.warmup_steps + constant_steps])
        print(f"LR Schedule: warmup {args.warmup_steps} → constant {constant_steps} → cosine decay {decay_steps}")
    else:
        scheduler = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=args.lr,
            warmup_steps=args.warmup_steps,
            decay_steps=total_steps,
            end_value=1e-5
        )

    # Optimizer swap logic
    if args.optimizer.lower() == "muon":
        print("Note: Muon requested. Initializing 2D orthogonal momentum partitioning via multi_transform.")
        from optimizers.muon import scale_by_muon
        
        adamw_chain = optax.adamw(learning_rate=scheduler, weight_decay=args.weight_decay)
        muon_chain = optax.chain(
            scale_by_muon(momentum=0.95, n_steps=5, use_bfloat16=True),
            optax.add_decayed_weights(args.weight_decay),
            optax.scale_by_schedule(scheduler),
            optax.scale(-1.0)
        )
        
        def label_fn(path, p):
            path_str = "".join(str(k) for k in path).lower()
            if len(jnp.shape(p)) < 2:
                return "adamw"
            if any(x in path_str for x in ["embed", "vocab", "router", "mem_tokens"]):
                return "adamw"
            return "muon"
            
        def create_mask(params):
            return jax.tree_util.tree_map_with_path(label_fn, params)
            
        # the Split-Clip Architecture (cage AdamW router, un-cage Muon)
        adamw_clipped = optax.chain(
            optax.clip_by_global_norm(args.clip_grad_norm),
            adamw_chain
        )
            
        optimizer_def = optax.multi_transform(
            {"muon": muon_chain, "adamw": adamw_clipped}, 
            create_mask
        )
    elif args.optimizer.lower() == "adamw_poprisk":
        from optimizers.poprisk import adamw_poprisk

        poprisk_chain = adamw_poprisk(
            learning_rate=scheduler,
            weight_decay=args.weight_decay,
            rho=args.poprisk_rho,
            alpha=args.poprisk_alpha,
            lambda_pop=args.poprisk_lambda_pop,
            eps_gate=args.poprisk_eps,
            gate=args.poprisk_gate,
        )

        if args.poprisk_skip_router:
            print(f"Note: AdamW-poprisk HYBRID (ADR_016). Router params -> plain AdamW; everything else -> poprisk. gate={args.poprisk_gate}, alpha={args.poprisk_alpha}")
            adamw_chain = optax.adamw(learning_rate=scheduler, weight_decay=args.weight_decay)

            def label_fn(path, p):
                path_str = "".join(str(k) for k in path).lower()
                return "adamw" if "router" in path_str else "adamw_poprisk"

            def create_mask(params):
                return jax.tree_util.tree_map_with_path(label_fn, params)

            optimizer_def = optax.chain(
                optax.clip_by_global_norm(args.clip_grad_norm),
                optax.multi_transform(
                    {"adamw": adamw_chain, "adamw_poprisk": poprisk_chain},
                    create_mask,
                ),
            )
        else:
            print(f"Note: AdamW + population-risk gate (Litman & Guo 2026, ADR_016). gate={args.poprisk_gate}, rho={args.poprisk_rho}, alpha={args.poprisk_alpha}, lambda_pop={args.poprisk_lambda_pop}")
            optimizer_def = optax.chain(
                optax.clip_by_global_norm(args.clip_grad_norm),
                poprisk_chain,
            )
    else:
        optimizer_def = optax.chain(
            optax.clip_by_global_norm(args.clip_grad_norm),
            optax.adamw(learning_rate=scheduler, weight_decay=args.weight_decay)
        )

    
    optimizer = nnx.ModelAndOptimizer(model, optimizer_def)

    # Optional EMA Model
    ema_model = None
    if args.use_ema:
        ema_model = nnx.clone(model)
        print("EMA tracking initialized.")

    # Checkpoint setup (run_ckpt_dir already created above for wandb ID)
    checkpointer = ocp.StandardCheckpointer()
    checkpoint_manager = ocp.CheckpointManager(run_ckpt_dir, checkpointer)

    # Resume from checkpoint if requested
    step = 0
    if args.resume and checkpoint_manager.latest_step() is not None:
        resume_step = checkpoint_manager.latest_step()
        print(f"Resuming from checkpoint at step {resume_step}...")
        target_state = {"model": nnx.state(model), "optimizer": nnx.state(optimizer)}
        if ema_model is not None:
            target_state["ema_model"] = nnx.state(ema_model)
        restored = checkpoint_manager.restore(resume_step, args=ocp.args.StandardRestore(target_state))
        nnx.update(model, restored["model"])
        nnx.update(optimizer, restored["optimizer"])
        if ema_model is not None and "ema_model" in restored:
            nnx.update(ema_model, restored["ema_model"])
        step = resume_step
        print(f"Resumed. Continuing from step {step}.")
    pbar = tqdm.tqdm()
    batch_iter = prefetch_generator(train_loader)

    # Fast-forward dataloader if resuming
    if step > 0:
        print(f"Fast-forwarding dataloader past {step} batches...")
        for _ in tqdm.tqdm(range(step), desc="Skipping", leave=False):
            next(batch_iter, None)
        print(f"Dataloader ready at step {step}.")

    for sharded_batch in batch_iter:
        # Step
        step_rngs = nnx.Rngs(args.seed + step)
        if args.lambda_warmup_steps > 0:
            effective_lambda = args.ponder_lambda * min(1.0, step / args.lambda_warmup_steps)
        else:
            effective_lambda = args.ponder_lambda
        metrics = train_step(model, optimizer, sharded_batch, effective_lambda, args.num_memory_tokens, metadata.pad_id, step_rngs)
        
        if ema_model is not None:
            update_ema_step(ema_model, model, decay=0.999)
            
        # Periodic Evaluation
        if step % args.eval_interval == 0 and step > 0:
            eval_model = ema_model if ema_model is not None else model
            eval_metrics_accum = []
            
            for _ in range(args.eval_steps):
                eval_batch = next(test_iterator)
                eval_batch_metrics = eval_step(eval_model, eval_batch, args.ponder_lambda, args.num_memory_tokens, metadata.pad_id)
                eval_metrics_accum.append(eval_batch_metrics)
                
            avg_metrics = {f"eval_{k}": float(jnp.mean(jnp.array([m[k] for m in eval_metrics_accum]))) for k in eval_metrics_accum[0].keys()}
            avg_metrics["step"] = step
            wandb.log(avg_metrics)
            print(f"\n[Eval Step {step}] Loss: {avg_metrics['eval_loss']:.3f} | Acc: {avg_metrics['eval_accuracy']:.3f}")
        
        log_metrics = {k: float(v) for k, v in metrics.items()}
        log_metrics["learning_rate"] = float(scheduler(step)) if 'scheduler' in locals() else args.lr
        log_metrics["effective_lambda"] = float(effective_lambda)
        log_metrics["step"] = step
        
        wandb.log(log_metrics)
        pbar.set_description(f"Loss: {log_metrics['loss']:.3f} | Acc: {log_metrics['accuracy']:.3f} | Halt Steps: {log_metrics['mean_halt_steps']:.2f}")
        pbar.update(1)
        
        step += 1
        # In a real run, save upon epoch / interval
        if step % args.checkpoint_interval == 0:
            states = {"model": nnx.state(model), "optimizer": nnx.state(optimizer)}
            if ema_model is not None:
                states["ema_model"] = nnx.state(ema_model)
            checkpoint_manager.save(step, args=ocp.args.StandardSave(states))

    # Save absolute final model unconditionally
    print(f"\nTraining exhausted. Saving final explicit checkpoint at step {step}...")
    final_states = {"model": nnx.state(model), "optimizer": nnx.state(optimizer)}
    if ema_model is not None:
        final_states["ema_model"] = nnx.state(ema_model)
    checkpoint_manager.save(step, args=ocp.args.StandardSave(final_states))
    checkpoint_manager.wait_until_finished()

if __name__ == "__main__":
    main()
