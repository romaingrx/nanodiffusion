# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "nanodiffusion @ file:///${PROJECT_ROOT}",
#     "matplotlib",
#     "optax",
# ]
# ///
"""Overfit a tiny model on a short pattern, then greedily unmask from scratch."""

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import optax

from nanodiffusion.config import ModelConfig
from nanodiffusion.model.transformer import Transformer
from nanodiffusion.pretrain import make_train_step
from nanodiffusion.schedule import LogLinearSchedule
from nanodiffusion.types import PRNGKeyArray

VOCAB_SIZE = 8
MASK_ID = VOCAB_SIZE - 1
SEQ_LEN = 16
BATCH_SIZE = 64
NUM_STEPS = 1500
LR = 3e-3

config = ModelConfig(
    vocab_size=VOCAB_SIZE,
    num_layers=2,
    hidden_dim=64,
    num_heads=4,
    max_seq_len=SEQ_LEN,
)
schedule = LogLinearSchedule()
pattern = jnp.array([0, 1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1, 0, 1, 2, 3])
batch = jnp.stack([pattern] * BATCH_SIZE)


def bounded_sampler(batch_size: int, *, key: PRNGKeyArray) -> jax.Array:
    """Stratified sampling with eps=0.05, capping NELBO weight at ~20."""
    eps = 0.05
    u = jax.random.uniform(key, (1,))
    t_batch = (u / batch_size + jnp.arange(batch_size) / batch_size) % 1
    return (1 - 2 * eps) * t_batch + eps


key = jax.random.PRNGKey(42)
key, model_key = jax.random.split(key)
model = Transformer(config, key=model_key)
ema_model = model

lr_schedule = optax.warmup_cosine_decay_schedule(
    init_value=0.0,
    peak_value=LR,
    warmup_steps=200,
    decay_steps=NUM_STEPS,
)
optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(lr_schedule))
opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

train_step = make_train_step(
    optimizer,
    schedule=schedule,
    mask_token_id=MASK_ID,
    ema_decay=0.0,
    sampler=bounded_sampler,
)

print("Training...")
losses = []
for step in range(NUM_STEPS):
    key, step_key = jax.random.split(key)
    model, ema_model, opt_state, metrics = train_step(
        model, ema_model, opt_state, batch, step_key
    )
    losses.append(float(metrics["loss"]))
    if step % 500 == 0 or step == NUM_STEPS - 1:
        print(f"  step {step:5d} | loss {losses[-1]:.4f}")

print("\nGreedy unmasking...")
xt = jnp.full(SEQ_LEN, MASK_ID)
trace = [np.array(xt)]

for _ in range(SEQ_LEN):
    t = jnp.array((xt == MASK_ID).sum() / SEQ_LEN).clip(0.05, 0.95)
    logits = model(xt, t)
    probs = jax.nn.softmax(logits[:, :MASK_ID], axis=-1)

    confidence = probs.max(axis=-1)
    confidence = jnp.where(xt == MASK_ID, confidence, -1.0)
    pos = int(jnp.argmax(confidence))
    xt = xt.at[pos].set(int(jnp.argmax(probs[pos])))
    trace.append(np.array(xt))

matches = int((trace[-1] == np.array(pattern)).sum())
print(f"Result: {trace[-1].tolist()}")
print(f"Target: {pattern.tolist()}")
print(f"Match:  {matches}/{SEQ_LEN}")

TOKEN_COLORS = [
    "#e6194b",
    "#3cb44b",
    "#4363d8",
    "#f58231",
    "#911eb4",
    "#42d4f4",
    "#f032e6",
]
MASK_COLOR = "#d3d3d3"

fig, (ax_loss, ax_grid) = plt.subplots(
    2, 1, figsize=(10, 7), height_ratios=[1, 2.2], gridspec_kw={"hspace": 0.35}
)

ax_loss.plot(losses, color="steelblue", linewidth=1.2)
ax_loss.set(
    xlabel="Step", ylabel="Loss", title="Training loss (overfit on single pattern)"
)
ax_loss.set_yscale("log")

grid = np.stack(trace)
n_rows, n_cols = grid.shape
img = np.zeros((n_rows + 1, n_cols, 3))

for i in range(n_rows):
    for j in range(n_cols):
        tok = int(grid[i, j])
        color = MASK_COLOR if tok == MASK_ID else TOKEN_COLORS[tok]
        img[i, j] = mcolors.to_rgb(color)

for j in range(n_cols):
    img[n_rows, j] = mcolors.to_rgb(TOKEN_COLORS[int(pattern[j])])

ax_grid.imshow(img, aspect="auto", interpolation="nearest")

for i in range(n_rows):
    for j in range(n_cols):
        tok = int(grid[i, j])
        if tok != MASK_ID:
            ax_grid.text(
                j,
                i,
                str(tok),
                ha="center",
                va="center",
                fontsize=7,
                fontweight="bold",
                color="white",
            )

for j in range(n_cols):
    ax_grid.text(
        j,
        n_rows,
        str(int(pattern[j])),
        ha="center",
        va="center",
        fontsize=7,
        fontweight="bold",
        color="white",
    )

ax_grid.axhline(y=n_rows - 0.5, color="black", linewidth=2)
ax_grid.set(
    xlabel="Position",
    ylabel="Unmasking step",
    title=f"Greedy unmasking ({matches}/{SEQ_LEN} correct)",
)
yticks = [*list(range(0, n_rows, 4)), n_rows]
ax_grid.set_yticks(yticks)
ax_grid.set_yticklabels([str(t) for t in yticks[:-1]] + ["target"])

out = "overfit_pattern.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"\nSaved to {out}")
