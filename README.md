# nanodiffusion

Diffusion-based chat language model built from scratch in JAX/Equinox. Uses masked discrete diffusion (MDLM-style) to generate text by iteratively unmasking tokens in parallel, rather than left-to-right autoregressive generation.

## Setup

```bash
uv sync --all-extras
```

## Development

```bash
just check    # lint + typecheck + test
just test     # pytest only
just lint     # ruff check
just format   # ruff format
just typecheck # basedpyright
```

## License

MIT
