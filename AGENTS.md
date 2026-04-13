# AGENTS.md

## Project Overview

Nanodiffusion is a diffusion-based chat language model built from scratch in JAX/Equinox. It uses masked discrete diffusion (MDLM-style) to generate text by iteratively unmasking tokens in parallel, rather than left-to-right autoregressive generation.

## Development Commands

- `just check` - Run lint + typecheck + tests
- `just test` - Run pytest
- `just lint` - Run ruff check
- `just format` - Run ruff format
- `just typecheck` - Run basedpyright

## Architecture

- **ML framework**: JAX + Equinox (functional, JIT-compiled)
- **Type safety**: jaxtyping with beartype for runtime checks in tests, basedpyright strict mode
- **Config**: Pydantic models loaded from YAML
- **Logging**: structlog
- **CLI**: click

## Code Style

- Follow SOLID principles: single responsibility, open/closed, Liskov substitution, interface segregation, dependency inversion
- No section separator comments (e.g. `# -- section ---`). No comments that restate what the code does. Only keep comments that explain a non-obvious *why*.
- Heavy use of jaxtyping annotations on all array-valued arguments and returns
- Equinox modules with `eqx.field(static=True)` for compile-time constants
- Functional patterns: `eqx.filter_jit`, `jax.lax.scan`, `jax.vmap`
- `nonlocal key` pattern for PRNG key management
