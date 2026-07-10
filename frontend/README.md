# PINN PDE Solver — web frontend

A React + TypeScript app that runs the trained models **entirely in the browser**. Three tabs:

- **Forward (heat equation):** the parametric PINN's predicted field for a chosen diffusivity α,
  with live analytical-error metrics against the closed-form solution.
- **Inverse problem:** the recovered diffusivity from sparse, noisy observations, shown against
  its Cramér–Rao lower-bound (CRLB) context.
- **Burgers' equation:** the PINN solution alongside a numerical (method-of-lines) reference.

## Run it

```bash
pnpm install
pnpm dev
```

## Where the models come from

The JSON model files in `public/` (`pinn_model.json`, `inverse_model.json`, `burgers_model.json`)
are exported by the Python training pipeline — `python/src/main.py` writes them into
`frontend/public/`. Inference runs fully in-browser via a pure-TypeScript forward pass
(`src/lib/inference.ts`); there is no runtime ML framework, ONNX, or server involved.
