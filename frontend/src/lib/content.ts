import type { TabMode, ViewMode } from './plotting.ts';

export const SITE_URL = 'https://pinn-pde-solver.vercel.app/';
export const REPO_URL = 'https://github.com/sahmed0/PINN-PDE-Solver';
export const PIPELINE_URL = `${REPO_URL}#3-the-azure-mlops-pipeline`;

export const ALPHA_HAT = 'α̂';

export const TAB_ORDER: readonly TabMode[] = ['forward', 'inverse', 'burgers'];

export const TAB_LABELS: Record<TabMode, string> = {
  forward: 'Heat equation',
  inverse: 'Inverse problem',
  burgers: 'Burgers’ equation',
};

export interface HeaderContent {
  eyebrow: string;
  title: string;
  tex: string;
}

export const HEADERS: Record<TabMode, HeaderContent> = {
  forward: {
    eyebrow: 'Forward problem · Parametric PINN',
    title: '1D heat equation',
    tex: String.raw`\partial_t u = \alpha\,\partial_{xx} u,\qquad x\in[-1,1],\; t\in[0,1],\qquad u(x,0)=\sin(\pi x),\; u(\pm 1,t)=0`,
  },
  inverse: {
    eyebrow: 'Inverse problem · Parameter estimation',
    title: 'Recovering diffusivity from noisy data',
    tex: String.raw`u_i^{\mathrm{obs}} = u(x_i,t_i;\alpha) + \varepsilon_i,\qquad \varepsilon_i\sim\mathcal{N}(0,\sigma^2)\quad\Longrightarrow\quad \hat{\alpha}`,
  },
  burgers: {
    eyebrow: 'Nonlinear PDE · Numerical reference',
    title: 'Viscous Burgers’ equation',
    tex: String.raw`\partial_t u + u\,\partial_x u = \nu\,\partial_{xx} u,\qquad \nu = 0.01/\pi,\qquad u(x,0)=-\sin(\pi x),\; u(\pm 1,t)=0`,
  },
};

export const FORWARD_LEDE =
  'One network, trained only on the PDE residual, represents the whole family of solutions for ' +
  'α ∈ [0.01, 0.1]. Drag α and the in-browser forward pass re-solves the field, checked live ' +
  'against the closed-form solution.';

export function inverseLede(nObs: number | undefined, sigma: number | undefined, seeds: number | undefined): string {
  const n = nObs ?? 'a few hundred';
  const noise = sigma != null ? ` with σ = ${sigma} Gaussian noise` : '';
  const reps = seeds != null ? `${seeds} noise realisations` : 'repeated noise realisations';
  return (
    `The diffusivity α is treated as unknown and recovered from ${n} sparse measurements${noise}. ` +
    `The estimator’s spread over ${reps} is compared with the Cramér–Rao lower bound: the best ` +
    'precision any unbiased estimator could achieve from this data.'
  );
}

export const BURGERS_LEDE =
  'A standard nonlinear benchmark (Raissi et al., 2019) with no closed-form solution. A steep front ' +
  'forms at x = 0: the profile steepens sharply after the inviscid breaking time t = 1/π ≈ 0.32 and ' +
  'is steepest near t ≈ 0.5. The PINN is validated against a method-of-lines reference ' +
  'integrated on a 512-point grid.';

export const BURGERS_CONTEXT_NOTE =
  'Most of the error sits on the steep front at x ≈ 0, which a smooth tanh network smears; away from ' +
  'it the model is more than an order of magnitude more accurate. The reference’s own discretisation ' +
  'error (right) accounts for only a small fraction of the figure quoted here.';

export const VIEW_OPTIONS: Record<TabMode, { value: ViewMode; label: string }[]> = {
  forward: [
    { value: 'pinn', label: 'PINN' },
    { value: 'exact', label: 'Analytic' },
    { value: 'error', label: 'Error' },
  ],
  inverse: [
    { value: 'pinn', label: 'PINN' },
    { value: 'exact', label: 'Analytic' },
    { value: 'error', label: 'Error' },
  ],
  burgers: [
    { value: 'pinn', label: 'PINN' },
    { value: 'exact', label: 'Reference' },
    { value: 'error', label: 'Error' },
  ],
};

export const PLOT_TITLES: Record<TabMode, Record<ViewMode, string>> = {
  forward: {
    pinn: 'PINN prediction u(x, t)',
    exact: 'Analytic solution u(x, t)',
    error: 'Pointwise error (PINN − analytic)',
  },
  inverse: {
    pinn: `PINN prediction at ${ALPHA_HAT}`,
    exact: `Analytic solution at ${ALPHA_HAT}`,
    error: `Pointwise error at ${ALPHA_HAT} (PINN − analytic)`,
  },
  burgers: {
    pinn: 'PINN prediction u(x, t)',
    exact: 'Method-of-lines reference u(x, t)',
    error: 'Pointwise error (PINN − reference)',
  },
};

export interface SubtitleContext {
  alpha: number;
  alphaEst: number | undefined;
  nObs: number;
  showObs: boolean;
  nx: number;
  nt: number;
}

export function plotSubtitle(tab: TabMode, view: ViewMode, c: SubtitleContext): string {
  const scale = view === 'error' ? 'symmetric scale ±L∞' : 'shared colour scale';
  if (tab === 'forward') return `α = ${c.alpha.toFixed(3)} · ${scale}`;
  if (tab === 'inverse') {
    const est = c.alphaEst != null ? `Recovered ${ALPHA_HAT} = ${c.alphaEst.toFixed(5)}` : `Recovered ${ALPHA_HAT}`;
    return c.showObs ? `${est} · ${c.nObs} observations overlaid` : est;
  }
  return `ν = 0.01/π · ${c.nx} × ${c.nt} reference grid · ${scale}`;
}

export const ENGINE_LABEL = 'Runs in your browser · no server';

export const ABOUT: Record<TabMode, string[]> = {
  forward: [
    'Hard-constraint ansatz: the output is built as u = sin(πx) + (1 − x²)·t·N, so the initial and boundary conditions hold exactly and the network only learns the interior dynamics.',
    'Inputs are normalised to roughly [−1, 1]. Without this the network collapses towards an α-averaged solution at the edges of the range.',
    'Trained in JAX/Equinox, exported as JSON weights and re-implemented as a ~30-line TypeScript forward pass, pinned to the Python reference by golden-vector parity tests at 1e-9.',
    'This model clears the same Azure ML evaluation gate that guards the registry: in-distribution rel. L² < 1e-2 and out-of-distribution < 5e-2, both measured on held-out α slices. Its measured gate result ships inside the weights file.',
  ],
  inverse: [
    'α is a trainable scalar optimised jointly with the network: the PDE residual ties the field to α, and the noisy observations pin both down.',
    'α gets its own faster optimiser, followed by an L-BFGS polish that seats it at the data optimum.',
    'The whole fit is repeated over independent noise draws to measure the estimator’s real spread, not a single lucky run.',
    'The Cramér–Rao bound is the information limit of the measurement design; the table shows how the design moves it.',
  ],
  burgers: [
    'No closed form exists, so the reference is a method-of-lines solution. Its own error is measured by grid refinement (512 vs 2048 points).',
    'Same shared MLP core and hard-constraint ansatz as the heat model, with a −sin(πx) initial profile.',
    'Nonlinear advection steepens the profile into a near-shock, a known hard case for smooth PINNs.',
  ],
};

export const ABOUT_LINK_LABEL = 'Read the full write-up';

export const ANSATZ_TEX: Record<'heat' | 'burgers', string> = {
  heat: String.raw`u = \sin(\pi x) + (1-x^2)\,t\,N_\theta(x,t,\alpha)`,
  burgers: String.raw`u = -\sin(\pi x) + (1-x^2)\,t\,N_\theta(x,t)`,
};

export const HEAT_EXACT_TEX = String.raw`u(x,t) = \sin(\pi x)\,e^{-\alpha\pi^2 t}`;

export const LOAD_LABELS: Record<TabMode, string> = {
  forward: 'The heat-equation model failed to load.',
  inverse: 'The inverse-problem result failed to load.',
  burgers: 'The Burgers’ model failed to load.',
};
