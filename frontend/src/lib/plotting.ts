// Pure plotting helpers shared by App and its panel components.
//
// These are framework-free transforms over the precomputed PINN / reference /
// error fields — no React, no fetching. The heatmap trace configuration lives
// here so the view logic can be unit-reasoned independently of the components.

import type { ErrorMetrics } from './inference.ts';

export type ViewMode = 'pinn' | 'exact' | 'error';
export type TabMode = 'forward' | 'inverse' | 'burgers';

export interface PlotState {
  pinn: number[][];
  exact: number[][];
  error: number[][];
  metrics: ErrorMetrics;
  x: number[];
  y: number[];
}

export const VIEW_LABELS: Record<ViewMode, string> = {
  pinn: 'PINN',
  exact: 'Exact',
  error: 'Error',
};

export interface HeatmapTrace {
  z: number[][];
  colorscale: 'RdBu' | 'Viridis';
  zmin: number;
  zmax: number;
  zmid: number | undefined;
  colorbarTitle: string;
  title: string;
}

// Build the Plotly trace settings for the selected view. PINN and Exact share a
// common Viridis colour range so they are directly comparable; Error uses a
// diverging scale centred at zero so over/under-prediction is obvious.
export function buildTrace(
  view: ViewMode,
  data: PlotState,
  exactLabel = 'Analytical Solution',
  fieldLabel = 'Temp (u)'
): HeatmapTrace {
  if (view === 'error') {
    const m = data.metrics.linf || 1e-9;
    return {
      z: data.error,
      colorscale: 'RdBu',
      zmin: -m,
      zmax: m,
      zmid: 0,
      colorbarTitle: 'Δu',
      title: 'Error (PINN − Exact)',
    };
  }

  // Shared range across both physical fields for a fair comparison.
  const [zmin, zmax] = sharedRange(data.pinn, data.exact);
  const isPinn = view === 'pinn';
  return {
    z: isPinn ? data.pinn : data.exact,
    colorscale: 'Viridis',
    zmin,
    zmax,
    zmid: undefined,
    colorbarTitle: fieldLabel,
    title: isPinn ? 'PINN Prediction' : exactLabel,
  };
}

export function sharedRange(a: number[][], b: number[][]): [number, number] {
  let min = Infinity;
  let max = -Infinity;
  for (const grid of [a, b]) {
    for (const row of grid) {
      for (const v of row) {
        if (v < min) min = v;
        if (v > max) max = v;
      }
    }
  }
  return [min, max];
}

export function formatPct(x: number): string {
  return `${(x * 100).toFixed(3)}%`;
}
