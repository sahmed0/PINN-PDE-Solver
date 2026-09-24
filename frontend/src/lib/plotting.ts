import type { ErrorMetrics } from './inference.ts';

export type ViewMode = 'pinn' | 'exact' | 'error';
export type TabMode = 'forward' | 'inverse' | 'burgers';

export interface InferenceTiming {
  points: number;
  ms: number;
}

export interface PlotState {
  pinn: number[][];
  exact: number[][];
  error: number[][];
  metrics: ErrorMetrics;
  x: number[];
  y: number[];
  timing: InferenceTiming;
}

export type Colorscale = Array<[number, string]>;

// ColorBrewer RdBu (11-class), reversed: negative = blue, positive = red.
export const FIELD_COLORSCALE: Colorscale = [
  [0, '#053061'], [0.1, '#2166ac'], [0.2, '#4393c3'], [0.3, '#92c5de'], [0.4, '#d1e5f0'],
  [0.5, '#f7f7f7'],
  [0.6, '#fddbc7'], [0.7, '#f4a582'], [0.8, '#d6604d'], [0.9, '#b2182b'], [1, '#67001f'],
];

// ColorBrewer PuOr (11-class), reversed: negative = purple, positive = orange.
export const ERROR_COLORSCALE: Colorscale = [
  [0, '#2d004b'], [0.1, '#542788'], [0.2, '#8073ac'], [0.3, '#b2abd2'], [0.4, '#d8daeb'],
  [0.5, '#f7f7f7'],
  [0.6, '#fee0b6'], [0.7, '#fdb863'], [0.8, '#e08214'], [0.9, '#b35806'], [1, '#7f3b08'],
];

// Mirrors tokens.css; Plotly styles are set in JS and cannot read CSS variables.
export const PLOT_THEME = {
  fontSans: "'Inter Variable', Inter, system-ui, sans-serif",
  fontMono: "'JetBrains Mono Variable', 'JetBrains Mono', ui-monospace, monospace",
  text: '#475569',
  title: '#0f172a',
  axisLine: '#cbd5e1',
  hoverBg: '#0f172a',
  hoverText: '#f8fafc',
  markerFill: '#ffffff',
  markerLine: '#0f172a',
} as const;

export interface HeatmapTrace {
  z: number[][];
  colorscale: Colorscale;
  zmin: number;
  zmax: number;
  colorbarTitle: string;
  isError: boolean;
}

export function maxAbs(grid: number[][]): number {
  let m = 0;
  for (const row of grid) for (const v of row) if (Math.abs(v) > m) m = Math.abs(v);
  return m;
}

export function buildTrace(view: ViewMode, data: PlotState): HeatmapTrace {
  if (view === 'error') {
    const m = data.metrics.linf || 1e-9;
    return { z: data.error, colorscale: ERROR_COLORSCALE, zmin: -m, zmax: m, colorbarTitle: 'Δu', isError: true };
  }
  // One symmetric range for PINN and reference so the two views are directly comparable.
  const m = Math.max(maxAbs(data.pinn), maxAbs(data.exact)) || 1;
  return {
    z: view === 'pinn' ? data.pinn : data.exact,
    colorscale: FIELD_COLORSCALE,
    zmin: -m,
    zmax: m,
    colorbarTitle: 'u',
    isError: false,
  };
}
