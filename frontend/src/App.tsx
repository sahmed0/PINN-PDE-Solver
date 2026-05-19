// src/App.tsx
import { useState, useEffect, useCallback } from 'react';
import createPlotlyComponentImport from 'react-plotly.js/factory';
import Plotly from 'plotly.js-dist-min';
import type { Data } from 'plotly.js';

// react-plotly.js/factory is CommonJS; under Vite's interop the function can
// arrive on `.default` instead of as the module's default binding.
const createPlotlyComponent =
  (createPlotlyComponentImport as unknown as { default?: typeof createPlotlyComponentImport })
    .default ?? createPlotlyComponentImport;

const Plot = createPlotlyComponent(Plotly);
import {
  loadModel,
  loadInverseResult,
  generateGrid,
  runInference,
  reshapeForPlotly,
  exactGrid,
  errorGrid,
  computeErrorMetrics,
  type PINNModel,
  type ErrorMetrics,
  type InverseResult,
} from './lib/inference.ts';
import styles from './App.module.css';

// Resolution of our grid
const NX = 50;
const NT = 50;

type ViewMode = 'pinn' | 'exact' | 'error';

interface PlotState {
  pinn: number[][];
  exact: number[][];
  error: number[][];
  metrics: ErrorMetrics;
  x: number[];
  y: number[];
}

const VIEW_LABELS: Record<ViewMode, string> = {
  pinn: 'PINN',
  exact: 'Exact',
  error: 'Error',
};

function App() {
  // --- State ---
  const [model, setModel] = useState<PINNModel | null>(null);
  const [inverse, setInverse] = useState<InverseResult | null>(null);
  const [showObs, setShowObs] = useState<boolean>(false);
  const [alpha, setAlpha] = useState<number>(0.05); // Default thermal diffusivity
  const [isInferencing, setIsInferencing] = useState<boolean>(false);
  const [view, setView] = useState<ViewMode>('pinn');

  // Data for Plotly
  const [plotData, setPlotData] = useState<PlotState | null>(null);

  // --- 1. Load Model on Mount ---
  useEffect(() => {
    async function initModel() {
      try {
        console.log("Loading PINN weights...");
        // Ensure pinn_model.json is inside your frontend/public folder
        // (the Python pipeline writes it there automatically).
        const m = await loadModel('/pinn_model.json');
        setModel(m);
        console.log("Model loaded successfully!");
      } catch (err) {
        console.error("Failed to load model. Did you run the Python training pipeline to create public/pinn_model.json?", err);
      }
    }
    initModel();
  }, []);

  // --- 1b. Load the inverse-problem result (recovered alpha + observations) ---
  useEffect(() => {
    async function initInverse() {
      try {
        const r = await loadInverseResult('/inverse_model.json');
        setInverse(r);
      } catch (err) {
        // Non-fatal: the forward viewer works without it (run the Python
        // pipeline to generate public/inverse_model.json).
        console.error("Failed to load inverse result.", err);
      }
    }
    initInverse();
  }, []);

  // --- 2. Run Inference (and compute the analytical reference + error) ---
  // We use useCallback so the function doesn't recreate on every render
  const updatePrediction = useCallback(() => {
    if (!model) return;

    setIsInferencing(true);
    try {
      // Step A: Generate the input grid.
      const { inputs, numPoints, xVals, tVals } = generateGrid(NX, NT, alpha);

      // Step B: Run the PINN forward pass and reshape to [nt][nx].
      const flatOutput = runInference(model, inputs, numPoints);
      const pinn = reshapeForPlotly(flatOutput, NX, NT);

      // Step C: Evaluate the closed-form solution on the same grid, and the
      // signed error field + scalar metrics (relative L2, L-infinity).
      const exact = exactGrid(xVals, tVals, alpha);
      const error = errorGrid(pinn, exact);
      const metrics = computeErrorMetrics(pinn, exact);

      setPlotData({ pinn, exact, error, metrics, x: xVals, y: tVals });
    } catch (err) {
      console.error("Inference failed:", err);
    } finally {
      setIsInferencing(false);
    }
  }, [model, alpha]);

  // Trigger prediction when the model loads or alpha changes
  useEffect(() => {
    updatePrediction();
  }, [updatePrediction]);

  // --- Plot configuration depends on the selected view ---
  const trace = plotData ? buildTrace(view, plotData) : null;

  // --- Render ---
  return (
    <div className={styles.container}>

      {/* SIDEBAR: Controls */}
      <aside className={styles.sidebar}>
        <div className={styles.header}>
          <h1>Neural PDE Solver</h1>
          <p>Physics-Informed Neural Network (1D Heat Equation)</p>
        </div>

        <div className={styles.controlGroup}>
          <label>
            <span>Thermal Diffusivity (&alpha;)</span>
            <span>{alpha.toFixed(3)}</span>
          </label>
          <input
            type="range"
            min="0.01"
            max="0.1"
            step="0.001"
            value={alpha}
            onChange={(e) => setAlpha(parseFloat(e.target.value))}
            className={styles.slider}
            disabled={!model}
          />
        </div>

        <div className={styles.controlGroup}>
          <label><span>View</span></label>
          <div className={styles.toggle}>
            {(Object.keys(VIEW_LABELS) as ViewMode[]).map((mode) => (
              <button
                key={mode}
                type="button"
                className={view === mode ? styles.toggleActive : styles.toggleButton}
                onClick={() => setView(mode)}
                disabled={!model}
              >
                {VIEW_LABELS[mode]}
              </button>
            ))}
          </div>
        </div>

        {/* Live validation metrics vs. the analytical solution */}
        <div className={styles.controlGroup}>
          <label><span>Validation vs. analytical</span></label>
          <div className={styles.metrics}>
            <div className={styles.metricRow}>
              <span>Relative L&#8322;</span>
              <span className={styles.metricValue}>
                {plotData ? formatPct(plotData.metrics.relL2) : '—'}
              </span>
            </div>
            <div className={styles.metricRow}>
              <span>L&#8734; (max abs)</span>
              <span className={styles.metricValue}>
                {plotData ? plotData.metrics.linf.toExponential(2) : '—'}
              </span>
            </div>
          </div>
          <p className={styles.metricNote}>
            Compared against u(x,t) = sin(&pi;x)&middot;e<sup>&minus;&alpha;&pi;&sup2;t</sup>
          </p>
        </div>

        {/* Inverse problem: alpha recovered from sparse, noisy observations */}
        <div className={styles.controlGroup}>
          <label><span>Inverse problem</span></label>
          <div className={styles.metrics}>
            <div className={styles.metricRow}>
              <span>True &alpha;</span>
              <span className={styles.metricValue}>
                {inverse ? inverse.alpha_true.toFixed(4) : '—'}
              </span>
            </div>
            <div className={styles.metricRow}>
              <span>Estimated &alpha;</span>
              <span className={styles.metricValue}>
                {inverse ? inverse.alpha_est.toFixed(4) : '—'}
              </span>
            </div>
            <div className={styles.metricRow}>
              <span>Absolute error</span>
              <span className={styles.metricValue}>
                {inverse ? Math.abs(inverse.alpha_est - inverse.alpha_true).toExponential(2) : '—'}
              </span>
            </div>
          </div>
          <label style={{ fontWeight: 400, cursor: inverse ? 'pointer' : 'not-allowed' }}>
            <span>
              <input
                type="checkbox"
                checked={showObs}
                onChange={(e) => setShowObs(e.target.checked)}
                disabled={!inverse}
                style={{ marginRight: '0.5rem' }}
              />
              Show observations
            </span>
            <span className={styles.metricValue}>
              {inverse ? inverse.observations.length : '—'}
            </span>
          </label>
          <p className={styles.metricNote}>
            &alpha; recovered from {inverse ? inverse.observations.length : 'N'} noisy
            measurements (overlaid as points).
          </p>
        </div>

        <div style={{ marginTop: 'auto', fontSize: '0.8rem', color: '#9ca3af' }}>
          <p>Compute Backend: In-browser tanh-MLP</p>
          <p>Latency: {isInferencing ? "Computing..." : "Idle"}</p>
        </div>
      </aside>

      {/* MAIN: Visualization */}
      <main className={styles.main}>
        {!model ? (
          <div className={styles.loading}>Loading AI Model into Browser...</div>
        ) : !plotData || !trace ? (
          <div className={styles.loading}>Running initial inference...</div>
        ) : (
          <Plot
            data={[
              {
                z: trace.z,
                x: plotData.x, // Space (-1 to 1)
                y: plotData.y, // Time (0 to 1)
                type: 'heatmap',
                colorscale: trace.colorscale,
                zmin: trace.zmin,
                zmax: trace.zmax,
                // zmid is a valid Plotly heatmap prop; cast covers older @types.
                zmid: trace.zmid,
                colorbar: { title: { text: trace.colorbarTitle } },
              } as Data,
              // Overlay the inverse-problem observations at their (x, t) so it is
              // visually clear the network inferred alpha from these sparse points.
              ...(showObs && inverse
                ? [{
                    x: inverse.observations.map((o) => o.x),
                    y: inverse.observations.map((o) => o.t),
                    type: 'scatter',
                    mode: 'markers',
                    name: 'observations',
                    marker: {
                      color: '#ffffff',
                      size: 7,
                      line: { color: '#111827', width: 1 },
                      symbol: 'circle',
                    },
                    hovertemplate: 'x=%{x:.2f}, t=%{y:.2f}<extra>obs</extra>',
                    showlegend: false,
                  } as Data]
                : []),
            ]}
            layout={{
              title: { text: trace.title },
              xaxis: { title: { text: 'Space (x)' } },
              yaxis: { title: { text: 'Time (t)' } },
              width: 700,
              height: 550,
              margin: { t: 50, b: 50, l: 50, r: 50 },
              paper_bgcolor: 'transparent',
              plot_bgcolor: 'transparent'
            }}
            config={{ responsive: true, displayModeBar: false }}
          />
        )}
      </main>

    </div>
  );
}

// Build the Plotly trace settings for the selected view. PINN and Exact share a
// common Viridis colour range so they are directly comparable; Error uses a
// diverging scale centred at zero so over/under-prediction is obvious.
function buildTrace(view: ViewMode, data: PlotState) {
  if (view === 'error') {
    const m = data.metrics.linf || 1e-9;
    return {
      z: data.error,
      colorscale: 'RdBu' as const,
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
    colorscale: 'Viridis' as const,
    zmin,
    zmax,
    zmid: undefined as number | undefined,
    colorbarTitle: 'Temp (u)',
    title: isPinn ? 'PINN Prediction' : 'Analytical Solution',
  };
}

function sharedRange(a: number[][], b: number[][]): [number, number] {
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

function formatPct(x: number): string {
  return `${(x * 100).toFixed(3)}%`;
}

export default App;
