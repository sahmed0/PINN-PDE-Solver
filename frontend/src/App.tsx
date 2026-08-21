// src/App.tsx
import { useState, useEffect, useCallback } from 'react';
import {
  loadModel,
  loadInverseResult,
  loadBurgersModel,
  runBurgersInference,
  generateGrid,
  runInference,
  reshapeForPlotly,
  exactGrid,
  errorGrid,
  computeErrorMetrics,
  type PINNModel,
  type InverseResult,
  type BurgersModel,
} from './lib/inference.ts';
import {
  buildTrace,
  type PlotState,
  type ViewMode,
  type TabMode,
} from './lib/plotting.ts';
import { HeatmapPanel } from './components/HeatmapPanel.tsx';
import { LoadErrorCard } from './components/LoadErrorCard.tsx';
import { Sidebar } from './components/Sidebar.tsx';
import { ForwardControls } from './components/ForwardControls.tsx';
import { InverseControls } from './components/InverseControls.tsx';
import { BurgersControls } from './components/BurgersControls.tsx';
import styles from './App.module.css';

// Resolution of our grid
const NX = 50;
const NT = 50;

// Per-tab load failures. A tab whose model failed to load shows an error card
// with a Retry button; the other tabs keep working (each model is independent).
interface LoadErrors {
  forward?: string;
  inverse?: string;
  burgers?: string;
}

const LOAD_LABELS: Record<keyof LoadErrors, string> = {
  forward: 'Failed to load the forward PINN.',
  inverse: 'Failed to load the inverse result.',
  burgers: "Failed to load the Burgers' model.",
};

function App() {
  // --- State ---
  const [model, setModel] = useState<PINNModel | null>(null);
  const [inverse, setInverse] = useState<InverseResult | null>(null);
  const [burgersModel, setBurgersModel] = useState<BurgersModel | null>(null);
  // The Burgers' fields don't depend on any slider (nu is fixed), so they are
  // computed once when the model loads rather than reactively on alpha changes.
  const [burgersPlotData, setBurgersPlotData] = useState<PlotState | null>(null);
  const [loadErrors, setLoadErrors] = useState<LoadErrors>({});
  const [tab, setTab] = useState<TabMode>('forward');
  const [showObs, setShowObs] = useState<boolean>(true);
  const [alpha, setAlpha] = useState<number>(0.05); // Default thermal diffusivity
  const [isInferencing, setIsInferencing] = useState<boolean>(false);
  const [view, setView] = useState<ViewMode>('pinn');

  // On the inverse tab the heatmap is rendered at the *recovered* alpha (so the
  // overlaid observations sit on the field they were inferred from); on the
  // forward tab it follows the slider.
  const displayAlpha = tab === 'inverse' && inverse ? inverse.alpha_est : alpha;
  const [plotData, setPlotData] = useState<PlotState | null>(null);

  // --- 1. Load all three models (retryable) -----------------------------------
  // Each model is loaded independently: a failure on one records a per-tab error
  // (surfaced as an error card + Retry) without blocking the others.
  const loadAll = useCallback(async () => {
    const guard = (key: keyof LoadErrors, run: () => Promise<void>) =>
      run()
        .then(() => setLoadErrors((e) => ({ ...e, [key]: undefined })))
        .catch((err: unknown) => {
          const detail = err instanceof Error ? err.message : String(err);
          setLoadErrors((e) => ({ ...e, [key]: `${LOAD_LABELS[key]} ${detail}` }));
        });

    await Promise.all([
      guard('forward', async () => setModel(await loadModel('/pinn_model.json'))),
      guard('inverse', async () => setInverse(await loadInverseResult('/inverse_model.json'))),
      guard('burgers', async () => {
        const m = await loadBurgersModel('/burgers_model.json');
        setBurgersModel(m);
        // The "Exact" field here is the embedded method-of-lines reference, not a
        // closed form. PINN/error fields are derived from it on the same grid.
        const xVals = m.reference.x;
        const tVals = m.reference.t;
        const pinn = runBurgersInference(m, xVals, tVals);
        const exact = m.reference.u;
        const metrics = computeErrorMetrics(pinn, exact);
        setBurgersPlotData({ pinn, exact, error: errorGrid(pinn, exact), metrics, x: xVals, y: tVals });
      }),
    ]);
  }, []);

  useEffect(() => {
    void loadAll();
  }, [loadAll]);

  // --- 2. Run Inference (and compute the analytical reference + error) ---
  // We use useCallback so the function doesn't recreate on every render
  const updatePrediction = useCallback(() => {
    if (!model) return;

    setIsInferencing(true);
    try {
      // Step A: Generate the input grid.
      const { inputs, numPoints, xVals, tVals } = generateGrid(NX, NT, displayAlpha);

      // Step B: Run the PINN forward pass and reshape to [nt][nx].
      const flatOutput = runInference(model, inputs, numPoints);
      const pinn = reshapeForPlotly(flatOutput, NX, NT);

      // Step C: Evaluate the closed-form solution on the same grid, and the
      // signed error field + scalar metrics (relative L2, L-infinity).
      const exact = exactGrid(xVals, tVals, displayAlpha);
      const error = errorGrid(pinn, exact);
      const metrics = computeErrorMetrics(pinn, exact);

      setPlotData({ pinn, exact, error, metrics, x: xVals, y: tVals });
    } catch (err) {
      console.error("Inference failed:", err);
    } finally {
      setIsInferencing(false);
    }
  }, [model, displayAlpha]);

  // Trigger prediction when the model loads or alpha changes
  useEffect(() => {
    updatePrediction();
  }, [updatePrediction]);

  // --- Plot configuration depends on the selected view ---
  // The Burgers' tab draws from its own precomputed fields; both share the same
  // buildTrace/sharedRange/RdBu pattern. Its "Exact" is a numerical reference.
  const activePlot = tab === 'burgers' ? burgersPlotData : plotData;
  const trace = activePlot
    ? tab === 'burgers'
      ? buildTrace(view, activePlot, 'Numerical Reference', 'Velocity (u)')
      : buildTrace(view, activePlot)
    : null;

  // The active tab's own model + error decide the main-area state: each tab is
  // gated on the artifact it actually needs (inverse on the inverse result, not
  // the heat MLP), so one failed load only darkens its own tab.
  const activeModel = tab === 'burgers' ? burgersModel : tab === 'inverse' ? inverse : model;
  const activeError = loadErrors[tab];

  // --- Render ---
  return (
    <div className={styles.container}>
      {/* SIDEBAR: Controls */}
      <Sidebar
        tab={tab}
        setTab={setTab}
        view={view}
        setView={setView}
        model={model}
        burgersModel={burgersModel}
        isInferencing={isInferencing}
      >
        {tab === 'forward' && (
          <ForwardControls model={model} alpha={alpha} setAlpha={setAlpha} plotData={plotData} />
        )}
        {tab === 'inverse' && (
          <InverseControls inverse={inverse} showObs={showObs} setShowObs={setShowObs} />
        )}
        {tab === 'burgers' && <BurgersControls burgersModel={burgersModel} />}
      </Sidebar>

      {/* MAIN: Visualization */}
      <main className={styles.main}>
        {!activeModel && activeError ? (
          <LoadErrorCard message={activeError} onRetry={() => void loadAll()} />
        ) : !activeModel ? (
          <div className={styles.loading}>Loading AI Model into Browser...</div>
        ) : !activePlot || !trace ? (
          <div className={styles.loading}>Running initial inference...</div>
        ) : (
          <HeatmapPanel
            trace={trace}
            plot={activePlot}
            observations={
              showObs && inverse && tab === 'inverse' ? inverse.observations : undefined
            }
          />
        )}
      </main>
    </div>
  );
}

export default App;
