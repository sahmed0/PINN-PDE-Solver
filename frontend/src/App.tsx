import { useState, useEffect, useCallback, type ReactNode } from 'react';
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
import {
  HEADERS,
  LOAD_LABELS,
  PLOT_TITLES,
  TAB_LABELS,
  VIEW_OPTIONS,
  BURGERS_LEDE,
  BURGERS_CONTEXT_NOTE,
  FORWARD_LEDE,
  inverseLede,
  plotSubtitle,
} from './lib/content.ts';
import { formatFixed, formatRatio } from './lib/format.ts';
import { Sci } from './components/Sci.tsx';
import { AppBar } from './components/AppBar.tsx';
import { PageHeader } from './components/PageHeader.tsx';
import { StatRow, StatTile, ContextNote } from './components/StatRow.tsx';
import { PlotCard } from './components/PlotCard.tsx';
import { HeatmapPanel } from './components/HeatmapPanel.tsx';
import { LoadingPanel, LoadErrorPanel } from './components/StatusPanel.tsx';
import { Inspector } from './components/Inspector.tsx';
import { ForwardInspector } from './components/ForwardInspector.tsx';
import { InverseInspector } from './components/InverseInspector.tsx';
import { BurgersInspector } from './components/BurgersInspector.tsx';
import { useHashTab } from './lib/useHashTab.ts';
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

interface TabStatsProps {
  tab: TabMode;
  alpha: number;
  plotData: PlotState | null;
  inverse: InverseResult | null;
  burgersModel: BurgersModel | null;
  burgersPlotData: PlotState | null;
}

function TabStats({ tab, alpha, plotData, inverse, burgersModel, burgersPlotData }: TabStatsProps) {
  if (tab === 'forward') {
    const nx = plotData?.x.length ?? NX;
    const nt = plotData?.y.length ?? NT;
    return (
      <StatRow>
        <StatTile label="Relative L² error" value={<Sci value={plotData?.metrics.relL2} />} sub="vs. closed-form solution" />
        <StatTile label="Max abs error (L∞)" value={<Sci value={plotData?.metrics.linf} />} sub={`over the ${nx} × ${nt} grid`} />
        <StatTile label="Diffusivity α" value={formatFixed(alpha, 3)} sub="trained range 0.010–0.100" />
      </StatRow>
    );
  }

  if (tab === 'inverse') {
    const ratio =
      inverse?.alpha_std != null && inverse.crlb_std != null ? inverse.alpha_std / inverse.crlb_std : undefined;
    return (
      <StatRow>
        <StatTile label="Recovered α̂" value={formatFixed(inverse?.alpha_est, 5)} sub={`true α = ${formatFixed(inverse?.alpha_true, 5)}`} />
        <StatTile
          label="Estimator spread (1σ)"
          value={<Sci value={inverse?.alpha_std} />}
          sub={inverse?.n_seeds != null ? `over ${inverse.n_seeds} noise realisations` : 'over repeated noise realisations'}
        />
        <StatTile
          label="Spread vs. Cramér–Rao floor"
          value={formatRatio(ratio)}
          sub={<>floor σ = <Sci value={inverse?.crlb_std} /></>}
        />
      </StatRow>
    );
  }

  const refUncertainty = burgersModel?.reference_uncertainty;
  const showRefTile = burgersModel == null || refUncertainty != null;
  return (
    <>
      <StatRow>
        <StatTile label="Relative L² error" value={<Sci value={burgersPlotData?.metrics.relL2} />} sub="vs. method-of-lines reference" />
        <StatTile label="Max abs error (L∞)" value={<Sci value={burgersPlotData?.metrics.linf} />} sub="on the steep front, x ≈ 0" />
        {showRefTile && (
          <StatTile
            label="Reference grid error"
            value={<Sci value={refUncertainty?.rel_l2_512_vs_2048} />}
            sub="rel. L², 512 vs 2048 points"
          />
        )}
      </StatRow>
      <ContextNote>
        {showRefTile ? BURGERS_CONTEXT_NOTE : `${BURGERS_CONTEXT_NOTE.split('. ')[0]}.`}
      </ContextNote>
    </>
  );
}

function App() {
  // --- State ---
  const [model, setModel] = useState<PINNModel | null>(null);
  const [inverse, setInverse] = useState<InverseResult | null>(null);
  const [burgersModel, setBurgersModel] = useState<BurgersModel | null>(null);
  // The Burgers' fields don't depend on any slider (nu is fixed), so they are
  // computed once when the model loads rather than reactively on alpha changes.
  const [burgersPlotData, setBurgersPlotData] = useState<PlotState | null>(null);
  const [loadErrors, setLoadErrors] = useState<LoadErrors>({});
  const [tab, setTab] = useHashTab();
  const [showObs, setShowObs] = useState<boolean>(true);
  const [alpha, setAlpha] = useState<number>(0.05); // Default thermal diffusivity
  const [view, setView] = useState<ViewMode>('pinn');

  useEffect(() => {
    document.title = `${TAB_LABELS[tab]} · PINN Solver`;
  }, [tab]);

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
          setLoadErrors((e) => ({ ...e, [key]: detail }));
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
        const t0 = performance.now();
        const pinn = runBurgersInference(m, xVals, tVals);
        const ms = performance.now() - t0;
        const exact = m.reference.u;
        const metrics = computeErrorMetrics(pinn, exact);
        setBurgersPlotData({
          pinn,
          exact,
          error: errorGrid(pinn, exact),
          metrics,
          x: xVals,
          y: tVals,
          timing: { points: xVals.length * tVals.length, ms },
        });
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

    try {
      // Step A: Generate the input grid.
      const { inputs, numPoints, xVals, tVals } = generateGrid(NX, NT, displayAlpha);

      // Step B: Run the PINN forward pass and reshape to [nt][nx].
      const t0 = performance.now();
      const flatOutput = runInference(model, inputs, numPoints);
      const ms = performance.now() - t0;
      const pinn = reshapeForPlotly(flatOutput, NX, NT);

      // Step C: Evaluate the closed-form solution on the same grid, and the
      // signed error field + scalar metrics (relative L2, L-infinity).
      const exact = exactGrid(xVals, tVals, displayAlpha);
      const error = errorGrid(pinn, exact);
      const metrics = computeErrorMetrics(pinn, exact);

      setPlotData({ pinn, exact, error, metrics, x: xVals, y: tVals, timing: { points: numPoints, ms } });
    } catch (err) {
      console.error('Inference failed:', err);
    }
  }, [model, displayAlpha]);

  // Trigger prediction when the model loads or alpha changes. The forward pass
  // is measured wall-clock time (performance.now), which is inherently a side
  // effect, so it belongs in an effect rather than a pure render-time useMemo.
  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect -- derived from (model, displayAlpha), not an event; recomputing here is the intended trigger.
    updatePrediction();
  }, [updatePrediction]);

  // --- Plot configuration depends on the selected view ---
  const activePlot = tab === 'burgers' ? burgersPlotData : plotData;
  const trace = activePlot ? buildTrace(view, activePlot) : null;

  // Inverse draws its field with the heat model, so it needs both artefacts.
  const ready =
    tab === 'burgers' ? burgersModel != null
    : tab === 'inverse' ? inverse != null && model != null
    : model != null;

  const errorKey: TabMode | undefined =
    tab === 'burgers' ? (loadErrors.burgers ? 'burgers' : undefined)
    : tab === 'inverse' ? (loadErrors.inverse ? 'inverse' : loadErrors.forward ? 'forward' : undefined)
    : loadErrors.forward ? 'forward' : undefined;

  const nx = activePlot?.x.length ?? NX;
  const nt = activePlot?.y.length ?? NT;
  const header = HEADERS[tab];
  const lede =
    tab === 'forward' ? FORWARD_LEDE
    : tab === 'inverse' ? inverseLede(inverse?.n_obs, inverse?.noise_sigma, inverse?.n_seeds)
    : BURGERS_LEDE;
  const subtitle = plotSubtitle(tab, view, {
    alpha,
    alphaEst: inverse?.alpha_est,
    nObs: inverse?.observations.length ?? 0,
    showObs: showObs && inverse != null,
    nx,
    nt,
  });

  let plotBody: ReactNode;
  if (!ready && errorKey) {
    plotBody = <LoadErrorPanel title={LOAD_LABELS[errorKey]} detail={loadErrors[errorKey] ?? ''} onRetry={() => void loadAll()} />;
  } else if (!ready) {
    plotBody = <LoadingPanel label="Loading model weights…" />;
  } else if (!activePlot || !trace) {
    plotBody = <LoadingPanel label="Running first forward pass…" />;
  } else {
    plotBody = (
      <HeatmapPanel
        trace={trace}
        plot={activePlot}
        observations={tab === 'inverse' && showObs && inverse ? inverse.observations : undefined}
      />
    );
  }

  return (
    <div className={styles.shell}>
      <AppBar tab={tab} onTabChange={setTab} />
      <main className={styles.main}>
        <div id="workspace" role="tabpanel" aria-labelledby={`tab-${tab}`} className={styles.workspace}>
          <div className={styles.primary}>
            <PageHeader eyebrow={header.eyebrow} title={header.title} tex={header.tex} lede={lede} />
            <TabStats tab={tab} alpha={alpha} plotData={plotData} inverse={inverse}
              burgersModel={burgersModel} burgersPlotData={burgersPlotData} />
            <PlotCard
              title={PLOT_TITLES[tab][view]}
              subtitle={subtitle}
              view={view}
              viewOptions={VIEW_OPTIONS[tab]}
              onViewChange={setView}
              timing={activePlot?.timing ?? null}
            >
              {plotBody}
            </PlotCard>
          </div>
          <Inspector>
            {tab === 'forward' && <ForwardInspector model={model} alpha={alpha} setAlpha={setAlpha} plotData={plotData} />}
            {tab === 'inverse' && <InverseInspector inverse={inverse} model={model} showObs={showObs} setShowObs={setShowObs} />}
            {tab === 'burgers' && <BurgersInspector burgersModel={burgersModel} plot={burgersPlotData} />}
          </Inspector>
        </div>
      </main>
    </div>
  );
}

export default App;
