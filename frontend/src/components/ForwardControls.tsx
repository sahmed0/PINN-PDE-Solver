// Forward-tab sidebar: the thermal-diffusivity slider and the live validation
// metrics against the analytical solution. Presentational — state lives in App.
import type { PlotState } from '../lib/plotting.ts';
import { formatPct } from '../lib/plotting.ts';
import type { PINNModel } from '../lib/inference.ts';
import styles from '../App.module.css';

interface ForwardControlsProps {
  model: PINNModel | null;
  alpha: number;
  setAlpha: (alpha: number) => void;
  plotData: PlotState | null;
}

export function ForwardControls({ model, alpha, setAlpha, plotData }: ForwardControlsProps) {
  return (
    <>
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

      {/* Live validation metrics vs. the analytical solution */}
      <div className={styles.controlGroup}>
        <label><span>Validation vs. analytical</span></label>
        <div className={styles.metrics}>
          <div className={styles.metricRow}>
            <span>Relative L&#8322;</span>
            <span className={styles.metricValue}>
              {plotData ? formatPct(plotData.metrics.relL2) : '-'}
            </span>
          </div>
          <div className={styles.metricRow}>
            <span>L&#8734; (max abs)</span>
            <span className={styles.metricValue}>
              {plotData ? plotData.metrics.linf.toExponential(2) : '-'}
            </span>
          </div>
        </div>
        <p className={styles.metricNote}>
          Compared against u(x,t) = sin(&pi;x)&middot;e<sup>&minus;&alpha;&pi;&sup2;t</sup>
        </p>
      </div>
    </>
  );
}
