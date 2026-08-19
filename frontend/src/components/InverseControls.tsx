// Inverse-tab sidebar: the recovered diffusivity with its uncertainty band and
// Cramer-Rao floor, an observations toggle, and the CRLB-by-design table.
// Presentational — state lives in App.
import type { InverseResult } from '../lib/inference.ts';
import styles from '../App.module.css';

interface InverseControlsProps {
  inverse: InverseResult | null;
  showObs: boolean;
  setShowObs: (show: boolean) => void;
}

export function InverseControls({ inverse, showObs, setShowObs }: InverseControlsProps) {
  return (
    <>
      {/* Inverse problem: alpha recovered from sparse, noisy observations */}
      <div className={styles.controlGroup}>
        <label><span>Recovered &alpha;</span></label>
        <div className={styles.metrics}>
          <div className={styles.metricRow}>
            <span>True &alpha;</span>
            <span className={styles.metricValue}>
              {inverse ? inverse.alpha_true.toFixed(4) : '-'}
            </span>
          </div>
          <div className={styles.metricRow}>
            <span>Estimate (1&sigma;)</span>
            <span className={styles.metricValue}>
              {inverse
                ? inverse.alpha_std != null
                  ? `${inverse.alpha_est.toFixed(4)} ± ${inverse.alpha_std.toFixed(4)}`
                  : inverse.alpha_est.toFixed(4)
                : '-'}
            </span>
          </div>
          <div className={styles.metricRow}>
            <span>Cramér–Rao floor</span>
            <span className={styles.metricValue}>
              {inverse && inverse.crlb_std != null ? `± ${inverse.crlb_std.toFixed(4)}` : '-'}
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
            {inverse ? inverse.observations.length : '-'}
          </span>
        </label>
        <p className={styles.metricNote}>
          &alpha; recovered from {inverse ? inverse.observations.length : 'N'} noisy
          measurements (overlaid as points). The ± band is the 1σ spread over
          {inverse?.n_seeds ? ` ${inverse.n_seeds}` : ''} noise realisations;
          the Cramér–Rao floor is the best precision any estimator could achieve
          from this data
          {inverse && inverse.crlb_std != null && inverse.alpha_std != null
            ? ` (we reach ${(inverse.alpha_std / inverse.crlb_std).toFixed(1)}× the floor).`
            : '.'}
        </p>
      </div>

      {/* CRLB floor by experiment design */}
      {inverse?.design_sweep && (
        <div className={styles.controlGroup}>
          <label><span>CRLB floor by experiment</span></label>
          <table className={styles.sweepTable}>
            <thead>
              <tr>
                <th>N</th>
                <th>&sigma;</th>
                <th>t&#8804;</th>
                <th>Floor</th>
              </tr>
            </thead>
            <tbody>
              {inverse.design_sweep.map((r, i) => {
                const active =
                  r.n_obs === inverse.n_obs &&
                  Math.abs(r.sigma - (inverse.noise_sigma ?? r.sigma)) < 1e-9 &&
                  Math.abs(r.t_max - 1.0) < 1e-9;
                return (
                  <tr key={i} className={active ? styles.sweepActive : undefined}>
                    <td>{r.n_obs}</td>
                    <td>{r.sigma}</td>
                    <td>{r.t_max}</td>
                    <td>{r.rel_pct.toFixed(2)}%</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
          <p className={styles.metricNote}>
            The information floor (best achievable 1σ on α, as % of α) versus
            the measurement design. The highlighted row is the live experiment;
            more points, lower noise, or a longer time window all lower it.
          </p>
        </div>
      )}
    </>
  );
}
