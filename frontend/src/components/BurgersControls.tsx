// Burgers-tab sidebar: the fixed viscosity and the validation metrics against
// the method-of-lines numerical reference. Presentational — state lives in App.
import { formatPct } from '../lib/plotting.ts';
import type { BurgersModel } from '../lib/inference.ts';
import styles from '../App.module.css';

interface BurgersControlsProps {
  burgersModel: BurgersModel | null;
}

export function BurgersControls({ burgersModel }: BurgersControlsProps) {
  return (
    <>
      <div className={styles.controlGroup}>
        <label>
          <span>Viscosity (&nu;)</span>
          <span>{burgersModel ? burgersModel.nu.toExponential(4) : '-'}</span>
        </label>
        <p className={styles.metricNote}>
          Fixed at &nu; = 0.01/&pi; (Raissi et al. 2019). A near-shock forms
          around t &asymp; 0.7 where the field steepens sharply.
        </p>
      </div>

      {/* Accuracy vs. the method-of-lines reference (computed in Python) */}
      <div className={styles.controlGroup}>
        <label><span>Validation vs. reference</span></label>
        <div className={styles.metrics}>
          <div className={styles.metricRow}>
            <span>Relative L&#8322;</span>
            <span className={styles.metricValue}>
              {burgersModel ? formatPct(burgersModel.rel_l2) : '-'}
            </span>
          </div>
          <div className={styles.metricRow}>
            <span>L&#8734; (max abs)</span>
            <span className={styles.metricValue}>
              {burgersModel ? burgersModel.linf.toExponential(2) : '-'}
            </span>
          </div>
        </div>
        <p className={styles.metricNote}>
          u<sub>t</sub> + u&middot;u<sub>x</sub> = &nu;&middot;u<sub>xx</sub>.
          &ldquo;Exact&rdquo; here is a method-of-lines numerical reference
          (512-point grid integrated in time), not a closed form.
        </p>
        {burgersModel?.reference_uncertainty && (
          <p className={styles.metricNote}>
            Reference&rsquo;s own grid-refinement error: ~
            {formatPct(burgersModel.reference_uncertainty.rel_l2_512_vs_2048)}
            {' '}(512 vs 2048 points) &mdash; the quoted rel-L&#8322; partially
            reflects reference error near the shock.
          </p>
        )}
      </div>
    </>
  );
}
