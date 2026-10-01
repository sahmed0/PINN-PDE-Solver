import type { InverseResult, PINNModel } from '../lib/inference.ts';
import { ALPHA_HAT } from '../lib/content.ts';
import { EM_DASH, formatFixed, formatInt, formatPercent, formatRatioWithSe } from '../lib/format.ts';
import { Sci } from './Sci.tsx';
import { InspectorSection, KeyValue, KeyValueList } from './Inspector.tsx';
import { ModelSection } from './ModelSection.tsx';
import { AboutSection } from './AboutSection.tsx';
import inspector from './Inspector.module.css';
import local from './InverseInspector.module.css';

interface InverseInspectorProps {
  inverse: InverseResult | null;
  model: PINNModel | null;
  showObs: boolean;
  setShowObs: (v: boolean) => void;
}

export function InverseInspector({ inverse, model, showObs, setShowObs }: InverseInspectorProps) {
  const ratio =
    inverse?.spread_to_crlb ??
    (inverse?.alpha_std != null && inverse.crlb_std != null
      ? inverse.alpha_std / inverse.crlb_std
      : undefined);

  return (
    <>
      <InspectorSection title="Observations">
        <label className={local.switchRow}>
          <span>Show on plot</span>
          <input
            type="checkbox"
            role="switch"
            className={local.switch}
            checked={showObs}
            onChange={(e) => setShowObs(e.target.checked)}
            disabled={!inverse}
          />
        </label>
        <KeyValueList>
          <KeyValue label="Measurements">
            {inverse ? formatInt(inverse.n_obs ?? inverse.observations.length) : EM_DASH}
          </KeyValue>
          <KeyValue label="Noise σ">{inverse ? formatFixed(inverse.noise_sigma, 3) : EM_DASH}</KeyValue>
          <KeyValue label="Noise realisations">{inverse?.n_seeds ?? EM_DASH}</KeyValue>
        </KeyValueList>
      </InspectorSection>

      <InspectorSection title="Estimate">
        <KeyValueList>
          <KeyValue label="True α">{inverse ? formatFixed(inverse.alpha_true, 5) : EM_DASH}</KeyValue>
          <KeyValue label={`Recovered ${ALPHA_HAT}`}>{inverse ? formatFixed(inverse.alpha_est, 5) : EM_DASH}</KeyValue>
          <KeyValue label="Spread (1σ)"><Sci value={inverse?.alpha_std} /></KeyValue>
          <KeyValue label="Cramér–Rao floor (1σ)"><Sci value={inverse?.crlb_std} /></KeyValue>
          <KeyValue label="Spread / floor">{formatRatioWithSe(ratio, inverse?.spread_to_crlb_se)}</KeyValue>
        </KeyValueList>
        <p className={inspector.caption}>
          The floor is the smallest standard deviation any unbiased estimator can reach from this data. A ratio of 1× would be statistically optimal.
          {inverse?.spread_to_crlb_se != null &&
            ` The ± is the sampling uncertainty of a standard deviation estimated from ${inverse.n_seeds ?? EM_DASH} noise realisations.`}
        </p>
      </InspectorSection>

      {inverse?.design_sweep && (
        <InspectorSection title="Experiment design">
          <div className={local.tableWrap}>
            <table className={local.table}>
              <caption className="visually-hidden">Cramér–Rao floor by measurement design</caption>
              <thead>
                <tr>
                  <th scope="col">Points</th>
                  <th scope="col">Noise σ</th>
                  <th scope="col">t ≤</th>
                  <th scope="col">Floor</th>
                </tr>
              </thead>
              <tbody>
                {inverse.design_sweep.map((r, i) => {
                  const active =
                    r.n_obs === inverse.n_obs &&
                    Math.abs(r.sigma - (inverse.noise_sigma ?? r.sigma)) < 1e-9 &&
                    Math.abs(r.t_max - 1.0) < 1e-9;
                  return (
                    <tr key={i} className={active ? local.active : undefined} aria-current={active ? 'true' : undefined}>
                      <td>{formatInt(r.n_obs)}</td>
                      <td>{String(r.sigma)}</td>
                      <td>{String(r.t_max)}</td>
                      <td>{formatPercent(r.rel_pct)}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
          <p className={inspector.caption}>
            Best achievable 1σ on α, as a percentage of α. Highlighted: this experiment. More points, less noise or a longer window all lower the floor.
          </p>
        </InspectorSection>
      )}

      <ModelSection
        variant="heat"
        model={model}
        title="Model (field display)"
        caption={`The inverse result ships no network weights. The field is drawn by the forward heat PINN at the recovered ${ALPHA_HAT}.`}
      />
      <AboutSection tab="inverse" />
    </>
  );
}
