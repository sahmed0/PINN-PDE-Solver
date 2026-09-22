import type { BurgersModel } from '../lib/inference.ts';
import type { PlotState } from '../lib/plotting.ts';
import { EM_DASH } from '../lib/format.ts';
import { Sci } from './Sci.tsx';
import { InspectorSection, KeyValue, KeyValueList } from './Inspector.tsx';
import { ModelSection } from './ModelSection.tsx';
import { AboutSection } from './AboutSection.tsx';
import styles from './Inspector.module.css';

interface BurgersInspectorProps {
  burgersModel: BurgersModel | null;
  plot: PlotState | null;
}

export function BurgersInspector({ burgersModel, plot }: BurgersInspectorProps) {
  const nx = plot?.x.length;
  const nt = plot?.y.length;
  const refUncertainty = burgersModel?.reference_uncertainty;

  return (
    <>
      <InspectorSection title="Parameters">
        <KeyValueList>
          <KeyValue label="Viscosity ν">
            <>0.01/π ≈ <Sci value={burgersModel?.nu} /></>
          </KeyValue>
          <KeyValue label="Domain">x ∈ [−1, 1], t ∈ [0, 1]</KeyValue>
        </KeyValueList>
        <p className={styles.caption}>ν is fixed at training time, so this network takes only (x, t).</p>
      </InspectorSection>

      <InspectorSection title="Validation">
        <KeyValueList>
          <KeyValue label="Reference">Method of lines, 512 pts</KeyValue>
          <KeyValue label="Display grid">{nx != null && nt != null ? `${nx} × ${nt}` : EM_DASH}</KeyValue>
          {refUncertainty && (
            <>
              <KeyValue label="Reference error (rel. L²)"><Sci value={refUncertainty.rel_l2_512_vs_2048} /></KeyValue>
              <KeyValue label="Reference error (L∞)"><Sci value={refUncertainty.linf_512_vs_2048} /></KeyValue>
            </>
          )}
        </KeyValueList>
        {refUncertainty && (
          <p className={styles.caption}>
            Measured by grid refinement: the 512-point reference compared with a 2048-point solution.
          </p>
        )}
      </InspectorSection>

      <ModelSection variant="burgers" model={burgersModel} />
      <AboutSection tab="burgers" />
    </>
  );
}
