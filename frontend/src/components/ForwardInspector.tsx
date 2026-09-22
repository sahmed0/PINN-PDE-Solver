import type { PINNModel } from '../lib/inference.ts';
import type { PlotState } from '../lib/plotting.ts';
import { HEAT_EXACT_TEX } from '../lib/content.ts';
import { formatFixed, formatInt } from '../lib/format.ts';
import { InspectorSection, KeyValue, KeyValueList } from './Inspector.tsx';
import { Tex } from './Tex.tsx';
import { RangeSlider } from './RangeSlider.tsx';
import { ModelSection } from './ModelSection.tsx';
import { AboutSection } from './AboutSection.tsx';
import styles from './Inspector.module.css';

interface ForwardInspectorProps {
  model: PINNModel | null;
  alpha: number;
  setAlpha: (a: number) => void;
  plotData: PlotState | null;
}

export function ForwardInspector({ model, alpha, setAlpha, plotData }: ForwardInspectorProps) {
  const nx = plotData?.x.length ?? 50;
  const nt = plotData?.y.length ?? 50;
  return (
    <>
      <InspectorSection title="Parameters">
        <RangeSlider
          id="alpha"
          label="Thermal diffusivity α"
          min={0.01}
          max={0.1}
          step={0.001}
          value={alpha}
          onChange={setAlpha}
          disabled={!model}
          formatValue={(v) => formatFixed(v, 3)}
        />
        <p className={styles.caption}>
          The network was trained across the whole range, so every value is a genuine interpolation, not a lookup.
        </p>
      </InspectorSection>
      <InspectorSection title="Validation">
        <p className={styles.formulaLabel}>Closed-form reference</p>
        <div className={styles.formula}><Tex tex={HEAT_EXACT_TEX} display /></div>
        <KeyValueList>
          <KeyValue label="Grid">{`${nx} × ${nt}`}</KeyValue>
          <KeyValue label="Points">{formatInt(nx * nt)}</KeyValue>
          <KeyValue label="Colour scale">Symmetric about 0</KeyValue>
        </KeyValueList>
      </InspectorSection>
      <ModelSection variant="heat" model={model} />
      <AboutSection tab="forward" />
    </>
  );
}
