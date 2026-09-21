import type { BurgersModel, PINNModel } from '../lib/inference.ts';
import { EM_DASH, formatInt } from '../lib/format.ts';
import { architectureString, parameterCount } from '../lib/modelStats.ts';
import { ANSATZ_TEX } from '../lib/content.ts';
import { InspectorSection, KeyValue, KeyValueList } from './Inspector.tsx';
import { Tex } from './Tex.tsx';
import styles from './Inspector.module.css';

interface ModelSectionProps {
  variant: 'heat' | 'burgers';
  model: PINNModel | BurgersModel | null;
  title?: string;
  caption?: string;
}

export function ModelSection({ variant, model, title = 'Model', caption }: ModelSectionProps) {
  return (
    <InspectorSection title={title}>
      <KeyValueList>
        <KeyValue label="Architecture">{model ? architectureString(model.layers) : EM_DASH}</KeyValue>
        <KeyValue label="Parameters">{model ? formatInt(parameterCount(model.layers)) : EM_DASH}</KeyValue>
        <KeyValue label="Activation">{model ? model.activation : EM_DASH}</KeyValue>
        <KeyValue label="Inputs">{variant === 'heat' ? 'x, t, α' : 'x, t'}</KeyValue>
        <KeyValue label="Runtime">TypeScript forward pass</KeyValue>
      </KeyValueList>
      <p className={styles.formulaLabel}>Hard-constraint output</p>
      <div className={styles.formula}><Tex tex={ANSATZ_TEX[variant]} display /></div>
      {caption != null && <p className={styles.caption}>{caption}</p>}
    </InspectorSection>
  );
}
