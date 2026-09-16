import type { CSSProperties } from 'react';
import { EM_DASH, formatSciText, toSciParts } from '../lib/format.ts';

const SUP: CSSProperties = { fontSize: '0.68em', verticalAlign: '0.6em', lineHeight: 0, marginLeft: '0.08em' };

interface SciProps {
  value: number | null | undefined;
  sig?: number;
  className?: string;
}

export function Sci({ value, sig = 3, className }: SciProps) {
  if (value == null || !Number.isFinite(value)) return <span className={className}>{EM_DASH}</span>;
  const { mantissa, exponent } = toSciParts(value, sig);
  if (exponent === 0) return <span className={className}>{mantissa}</span>;
  return (
    <span className={className} title={formatSciText(value, sig)}>
      {mantissa}
      {' × 10'}
      <sup style={SUP}>{String(exponent).replace('-', '−')}</sup>
    </span>
  );
}
