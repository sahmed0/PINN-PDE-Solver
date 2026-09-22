import type { CSSProperties } from 'react';
import styles from './RangeSlider.module.css';

interface RangeSliderProps {
  id: string;
  label: string;
  min: number;
  max: number;
  step: number;
  value: number;
  onChange: (value: number) => void;
  formatValue: (value: number) => string;
  disabled?: boolean;
}

export function RangeSlider({ id, label, min, max, step, value, onChange, formatValue, disabled }: RangeSliderProps) {
  const fill = `${((value - min) / (max - min)) * 100}%`;
  return (
    <div className={styles.slider}>
      <div className={styles.header}>
        <label htmlFor={id} className={styles.label}>{label}</label>
        <output htmlFor={id} className={styles.value}>{formatValue(value)}</output>
      </div>
      <input
        id={id}
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        disabled={disabled}
        aria-valuetext={formatValue(value)}
        className={styles.input}
        style={{ '--fill': fill } as CSSProperties}
      />
      <div className={styles.scale} aria-hidden="true">
        <span>{formatValue(min)}</span>
        <span>{formatValue(max)}</span>
      </div>
    </div>
  );
}
