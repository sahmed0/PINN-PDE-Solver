import type { ReactNode } from 'react';
import type { ViewMode } from '../lib/plotting.ts';
import { ENGINE_LABEL } from '../lib/content.ts';
import { formatInt, formatMs } from '../lib/format.ts';
import { SegmentedControl } from './SegmentedControl.tsx';
import styles from './PlotCard.module.css';

interface PlotCardProps {
  title: string;
  subtitle: string;
  view: ViewMode;
  viewOptions: { value: ViewMode; label: string }[];
  onViewChange: (view: ViewMode) => void;
  timing: { points: number; ms: number } | null;
  children: ReactNode;
}

export function PlotCard({ title, subtitle, view, viewOptions, onViewChange, timing, children }: PlotCardProps) {
  return (
    <section className={styles.card} aria-labelledby="plot-title">
      <div className={styles.toolbar}>
        <div className={styles.heading}>
          <h2 id="plot-title" className={styles.title}>{title}</h2>
          <p className={styles.subtitle}>{subtitle}</p>
        </div>
        <div className={styles.toggle}>
          <SegmentedControl label="Field shown" options={viewOptions} value={view} onChange={onViewChange} />
        </div>
      </div>
      <div className={styles.plotArea}>{children}</div>
      <div className={styles.footer}>
        <span className={styles.timing}>
          <span className={styles.dot} aria-hidden="true" />
          {timing ? (
            <>Forward pass · <span className={styles.num}>{formatInt(timing.points)}</span> points in{' '}
              <span className={styles.num}>{formatMs(timing.ms)}</span></>
          ) : (
            'Forward pass · waiting for model'
          )}
        </span>
        <span>{ENGINE_LABEL}</span>
      </div>
    </section>
  );
}
