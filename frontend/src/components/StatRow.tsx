import type { ReactNode } from 'react';
import styles from './StatRow.module.css';

export function StatRow({ children }: { children: ReactNode }) {
  return <div className={styles.row}>{children}</div>;
}

interface StatTileProps {
  label: string;
  value: ReactNode;
  sub?: ReactNode;
}

export function StatTile({ label, value, sub }: StatTileProps) {
  return (
    <div className={styles.tile}>
      <p className={styles.label}>{label}</p>
      <p className={styles.value}>{value}</p>
      {sub != null && <p className={styles.sub}>{sub}</p>}
    </div>
  );
}

export function ContextNote({ children }: { children: ReactNode }) {
  return (
    <p className={styles.note}>
      <svg className={styles.noteIcon} viewBox="0 0 16 16" width="16" height="16" aria-hidden="true">
        <circle cx="8" cy="8" r="6.5" fill="none" stroke="currentColor" strokeWidth="1.5" />
        <path d="M8 7.25v3.5" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
        <circle cx="8" cy="5" r="0.9" fill="currentColor" />
      </svg>
      <span>{children}</span>
    </p>
  );
}
