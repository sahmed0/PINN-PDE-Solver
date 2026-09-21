import styles from './StatusPanel.module.css';

export function LoadingPanel({ label }: { label: string }) {
  return (
    <div className={styles.panel} role="status" aria-live="polite">
      <div className={styles.skeleton} aria-hidden="true" />
      <p className={styles.loadingLabel}>{label}</p>
    </div>
  );
}

interface LoadErrorPanelProps {
  title: string;
  detail: string;
  onRetry: () => void;
}

export function LoadErrorPanel({ title, detail, onRetry }: LoadErrorPanelProps) {
  return (
    <div className={styles.panel}>
      <div className={styles.errorCard} role="alert">
        <p className={styles.errorTitle}>{title}</p>
        <p className={styles.errorDetail}>{detail}</p>
        <p className={styles.errorHint}>
          Model files are served from <code>frontend/public/</code>. Run <code>pnpm dev</code> from <code>frontend/</code>.
        </p>
        <button type="button" className={styles.retry} onClick={onRetry}>Retry</button>
      </div>
    </div>
  );
}
