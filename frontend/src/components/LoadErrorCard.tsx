// Shown in the main area when the active tab's model failed to load: the error
// message, a hint about where the JSONs come from, and a Retry button.
import styles from '../App.module.css';

interface LoadErrorCardProps {
  message: string;
  onRetry: () => void;
}

export function LoadErrorCard({ message, onRetry }: LoadErrorCardProps) {
  return (
    <div className={styles.errorCard}>
      <p className={styles.errorMessage}>{message}</p>
      <p className={styles.errorHint}>
        The model JSONs are served from <code>frontend/public/</code> — run{' '}
        <code>pnpm dev</code> from <code>frontend/</code>.
      </p>
      <button type="button" className={styles.retryButton} onClick={onRetry}>
        Retry
      </button>
    </div>
  );
}
