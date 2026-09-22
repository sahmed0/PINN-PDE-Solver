import type { TabMode } from '../lib/plotting.ts';
import { ABOUT, ABOUT_LINK_LABEL, PIPELINE_URL, REPO_URL } from '../lib/content.ts';
import inspector from './Inspector.module.css';
import styles from './AboutSection.module.css';

function ChevronIcon() {
  return (
    <svg className={styles.chevron} viewBox="0 0 12 12" width="12" height="12" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d="M3 4.5 6 7.5 9 4.5" />
    </svg>
  );
}

function ExternalIcon() {
  return (
    <svg viewBox="0 0 12 12" width="12" height="12" aria-hidden="true" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
      <path d="M4.5 2.5h5v5M9.5 2.5 3 9" />
    </svg>
  );
}

export function AboutSection({ tab }: { tab: TabMode }) {
  return (
    <div className={inspector.section}>
      <details className={styles.details}>
        <summary className={styles.summary}>
          <span className={styles.summaryLabel}>How this works</span>
          <ChevronIcon />
        </summary>
        <ul className={styles.list}>
          {ABOUT[tab].map((item) => <li key={item}>{item}</li>)}
        </ul>
        <a className={styles.link} href={tab === 'forward' ? PIPELINE_URL : REPO_URL} target="_blank" rel="noreferrer">
          {ABOUT_LINK_LABEL} <ExternalIcon />
        </a>
      </details>
    </div>
  );
}
