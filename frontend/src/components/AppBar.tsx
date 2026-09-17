import { useRef, type KeyboardEvent } from 'react';
import type { TabMode } from '../lib/plotting.ts';
import { PIPELINE_URL, REPO_URL, TAB_LABELS, TAB_ORDER } from '../lib/content.ts';
import { nextIndex } from '../lib/keyboard.ts';
import styles from './AppBar.module.css';

interface AppBarProps {
  tab: TabMode;
  onTabChange: (tab: TabMode) => void;
}

function BrandMark() {
  return (
    <svg className={styles.mark} viewBox="0 0 32 32" width="28" height="28" aria-hidden="true">
      <rect width="32" height="32" rx="8" fill="#0F766E" />
      <path d="M6 16c2.5-6.5 7.5-6.5 10 0s7.5 6.5 10 0" fill="none" stroke="#fff" strokeWidth="2.5" strokeLinecap="round" />
    </svg>
  );
}

function GitHubIcon() {
  return (
    <svg viewBox="0 0 16 16" width="16" height="16" aria-hidden="true" fill="currentColor">
      <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z" />
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

export function AppBar({ tab, onTabChange }: AppBarProps) {
  const refs = useRef<(HTMLButtonElement | null)[]>([]);

  const onKeyDown = (e: KeyboardEvent<HTMLButtonElement>, i: number) => {
    if (e.key === 'ArrowUp' || e.key === 'ArrowDown') return;
    const next = nextIndex(e.key, i, TAB_ORDER.length);
    if (next === null) return;
    e.preventDefault();
    refs.current[next]?.focus();
    onTabChange(TAB_ORDER[next]);
  };

  return (
    <header className={styles.appBar}>
      <div className={styles.inner}>
        <div className={styles.brand}>
          <BrandMark />
          <span className={styles.brandName}>PINN Solver</span>
        </div>

        <nav className={styles.nav} aria-label="Problems">
          <div role="tablist" aria-label="Problem" className={styles.tablist}>
            {TAB_ORDER.map((mode, i) => (
              <button
                key={mode}
                ref={(el) => { refs.current[i] = el; }}
                id={`tab-${mode}`}
                type="button"
                role="tab"
                aria-selected={tab === mode}
                aria-controls="workspace"
                tabIndex={tab === mode ? 0 : -1}
                className={styles.tab}
                onClick={() => onTabChange(mode)}
                onKeyDown={(e) => onKeyDown(e, i)}
              >
                {TAB_LABELS[mode]}
              </button>
            ))}
          </div>
        </nav>

        <div className={styles.links}>
          <a className={styles.textLink} href={PIPELINE_URL} target="_blank" rel="noreferrer">
            MLOps pipeline <ExternalIcon />
          </a>
          <a className={styles.githubButton} href={REPO_URL} target="_blank" rel="noreferrer" aria-label="Source code on GitHub">
            <GitHubIcon />
            <span className={styles.githubLabel}>Source</span>
          </a>
        </div>
      </div>
    </header>
  );
}
