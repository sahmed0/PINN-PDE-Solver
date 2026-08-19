// Sidebar chrome: title, the problem-tab bar, the shared PINN/Exact/Error view
// toggle, and the compute footer. The active tab's control body is supplied as
// children, so App composes the panels and the sidebar stays presentational.
import type { ReactNode } from 'react';
import { VIEW_LABELS, type TabMode, type ViewMode } from '../lib/plotting.ts';
import type { BurgersModel, PINNModel } from '../lib/inference.ts';
import styles from '../App.module.css';

interface SidebarProps {
  tab: TabMode;
  setTab: (tab: TabMode) => void;
  view: ViewMode;
  setView: (view: ViewMode) => void;
  model: PINNModel | null;
  burgersModel: BurgersModel | null;
  isInferencing: boolean;
  children: ReactNode;
}

export function Sidebar({
  tab,
  setTab,
  view,
  setView,
  model,
  burgersModel,
  isInferencing,
  children,
}: SidebarProps) {
  return (
    <aside className={styles.sidebar}>
      <div className={styles.header}>
        <h1>Neural PDE Solver</h1>
        <p>Physics-Informed Neural Network (1D Heat Equation)</p>
      </div>

      {/* Forward / Inverse / Burgers' problem tabs */}
      <div className={`${styles.toggle} ${styles.tabBar}`}>
        {(['forward', 'inverse', 'burgers'] as TabMode[]).map((mode) => (
          <button
            key={mode}
            type="button"
            className={tab === mode ? styles.toggleActive : styles.toggleButton}
            onClick={() => setTab(mode)}
            disabled={mode === 'burgers' ? !burgersModel : !model}
          >
            {mode === 'forward' ? 'Forward' : mode === 'inverse' ? 'Inverse' : "Burgers'"}
          </button>
        ))}
      </div>

      {/* View toggle (shared by all tabs) */}
      <div className={styles.controlGroup}>
        <label><span>View</span></label>
        <div className={styles.toggle}>
          {(Object.keys(VIEW_LABELS) as ViewMode[]).map((mode) => (
            <button
              key={mode}
              type="button"
              className={view === mode ? styles.toggleActive : styles.toggleButton}
              onClick={() => setView(mode)}
              disabled={tab === 'burgers' ? !burgersModel : !model}
            >
              {VIEW_LABELS[mode]}
            </button>
          ))}
        </div>
      </div>

      {children}

      <div style={{ marginTop: 'auto', fontSize: '0.8rem', color: '#9ca3af' }}>
        <p>Compute Backend: In-browser tanh-MLP</p>
        <p>Latency: {isInferencing ? 'Computing...' : 'Idle'}</p>
      </div>
    </aside>
  );
}
