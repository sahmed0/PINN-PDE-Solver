import { useCallback, useEffect, useState } from 'react';
import type { TabMode } from './plotting.ts';
import { TAB_ORDER } from './content.ts';

function readHash(): TabMode {
  const h = window.location.hash.slice(1);
  return (TAB_ORDER as readonly string[]).includes(h) ? (h as TabMode) : 'forward';
}

export function useHashTab(): [TabMode, (tab: TabMode) => void] {
  const [tab, setTabState] = useState<TabMode>(readHash);

  useEffect(() => {
    const sync = () => setTabState(readHash());
    window.addEventListener('hashchange', sync);
    window.addEventListener('popstate', sync);
    return () => {
      window.removeEventListener('hashchange', sync);
      window.removeEventListener('popstate', sync);
    };
  }, []);

  const setTab = useCallback((next: TabMode) => {
    setTabState(next);
    if (window.location.hash !== `#${next}`) window.history.pushState(null, '', `#${next}`);
  }, []);

  return [tab, setTab];
}
