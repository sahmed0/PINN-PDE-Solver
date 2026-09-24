import { expect, test } from 'vitest';

import { buildTrace, maxAbs } from './plotting';
import type { PlotState } from './plotting';

function makeState(): PlotState {
  return {
    pinn: [[1, -3], [2, 0]],
    exact: [[1, -2], [2, 0]],
    error: [[0, -1], [0, 0]],
    metrics: { relL2: 0.1, linf: 1 },
    x: [-1, 1],
    y: [0, 1],
    timing: { points: 4, ms: 1 },
  };
}

test('maxAbs finds the largest magnitude in a grid', () => {
  expect(maxAbs([[1, -3], [2, 0]])).toBe(3);
});

test('buildTrace: pinn/exact views use a symmetric range', () => {
  const trace = buildTrace('pinn', makeState());
  expect(trace.zmin).toBe(-trace.zmax);
});

test('buildTrace: error view is scaled to L-infinity', () => {
  const s = makeState();
  const trace = buildTrace('error', s);
  expect(trace.zmax).toBe(s.metrics.linf);
  expect(trace.colorbarTitle).toBe('Δu');
});
