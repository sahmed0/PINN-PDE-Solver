// Golden-vector parity tests: the pure-TS forward pass must reproduce the float64
// reference vectors embedded by the Python exporter (see python/src/json_forward.py).
// Heat vectors live inside public/pinn_model.json; Burgers vectors live in a fixture
// (the shipped burgers_model.json has no recoverable checkpoint, so its vectors are
// derived from the file itself). Tolerance 1e-9 = both sides are float64.

import { readFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

import { expect, test } from 'vitest';

import { forwardBurgers, forwardOne } from './inference';
import type { BurgersModel, InverseResult, PINNModel } from './inference';

const here = dirname(fileURLToPath(import.meta.url));

interface TestVectors {
  inputs: number[][];
  outputs: number[];
}

function readJson<T>(relPath: string): T {
  return JSON.parse(readFileSync(resolve(here, relPath), 'utf-8')) as T;
}

test('heat: forwardOne matches the embedded float64 reference vectors', () => {
  const model = readJson<PINNModel & { test_vectors: TestVectors }>(
    '../../public/pinn_model.json'
  );
  const { inputs, outputs } = model.test_vectors;
  expect(inputs.length).toBeGreaterThan(0);
  expect(inputs.length).toBe(outputs.length);
  for (let i = 0; i < inputs.length; i++) {
    expect(Math.abs(forwardOne(model, inputs[i]) - outputs[i])).toBeLessThan(1e-9);
  }
});

test('burgers: forwardBurgers matches the derived float64 reference vectors', () => {
  const model = readJson<BurgersModel>('../../public/burgers_model.json');
  const { inputs, outputs } = readJson<TestVectors>('./__fixtures__/burgers_test_vectors.json');
  expect(inputs.length).toBeGreaterThan(0);
  expect(inputs.length).toBe(outputs.length);
  for (let i = 0; i < inputs.length; i++) {
    expect(Math.abs(forwardBurgers(model, inputs[i]) - outputs[i])).toBeLessThan(1e-9);
  }
});

test('heat: forwardOne at t=0 equals sin(pi x) exactly (ansatz guarantee)', () => {
  const model = readJson<PINNModel>('../../public/pinn_model.json');
  for (const x of [-0.85, -0.3, 0.0, 0.42, 0.777]) {
    expect(forwardOne(model, [x, 0.0, 0.05])).toBe(Math.sin(Math.PI * x));
  }
});

test('inverse: result JSON matches the display schema and honest CRLB ratio', () => {
  const result = readJson<InverseResult>('../../public/inverse_model.json');
  expect(result.format).toBe('inverse-heat-v2');
  expect(typeof result.alpha_true).toBe('number');
  expect(typeof result.alpha_est).toBe('number');
  expect(typeof result.alpha_std).toBe('number');
  expect(typeof result.crlb_std).toBe('number');

  expect(Array.isArray(result.observations)).toBe(true);
  expect(result.observations.length).toBeGreaterThan(0);
  for (const obs of result.observations) {
    expect(typeof obs.x).toBe('number');
    expect(typeof obs.t).toBe('number');
    expect(typeof obs.u).toBe('number');
  }

  // The measured spread sits at ~1.6x the Cramer-Rao floor; pin it to [1, 3] so the
  // README's honest "1.6x the floor" claim can't silently drift.
  const ratio = (result.alpha_std as number) / (result.crlb_std as number);
  expect(ratio).toBeGreaterThanOrEqual(1);
  expect(ratio).toBeLessThanOrEqual(3);
});
