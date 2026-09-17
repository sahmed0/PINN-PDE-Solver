import { readFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

import { expect, test } from 'vitest';

import { architectureString, parameterCount } from './modelStats';
import type { BurgersModel, PINNModel } from './inference';

const here = dirname(fileURLToPath(import.meta.url));

function readJson<T>(relPath: string): T {
  return JSON.parse(readFileSync(resolve(here, relPath), 'utf-8')) as T;
}

test('heat model parameter count and architecture string', () => {
  const model = readJson<PINNModel>('../../public/pinn_model.json');
  expect(parameterCount(model.layers)).toBe(2273);
  expect(architectureString(model.layers)).toBe('3 → 32 → 32 → 32 → 1');
});

test('burgers model parameter count and architecture string', () => {
  const model = readJson<BurgersModel>('../../public/burgers_model.json');
  expect(parameterCount(model.layers)).toBe(12737);
  expect(architectureString(model.layers)).toBe('2 → 64 → 64 → 64 → 64 → 1');
});

test('architectureString of an empty model is empty', () => {
  expect(architectureString([])).toBe('');
});
