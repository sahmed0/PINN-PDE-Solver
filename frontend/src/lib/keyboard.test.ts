import { expect, test } from 'vitest';

import { nextIndex } from './keyboard';

test('nextIndex over a length-3 list', () => {
  expect(nextIndex('ArrowRight', 2, 3)).toBe(0);
  expect(nextIndex('ArrowLeft', 0, 3)).toBe(2);
  expect(nextIndex('ArrowDown', 0, 3)).toBe(1);
  expect(nextIndex('Home', 1, 3)).toBe(0);
  expect(nextIndex('End', 1, 3)).toBe(2);
  expect(nextIndex('a', 1, 3)).toBeNull();
});
