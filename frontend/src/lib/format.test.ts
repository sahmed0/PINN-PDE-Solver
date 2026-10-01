import { expect, test } from 'vitest';

import { formatFixed, formatInt, formatMs, formatPercent, formatRatio, formatRatioWithSe, formatSciText, toSciParts } from './format';

test('toSciParts', () => {
  expect(toSciParts(2.4812e-4)).toEqual({ mantissa: '2.48', exponent: -4 });
  expect(toSciParts(0.10465771704912186)).toEqual({ mantissa: '1.05', exponent: -1 });
  expect(toSciParts(1.08327317237854)).toEqual({ mantissa: '1.08', exponent: 0 });
  expect(toSciParts(9.9996e-5)).toEqual({ mantissa: '1.00', exponent: -4 });
  expect(toSciParts(0)).toEqual({ mantissa: '0.00', exponent: 0 });
});

test('formatSciText', () => {
  expect(formatSciText(-3.97e-4)).toBe('−3.97 × 10⁻⁴');
  expect(formatSciText(1.0833)).toBe('1.08');
  expect(formatSciText(undefined)).toBe('—');
  expect(formatSciText(NaN)).toBe('—');
});

test('formatFixed', () => {
  expect(formatFixed(-0.5, 2)).toBe('−0.50');
});

test('formatInt', () => {
  expect(formatInt(2500)).toBe('2,500');
});

test('formatMs', () => {
  expect(formatMs(0.4)).toBe('<1 ms');
  expect(formatMs(6.13)).toBe('6.1 ms');
  expect(formatMs(111.9)).toBe('112 ms');
});

test('formatPercent', () => {
  expect(formatPercent(0.5688286771709542)).toBe('0.57%');
});

test('formatRatio', () => {
  // Formatting only: the real spread / floor values live in inverse_model.json.
  expect(formatRatio(2.3456)).toBe('2.35×');
  expect(formatRatio(undefined)).toBe('—');
});

test('formatRatioWithSe', () => {
  expect(formatRatioWithSe(2.3456, 0.6271)).toBe('2.35× ± 0.63');
  expect(formatRatioWithSe(2.3456, undefined)).toBe('2.35×');
  expect(formatRatioWithSe(undefined, 0.44)).toBe('—');
});
