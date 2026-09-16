const SUPERSCRIPT: Record<string, string> = {
  '0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴',
  '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹', '-': '⁻',
};

export const MINUS = '−';
export const EM_DASH = '—';

export interface SciParts {
  mantissa: string;
  exponent: number;
}

export function toSciParts(value: number, sig = 3): SciParts {
  if (value === 0) return { mantissa: (0).toFixed(sig - 1), exponent: 0 };
  const [m, e] = value.toExponential(sig - 1).split('e');
  return { mantissa: m.replace('-', MINUS), exponent: Number(e) };
}

export function formatSciText(value: number | null | undefined, sig = 3): string {
  if (value == null || !Number.isFinite(value)) return EM_DASH;
  const { mantissa, exponent } = toSciParts(value, sig);
  if (exponent === 0) return mantissa;
  const sup = String(exponent).split('').map((ch) => SUPERSCRIPT[ch]).join('');
  return `${mantissa} × 10${sup}`;
}

export function formatFixed(value: number | null | undefined, dp: number): string {
  if (value == null || !Number.isFinite(value)) return EM_DASH;
  return value.toFixed(dp).replace('-', MINUS);
}

export function formatInt(value: number): string {
  return value.toLocaleString('en-GB');
}

export function formatMs(ms: number): string {
  if (ms < 1) return '<1 ms';
  if (ms < 10) return `${ms.toFixed(1)} ms`;
  return `${Math.round(ms)} ms`;
}

export function formatPercent(value: number, dp = 2): string {
  return `${value.toFixed(dp)}%`;
}

export function formatRatio(value: number | null | undefined): string {
  if (value == null || !Number.isFinite(value)) return EM_DASH;
  return `${value.toFixed(2)}×`;
}
