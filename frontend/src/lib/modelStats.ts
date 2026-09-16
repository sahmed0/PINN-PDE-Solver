import type { PINNLayer } from './inference.ts';

export function parameterCount(layers: PINNLayer[]): number {
  return layers.reduce((n, l) => n + l.weight.length * (l.weight[0]?.length ?? 0) + l.bias.length, 0);
}

export function architectureString(layers: PINNLayer[]): string {
  if (layers.length === 0) return '';
  return [layers[0].weight[0].length, ...layers.map((l) => l.weight.length)].join(' → ');
}
