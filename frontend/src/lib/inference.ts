// Pure-TypeScript inference for the Parametric PINN.
//
// The trained model is just a small tanh-MLP (inputs [x, t, alpha] -> [u]),
// exported as JSON weights by the Python backend (see train.export_to_json).
// Running it here directly means the browser needs no ONNX runtime,
// TensorFlow, or tf2onnx toolchain.

export interface PINNLayer {
  weight: number[][]; // shape (out, in)
  bias: number[]; // shape (out,)
}

export interface PINNModel {
  format: string;
  in_size: number;
  out_size: number;
  activation: string;
  input_names: string[];
  output_names: string[];
  // Input normalisation: norm[i] = (raw[i] - input_center[i]) / input_scale[i].
  input_center: number[];
  input_scale: number[];
  // Hard-constraint reconstruction applied to the MLP output (see forwardOne).
  ansatz: string;
  layers: PINNLayer[];
}

// 0. Load the exported weights (served from frontend/public/pinn_model.json).
export async function loadModel(url = '/pinn_model.json'): Promise<PINNModel> {
  const res = await fetch(url);
  if (!res.ok) {
    throw new Error(`Failed to load model from ${url}: ${res.status} ${res.statusText}`);
  }
  const model = (await res.json()) as PINNModel;
  if (model.format !== 'tanh-mlp-heat-v2') {
    throw new Error(`Unexpected model format: ${model.format}`);
  }
  return model;
}

// Forward pass for a single input vector [x, t, alpha] -> u.
//
// This must mirror the Python model (see model.ParametricPINN.__call__):
//   1. Normalise raw inputs to ~[-1, 1] before the MLP.
//   2. Run the tanh-MLP (tanh after every layer except the linear output).
//   3. Reconstruct u via the hard-constraint ansatz so the IC/BCs are exact:
//        u = sin(pi x) + (1 - x^2) * t * N
export function forwardOne(model: PINNModel, input: number[]): number {
  const x = input[0];
  const t = input[1];

  // 1. Input normalisation.
  let activations = input.map((v, i) => (v - model.input_center[i]) / model.input_scale[i]);

  // 2. MLP.
  const layers = model.layers;
  for (let l = 0; l < layers.length; l++) {
    const { weight, bias } = layers[l];
    const out = new Array<number>(weight.length);

    for (let i = 0; i < weight.length; i++) {
      const row = weight[i];
      let sum = bias[i];
      for (let j = 0; j < row.length; j++) {
        sum += row[j] * activations[j];
      }
      // Linear output on the final layer, tanh on the hidden layers.
      out[i] = l < layers.length - 1 ? Math.tanh(sum) : sum;
    }
    activations = out;
  }
  const n = activations[0];

  // 3. Hard-constraint ansatz.
  return Math.sin(Math.PI * x) + (1 - x * x) * t * n;
}

// --- Inverse problem result -------------------------------------------------
//
// The Python backend (see inverse.export_inverse_to_json) recovers an unknown
// diffusivity alpha from sparse, noisy measurements and exports the true and
// estimated alpha plus the (x, t, u) observations it was trained on. The browser
// only needs to display these, so no network weights are shipped here.
export interface InverseObservation {
  x: number;
  t: number;
  u: number;
}

// One row of the Cramer-Rao floor-by-design table: the expected relative CRLB
// (% of alpha) for an experiment with n_obs points, noise sigma, and time
// horizon t_max. Shows how the information limit moves with the measurement.
export interface DesignRow {
  n_obs: number;
  sigma: number;
  t_max: number;
  rel_pct: number;
}

export interface InverseResult {
  format: string;
  alpha_true: number;
  alpha_est: number; // mean estimate over the noise realisations
  // Uncertainty fields (v2+): empirical 1-sigma spread of the estimate over
  // independent noise draws, and the Cramer-Rao lower bound it is measured
  // against (the best std physically attainable from this noisy data).
  alpha_std?: number;
  crlb_std?: number;
  n_obs?: number;
  noise_sigma?: number;
  n_seeds?: number;
  design_sweep?: DesignRow[];
  observations: InverseObservation[];
}

export async function loadInverseResult(url = '/inverse_model.json'): Promise<InverseResult> {
  const res = await fetch(url);
  if (!res.ok) {
    throw new Error(`Failed to load inverse result from ${url}: ${res.status} ${res.statusText}`);
  }
  const result = (await res.json()) as InverseResult;
  if (result.format !== 'inverse-heat-v2') {
    throw new Error(`Unexpected inverse result format: ${result.format}`);
  }
  return result;
}

// 1. Grid Generation (mapped to the Python training domains).
export function generateGrid(
  nx: number,
  nt: number,
  alpha: number
): { inputs: Float32Array; numPoints: number; xVals: number[]; tVals: number[] } {
  const numPoints = nx * nt;
  const inputs = new Float32Array(numPoints * 3);

  const xVals: number[] = [];
  const tVals: number[] = [];
  for (let j = 0; j < nx; j++) xVals.push(-1.0 + (j / (nx - 1)) * 2.0);
  for (let i = 0; i < nt; i++) tVals.push((i / (nt - 1)) * 1.0);

  let idx = 0;
  // Outer loop: time (0.0 to 1.0). Inner loop: space (-1.0 to 1.0).
  for (let i = 0; i < nt; i++) {
    for (let j = 0; j < nx; j++) {
      inputs[idx] = xVals[j];
      inputs[idx + 1] = tVals[i];
      inputs[idx + 2] = alpha;
      idx += 3;
    }
  }

  return { inputs, numPoints, xVals, tVals };
}

// 2. Run inference over the whole grid -> flat array of u values.
export function runInference(model: PINNModel, inputs: Float32Array, numPoints: number): Float32Array {
  const out = new Float32Array(numPoints);
  const vec = [0, 0, 0];
  for (let p = 0; p < numPoints; p++) {
    vec[0] = inputs[p * 3];
    vec[1] = inputs[p * 3 + 1];
    vec[2] = inputs[p * 3 + 2];
    out[p] = forwardOne(model, vec);
  }
  return out;
}

// 3. Reshape the flat output into a [nt][nx] grid for Plotly.
export function reshapeForPlotly(flatOutput: Float32Array, nx: number, nt: number): number[][] {
  const zData: number[][] = [];
  let idx = 0;
  for (let i = 0; i < nt; i++) {
    const row: number[] = [];
    for (let j = 0; j < nx; j++) {
      row.push(flatOutput[idx]);
      idx++;
    }
    zData.push(row);
  }
  return zData;
}

// --- Analytical reference solution -----------------------------------------
//
// The training problem (IC u(x,0)=sin(pi x), zero Dirichlet BCs) has the exact
// closed form  u(x,t) = sin(pi x) * exp(-alpha * pi^2 * t).  Mirrors the Python
// `analytical.u_exact`, so the browser can show ground truth and a live error
// field next to the PINN prediction.
export function uExact(x: number, t: number, alpha: number): number {
  return Math.sin(Math.PI * x) * Math.exp(-alpha * Math.PI * Math.PI * t);
}

// Exact field on the same grid as the PINN prediction, shaped [nt][nx].
export function exactGrid(xVals: number[], tVals: number[], alpha: number): number[][] {
  return tVals.map((t) => xVals.map((x) => uExact(x, t, alpha)));
}

// Signed difference field (pred - exact), shaped [nt][nx].
export function errorGrid(pred: number[][], exact: number[][]): number[][] {
  return pred.map((row, i) => row.map((v, j) => v - exact[i][j]));
}

export interface ErrorMetrics {
  relL2: number; // ||pred - exact||_2 / ||exact||_2
  linf: number; // max |pred - exact|
}

// Relative L2 and L-infinity error between two [nt][nx] grids.
export function computeErrorMetrics(pred: number[][], exact: number[][]): ErrorMetrics {
  let sqDiff = 0;
  let sqRef = 0;
  let linf = 0;
  for (let i = 0; i < pred.length; i++) {
    for (let j = 0; j < pred[i].length; j++) {
      const d = pred[i][j] - exact[i][j];
      sqDiff += d * d;
      sqRef += exact[i][j] * exact[i][j];
      const ad = Math.abs(d);
      if (ad > linf) linf = ad;
    }
  }
  return { relL2: sqRef > 0 ? Math.sqrt(sqDiff / sqRef) : 0, linf };
}

// --- Burgers' equation -----------------------------------------------------
//
// A second, nonlinear PDE (u_t + u u_x = nu u_xx). Like the heat model it ships
// as a tanh-MLP of JSON weights, but the inputs are [x, t] (nu is fixed) and the
// ansatz uses the -sin initial profile. There is no closed form, so the Python
// backend embeds a method-of-lines reference field and the rel_l2 / linf it
// scored against it; the browser only re-runs the forward pass for display.
export interface BurgersModel {
  format: string; // "tanh-mlp-burgers-v1"
  in_size: number;
  out_size: number;
  activation: string;
  input_center: number[];
  input_scale: number[];
  ansatz: string; // "burgers_dirichlet_negsin"
  nu: number;
  reference: { x: number[]; t: number[]; u: number[][] };
  rel_l2: number;
  linf: number;
  layers: PINNLayer[];
}

export async function loadBurgersModel(url = '/burgers_model.json'): Promise<BurgersModel> {
  const res = await fetch(url);
  if (!res.ok) {
    throw new Error(`Failed to load Burgers' model from ${url}: ${res.status} ${res.statusText}`);
  }
  const model = (await res.json()) as BurgersModel;
  if (model.format !== 'tanh-mlp-burgers-v1') {
    throw new Error(`Unexpected Burgers' model format: ${model.format}`);
  }
  return model;
}

// Forward pass for a single [x, t] -> u. Mirrors forwardOne but with 2 inputs
// and the Burgers' ansatz  u = -sin(pi x) + (1 - x^2) * t * N.
export function forwardBurgers(model: BurgersModel, input: number[]): number {
  const x = input[0];
  const t = input[1];

  let activations = input.map((v, i) => (v - model.input_center[i]) / model.input_scale[i]);

  const layers = model.layers;
  for (let l = 0; l < layers.length; l++) {
    const { weight, bias } = layers[l];
    const out = new Array<number>(weight.length);
    for (let i = 0; i < weight.length; i++) {
      const row = weight[i];
      let sum = bias[i];
      for (let j = 0; j < row.length; j++) {
        sum += row[j] * activations[j];
      }
      out[i] = l < layers.length - 1 ? Math.tanh(sum) : sum;
    }
    activations = out;
  }
  const n = activations[0];

  return -Math.sin(Math.PI * x) + (1 - x * x) * t * n;
}

// Evaluate the PINN over the full (nt x nx) grid, returning [nt][nx].
export function runBurgersInference(
  model: BurgersModel,
  xVals: number[],
  tVals: number[]
): number[][] {
  return tVals.map((t) => xVals.map((x) => forwardBurgers(model, [x, t])));
}
