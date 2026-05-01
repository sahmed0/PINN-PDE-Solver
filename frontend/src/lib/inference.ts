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
function forwardOne(model: PINNModel, input: number[]): number {
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
