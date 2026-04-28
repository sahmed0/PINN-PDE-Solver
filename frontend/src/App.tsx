// src/App.tsx
import { useState, useEffect, useCallback } from 'react';
import createPlotlyComponentImport from 'react-plotly.js/factory';
import Plotly from 'plotly.js-dist-min';

// react-plotly.js/factory is CommonJS; under Vite's interop the function can
// arrive on `.default` instead of as the module's default binding.
const createPlotlyComponent =
  (createPlotlyComponentImport as unknown as { default?: typeof createPlotlyComponentImport })
    .default ?? createPlotlyComponentImport;

const Plot = createPlotlyComponent(Plotly);
import {
  loadModel,
  generateGrid,
  runInference,
  reshapeForPlotly,
  type PINNModel,
} from './lib/inference.ts';
import styles from './App.module.css';

// Resolution of our grid
const NX = 50;
const NT = 50;

function App() {
  // --- State ---
  const [model, setModel] = useState<PINNModel | null>(null);
  const [alpha, setAlpha] = useState<number>(0.05); // Default thermal diffusivity
  const [isInferencing, setIsInferencing] = useState<boolean>(false);

  // Data for Plotly
  const [plotData, setPlotData] = useState<{ z: number[][]; x: number[]; y: number[] } | null>(null);

  // --- 1. Load Model on Mount ---
  useEffect(() => {
    async function initModel() {
      try {
        console.log("Loading PINN weights...");
        // Ensure pinn_model.json is inside your frontend/public folder
        // (the Python pipeline writes it there automatically).
        const m = await loadModel('/pinn_model.json');
        setModel(m);
        console.log("Model loaded successfully!");
      } catch (err) {
        console.error("Failed to load model. Did you run the Python training pipeline to create public/pinn_model.json?", err);
      }
    }
    initModel();
  }, []);

  // --- 2. Run Inference ---
  // We use useCallback so the function doesn't recreate on every render
  const updatePrediction = useCallback(() => {
    if (!model) return;

    setIsInferencing(true);
    try {
      // Step A: Generate the input grid.
      const { inputs, numPoints, xVals, tVals } = generateGrid(NX, NT, alpha);

      // Step B: Run the forward pass.
      const flatOutput = runInference(model, inputs, numPoints);

      // Step C: Reshape for Plotly.
      const zData = reshapeForPlotly(flatOutput, NX, NT);

      setPlotData({ z: zData, x: xVals, y: tVals });
    } catch (err) {
      console.error("Inference failed:", err);
    } finally {
      setIsInferencing(false);
    }
  }, [model, alpha]);

  // Trigger prediction when the model loads or alpha changes
  useEffect(() => {
    updatePrediction();
  }, [updatePrediction]);

  // --- Render ---
  return (
    <div className={styles.container}>
      
      {/* SIDEBAR: Controls */}
      <aside className={styles.sidebar}>
        <div className={styles.header}>
          <h1>Neural PDE Solver</h1>
          <p>Physics-Informed Neural Network (1D Heat Equation)</p>
        </div>

        <div className={styles.controlGroup}>
          <label>
            <span>Thermal Diffusivity (&alpha;)</span>
            <span>{alpha.toFixed(3)}</span>
          </label>
          <input
            type="range"
            min="0.01"
            max="0.1"
            step="0.001"
            value={alpha}
            onChange={(e) => setAlpha(parseFloat(e.target.value))}
            className={styles.slider}
            disabled={!model}
          />
        </div>
        
        <div style={{ marginTop: 'auto', fontSize: '0.8rem', color: '#9ca3af' }}>
          <p>Compute Backend: In-browser tanh-MLP</p>
          <p>Latency: {isInferencing ? "Computing..." : "Idle"}</p>
        </div>
      </aside>

      {/* MAIN: Visualization */}
      <main className={styles.main}>
        {!model ? (
          <div className={styles.loading}>Loading AI Model into Browser...</div>
        ) : !plotData ? (
          <div className={styles.loading}>Running initial inference...</div>
        ) : (
          <Plot
            data={[
              {
                z: plotData.z,
                x: plotData.x, // Space (-1 to 1)
                y: plotData.y, // Time (0 to 1)
                type: 'heatmap',
                colorscale: 'Viridis',
                colorbar: { title: { text: 'Temp (u)' } }
              }
            ]}
            layout={{
              title: { text: 'Real-Time Solution Prediction' },
              xaxis: { title: { text: 'Space (x)' } },
              yaxis: { title: { text: 'Time (t)' } },
              width: 700,
              height: 550,
              margin: { t: 50, b: 50, l: 50, r: 50 },
              paper_bgcolor: 'transparent',
              plot_bgcolor: 'transparent'
            }}
            config={{ responsive: true, displayModeBar: false }}
          />
        )}
      </main>
      
    </div>
  );
}

export default App;