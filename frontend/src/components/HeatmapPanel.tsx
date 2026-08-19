// The main visualization: a responsive Plotly heatmap of the active field
// (PINN / Exact / Error), with an optional overlay of the inverse-problem
// observations. Presentational — all field data arrives via props.
import createPlotlyComponentImport from 'react-plotly.js/factory';
import Plotly from 'plotly.js-dist-min';
import type { Data } from 'plotly.js';

import type { HeatmapTrace, PlotState } from '../lib/plotting.ts';
import type { InverseObservation } from '../lib/inference.ts';
import styles from '../App.module.css';

// react-plotly.js/factory is CommonJS; under Vite's interop the function can
// arrive on `.default` instead of as the module's default binding.
const createPlotlyComponent =
  (createPlotlyComponentImport as unknown as { default?: typeof createPlotlyComponentImport })
    .default ?? createPlotlyComponentImport;

const Plot = createPlotlyComponent(Plotly);

interface HeatmapPanelProps {
  trace: HeatmapTrace;
  plot: PlotState;
  // When set, the inverse observations are overlaid as scatter points (the
  // caller decides whether the inverse tab is active and the toggle is on).
  observations?: InverseObservation[];
}

export function HeatmapPanel({ trace, plot, observations }: HeatmapPanelProps) {
  return (
    <div className={styles.plotWrap}>
      <Plot
        data={[
          {
            z: trace.z,
            x: plot.x, // Space (-1 to 1)
            y: plot.y, // Time (0 to 1)
            type: 'heatmap',
            colorscale: trace.colorscale,
            zmin: trace.zmin,
            zmax: trace.zmax,
            // zmid is a valid Plotly heatmap prop; cast covers older @types.
            zmid: trace.zmid,
            colorbar: { title: { text: trace.colorbarTitle } },
          } as Data,
          // Overlay the inverse-problem observations at their (x, t) so it is
          // visually clear the network inferred alpha from these sparse points.
          ...(observations
            ? [{
                x: observations.map((o) => o.x),
                y: observations.map((o) => o.t),
                type: 'scatter',
                mode: 'markers',
                name: 'observations',
                marker: {
                  color: '#ffffff',
                  size: 7,
                  line: { color: '#111827', width: 1 },
                  symbol: 'circle',
                },
                hovertemplate: 'x=%{x:.2f}, t=%{y:.2f}<extra>obs</extra>',
                showlegend: false,
              } as Data]
            : []),
        ]}
        layout={{
          title: { text: trace.title },
          xaxis: { title: { text: 'Space (x)' } },
          yaxis: { title: { text: 'Time (t)' } },
          autosize: true,
          margin: { t: 50, b: 50, l: 50, r: 50 },
          paper_bgcolor: 'transparent',
          plot_bgcolor: 'transparent',
        }}
        config={{ responsive: true, displayModeBar: false }}
        useResizeHandler
        style={{ width: '100%', height: '100%' }}
      />
    </div>
  );
}
