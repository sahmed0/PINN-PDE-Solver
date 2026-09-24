import { useMemo } from 'react';
import createPlotlyComponentImport from 'react-plotly.js/factory';
import Plotly from 'plotly.js-dist-min';
import type { Config, Data, Layout } from 'plotly.js';

import { PLOT_THEME, type HeatmapTrace, type PlotState } from '../lib/plotting.ts';
import type { InverseObservation } from '../lib/inference.ts';

// react-plotly.js/factory is CommonJS; under Vite's interop the function can arrive on `.default`.
const createPlotlyComponent =
  (createPlotlyComponentImport as unknown as { default?: typeof createPlotlyComponentImport })
    .default ?? createPlotlyComponentImport;

const Plot = createPlotlyComponent(Plotly);

const CONFIG: Partial<Config> = { responsive: true, displayModeBar: false, scrollZoom: false, doubleClick: false };

function axis(title: string): Partial<Layout['xaxis']> {
  return {
    title: { text: title, standoff: 8, font: { family: PLOT_THEME.fontSans, size: 13, color: PLOT_THEME.title } },
    showgrid: false,
    zeroline: false,
    showline: true,
    mirror: true,
    linecolor: PLOT_THEME.axisLine,
    ticks: 'outside',
    ticklen: 4,
    tickcolor: PLOT_THEME.axisLine,
    tickfont: { family: PLOT_THEME.fontMono, size: 11, color: PLOT_THEME.text },
    fixedrange: true,
  };
}

// Built per mount: Plotly may write computed ranges back into the layout object it is given.
function makeLayout(): Partial<Layout> {
  return {
    autosize: true,
    margin: { t: 12, r: 12, b: 48, l: 52, pad: 0 },
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
    font: { family: PLOT_THEME.fontSans, size: 12, color: PLOT_THEME.text },
    xaxis: axis('x'),
    yaxis: axis('t'),
    hoverlabel: {
      bgcolor: PLOT_THEME.hoverBg,
      bordercolor: PLOT_THEME.hoverBg,
      font: { family: PLOT_THEME.fontMono, size: 12, color: PLOT_THEME.hoverText },
    },
    dragmode: false,
  };
}

interface HeatmapPanelProps {
  trace: HeatmapTrace;
  plot: PlotState;
  observations?: InverseObservation[];
}

export function HeatmapPanel({ trace, plot, observations }: HeatmapPanelProps) {
  const layout = useMemo(() => makeLayout(), []);

  const heatmap = {
    type: 'heatmap',
    z: trace.z,
    x: plot.x,
    y: plot.y,
    colorscale: trace.colorscale,
    zmin: trace.zmin,
    zmax: trace.zmax,
    zsmooth: 'best',
    hovertemplate: trace.isError
      ? 'x = %{x:.3f}<br>t = %{y:.3f}<br>Δu = %{z:.2e}<extra></extra>'
      : 'x = %{x:.3f}<br>t = %{y:.3f}<br>u = %{z:.4f}<extra></extra>',
    colorbar: {
      title: { text: trace.colorbarTitle, side: 'right', font: { family: PLOT_THEME.fontSans, size: 13, color: PLOT_THEME.title } },
      thickness: 10,
      len: 1,
      outlinewidth: 0,
      xpad: 12,
      ypad: 0,
      ticks: 'outside',
      ticklen: 3,
      tickcolor: PLOT_THEME.axisLine,
      tickfont: { family: PLOT_THEME.fontMono, size: 11, color: PLOT_THEME.text },
      exponentformat: 'power',
      ...(trace.isError ? {} : { tickformat: '.1f' }),
    },
  } as Data;

  const markers = observations
    ? [{
        type: 'scatter',
        mode: 'markers',
        x: observations.map((o) => o.x),
        y: observations.map((o) => o.t),
        customdata: observations.map((o) => o.u),
        marker: { color: PLOT_THEME.markerFill, size: 6, opacity: 0.95, line: { color: PLOT_THEME.markerLine, width: 1 } },
        hovertemplate: 'Observation<br>x = %{x:.3f}<br>t = %{y:.3f}<br>u = %{customdata:.4f}<extra></extra>',
        showlegend: false,
      } as Data]
    : [];

  return (
    <Plot
      data={[heatmap, ...markers]}
      layout={layout}
      config={CONFIG}
      useResizeHandler
      style={{ width: '100%', height: '100%' }}
    />
  );
}
