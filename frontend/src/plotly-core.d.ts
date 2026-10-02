// plotly.js ships its per-module entry points (lib/core, lib/<trace>) without
// type declarations. @types/plotly.js (a transitive dependency of
// @types/react-plotly.js) types the package root, so re-export those for the
// core, and type the trace modules as the opaque values Plotly.register takes.
declare module 'plotly.js/lib/core' {
  import Plotly from 'plotly.js';
  export default Plotly;
  export * from 'plotly.js';
}

declare module 'plotly.js/lib/heatmap' {
  import type { PlotlyModule } from 'plotly.js';
  const heatmap: PlotlyModule;
  export default heatmap;
}

declare module 'plotly.js/lib/scatter' {
  import type { PlotlyModule } from 'plotly.js';
  const scatter: PlotlyModule;
  export default scatter;
}
