// plotly.js-dist-min ships the bundled runtime without its own type
// declarations. It is API-compatible with plotly.js, whose types are provided
// by @types/plotly.js (a transitive dependency of @types/react-plotly.js), so
// re-export those here to give the default import a real type.
declare module 'plotly.js-dist-min' {
  import Plotly from 'plotly.js';
  export default Plotly;
  export * from 'plotly.js';
}
