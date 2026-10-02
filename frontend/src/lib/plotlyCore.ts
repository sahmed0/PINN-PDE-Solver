// Plotly is bundled from its module core with only the two trace types this app draws,
// instead of the ~5 MB prebuilt distribution. Adding a new chart type means registering
// its module here.
import Plotly from 'plotly.js/lib/core';
import heatmap from 'plotly.js/lib/heatmap';
import scatter from 'plotly.js/lib/scatter';

Plotly.register([heatmap, scatter]);

export default Plotly;
