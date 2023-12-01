import React, {useEffect} from 'react';
import Loadable from 'react-loadable';
import BrowserOnly from '@docusaurus/BrowserOnly';
import vegaEmbed from 'vega-embed';

export const BokehFigure = React.memo(({data}) => {
  const targetId = data['target_id'];
  return (
    <div
      className="bk-root thin-scrollbar"
      id={targetId}
      style={{overflow: 'auto', width: '100%'}}>
      <BrowserOnly fallback={<div>loading...</div>}>
        {() => {
          {
            window.Bokeh.embed.embed_item(data, targetId);
          }
        }}
      </BrowserOnly>
    </div>
  );
});

const Plotly = Loadable({
  loader: () => import(`react-plotly.js`),
  loading: ({timedOut}) =>
    timedOut ? (
      <blockquote>Error: Loading Plotly timed out.</blockquote>
    ) : (
      <div>loading...</div>
    ),
  timeout: 10000,
});

export const PlotlyFigure = React.memo(({data}) => {
  return (
    <div className="plotly-figure">
      <Plotly data={data['data']} layout={data['layout']} />
    </div>
  );
});

