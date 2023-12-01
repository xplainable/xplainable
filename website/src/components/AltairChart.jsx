import React, { useEffect, useRef } from 'react';
import vegaEmbed from 'vega-embed';
import BrowserOnly from '@docusaurus/BrowserOnly';

export const AltairChart = React.memo(({ spec }) => {
  const ref = useRef(null);

  useEffect(() => {
    if (ref.current) {
      vegaEmbed(ref.current, spec, { actions: false }).catch(console.error);
    }
  }, [spec]);

  return (
    <BrowserOnly fallback={<div>Loading chart...</div>}>
      {() => <div ref={ref} />}
    </BrowserOnly>
  );
});
