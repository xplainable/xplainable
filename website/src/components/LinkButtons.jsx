/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

import React from 'react';
import Link from '@docusaurus/Link';

const LinkButtons = ({githubUrl, colabUrl}) => {
  return (
    <div style={{
      display: 'flex',
      gap: '8px',
      marginBottom: '16px',
      flexWrap: 'wrap'
    }}>
      <Link 
        to={githubUrl}
        style={{
          display: 'inline-flex',
          alignItems: 'center',
          gap: '6px',
          padding: '6px 12px',
          backgroundColor: '#24292f',
          color: 'white',
          borderRadius: '16px',
          textDecoration: 'none',
          fontSize: '12px',
          fontWeight: '500',
          transition: 'all 0.2s ease',
          border: 'none'
        }}
        onMouseEnter={(e) => {
          e.target.style.backgroundColor = '#32383f';
          e.target.style.transform = 'translateY(-1px)';
        }}
        onMouseLeave={(e) => {
          e.target.style.backgroundColor = '#24292f';
          e.target.style.transform = 'translateY(0)';
        }}
      >
        <svg width="14" height="14" viewBox="0 0 16 16" fill="currentColor">
          <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"/>
        </svg>
        Open in GitHub
      </Link>
      
      <Link 
        to={colabUrl}
        style={{
          display: 'inline-flex',
          alignItems: 'center',
          gap: '6px',
          padding: '6px 12px',
          backgroundColor: '#f9ab00',
          color: 'white',
          borderRadius: '16px',
          textDecoration: 'none',
          fontSize: '12px',
          fontWeight: '500',
          transition: 'all 0.2s ease',
          border: 'none'
        }}
        onMouseEnter={(e) => {
          e.target.style.backgroundColor = '#e69100';
          e.target.style.transform = 'translateY(-1px)';
        }}
        onMouseLeave={(e) => {
          e.target.style.backgroundColor = '#f9ab00';
          e.target.style.transform = 'translateY(0)';
        }}
      >
        <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor">
          <path d="M16.9414 4.9757a7.033 7.033 0 0 0-4.9308 2.0324 7.033 7.033 0 0 0-.1232 9.8068l2.395-2.395a3.6455 3.6455 0 0 1 5.1497-5.1478l2.397-2.3989a7.033 7.033 0 0 0-4.8877-1.9175zM7.07 4.9855a7.033 7.033 0 0 0-4.8878 1.9175l2.395 2.395a3.6434 3.6434 0 0 1 5.1497 5.1497l2.395 2.395a7.033 7.033 0 0 0 .1271-9.8068 7.033 7.033 0 0 0-4.9308-2.0324h-.2532zm5.1478 10.534a3.6434 3.6434 0 0 1-5.1497-5.1497l-2.395-2.395a7.033 7.033 0 0 0-.1271 9.8068 7.033 7.033 0 0 0 9.8068.1271l-2.395-2.395z"/>
        </svg>
        Run in Google Colab
      </Link>
    </div>
  );
};

export default LinkButtons;