import type {SidebarsConfig} from '@docusaurus/plugin-content-docs';

// This runs in Node.js - Don't use client-side code here (browser APIs, JSX...)

/**
 * Creating a sidebar enables you to:
 - create an ordered group of docs
 - render a sidebar for each doc of that group
 - provide next/previous navigation

 The sidebars can be generated from the filesystem, or explicitly defined here.

 Create as many sidebars as you want.
 */
const sidebars: SidebarsConfig = {
  copilotSidebar: [
    'copilot/index', // Ensure index.md is the first document
    'copilot/copilot-feature-pattern',
  ],
  systemDesignSidebar: [
    'systemdesign/index',
    // Add other documents here in the desired order
  ],
  mlSidebar: [
    'ml/index', // Added index.md as the first document for ML
    {
      type: 'category',
      label: 'Supervised Learning',
      items: [
        'ml/supervised/intro',
        'ml/supervised/examples',
      ],
    },
    {
      type: 'category',
      label: 'Unsupervised Learning',
      items: [
        'ml/unsupervised/intro',
        'ml/unsupervised/examples',
      ],
    },
    {
      type: 'category',
      label: 'Reinforcement Learning',
      items: [
        'ml/reinforcement/intro',
        'ml/reinforcement/examples',
      ],
    },
  ],
  genaiSidebar: [
    'genai/index',
    // Add other documents here in the desired order
  ],
  algorithmsSidebar: [
    'algorithms/index',
    // Add other documents here in the desired order
  ],
};

export default sidebars;
