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
    'ml/development-lifecycle', // Added ML Development Lifecycle page
    {
      type: 'category',
      label: 'Supervised Learning',
      items: [
        'ml/supervised/intro',
        {
          type: 'category',
          label: 'Regression Algorithms',
          items: [
            'ml/supervised/linear-regression',
            'ml/supervised/polynomial-regression',
            'ml/supervised/ridge-regression',
            'ml/supervised/lasso-regression',
            'ml/supervised/decision-tree-regression',
            'ml/supervised/random-forest-regression',
            'ml/supervised/svm-regression',
            'ml/supervised/neural-networks-regression',
          ]
        },
        {
          type: 'category',
          label: 'Classification Algorithms',
          items: [
            'ml/supervised/logistic-regression',
            'ml/supervised/decision-tree-classification',
            'ml/supervised/random-forest-classification',
            'ml/supervised/naive-bayes',
            'ml/supervised/k-nearest-neighbors',
            'ml/supervised/svm-classification',
            'ml/supervised/neural-networks-classification',
          ]
        },
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
      label: 'Semi-supervised Learning',
      items: [
        'ml/semi-supervised/intro',
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
    'ml/tools-frameworks', // Added Tools and Frameworks page
    'ml/industry-applications', // Moved Industry Applications page to the bottom
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
