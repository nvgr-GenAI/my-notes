// Math plugin for Docusaurus
const remarkMath = require('remark-math');
const rehypeKatex = require('rehype-katex');

/** @type {import('@docusaurus/types').PluginModule} */
module.exports = function (context, options) {
  return {
    name: 'docusaurus-math-plugin',
    configureWebpack() {
      return {
        resolve: {
          symlinks: false,
        },
      };
    },
    extendDefaultPlugins: true,
    beforeDefaultRemarkPlugins: [remarkMath],
    beforeDefaultRehypePlugins: [[rehypeKatex, { strict: false, throwOnError: false, trust: true }]],
  };
};