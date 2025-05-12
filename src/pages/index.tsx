import type {ReactNode} from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import HomepageFeatures from '@site/src/components/HomepageFeatures';
import Heading from '@theme/Heading';

import styles from './index.module.css';

function HomepageHeader() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <header className={clsx('hero hero--primary', styles.heroBanner)}>
      <div className="container">
        <Heading as="h1" className="hero__title">
          Welcome to My Notes
        </Heading>
        <p className="hero__subtitle">
          Your ultimate resource for mastering System Design, Machine Learning, Generative AI, Algorithms, and more.
        </p>
      </div>
    </header>
  );
}

export default function Home(): ReactNode {
  const {siteConfig} = useDocusaurusContext();
  return (
    <Layout
      title="Welcome to My Notes"
      description="Your ultimate resource for mastering System Design, Machine Learning, Generative AI, Algorithms, and more.">
      <HomepageHeader />
      <main>
        <section className={styles.features}>
          <div className="container">
            <div className="row">
              <div className="col col--4">
                <Link to="/docs/systemdesign">
                  <img src="/img/undraw_docusaurus_tree.svg" alt="System Design" className={styles.featureImage} />
                  <h3>System Design</h3>
                </Link>
                <p>Learn how to design scalable, fault-tolerant systems with real-world examples and best practices.</p>
              </div>
              <div className="col col--4">
                <Link to="/docs/ml">
                  <img src="/img/undraw_docusaurus_react.svg" alt="Machine Learning" className={styles.featureImage} />
                  <h3>Machine Learning</h3>
                </Link>
                <p>Master the fundamentals of ML, from supervised learning to neural networks and beyond.</p>
              </div>
              <div className="col col--4">
                <Link to="/docs/genai">
                  <img src="/img/undraw_docusaurus_mountain.svg" alt="Generative AI" className={styles.featureImage} />
                  <h3>Generative AI</h3>
                </Link>
                <p>Explore the cutting-edge world of Generative AI, including GPT models, DALL-E, and more.</p>
              </div>
            </div>
            <div className="row">
              <div className="col col--4">
                <Link to="/docs/algorithms">
                  <img src="/img/feature-development-pattern.png" alt="Algorithms" className={styles.featureImage} />
                  <h3>Algorithms</h3>
                </Link>
                <p>Dive into algorithms, from sorting and searching to graph algorithms and dynamic programming.</p>
              </div>
              <div className="col col--4">
                <Link to="/docs/copilot">
                  <img src="/img/logo.svg" alt="Copilot" className={styles.featureImage} />
                  <h3>Copilot</h3>
                </Link>
                <p>Discover tips, tricks, and cheatsheets to make the most of GitHub Copilot in your development workflow.</p>
              </div>
              <div className="col col--4">
                <Link to="/blog">
                  <img src="/img/docusaurus-social-card.jpg" alt="Blogs" className={styles.featureImage} />
                  <h3>Blogs</h3>
                </Link>
                <p>Stay updated with the latest insights, tutorials, and trends in technology through our curated blogs.</p>
              </div>
            </div>
          </div>
        </section>
      </main>
    </Layout>
  );
}
