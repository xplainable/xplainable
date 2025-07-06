import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import HomepageFeatures from '@site/src/components/HomepageFeatures';
import { useState } from 'react';

import Heading from '@theme/Heading';
import styles from './index.module.css';

function HomepageHeader() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <header className={clsx('hero', styles.heroBanner)}>
      <div className="container">
        <div className={styles.heroContent}>
          <Heading as="h1" className={styles.heroTitle}>
            {siteConfig.title}
          </Heading>
          <p className={styles.heroSubtitle}>{siteConfig.tagline}</p>
          <div className={styles.buttons}>
            <Link
              className={clsx('button button--primary button--lg', styles.heroButton)}
              to="/docs/category/getting-started">
              Start Tutorial
            </Link>
            <Link
              className={clsx('button button--secondary button--outline button--lg', styles.heroButtonSecondary)}
              to="https://xplainable.readthedocs.io/en/latest/">
              View API Docs
            </Link>
          </div>
        </div>
      </div>
    </header>
  );
}

function QuickStartSection() {
  const [copied, setCopied] = useState(false);
  
  const copyToClipboard = () => {
    navigator.clipboard.writeText('pip install xplainable');
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <section className={styles.quickStart}>
      <div className="container">
        <div className="row">
          <div className="col col--12">
            <div className={styles.quickStartContent}>
              <Heading as="h2" className={styles.sectionTitle}>
                Get Started in Minutes
              </Heading>
              <p className={styles.sectionSubtitle}>
                Install xplainable and start building transparent machine learning models
              </p>
              <div className={styles.codeBlock}>
                <button 
                  className={styles.copyButton}
                  onClick={copyToClipboard}
                  title="Copy to clipboard"
                >
                  {copied ? '✓' : '📋'}
                </button>
                <pre>
                  <code>pip install xplainable</code>
                </pre>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}

function FeatureHighlights() {
  const features = [
    {
      title: 'Transparent AI',
      icon: '🔍',
      description: 'Build machine learning models that are fully interpretable and explainable by design.'
    },
    {
      title: 'Easy Integration',
      icon: '🔧',
      description: 'Drop-in replacement for scikit-learn with familiar APIs and seamless integration.'
    },
    {
      title: 'Production Ready',
      icon: '🚀',
      description: 'Deploy models to production with confidence using our cloud platform and monitoring tools.'
    },
    {
      title: 'Advanced Features',
      icon: '⚡',
      description: 'Leverage partitioned models, rapid refitting, and evolutionary optimization networks.'
    }
  ];

  return (
    <section className={styles.features}>
      <div className="container">
        <div className="row">
          <div className="col col--12">
            <Heading as="h2" className={styles.sectionTitle}>
              Why Choose Xplainable?
            </Heading>
          </div>
        </div>
        <div className="row">
          {features.map((feature, idx) => (
            <div key={idx} className="col col--6 col--lg-3">
              <div className={styles.featureCard}>
                <div className={styles.featureIcon}>{feature.icon}</div>
                <Heading as="h3" className={styles.featureTitle}>
                  {feature.title}
                </Heading>
                <p className={styles.featureDescription}>
                  {feature.description}
                </p>
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}

export default function Home() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <Layout
      title={`${siteConfig.title}`}
      description="Build transparent, interpretable machine learning models with xplainable - the explainable AI framework">
      <HomepageHeader />
      <main>
        <QuickStartSection />
        <FeatureHighlights />
        <HomepageFeatures />
      </main>
    </Layout>
  );
}
