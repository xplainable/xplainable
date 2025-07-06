import clsx from 'clsx';
import Heading from '@theme/Heading';
import Link from '@docusaurus/Link';
import styles from './styles.module.css';

export default function HomepageFeatures() {
  return (
    <section className={styles.ctaSection}>
      <div className="container">
        <div className="row">
          <div className="col col--12">
            <div className={styles.ctaContent}>
              <Heading as="h2" className={styles.ctaTitle}>
                Ready to Build Transparent AI?
              </Heading>
              <p className={styles.ctaDescription}>
                Join thousands of developers and data scientists who are building more interpretable machine learning models with xplainable.
              </p>
              <div className={styles.ctaButtons}>
                <Link
                  className="button button--primary button--lg"
                  to="/docs/category/getting-started">
                  Get Started Now
                </Link>
                <Link
                  className="button button--secondary button--outline button--lg"
                  to="/docs/category/tutorials">
                  View Examples
                </Link>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
