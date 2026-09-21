import { useId, type ReactNode } from 'react';
import styles from './Inspector.module.css';

export function Inspector({ children }: { children: ReactNode }) {
  return <aside className={styles.inspector} aria-label="Details">{children}</aside>;
}

export function InspectorSection({ title, children }: { title: string; children: ReactNode }) {
  const id = useId();
  return (
    <section className={styles.section} aria-labelledby={id}>
      <h2 id={id} className={styles.sectionTitle}>{title}</h2>
      {children}
    </section>
  );
}

export function KeyValueList({ children }: { children: ReactNode }) {
  return <dl className={styles.kv}>{children}</dl>;
}

export function KeyValue({ label, children }: { label: ReactNode; children: ReactNode }) {
  return (
    <div className={styles.kvRow}>
      <dt className={styles.kvLabel}>{label}</dt>
      <dd className={styles.kvValue}>{children}</dd>
    </div>
  );
}
