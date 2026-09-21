import { Tex } from './Tex.tsx';
import styles from './PageHeader.module.css';

interface PageHeaderProps {
  eyebrow: string;
  title: string;
  tex: string;
  lede: string;
}

export function PageHeader({ eyebrow, title, tex, lede }: PageHeaderProps) {
  return (
    <header className={styles.header}>
      <p className={styles.eyebrow}>{eyebrow}</p>
      <h1 className={styles.title}>{title}</h1>
      <Tex tex={tex} display className={styles.equation} />
      <p className={styles.lede}>{lede}</p>
    </header>
  );
}
