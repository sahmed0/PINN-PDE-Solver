import katex from 'katex';
import { useMemo } from 'react';
import styles from './Tex.module.css';

interface TexProps {
  tex: string;
  display?: boolean;
  className?: string;
}

export function Tex({ tex, display = false, className }: TexProps) {
  const html = useMemo(
    () => katex.renderToString(tex, { displayMode: display, throwOnError: false, output: 'htmlAndMathml' }),
    [tex, display],
  );
  const cls = [styles.tex, display ? styles.block : '', className ?? ''].filter(Boolean).join(' ');
  // TeX strings are static constants from content.ts, never user input.
  return <span className={cls} dangerouslySetInnerHTML={{ __html: html }} />;
}
