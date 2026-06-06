// @ts-nocheck
import { describe, expect, it } from 'vitest';
import { i18n, t } from '../src/core/I18n.js';

describe('I18n', () => {
  it('should translate', () => {
    i18n.setLanguage('en');
    expect(t('lhs.title')).toBe('LHS Container');

    i18n.setLanguage('fr');
    expect(t('lhs.title')).toBe('Conteneur Gauche');
  });
});
