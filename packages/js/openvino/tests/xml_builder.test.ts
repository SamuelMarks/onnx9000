import { describe, it, expect } from 'vitest';
import { XmlBuilder, XmlNode } from '../src/xml_builder.js';

describe('XmlBuilder', () => {
  it('should build simple XML with string children', () => {
    const node = new XmlNode('root');
    node.setAttribute('attr', 'val');
    node.addChild('text content');

    const builder = new XmlBuilder();
    builder.setRoot(node);
    const xml = builder.toString(true);
    expect(xml).toContain('<root attr=\"val\">');
    expect(xml).toContain('text content');
  });

  it('should handle nested nodes and pretty print', () => {
    const root = new XmlNode('root');
    const child1 = new XmlNode('child1');
    child1.addChild('line1');
    child1.addChild('line2');
    const child2 = new XmlNode('child2');
    root.addChild(child1);
    root.addChild(child2);

    const builder = new XmlBuilder();
    builder.setRoot(root);
    const xml = builder.toString(true);
    expect(xml).toContain('        line1');
    expect(xml).toContain('    <child2 />');

    // Test pretty: false with multiple children
    const xmlCompact = builder.toString(false);
    expect(xmlCompact).not.toContain('\n');
    expect(xmlCompact).toContain('line1line2');
  });

  it('should handle escaping', () => {
    const node = new XmlNode('test');
    node.setAttribute('key', '\"quoted\" & <angled>');
    node.addChild('text & <angled>');
    expect(node.toString()).toContain('&quot;quoted&quot; &amp; &lt;angled&gt;');
    expect(node.toString()).toContain('text &amp; &lt;angled&gt;');
  });

  it('should handle XmlBuilder constructor and setDeclaration', () => {
    const builder = new XmlBuilder('root');
    builder.setDeclaration('<?xml version=\"2.0\" ?>');
    expect(builder.toString()).toContain('version=\"2.0\"');
    expect(builder.toString()).toContain('<root />');
  });

  it('should handle setAttribute and addChild chaining', () => {
    const node = new XmlNode('test');
    node.setAttribute('a', '1').addChild('child');
    expect(node.attributes['a']).toBe('1');
    expect(node.children).toContain('child');
  });

  it('should handle empty declaration', () => {
    const builder = new XmlBuilder('root');
    builder.setDeclaration('');
    expect(builder.toString()).toBe('<root />');
  });
});
