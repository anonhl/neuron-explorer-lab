import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';
import { contextFrequency, countLens, countWords } from '../explorer/static/frequency.js';
const token = (id, text, score = 1) => ({ id, token: text, score });
const context = (label, text, input, positive, negative) => ({ label, text,
  residual: { stages: { pre: { tokens: input } }, selected: { layer: 2, n: 5,
    write: { tokens: { positive, negative } } } } });
const a = context('A', 'cat cat', [token(1, 'cat'), token(2, 'pet')], [token(3, 'kitten')], [token(4, 'car', -1)]);
const b = context('B', 'cat dog', [token(1, 'cat')], [token(3, 'kitten'), token(5, 'dog')], []);
const combined = contextFrequency([a, b]);
assert.equal(countWords(combined.replies, 'en').rows.find(r => r.text === 'cat').count, 3);
const lens = countLens(combined.connections);
assert.deepEqual(lens.totals, { up: 3, down: 4 });
assert.equal(lens.rows.find(r => r.id === 1).up.count, 2);
assert.equal(lens.rows.find(r => r.id === 3).down.count, 2);
assert.equal(lens.rows.find(r => r.id === 4).down.positive, 0);
assert.equal(countWords(combined.replies, 'en').rows.find(r => r.text === 'cat').sources.size, 2);
assert.deepEqual(contextFrequency([]), { connections: { up: [], down: [] }, replies: [] });
// Exercise the offline API: cached A and B remain distinct, without replacing active B.
const window = {};
vm.runInNewContext(fs.readFileSync(new URL('../explorer/demo/mock-api.js', import.meta.url), 'utf8'),
  { window, localStorage: { getItem: () => null, setItem() {} }, structuredClone, Response, URL, URLSearchParams,
    TextEncoder, ReadableStream, Uint8Array, Uint16Array, ArrayBuffer, setTimeout, clearTimeout,
    btoa: s => Buffer.from(s, 'binary').toString('base64'), crypto: globalThis.crypto });
const post = async (path, body) => (await window.fetch(path, { body: JSON.stringify(body) })).json();
await post('/api/infer', { text: 'cat cat', texts: ['cat cat', 'dog dog'] });
const active = await post('/api/infer', { text: 'dog dog', texts: ['cat cat', 'dog dog'] });
const groups = await post('/api/infer/cached-texts', { texts: ['cat cat', 'dog dog'] });
assert.deepEqual(groups.texts.map(row => row.text), ['cat cat', 'dog dog']);
assert.deepEqual(groups.texts.map(row => row.tokens), [['cat', 'cat'], ['dog', 'dog']]);
const result = await post('/api/infer/contexts', { texts: ['cat cat', 'dog dog'], layer: 2, n: 5 });
assert.equal(result.contexts.length, 2);
assert.equal(result.contexts[0].text, 'cat cat');
assert.equal(result.contexts[1].text, 'dog dog');
assert.equal(result.contexts[1].capture, active.capture);
assert.notEqual(result.contexts[0].capture, active.capture);
const restored = await post('/api/infer', { text: 'dog dog', texts: ['cat cat', 'dog dog'] });
assert.equal(restored.capture, active.capture);
assert.equal((await post('/api/infer/contexts', { texts: ['new'], layer: 2, n: 5 })).ready, false);
const exported = await post('/api/infer/contexts/export', { text: 'cat cat', pos: 0 });
assert.equal(exported.text, 'cat cat');
assert.equal(exported.layers.length, 32);
console.log('Passed pooled words and input/output tokens, text provenance, cached contexts, active-capture isolation, and grouped export.');
