// Run the real lab UI against the demo transport, without a browser or model dependencies.
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import vm from 'node:vm';
import { initResidualLab } from '../explorer/static/residual-lab.js';
import { t } from '../explorer/static/i18n.js';

class Element {
  constructor() { this.value = ''; this.dataset = {}; this.listeners = {}; this.children = new Map(); }
  addEventListener(name, handler) { this.listeners[name] = handler; }
  querySelector(selector) {
    if (!this.children.has(selector)) this.children.set(selector, new Element());
    return this.children.get(selector);
  }
}
const texts = Array.from({ length: 5 }, (_, i) => ({ id: `text-${i + 1}`, label: String.fromCharCode(65 + i), text: `法国的首都是 ${i}` }));
const page = new Element(), root = new Element(), requests = [], streams = [];
const storage = { getItem: key => key === 'ne.residual-inputs.v1' ? JSON.stringify({ texts, serial: 5 }) : null, setItem() {} };
globalThis.document = { querySelector: selector => page.querySelector(selector), dispatchEvent() {}, createElement: () => ({ click() {} }) };
globalThis.localStorage = storage;
globalThis.CustomEvent = class { constructor(type, options) { this.type = type; this.detail = options.detail; } };
const demo = vm.createContext({ window: {}, localStorage: storage, URL, Response, ReadableStream, TextEncoder,
  DOMException, structuredClone, setTimeout: callback => { callback(); return 0; } });
vm.runInContext(await readFile(new URL('../explorer/demo/mock-api.js', import.meta.url), 'utf8'), demo);
const parseEvents = stream => stream.split('\n\n').filter(frame => frame.startsWith('data: ')).map(frame => JSON.parse(frame.slice(6)));
let failAfter = null, stopAfter = null;
globalThis.fetch = async (path, options) => {
  const body = JSON.parse(options.body);
  requests.push(body);
  let response = await demo.window.fetch(path, options);
  if (failAfter !== null || stopAfter !== null) {
    const events = parseEvents(await response.text());
    const completed = events.filter(e => e.type === 'result').slice(0, failAfter ?? stopAfter);
    const frames = [events[0], ...completed, ...(failAfter !== null ? [{ type: 'error', error: 'Example failure' }, { type: 'done' }] : [])];
    response = new Response(frames.map(event => `data: ${JSON.stringify(event)}\n\n`).join(''));
  }
  streams.push(response.clone().text());
  return response;
};
initResidualLab(root, {
  api: async (path, body) => {
    const response = await demo.window.fetch(path, { body: JSON.stringify(body) });
    assert.equal(response.status, 200);
    return response.json();
  },
  t, esc: value => String(value).replaceAll('&', '&amp;').replaceAll('"', '&quot;').replaceAll('<', '&lt;'),
  arch: { n_layers: 32, inter: 9216, hidden: 2560, layer_types: Array(32).fill('full_attention'), attn_heads: 16 },
  errorText: error => error.message, onActivate() {}, onOpen() {}, getSelected: () => null, onNeuron() {},
});
const click = dataset => root.listeners.click({ target: { closest: () => ({ dataset }) } });
const run = () => root.querySelector('#rl-form').onsubmit({ preventDefault() {} });
const html = () => root.querySelector('#rl-results').innerHTML;
const cards = () => (html().match(/<article class="rl-result"/g) || []).length;
const filter = value => root.listeners.change({ target: { id: 'rl-result-group', value } });
root.querySelector('#rl-neuron-address').value = 'L2 N7, L7 N8, L13 N9, L21 N10, L31 N11';
await click({ action: 'add-address' });
await run();
assert.equal(requests.at(-1).mode, 'joint');
assert.equal(cards(), 5);
let events = parseEvents(await streams.at(-1));
assert.equal(events[0].total, 5);
assert.ok(events.filter(e => e.type === 'result').every(e => e.group_id === 'joint' && e.plan_indices.length === 5));

await click({ mode: 'independent' });
assert.match(html(), /Inputs, settings or model configuration changed/);
await run();
const independentRequest = requests.at(-1);
assert.equal(independentRequest.mode, 'independent');
assert.equal(independentRequest.plans.length, 5);
assert.equal(cards(), 25);
events = parseEvents(await streams.at(-1));
assert.equal(events[0].total, 25);
assert.equal(events[0].groups.length, 5);
const rows = events.filter(e => e.type === 'result');
assert.equal(new Set(rows.map(r => `${r.group_id}:${r.id}`)).size, 25);
for (let i = 0; i < 5; i++) {
  const groupRows = rows.filter(r => r.group_id === `setting-${i + 1}`);
  assert.deepEqual(groupRows.map(r => r.id), texts.map(r => r.id));
  assert.ok(groupRows.every(r => r.plan_indices.length === 1 && r.plan_indices[0] === i));
  // Independent runs must match selecting only that neuron, including all candidate scores.
  const solo = await demo.window.fetch('/api/residual/experiment', { body: JSON.stringify({ ...independentRequest,
    mode: 'joint', plans: [independentRequest.plans[i]], plan: independentRequest.plans[i] }) });
  const soloRows = parseEvents(await solo.text()).filter(e => e.type === 'result');
  assert.deepEqual(groupRows.map(r => [r.baseline, r.intervened, r.kl]), soloRows.map(r => [r.baseline, r.intervened, r.kl]));
}
const ids = [...html().matchAll(/<article class="rl-result" id="([^"]+)"/g)].map(m => m[1]);
assert.equal(new Set(ids).size, 25);
for (const id of ids) assert.ok(html().includes(`href="#${id}"`));
filter('setting-3');
assert.equal(cards(), 5);
assert.ok(html().includes('L13 N9'));
assert.ok(!html().includes('id="rl-result-setting-1-'));
filter('all'); assert.equal(cards(), 25);
let exported;
const createURL = URL.createObjectURL, revokeURL = URL.revokeObjectURL;
try {
  URL.createObjectURL = blob => { exported = blob; return 'blob:test'; };
  URL.revokeObjectURL = () => {};
  await click({ action: 'export' });
  const snapshot = JSON.parse(await exported.text());
  assert.equal(snapshot.request.mode, 'independent');
  assert.equal(snapshot.results.length, 25);
  assert.equal(snapshot.context.groups.length, 5);
} finally { URL.createObjectURL = createURL; URL.revokeObjectURL = revokeURL; }

// An unchecked neuron is absent from both the request and the result groups.
root.listeners.change({ target: { dataset: { enable: '1' }, checked: false } });
await run();
assert.equal(requests.at(-1).plans.length, 4);
assert.equal(cards(), 20);
assert.ok(!html().includes('L7 N8'));

failAfter = 7;
await run();
assert.equal(cards(), 7);
assert.equal(root.querySelector('#rl-error').textContent, 'Example failure');
filter('setting-2'); assert.equal(cards(), 2);
filter('all'); assert.equal(cards(), 7);
failAfter = null; stopAfter = 3;
await run();
assert.equal(cards(), 3);
assert.match(root.querySelector('#rl-error').textContent, /before completion/);
stopAfter = null;
await click({ mode: 'joint' });
await run();
assert.equal(cards(), 5);
assert.equal(requests.at(-1).mode, 'joint');
assert.ok(!html().includes('id="rl-result-group"'));
console.log('Passed joint/independent transport, 5 × 5 results, per-neuron equivalence, filtering, unique anchors, exports, deselection and partial-result preservation.');
