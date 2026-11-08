import assert from 'node:assert/strict';
import { initResidualLab } from '../explorer/static/residual-lab.js';

class Element {
  constructor() { this.value = ''; this.dataset = {}; this.listeners = {}; this.children = new Map(); }
  addEventListener(name, fn) { this.listeners[name] = fn; }
  querySelector(selector) {
    if (!this.children.has(selector)) this.children.set(selector, new Element());
    return this.children.get(selector);
  }
}
const page = new Element(), root = new Element(), notifications = [], selections = [], requests = [];
globalThis.document = { querySelector: selector => page.querySelector(selector), dispatchEvent: event => notifications.push(event.detail) };
globalThis.localStorage = { getItem: () => null, setItem: () => {} };
globalThis.CustomEvent = class { constructor(type, options) { this.type = type; this.detail = options.detail; } };
let validate = async body => structuredClone(body), selectedOnMap = null;
globalThis.fetch = async (path, options) => {
  assert.equal(path, '/api/residual/experiment');
  requests.push(JSON.parse(options.body));
  return new Response('data: {"type":"done"}\n\n');
};
const lab = initResidualLab(root, {
  api: async (path, body) => {
    if (path === '/api/residual/tokens') return { texts: body.texts.map(row => ({ ...row, count: 3, tokens: [{ text: 'a', positions: [0, 1, 2] }] })) };
    assert.equal(path, '/api/residual/validate-held'); return validate(body);
  },
  t: (key, params) => `${key}${params ? JSON.stringify(params) : ''}`, esc: String,
  arch: { n_layers: 32, inter: 48, hidden: 4, layer_types: Array(32).fill('full_attention'), attn_heads: 2 },
  errorText: error => error.message, onActivate() {}, onOpen() {}, getSelected: () => selectedOnMap, onNeuron() {},
  onSelectionChange: value => selections.push(value),
});
const click = dataset => root.listeners.click({ target: { closest: () => ({ dataset }) } });
const input = (name, value) => root.listeners.input({ target: { name, value: String(value), type: 'number', tagName: 'INPUT', dataset: {} } });
const enable = (index, checked) => root.listeners.change({ target: { dataset: { enable: String(index) }, checked } });
const add = async value => { root.querySelector('#rl-neuron-address').value = value; await click({ action: 'add-address' }); };
const run = () => root.querySelector('#rl-form').onsubmit({ preventDefault() {} });
assert.deepEqual(lab.getSelection(), []); // No phantom L0 N0 intervention.
await add('L3 N7, L4 N8, L5 N9, L6 N10, L7 N11');
assert.equal(lab.getSelection().length, 5);
assert.match(root.querySelector('#rl-selected-neurons').textContent, /L3 N7, L4 N8, L5 N9, L6 N10, L7 N11/);
await click({ config: '0' }); input('strength', .2);
await click({ config: '1' }); input('strength', .6);
await click({ config: '2' }); input('strength', -.4);
await click({ config: '3' }); input('strength', .7);
await click({ config: '4' }); input('strength', -.1);
await add('L3 N7, L4 N8, L3 N7'); // Adding existing neurons selects them without duplicating/resetting settings.
assert.equal(lab.getSelection().length, 5);
await click({ action: 'hold' });
assert.match(root.innerHTML, /lab.release/);
assert.deepEqual(lab.getHeld().map(x => [x.plan.layer, x.plan.neuron, x.plan.strength]), [[3, 7, .2], [4, 8, .6], [5, 9, -.4], [6, 10, .7], [7, 11, -.1]]);
assert.ok(lab.getHeld().every(x => x.texts[0].text.trim()));
await click({ config: '1' });
input('strength', .9);
assert.equal(lab.getHeld()[1].plan.strength, .6);
assert.match(root.querySelector('#rl-held-edited').textContent, /lab.heldEdited/);
const copy = lab.getHeld(); copy[0].plan.strength = 99;
assert.equal(lab.getHeld()[0].plan.strength, .2);
await click({ action: 'hold' }); assert.deepEqual(lab.getHeld(), []);
assert.match(root.innerHTML, /lab.hold/);
enable(1, false); enable(3, false);
assert.deepEqual(lab.getSelection().map(c => c.layer), [3, 5, 7]);
assert.deepEqual(selections.at(-1), lab.getSelection());
await click({ action: 'hold' });
assert.deepEqual(lab.getHeld().map(c => c.plan.layer), [3, 5, 7]);
lab.invalidate(); assert.deepEqual(lab.getHeld(), []);
validate = async () => { throw new Error('Invalid residual strength'); };
await click({ action: 'hold' }); assert.deepEqual(lab.getHeld(), []);
assert.equal(root.querySelector('#rl-error').textContent, 'Invalid residual strength');
assert.ok(notifications.some(x => x.count === 5));
let resolveValidation;
validate = () => new Promise(resolve => { resolveValidation = resolve; });
const pending = click({ action: 'hold' });
lab.invalidate(); resolveValidation({ residuals: copy }); await pending;
assert.deepEqual(lab.getHeld(), []);
await click({ action: 'select-none' });
await click({ action: 'hold' });
assert.equal(root.querySelector('#rl-error').textContent, 'lab.noSelection');
await click({ action: 'select-all' });
lab.externalText('hello');
// A form submission returns its run promise so both the UI and the tests observe completion.
await run();
assert.deepEqual(requests.at(-1).plans.map(p => [p.layer, p.neuron, p.strength]), [[3, 7, .2], [4, 8, .9], [5, 9, -.4], [6, 10, .7], [7, 11, -.1]]);
assert.ok(requests.at(-1).plans.every(p => !('selected' in p) && !('chat_position' in p)));
enable(0, false); enable(2, false); enable(4, false);
await run();
assert.deepEqual(requests.at(-1).plans.map(p => p.layer), [4, 6]);
await click({ action: 'select-none' });
const prior = requests.length; await run(); assert.equal(requests.length, prior);
selectedOnMap = { layer: 9, n: 12 }; await click({ action: 'selected' });
assert.deepEqual(lab.getSelection(), [{ layer: 9, n: 12 }]);
await click({ action: 'remove-config' });
assert.deepEqual(lab.getSelection(), []);
assert.equal((root.querySelector('#rl-configurations').innerHTML.match(/data-config=/g) || []).length, 5);
console.log('Passed five independent settings, multi-selection, ID synchronization, combined experiment requests, held snapshots, release and invalidation.');
