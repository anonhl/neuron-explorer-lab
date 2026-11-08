// Exercise the real chat submit handler without a map selection or model dependencies.
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import vm from 'node:vm';

const source = await readFile(new URL('../explorer/static/app.js', import.meta.url), 'utf8');
const start = source.indexOf('$("#c-form").addEventListener("submit", async e => {');
const end = source.indexOf('\nasync function runReads', start);
assert.ok(start > 0 && end > start);
const residuals = [2, 7, 13, 21, 31].map(layer => ({
  texts: [{ id: 'a', text: 'Chat' }], plan: { layer, neuron: layer, stage: 'post', method: 'add', strength: .2 },
}));
for (const group of ['evaluation', 'conversation']) {
  let submit, request;
  const config = { neurons: [], thread: [] }, input = { value: 'hello', placeholder: '' };
  const state = { focus: null, chat: { group, groups: { [group]: config } } };
  const context = vm.createContext({
    $: selector => selector === '#c-form' ? { addEventListener: (name, handler) => { assert.equal(name, 'submit'); submit = handler; } } : input,
    state, running: null, reading: false,
    canStopChat: () => false, isConversation: () => group === 'conversation', isChatReading: () => false,
    validChatOperations: () => [], chatConfig: () => config,
    residualLab: { getHeld: () => structuredClone(residuals) },
    batchPanel: { getActive: () => null },
    renderThread() {}, chatThreadEl: () => ({ lastElementChild: { querySelectorAll: () => [] } }),
    syncChatButton() {}, renderFrequency() {},
    AbortController, TextDecoder, JSON, Response,
    fetch: async (url, options) => {
      assert.equal(url, '/api/steer'); request = JSON.parse(options.body);
      return new Response('data: {"row":null,"t":null}\n\n');
    },
  });
  vm.runInContext(source.slice(start, end), context);
  await submit({ preventDefault() {} });
  assert.equal(request.group, group);
  assert.deepEqual(request.operations, []);
  assert.deepEqual(request.residuals, residuals);
  assert.equal(config.thread[0].residualCount, 5);
  assert.equal(config.thread[0].replies.length, group === 'evaluation' ? 3 : 1);
  assert.equal(config.thread[0].completed, true);
  assert.equal(config.thread[0].error, undefined);
}
console.log('Passed real evaluation and Chat only submit handlers: five held settings reach the API without a map selection.');
