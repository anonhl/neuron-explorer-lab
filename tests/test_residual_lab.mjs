import assert from 'node:assert/strict';
import { parseResidualVector, readResidualEvents } from '../explorer/static/residual-lab.js';

assert.deepEqual(parseResidualVector('[1, -2, 0]', 3), [1, -2, 0]);
for (const value of ['[1,2]', '[1,"2",3]', '[1e999,2,3]', '{}']) {
  assert.throws(() => parseResidualVector(value, 3));
}
const encoder = new TextEncoder();
const payload = encoder.encode(': keepalive\n\ndata: {"type":"result","text":"中文"}\n\ndata: {"type":"done"}\n\n');
const events = [];
// Split every UTF-8 byte and every frame boundary to exercise real streaming transport.
await readResidualEvents(new Response(new ReadableStream({ start(controller) {
  for (const byte of payload) controller.enqueue(Uint8Array.of(byte));
  controller.close();
} })), event => events.push(event));
assert.deepEqual(events, [{ type: 'result', text: '中文' }, { type: 'done' }]);
await assert.rejects(() => readResidualEvents(new Response('data: {"type":"result"}\n\n'), () => {}), /before completion/);
await assert.rejects(() => readResidualEvents(new Response('bad input', { status: 400 }), () => {}), /bad input/);
console.log('Passed custom vectors, byte-split Chinese events, keepalives, incomplete-stream and HTTP-error handling.');
