// Run with: node tests/test_residual_vector.mjs
import assert from 'node:assert/strict';
import { vectorCells, decodeResidualVectors } from '../explorer/static/residual-vector.js';

const source = Float32Array.from({ length: 2560 }, (_, i) => i % 2 ? -(i + 1) : i + 1);
const overview = vectorCells(source);
assert.equal(overview.length, 64);
assert.equal(overview[0].start, 0); assert.equal(overview.at(-1).end, 2559);
assert.equal(overview.reduce((sum, c) => sum + c.value * (c.end - c.start + 1), 0), source.reduce((a, b) => a + b, 0), 'Binning preserves the coordinate sum, including sign cancellation');
for (let i = 1; i < overview.length; i++) assert.equal(overview[i].start, overview[i - 1].end + 1, 'No hidden coordinates may be skipped');
const detail = vectorCells(source, 128, true);
assert.equal(detail.length, 64);
assert.equal(detail[0].start, 128); assert.equal(detail.at(-1).end, 191);
for (const cell of detail) { assert.equal(cell.start, cell.end); assert.equal(cell.value, source[cell.start]); }
assert.equal(vectorCells(source, 9999, true).at(-1).end, 2559, 'The last window stays inside the vector');
assert.equal(vectorCells([3, -2, 1], 100, true).length, 3, 'Tiny models expose all coordinates');
const uneven = vectorCells(Float32Array.from({ length: 129 }, (_, i) => i));
assert.equal(uneven.at(-1).end, 128); assert.equal(uneven.reduce((s, c) => s + c.end - c.start + 1, 0), 129);
const bytes = Buffer.alloc(24), values = [1.25, -3.5, 0, 1e6, -1e-8, 2.75];
values.forEach((v, i) => bytes.writeFloatLE(v, i * 4));
const vectors = decodeResidualVectors({ shape: [2, 3], dtype: 'float32-le', data: bytes.toString('base64') });
assert.equal(vectors.length, 2); assert.deepEqual([...vectors[0]], values.slice(0, 3));
assert(Math.abs(vectors[1][1] + 1e-8) < 1e-14);
assert.throws(() => decodeResidualVectors({ shape: [65, 2560], dtype: 'float32-le', data: '' }), /length/);
assert.throws(() => decodeResidualVectors({ shape: [-1, 2], dtype: 'float32-le', data: '' }), /layout/);
console.log('Passed vector bands: complete 2560-dimension coverage, exact coordinate windows, signed grouping, little-endian transport and invalid payload handling.');
