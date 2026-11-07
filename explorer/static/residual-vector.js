// A band shows ordered coordinates. Overview bins and exact detail cells stay distinguishable.
export const RESIDUAL_WINDOW = 64;

export function vectorCells(vector, start = 0, detail = false, count = RESIDUAL_WINDOW) {
  if (!vector?.length) return [];
  const length = vector.length;
  if (detail) {
    const first = Math.max(0, Math.min(length - Math.min(count, length), Math.floor(start)));
    return Array.from({ length: Math.min(count, length) }, (_, i) => ({ start: first + i, end: first + i, value: vector[first + i] }));
  }
  const size = Math.min(count, length), width = Math.floor(length / size), extra = length % size;
  let first = 0;
  return Array.from({ length: size }, (_, i) => {
    const next = first + width + (i < extra ? 1 : 0);
    let sum = 0; for (let j = first; j < next; j++) sum += vector[j];
    const cell = { start: first, end: next - 1, value: sum / (next - first) };
    first = next; return cell;
  });
}

export function decodeResidualVectors(payload) {
  if (!payload) return null;
  if (payload.vectors) return payload.vectors; // The standalone practice page uses explicitly simulated arrays.
  const [rows, hidden] = payload.shape || [];
  if (payload.dtype !== 'float32-le' || !Number.isInteger(rows) || !Number.isInteger(hidden) || rows < 1 || hidden < 1) throw new Error('Invalid residual vector layout');
  const raw = atob(payload.data);
  if (raw.length !== rows * hidden * 4) throw new Error('Invalid residual vector length');
  const buffer = new ArrayBuffer(raw.length), bytes = new Uint8Array(buffer);
  for (let i = 0; i < raw.length; i++) bytes[i] = raw.charCodeAt(i);
  const view = new DataView(buffer), values = new Float32Array(rows * hidden);
  for (let i = 0; i < values.length; i++) values[i] = view.getFloat32(i * 4, true);
  return Array.from({ length: rows }, (_, i) => values.subarray(i * hidden, (i + 1) * hidden));
}
