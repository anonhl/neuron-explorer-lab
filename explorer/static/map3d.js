import * as THREE from "three";
import { vectorCells, decodeResidualVectors, RESIDUAL_WINDOW } from "./residual-vector.js?v=1";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { LineSegments2 } from "three/addons/lines/LineSegments2.js";
import { LineSegmentsGeometry } from "three/addons/lines/LineSegmentsGeometry.js";
import { LineMaterial } from "three/addons/lines/LineMaterial.js";

// geometry: layers run left to right along x. Each layer is the residual stream passing through a token-mixer slab
// (linear or full attention) and then the MLP, drawn as a vertical sheet of its neurons.
const SHEET = 4.2, GAP = 1.9, SHEET_Y0 = 0.7, ATTN_DX = -0.62, MLP_DX = 0.18;
const PAL = {
  bg: 0x000000,
  idle: new THREE.Color(0.036, 0.046, 0.052), swept: new THREE.Color(0.2, 0.24, 0.26),
  lo: new THREE.Color(0.42, 0.30, 0.10), hi: new THREE.Color(1.0, 0.71, 0.28), named: new THREE.Color(1, 1, 1),
  link: new THREE.Color(0.22, 0.78, 1.0),
  wire: 0x1c2a30, wireData: 0x3a5a66, slabEdge: 0x3c4a52, slabEdgeFull: 0x6a8592, head: 0x2f4a56, headFull: 0x5f8fa3,
};
// overall brightness of everything drawn in the map (points, lines, slabs); 1 = original
const BRIGHT = Number(new URLSearchParams(location.search).get("bright")) || 2.5;
// the boost lifts the structure (sheets, wires, idle points); data colours keep their hue
for (const k in PAL) if (!["bg", "swept", "lo", "hi", "named"].includes(k)) PAL[k] = new THREE.Color(PAL[k]).multiplyScalar(BRIGHT);
const SELECTED_LAYER_COLOR = 0x66d9ff, ACTIVE_LAYER_COLOR = 0xbadbe5;
const LAYER_BRIGHTNESS = { idle: 0.24, active: 0.72, selected: 1, otherActive: 0.4 };

function glowPoints(sizePx, maxScale, layerBrightness) {
  return new THREE.RawShaderMaterial({
    uniforms: { size: { value: sizePx * Math.min(devicePixelRatio, 2) }, ref: { value: 1 }, maxScale: { value: maxScale },
      tdist: { value: 1 }, layerBrightness: { value: layerBrightness } },
    vertexShader: `precision highp float;
      uniform mat4 modelViewMatrix, projectionMatrix; uniform float size, ref, maxScale, tdist;
      uniform float layerBrightness[${layerBrightness.length}];
      attribute vec3 position; attribute vec3 color; varying vec3 vC;
      void main() {
        vec4 mv = modelViewMatrix * vec4(position, 1.0); gl_Position = projectionMatrix * mv;
        int layer = int(floor((position.x - ${MLP_DX.toFixed(3)}) / ${GAP.toFixed(3)} + 0.5));
        // Keep the display luminance constant when tints change within a layer.
        float luma = dot(color, vec3(0.2126, 0.7152, 0.0722));
        vec3 tint = mix(vec3(1.0), color / max(luma, 0.001), 0.7);
        float near = clamp(sqrt(ref / tdist), 1.0, 2.0);
        vC = tint * (0.26 * layerBrightness[layer] * near);
        gl_PointSize = size * clamp(sqrt(ref / -mv.z), 0.6, maxScale); }`,
    fragmentShader: `precision highp float; varying vec3 vC;
      void main() { float r = length(gl_PointCoord - 0.5); if (r > 0.5) discard;
        float edge = 1.0 - smoothstep(0.38, 0.5, r); gl_FragColor = vec4(vC * edge, 1.0); }`,
    transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
  });
}

export function createMap(canvas, labelsDiv, arch, cb) {
  const L = arch.n_layers, N = arch.inter;
  const layerBrightness = new Float32Array(L);
  let activeLayers = new Set();
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  renderer.setClearColor(PAL.bg);
  const scene = new THREE.Scene();
  const layerX = l => l * GAP;
  const X0 = ATTN_DX - 1.6, X1 = (L - 1) * GAP + MLP_DX + 1.4;

  const insets = { left: 410, right: 0 };
  const TARGET_X = Number(new URLSearchParams(location.search).get("targetx")) || 0;
  let fitPerPx = null;
  const camera = new THREE.PerspectiveCamera(32, 1, 0.02, 2000);
  let homeDist = 60;
  const home = { target: new THREE.Vector3((L - 1) * GAP / 2, SHEET_Y0 + SHEET / 2 - 0.4, 0), dir: new THREE.Vector3(0.62, 0.4, 1).normalize() };
  camera.position.copy(home.target).addScaledVector(home.dir, homeDist);
  const controls = new OrbitControls(camera, canvas);
  controls.target.copy(home.target);
  controls.enableDamping = false;
  controls.zoomToCursor = true;
  controls.minDistance = 1.6; controls.maxDistance = 400;
  controls.zoomSpeed = 0.6;
  controls.update();

  const gridSide = Math.ceil(Math.sqrt(N));
  let layoutCells = null;  // Int16Array [L * N] of grid cells, or null for neuron-index order
  const neuronPos = (layer, n) => {
    const c = layoutCells ? layoutCells[layer * N + n] : n;
    const i = c % gridSide, j = Math.floor(c / gridSide);
    return new THREE.Vector3(layerX(layer) + MLP_DX, SHEET_Y0 + (j / (gridSide - 1)) * SHEET, (i / (gridSide - 1) - 0.5) * SHEET);
  };

  // every neuron of every layer, one draw call
  const pos = new Float32Array(L * N * 3), col = new Float32Array(L * N * 3);
  for (let l = 0; l < L; l++) for (let n = 0; n < N; n++) {
    const p = neuronPos(l, n), k = (l * N + n) * 3;
    pos[k] = p.x; pos[k + 1] = p.y; pos[k + 2] = p.z;
    PAL.idle.toArray(col, k);
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.BufferAttribute(pos, 3));
  const colorAttr = new THREE.BufferAttribute(col, 3).setUsage(THREE.DynamicDrawUsage);
  geo.setAttribute("color", colorAttr);
  const pointMat = glowPoints(2.0, 5, layerBrightness);
  const points = new THREE.Points(geo, pointMat);
  scene.add(points);
  const pointColors = { units: new Map(), web: new Map(), marks: new Map(), links: new Map() };
  function setPointColors(source, colors) {
    const changed = new Set(pointColors[source].keys());
    for (const i of colors.keys()) changed.add(i);
    pointColors[source] = colors;
    let first = Infinity, last = -1;
    for (const i of changed) {
      const color = pointColors.marks.get(i) ?? pointColors.links.get(i) ?? pointColors.web.get(i) ?? pointColors.units.get(i) ?? PAL.idle;
      color.toArray(col, i * 3);
      first = Math.min(first, i); last = Math.max(last, i);
    }
    if (last >= 0) {
      colorAttr.addUpdateRange(first * 3, (last - first + 1) * 3);
      colorAttr.needsUpdate = true;
    }
  }

  // structure
  const wires = [];
  const seg = (a, b) => wires.push(a.x, a.y, a.z, b.x, b.y, b.z);
  const V = (x, y, z) => new THREE.Vector3(x, y, z);
  const h = SHEET / 2;
  const outlineGeo = new THREE.BufferGeometry().setFromPoints([
    V(0, SHEET_Y0, -h), V(0, SHEET_Y0, h), V(0, SHEET_Y0 + SHEET, h), V(0, SHEET_Y0 + SHEET, -h),
  ]);
  seg(V(X0, 0, 0), V(X1, 0, 0));
  // heads are horizontal lines stacked down the slab's face, head 0 at the top
  const HEAD_SPAN = SHEET * 0.82, HEAD_W = SHEET * 0.41;
  const headY = (h, H) => SHEET_Y0 + SHEET / 2 + (0.5 - (h + 0.5) / H) * HEAD_SPAN;
  const outlines = [], slabs = [], headPos = [], headBase = [], headStart = [];  // headStart[l]: first line of layer l's heads
  for (let l = 0; l < L; l++) {
    const x = layerX(l), mx = x + MLP_DX, ax = x + ATTN_DX;
    const outline = new THREE.LineLoop(outlineGeo,
      new THREE.LineBasicMaterial({ transparent: true, depthWrite: false }));
    outline.position.x = mx;
    scene.add(outline); outlines.push(outline);
    seg(V(ax, 0, 0), V(ax, SHEET_Y0, 0)); seg(V(mx, 0, 0), V(mx, SHEET_Y0, 0));
    const full = arch.layer_types[l] === "full_attention", t = full ? 0.3 : 0.1;
    const slab = new THREE.Mesh(new THREE.BoxGeometry(t, SHEET * 0.92, SHEET * 0.92), new THREE.MeshBasicMaterial({
      color: full ? 0x8fa6b2 : 0x6f8590, transparent: true, opacity: (full ? 0.075 : 0.045) * BRIGHT, depthWrite: false }));
    slab.position.set(ax, SHEET_Y0 + SHEET / 2, 0);
    slab.userData = { layer: l, full };
    scene.add(slab); slabs.push(slab);
    const edges = new THREE.LineSegments(new THREE.EdgesGeometry(slab.geometry),
      new THREE.LineBasicMaterial({ color: full ? PAL.slabEdgeFull : PAL.slabEdge, transparent: true }));
    edges.position.copy(slab.position); scene.add(edges);
    slab.userData.edges = edges; slab.userData.baseOpacity = slab.material.opacity;
    const heads = full ? arch.attn_heads : arch.linear_v_heads;
    slab.userData.heads = heads;
    headStart.push(headPos.length / 6);
    for (let h = 0; h < heads; h++) {
      const y = headY(h, heads);
      headPos.push(ax + t / 2 + 0.003, y, -HEAD_W, ax + t / 2 + 0.003, y, HEAD_W);
      headBase.push(full ? PAL.headFull : PAL.head);
    }
  }
  const fadeLines = [];  // whole-scene line work that dims as the camera closes in on one layer
  // the attention heads: one line per head on its layer's slab, coloured one by one
  const headGeo = new THREE.BufferGeometry(); headGeo.setAttribute("position", new THREE.Float32BufferAttribute(headPos, 3));
  const headCol = new Float32Array(headPos.length);
  headGeo.setAttribute("color", new THREE.BufferAttribute(headCol, 3));
  const headMat = new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.55 }); headMat.userData.base = 0.55;
  scene.add(new THREE.LineSegments(headGeo, headMat)); fadeLines.push(headMat);
  let headLevels = null, headSel = null, headHover = null;
  const band = op => { const m = new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1), new THREE.MeshBasicMaterial({
    color: PAL.named, transparent: true, opacity: op, depthWrite: false })); m.visible = false; m.renderOrder = 1; scene.add(m); return m; };
  const hoverBand = band(0.1), selBand = band(0.18);
  function placeBand(m, at) {
    m.visible = !!at;
    if (!at) return;
    const s = slabs[at.layer], H = s.userData.heads;
    m.position.set(s.position.x, headY(at.head, H), 0);
    m.scale.set(s.geometry.parameters.width + 0.02, HEAD_SPAN / H * 0.9, HEAD_W * 2 + 0.1);
  }
  function paintHeads() {
    placeBand(hoverBand, headHover && !(headSel && headSel.layer === headHover.layer && headSel.head === headHover.head) ? headHover : null);
    placeBand(selBand, headSel);
    const c = new THREE.Color();
    for (let l = 0; l < L; l++) {
      const H = (headStart[l + 1] ?? headBase.length) - headStart[l];
      for (let h = 0; h < H; h++) {
        const i = headStart[l] + h, lv = headLevels?.[l]?.[h] || 0;
        c.copy(headBase[i]);
        if (headLevels) c.multiplyScalar(0.3).lerp(PAL.hi, Math.min(1, lv));
        if (headHover && headHover.layer === l && headHover.head === h) c.lerp(PAL.named, 0.5);
        if (headSel && headSel.layer === l && headSel.head === h) c.copy(PAL.named);
        c.toArray(headCol, i * 6); c.toArray(headCol, i * 6 + 3);
      }
    }
    headGeo.attributes.color.needsUpdate = true;
  }
  paintHeads();
  const endSlabs = [];
  for (const x of [X0, X1]) {
    const m = new THREE.LineBasicMaterial({ color: PAL.slabEdgeFull, transparent: true }); m.userData.base = 1;
    const slab = new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.BoxGeometry(0.14, 1.4, 2.2)), m);
    slab.position.set(x, 0, 0); scene.add(slab); endSlabs.push(m);
  }
  const wg = new THREE.BufferGeometry(); wg.setAttribute("position", new THREE.Float32BufferAttribute(wires, 3));
  const wireMat = new THREE.LineBasicMaterial({ color: PAL.wire, transparent: true }); wireMat.userData.base = 1;
  scene.add(new THREE.LineSegments(wg, wireMat)); fadeLines.push(wireMat, ...endSlabs);
  const bounds = [];
  for (const x of [X0, X1]) for (const y of [-0.8, SHEET_Y0 + SHEET]) for (const z of [-SHEET / 2, SHEET / 2]) bounds.push(V(x, y, z));

  // The open attention head's marker and connections (their geometry changes on updates).
  const hiGeo = new THREE.BufferGeometry();
  const hiMat = glowPoints(5, 2, layerBrightness);
  const hiPts = new THREE.Points(hiGeo, hiMat);
  hiPts.frustumCulled = false; hiPts.renderOrder = 3;
  scene.add(hiPts);
  const conGeo = new THREE.BufferGeometry();
  const conLines = new THREE.LineSegments(conGeo, new THREE.LineBasicMaterial({
    vertexColors: true, transparent: true, depthWrite: false, blending: THREE.AdditiveBlending }));
  conLines.frustumCulled = false; conLines.renderOrder = 2;
  scene.add(conLines);
  // connection web of the selected neuron: upstream and downstream cones, coloured by link sign, fading per hop.
  const qs = new URLSearchParams(location.search);
  const WEB_ALPHA = Number(qs.get("webalpha")) || 0.5, WEB_DECAY = Number(qs.get("webdecay")) || 0.85;
  // by the sign of the link: yellow when the writer pushes the reader up, magenta when it pushes it down
  const WEB_UP = new THREE.Color(1.0, 0.82, 0.2), WEB_DOWN = new THREE.Color(1.0, 0.25, 0.85), HEAD_LINK = new THREE.Color(0.4, 0.85, 1.0);
  // a head's links by role: q and k (where it looks) apart from v and o (what it moves)
  const HEAD_ROLE = { q: new THREE.Color(0.45, 1.0, 0.55), k: new THREE.Color(0.7, 0.6, 1.0), v: HEAD_LINK, o: HEAD_LINK };
  // fat lines (WebGL draws plain lines 1px wide), additive with the intensity folded into the colour, and outside the
  // zoom fade, so the web stays bright up close
  const webMat = new LineMaterial({ vertexColors: true, linewidth: Number(qs.get("webwidth")) || 2, transparent: true,
    depthWrite: false, blending: THREE.AdditiveBlending });
  const webLines = new LineSegments2(new LineSegmentsGeometry(), webMat);
  webLines.frustumCulled = false; webLines.renderOrder = 1; webLines.visible = false;
  scene.add(webLines);
  let webEdges = null;
  const webFed = new Set(), webFeeds = new Set();  // neurons the drawn web reads into / out of (layer * N + n)
  const DEAD = 0.08;  // brightness of lines the web doesn't continue through
  const DIRECT = 0.6;  // the open neuron's direct links, relative to full brightness
  function buildWeb() {
    if (!webEdges || !webEdges.length) {
      webFed.clear(); webFeeds.clear(); webLines.visible = false;
      setPointColors("web", new Map());
      return;
    }
    const m = webEdges.length / 6, p = new Float32Array(m * 6), c = new Float32Array(m * 6);
    let zhi = 1;
    for (let i = 0; i < m; i++) zhi = Math.max(zhi, Math.abs(webEdges[i * 6 + 4]));
    // a line is "through" when the web continues past both ends: its writer is read into by another drawn line (or is
    // the open neuron) and its reader writes on into another (or is the open neuron); dead ends draw dimmer
    const fed = webFed, feeds = webFeeds, nk = (l, n) => l * N + n;
    fed.clear(); feeds.clear();
    for (let i = 0; i < m; i++) { const e = webEdges.subarray(i * 6, i * 6 + 6); feeds.add(nk(e[0], e[1])); fed.add(nk(e[2], e[3])); }
    const root = selected ? nk(selected.layer, selected.n) : -1;
    const touched = new Map();
    for (let i = 0; i < m; i++) {
      const e = webEdges.subarray(i * 6, i * 6 + 6), a = neuronPos(e[0], e[1]), b = neuronPos(e[2], e[3]);
      const k = e[4] > 0 ? WEB_UP : WEB_DOWN;
      const s0 = nk(e[0], e[1]), d0 = nk(e[2], e[3]);
      const through = (s0 === root || fed.has(s0)) && (d0 === root || feeds.has(d0));
      const al = WEB_ALPHA * (0.6 + 0.4 * Math.abs(e[4]) / zhi) * Math.max(0.35, WEB_DECAY ** (Math.abs(e[5]) - 1)) * (through ? 1 : DEAD);
      p.set([a.x, a.y, a.z, b.x, b.y, b.z], i * 6); c.set([k.r * al, k.g * al, k.b * al, k.r * al, k.g * al, k.b * al], i * 6);
      // each touched neuron takes the colour of the hop that first reached it (edges arrive in hop order)
      const key = e[5] > 0 ? e[2] * N + e[3] : e[0] * N + e[1];
      if (!touched.has(key)) touched.set(key, k);
    }
    setPointColors("web", touched);
    webLines.geometry.dispose();
    webLines.geometry = new LineSegmentsGeometry().setPositions(p).setColors(c);
    webLines.visible = true;
  }
  let selected = null, hovered = null, connections = [], hoverCon = null, headSrc = null;  // headSrc: an open head's line middle
  let neuronHeads = [];  // the open neuron's strongest linked heads: {layer, head, z}
  const headMid = (layer, head) => { const i = (headStart[layer] + head) * 6;
    return new THREE.Vector3((headPos[i] + headPos[i + 3]) / 2, (headPos[i + 1] + headPos[i + 4]) / 2, (headPos[i + 2] + headPos[i + 5]) / 2); };

  function refreshHighlights() {
    const src = hoverCon ? hoverCon.src : selected, shown = hoverCon ? hoverCon.list : connections;
    const zmax = Math.max(...shown.map(k => k.z), 1);
    // direct links coloured like the web: yellow feeds (pushes the reader up), magenta holds back
    // head links carry a role instead of a sign
    const linkCol = k => (k.role ? HEAD_ROLE[k.role] : k.sign > 0 ? WEB_UP : k.sign < 0 ? WEB_DOWN : HEAD_LINK).clone();
    const colors = new Map(shown.map(k => [k.layer * N + k.n, linkCol(k)]));
    if (hovered && (hovered.layer !== selected?.layer || hovered.n !== selected?.n)) {
      colors.set(hovered.layer * N + hovered.n, PAL.link);
    }
    setPointColors("links", colors);
    const headOpen = headSrc && !selected;
    hiGeo.setAttribute("position", new THREE.Float32BufferAttribute(headOpen ? headSrc.toArray() : [], 3));
    hiGeo.setAttribute("color", new THREE.Float32BufferAttribute(headOpen ? PAL.named.toArray() : [], 3));
    const lp = [], lc = [];
    const a = src ? neuronPos(src.layer, src.n) : hoverCon ? hoverCon.from : headSrc;
    if (a) {
      for (const k of shown) {
        // with the web on, a direct link stays bright only if the web goes on past its far end
        const far = k.layer * N + k.n, on = !webEdges || !webEdges.length || hoverCon
          || (k.side === "up" ? webFed.has(far) : webFeeds.has(far));
        const b = neuronPos(k.layer, k.n), s = DIRECT * (0.2 + 0.8 * k.z / zmax) * (on ? 1 : DEAD), cl = linkCol(k).multiplyScalar(s);
        lp.push(a.x, a.y, a.z, b.x, b.y, b.z); lc.push(cl.r, cl.g, cl.b, cl.r, cl.g, cl.b);
      }
    }
    if (selected && !hoverCon && neuronHeads.length) {
      const o = neuronPos(selected.layer, selected.n), hz = Math.max(...neuronHeads.map(k => k.z));
      for (const k of neuronHeads) {
        const b = headMid(k.layer, k.head), cl = HEAD_LINK.clone().multiplyScalar(DIRECT * (0.2 + 0.8 * k.z / hz));
        lp.push(o.x, o.y, o.z, b.x, b.y, b.z); lc.push(cl.r, cl.g, cl.b, cl.r, cl.g, cl.b);
      }
    }
    conGeo.setAttribute("position", new THREE.Float32BufferAttribute(lp, 3));
    conGeo.setAttribute("color", new THREE.Float32BufferAttribute(lc, 3));
    render();
  }

  // layer labels: text drawn in each layer's plane, just above the top-right corner of its neuron sheet
  const LABEL_H = Number(new URLSearchParams(location.search).get("labelsize")) || 0.2;
  const labels = [];
  for (let l = 0; l < L; l++) {
    const cv = document.createElement("canvas"), ctx = cv.getContext("2d"), font = "600 64px ui-monospace, Menlo, monospace";
    ctx.font = font; const tw = Math.ceil(ctx.measureText(`L${l}`).width) + 8;
    cv.width = tw; cv.height = 80;
    ctx.font = font; ctx.fillStyle = "#fff"; ctx.textBaseline = "middle"; ctx.fillText(`L${l}`, 4, 42);
    const tex = new THREE.CanvasTexture(cv); tex.colorSpace = THREE.SRGBColorSpace; tex.anisotropy = 8;
    const w = LABEL_H * tw / 80;
    const mat = new THREE.MeshBasicMaterial({ map: tex, transparent: true, depthWrite: false, side: THREE.DoubleSide, color: 0xc8d4da });
    mat.userData.base = 0.85;
    const m = new THREE.Mesh(new THREE.PlaneGeometry(w, LABEL_H), mat);
    // the plane's +x runs toward -z after this turn, so the text reads toward the sheet's right edge seen from +x
    m.rotation.y = Math.PI / 2;
    m.position.set(layerX(l) + MLP_DX, SHEET_Y0 + SHEET + 0.08 + LABEL_H / 2, -SHEET / 2 + w / 2);
    m.userData = { layer: l };
    scene.add(m); labels.push(m);
  }
  const dataLayers = new Set();

  // Actual residual states are ordered vector bands, with coordinate detail on zoom.
  const residualGroup = new THREE.Group(); scene.add(residualGroup);
  let residualData = null, residualNodes = [], residualLabels = [], residualBands = [], residualLOD = null;
  let residualVectors = null, vectorPayload = null, residualFlowGeo = null, residualBasePos = [], residualBaseCol = [];
  const BAND_WIDTH = SHEET, BAND_Y = -0.3;
  const residualPos = (l, stage) => V(layerX(l) + (stage === "pre" ? ATTN_DX - 0.35 : stage === "mid" ? MLP_DX - 0.32 : MLP_DX + 0.55), BAND_Y, 0);
  const coordinateVector = (l, stage) => residualVectors?.[stage === "pre" ? 2 * l : 2 * l + (stage === "mid" ? 1 : 2)]
    ?? (l === residualData?.layer ? residualData.stages?.[stage]?.vector : null);
  function clearResidual() {
    residualGroup.traverse(o => { o.geometry?.dispose(); if (o.material) { o.material.map?.dispose(); o.material.dispose(); } });
    residualGroup.clear(); residualNodes = []; residualBands = []; residualLOD = null; residualFlowGeo = null;
    for (const label of residualLabels) { label.el.remove(); label.axis.remove(); }
    residualLabels = [];
  }
  function refreshResidualBands(detail) {
    if (!residualData) return;
    residualLOD = detail;
    const data = residualData, hidden = data.hidden || coordinateVector(data.layer, "post")?.length || 2560;
    const windowStart = Math.max(0, Math.min(hidden - Math.min(RESIDUAL_WINDOW, hidden), data.windowStart || 0));
    const windowSize = Math.min(hidden, RESIDUAL_WINDOW);
    for (const band of residualBands) {
      const { layer, stage } = band.mesh.userData, vector = coordinateVector(layer, stage);
      let cells = vectorCells(vector, windowStart, detail);
      if (!cells.length) {
        const bins = data.layers[layer].bins?.[stage] || [];
        const ranges = vectorCells(new Float32Array(hidden), 0, false, Math.max(1, bins.length));
        cells = ranges.map((c, i) => ({ ...c, value: bins[i] || 0 }));
      }
      band.mesh.userData.cells = cells;
      band.mesh.userData.detail = !!vector && detail;
      const peak = Math.max(...cells.map(c => Math.abs(c.value)), 1e-12);
      const positions = [], colors = [], uv = [], borders = [];
      function quad(left, right, bottom, top, value, scale, targetPos, targetCol, targetUV) {
        const color = value === 0 ? new THREE.Color(0x25353f) : new THREE.Color(value < 0 ? 0xff65a2 : 0xffc16c).multiplyScalar(0.18 + 0.82 * Math.sqrt(Math.abs(value) / scale));
        for (const [x, y, u, v] of [[left,bottom,0,0],[right,bottom,1,0],[right,top,1,1],[left,bottom,0,0],[right,top,1,1],[left,top,0,1]]) {
          targetPos.push(x,y,0); targetCol.push(...color.toArray());
          if (targetUV) targetUV.push((x / BAND_WIDTH + 0.5), v);
        }
      }
      const height = band.focused ? 0.24 : 0.18;
      cells.forEach((cell, i) => {
        const left = -BAND_WIDTH / 2 + i / cells.length * BAND_WIDTH, right = left + BAND_WIDTH / cells.length * 0.94;
        quad(left, right, -height / 2, height / 2, cell.value, peak, positions, colors, uv);
        if (Number.isInteger(data.dimension) && data.dimension >= cell.start && data.dimension <= cell.end) {
          for (const [a,b] of [[[left,-height/2],[right,-height/2]],[[right,-height/2],[right,height/2]],[[right,height/2],[left,height/2]],[[left,height/2],[left,-height/2]]]) borders.push(...a,0,...b,0);
        }
      });
      band.mesh.geometry.dispose(); band.mesh.geometry = new THREE.BufferGeometry();
      band.mesh.geometry.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
      band.mesh.geometry.setAttribute("color", new THREE.Float32BufferAttribute(colors, 3));
      band.mesh.geometry.setAttribute("uv", new THREE.Float32BufferAttribute(uv, 2));
      band.border.geometry.dispose(); band.border.geometry = new THREE.BufferGeometry();
      band.border.geometry.setAttribute("position", new THREE.Float32BufferAttribute(borders, 3));
      if (band.write) {
        const contributions = cells.map(c => {
          let total = 0; for (let i = c.start; i <= c.end; i++) total += data.selected.write?.vector?.[i] || 0;
          return total / (c.end - c.start + 1);
        });
        const max = Math.max(...contributions.map(Math.abs), 1e-12), wp = [], wc = [], wu = [];
        contributions.forEach((v, i) => { const left = -BAND_WIDTH / 2 + i / cells.length * BAND_WIDTH; quad(left, left + BAND_WIDTH / cells.length * 0.94, -0.24, -0.18, v, max, wp, wc, wu); });
        band.write.geometry.dispose(); band.write.geometry = new THREE.BufferGeometry();
        band.write.geometry.setAttribute("position", new THREE.Float32BufferAttribute(wp, 3));
        band.write.geometry.setAttribute("color", new THREE.Float32BufferAttribute(wc, 3));
        band.write.geometry.setAttribute("uv", new THREE.Float32BufferAttribute(wu, 2));
        band.write.userData.cells = cells.map((c, i) => ({ ...c, value: contributions[i] }));
      }
      band.axis.textContent = cells.length ? `d${cells[0].start} — d${cells.at(-1).end} · ${band.mesh.userData.detail ? "1 : 1" : `Σ/${Math.ceil(hidden / cells.length)}`}` : "";
    }
    // Selected neuron links end at the actual written coordinates, not a shared box centre.
    const positions = [...residualBasePos], colors = [...residualBaseCol], s = data.selected;
    const write = s.write?.vector || [];
    const dims = Array.from({ length: write.length }, (_, dim) => dim)
      .filter(dim => write[dim] !== 0 && (!detail || dim >= windowStart && dim < windowStart + windowSize))
      .sort((a, b) => Math.abs(write[b]) - Math.abs(write[a])).slice(0, 8);
    for (const dim of dims) {
      const fraction = detail ? (dim - windowStart + 0.5) / windowSize : (dim + 0.5) / hidden;
      const a = neuronPos(s.layer, s.n), b = residualPos(s.layer, "post"); b.z = BAND_WIDTH * (0.5 - fraction); b.y -= 0.2;
      const color = new THREE.Color(write[dim] < 0 ? 0xff65a2 : 0xffc16c);
      positions.push(...a.toArray(), ...b.toArray()); colors.push(...color.toArray(), ...color.toArray());
    }
    residualFlowGeo?.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
    residualFlowGeo?.setAttribute("color", new THREE.Float32BufferAttribute(colors, 3));
  }
  function setResidualTrace(data) {
    clearResidual(); residualData = data;
    if (!data) { residualVectors = null; vectorPayload = null; render(); return; }
    if (data.map_vectors !== vectorPayload) { residualVectors = decodeResidualVectors(data.map_vectors); vectorPayload = data.map_vectors; }
    const lp = [], lc = [];
    function line(a, b, color, arrow = true) {
      const c = new THREE.Color(color); lp.push(...a.toArray(), ...b.toArray()); lc.push(...c.toArray(), ...c.toArray());
      if (arrow && a.distanceTo(b) > 0.12) {
        const m = new THREE.Mesh(new THREE.ConeGeometry(0.035, 0.12, 5), new THREE.MeshBasicMaterial({ color, depthTest: false }));
        m.position.copy(b).addScaledVector(b.clone().sub(a).normalize(), -0.15);
        m.quaternion.setFromUnitVectors(V(0,1,0), b.clone().sub(a).normalize()); residualGroup.add(m);
      }
    }
    function node(l, stage, norm) {
      const focused = l === data.layer && stage === data.stage || data.stage === "pre" && l === data.layer - 1 && stage === "post";
      const m = new THREE.Mesh(new THREE.BufferGeometry(), new THREE.MeshBasicMaterial({ vertexColors: true, side: THREE.DoubleSide, depthTest: false }));
      m.rotation.y = Math.PI / 2; m.position.copy(residualPos(l, stage));
      m.userData = { type: "residual", layer: l, stage, norm };
      residualGroup.add(m); residualNodes.push(m);
      const border = new THREE.LineSegments(new THREE.BufferGeometry(), new THREE.LineBasicMaterial({ color: 0xffffff, depthTest: false }));
      border.position.copy(m.position); border.rotation.copy(m.rotation); residualGroup.add(border);
      let write = null;
      if (l === data.selected.layer && stage === "post") {
        write = new THREE.Mesh(new THREE.BufferGeometry(), new THREE.MeshBasicMaterial({ vertexColors: true, side: THREE.DoubleSide, depthTest: false }));
        write.position.copy(m.position); write.rotation.copy(m.rotation);
        write.userData = { type: "residual-write", layer: l, stage, norm: data.selected.write?.norm || 0, contribution: true, n: data.selected.n };
        residualGroup.add(write); residualNodes.push(write);
      }
      // A dimension ruler runs across the entire band, regardless of the number of layers.
      const ticks = [], baseline = -0.135;
      ticks.push(-BAND_WIDTH/2,baseline,0,BAND_WIDTH/2,baseline,0);
      for (let i = 0; i <= 4; i++) { const x = -BAND_WIDTH/2 + i/4*BAND_WIDTH; ticks.push(x,baseline,0,x,baseline-0.055,0); }
      const ruler = new THREE.LineSegments(new THREE.BufferGeometry(), new THREE.LineBasicMaterial({ color: focused ? 0xc8e9f1 : 0x3f626b, depthTest: false }));
      ruler.geometry.setAttribute("position", new THREE.Float32BufferAttribute(ticks,3)); ruler.position.copy(m.position); ruler.rotation.copy(m.rotation); residualGroup.add(ruler);
      const el = document.createElement("button"), axis = document.createElement("span");
      el.className = "res-map-label"; axis.className = "res-map-axis";
      el.textContent = `L${l} ${stage === "mid" ? "hᵐ" : stage === "post" ? "h⁺" : "h⁰"}`;
      el.title = `‖h‖ ${norm.toPrecision(5)}`; el.onclick = () => cb.onPickResidual?.(l,stage);
      labelsDiv.append(el,axis); residualLabels.push({ el, axis, point:m.position.clone(), layer:l, stage });
      residualBands.push({ mesh:m, border, write, focused, axis });
      return m.position;
    }
    let previous = node(0,"pre",data.layers[0].pre);
    for (const r of data.layers) {
      const mid = node(r.layer,"mid",r.mid), post = node(r.layer,"post",r.post);
      line(previous,mid,0x677bce); line(mid,post,0x45aebf);
      for (let i = 0; i <= 8; i++) {
        const z = BAND_WIDTH*(i/8-0.5), a=previous.clone(), b=mid.clone(), c=post.clone(); a.z=b.z=c.z=z;
        line(a,b,0x233c49,false); line(b,c,0x233c49,false);
      }
      previous=post;
    }
    for (const w of data.writers) if (w.n !== data.selected.n || w.layer !== data.selected.layer) line(neuronPos(w.layer,w.n),residualPos(w.layer,"post"),0x3d5661,false);
    for (const r of data.readers) line(residualPos(r.layer,"mid"),neuronPos(r.layer,r.n),r.local_delta<0?0xff65a2:0xffc16c);
    residualBasePos=lp; residualBaseCol=lc; residualFlowGeo=new THREE.BufferGeometry();
    residualGroup.add(new THREE.LineSegments(residualFlowGeo,new THREE.LineBasicMaterial({vertexColors:true,transparent:true,opacity:0.85,depthTest:false})));
    refreshResidualBands(curDist()<homeDist*0.4); render();
  }
  function placeResidualLabels(w,h) {
    residualGroup.visible=!!residualData&&layerMode===null;
    const close=curDist()<homeDist*0.4;
    if (residualData && close!==residualLOD) refreshResidualBands(close);
    for (const a of residualLabels) {
      const v=a.point.clone().project(camera);
      const important=a.layer===residualData?.layer||a.layer===residualData?.layer+1||a.layer===0||a.layer===L-1;
      a.el.hidden=layerMode!==null||(!close&&!important&&(a.layer%4||a.stage!=="post"))||Math.abs(v.z)>1||Math.abs(v.x)>1||Math.abs(v.y)>1;
      a.el.style.left=`${(v.x+1)/2*w}px`; a.el.style.top=`${(1-v.y)/2*h+14}px`;
      a.axis.hidden=a.el.hidden||!close||!important;
      a.axis.style.left=a.el.style.left; a.axis.style.top=`${(1-v.y)/2*h+30}px`;
    }
  }

  // Project overlays after the camera updates so they stay pinned to the same neuron as the map moves.
  const selectionArrow = labelsDiv.querySelector("#neuron-pointer");
  const arrowHeight = Number(selectionArrow.firstElementChild.getAttribute("height"));
  function placeSelectionArrow(w, h) {
    if (!selected || !w || !h || (layerMode !== null && selected.layer !== layerMode)) {
      selectionArrow.hidden = true;
      return;
    }
    const v = neuronPos(selected.layer, selected.n).project(camera);
    selectionArrow.hidden = !(Math.abs(v.x) <= 1 && Math.abs(v.y) <= 1 && Math.abs(v.z) <= 1);
    if (selectionArrow.hidden) return;
    const x = (v.x + 1) / 2 * w, y = (1 - v.y) / 2 * h;
    selectionArrow.style.left = `${x}px`;
    selectionArrow.style.top = `${y}px`;
    selectionArrow.classList.toggle("below", y < arrowHeight + 2);
  }

  // Claude's annotations use the same projection for their labels.
  let notes = [];
  function placeNotes(w, h) {
    for (const a of notes) {
      const v = neuronPos(a.layer, a.n).project(camera);
      a.el.style.display = v.z > 1 || layerMode !== null && a.layer !== layerMode ? "none" : "";
      a.el.style.left = `${(v.x + 1) / 2 * w}px`; a.el.style.top = `${(1 - v.y) / 2 * h}px`;
    }
  }
  function setNotes(items) {
    for (const a of notes) a.el.remove();
    notes = (items || []).map(it => {
      // a zero-size anchor at the neuron: the ring is centred on it, the text sits to its right
      const el = document.createElement("div"); el.className = "note";
      const t = document.createElement("span"); t.textContent = it.text || `L${it.layer} N${it.n}`; el.appendChild(t);
      labelsDiv.appendChild(el); return { layer: it.layer, n: it.n, el };
    });
    render();
  }

  let pending = false;
  function render() {
    if (pending) return; pending = true;
    requestAnimationFrame(() => {
      pending = false;
      for (const m of [pointMat, hiMat]) { m.uniforms.ref.value = homeDist; m.uniforms.tdist.value = curDist(); }
      applyFade();
      applyLayerLighting();
      placeResidualLabels(canvas.clientWidth, canvas.clientHeight);
      renderer.render(scene, camera);
      const w = canvas.clientWidth, h = canvas.clientHeight;
      placeNotes(w, h);
      placeSelectionArrow(w, h);
    });
  }
  function resize() {
    const w = canvas.clientWidth, h = canvas.clientHeight;
    renderer.setSize(w, h, false);
    webMat.resolution.set(w, h);
    const narrow = w <= 760, li = narrow ? 0 : insets.left, ri = narrow ? 0 : insets.right;
    const availW = Math.max(200, w - li - ri);
    camera.aspect = w / h;
    if (fitPerPx === null) {
      // distance at which the model's bounding sphere fits the visible gap, then place the camera there once
      const center = home.target, r = Math.max(...bounds.map(b => b.distanceTo(center)));
      const vHalf = THREE.MathUtils.degToRad(camera.fov / 2), hHalf = Math.atan(Math.tan(vHalf) * availW / h);
      homeDist = r / Math.sin(Math.min(vHalf, hHalf)) * 0.82;
      camera.position.copy(home.target).addScaledVector(home.dir, homeDist);
      controls.update();
      fitPerPx = homeDist;
    }
    // the orbit target (and vanishing point) sits in the middle of the visible gap between the list and the neuron
    // panel, or at ?targetx= of the canvas width
    const tx = TARGET_X ? w * TARGET_X : li + availW / 2;
    camera.setViewOffset(w, h, w / 2 - tx, 0, w, h);
    camera.updateProjectionMatrix(); render();
  }
  new ResizeObserver(resize).observe(canvas);
  controls.addEventListener("change", render);

  // picking
  const ray = new THREE.Raycaster(); ray.params.Points.threshold = 0.05;
  const mouse = new THREE.Vector2();
  function pick(ev) {
    const r = canvas.getBoundingClientRect();
    mouse.set((ev.clientX - r.left) / r.width * 2 - 1, -(ev.clientY - r.top) / r.height * 2 + 1);
    ray.setFromCamera(mouse, camera);
    if (residualGroup.visible) {
      const hit = ray.intersectObjects(residualNodes)[0];
      if (hit) {
        const meta=hit.object.userData, cells=meta.cells||[];
        const index=Math.min(cells.length-1,Math.max(0,Math.floor((hit.uv?.x||0)*cells.length)));
        const cell=cells[index];
        return {...meta,type:"residual",...(cell?{dimension:cell.start,endDimension:cell.end,value:cell.value}: {})};
      }
    }
    // hits come sorted by depth; on a sheet seen at an angle several neighbours fall within the threshold, so take
    // the frontmost layer's hits and pick the point closest to the ray
    // a head is under the pointer when its line is within a few pixels on screen (on the nearest visible slab that
    // has one); it wins over a neuron only when its slab is in front of that neuron
    const px = v => { v.project(camera); return Math.hypot((v.x + 1) / 2 * r.width - (ev.clientX - r.left), (1 - v.y) / 2 * r.height - (ev.clientY - r.top)); };
    let headHit = null;
    // the open head's dot, where its links meet
    if (headSrc && headSel && px(headSrc.clone()) < 8) {
      const sl = slabs[headSel.layer].userData;
      headHit = { type: "head", layer: headSel.layer, full: sl.full, head: headSel.head, distance: ray.ray.origin.distanceTo(headSrc) };
    }
    for (const sh of headHit ? [] : ray.intersectObjects(slabs)) {
      if (sh.object.material.opacity <= 0.005) continue;  // faded slabs don't pick
      const { layer, full, heads } = sh.object.userData;
      // where the ray crosses the face the head lines are drawn on
      const lx = sh.object.position.x + sh.object.geometry.parameters.width / 2;
      if (Math.abs(ray.ray.direction.x) < 1e-6) continue;
      const on = ray.ray.origin.clone().addScaledVector(ray.ray.direction, (lx - ray.ray.origin.x) / ray.ray.direction.x);
      if (Math.abs(on.z) > HEAD_W) continue;  // past the ends of the lines
      const head = Math.floor((0.5 - (on.y - SHEET_Y0 - SHEET / 2) / HEAD_SPAN) * heads);
      if (head < 0 || head >= heads) continue;
      if (px(new THREE.Vector3(lx, headY(head, heads), on.z)) < 6) { headHit = { type: "head", layer, full, head, distance: sh.distance }; break; }
    }
    const hits = ray.intersectObject(points).filter(h => layerMode === null || Math.floor(h.index / N) === layerMode);
    if (hits.length && !(headHit && headHit.distance < hits[0].distance)) {
      // the frontmost layer that is actually visible at the current fade, then the point closest to the ray in it
      const layers = [...new Set(hits.map(h => Math.floor(h.index / N)))];
      const front = layers.find(l => layerVisibility(l) > 0.5) ?? layers.reduce((a, b) => layerVisibility(b) > layerVisibility(a) ? b : a);
      let best = null;
      for (const h of hits) if (Math.floor(h.index / N) === front && (!best || h.distanceToRay < best.distanceToRay)) best = h;
      return { type: "neuron", layer: front, n: best.index % N };
    }
    if (headHit) return headHit;
    const lb = ray.intersectObjects(labels).find(h => h.object.material.opacity > 0);
    if (lb) return { type: "block", layer: lb.object.userData.layer };
    const b = ray.intersectObjects(slabs).find(x => x.object.material.opacity > 0.005);
    if (b) return { type: "block", ...b.object.userData };
    return null;
  }
  let moveQueued = null, downAt = null;
  canvas.addEventListener("pointermove", ev => {
    if (moveQueued) { moveQueued = ev; return; }
    moveQueued = ev;
    requestAnimationFrame(() => {
      const e = moveQueued; moveQueued = null;
      const p = pick(e);
      const h = p && p.type === "neuron" ? { layer: p.layer, n: p.n } : null;
      if (JSON.stringify(h) !== JSON.stringify(hovered)) { hovered = h; refreshHighlights(); }
      const hh = p && p.type === "head" ? { layer: p.layer, head: p.head } : null;
      if (JSON.stringify(hh) !== JSON.stringify(headHover)) { headHover = hh; paintHeads(); render(); }
      cb.onHover(p, e.clientX, e.clientY);
    });
  });
  canvas.addEventListener("pointerleave", () => { hovered = null; refreshHighlights(); if (headHover) { headHover = null; paintHeads(); render(); } cb.onHover(null); });
  canvas.addEventListener("pointerdown", ev => { downAt = [ev.clientX, ev.clientY]; });
  canvas.addEventListener("pointerup", ev => {
    if (!downAt || Math.hypot(ev.clientX - downAt[0], ev.clientY - downAt[1]) > 4) return;
    const p = pick(ev);
    if (p && p.type === "neuron") cb.onPick(p.layer, p.n);
    else if (p && p.type === "residual") cb.onPickResidual?.(p.layer, p.stage, p.dimension);
    else if (p && p.type === "head") cb.onPickHead && cb.onPickHead(p.layer, p.head);
    else if (p && p.type === "block") cb.onPickLayer && cb.onPickLayer(p.layer);
  });

  // the closer the camera is to the orbit target, the more the layers away from the target's x fade out
  const fade = { fx: 0, fw: 1, amt: 0, side: 1 };
  const layerVisibility = l => {
    const dx = layerX(l) + MLP_DX - fade.fx, d = dx / fade.fw;
    if (dx * fade.side / GAP > 0.45) return 1 - fade.amt;
    return 1 - 0.75 * fade.amt * (1 - Math.exp(-d * d));
  };
  let layerMode = null, saved = null;
  function enterLayer(l) {
    if (layerMode === null) saved = { pos: camera.position.clone(), tgt: controls.target.clone() };
    layerMode = l;
    const c = new THREE.Vector3(layerX(l) + MLP_DX, SHEET_Y0 + SHEET / 2, 0);
    const dist = (SHEET / 2) / Math.tan(THREE.MathUtils.degToRad(camera.fov / 2)) * 1.12;
    controls.target.copy(c); camera.position.copy(c).add(new THREE.Vector3(-dist, 0, 0));
    controls.enableRotate = false;
    controls.mouseButtons = { LEFT: THREE.MOUSE.PAN, MIDDLE: THREE.MOUSE.DOLLY, RIGHT: THREE.MOUSE.PAN };
    controls.update(); render(); cb.onLayerMode && cb.onLayerMode(l);
  }
  function exitLayer() {
    if (layerMode === null) return;
    layerMode = null;
    controls.enableRotate = true;
    controls.mouseButtons = { LEFT: THREE.MOUSE.ROTATE, MIDDLE: THREE.MOUSE.DOLLY, RIGHT: THREE.MOUSE.PAN };
    camera.position.copy(saved.pos); controls.target.copy(saved.tgt);
    controls.update(); render(); cb.onLayerMode && cb.onLayerMode(null);
  }

  function applyFade() {
    if (layerMode !== null) {
      const fx = layerX(layerMode) + MLP_DX;
      fade.fx = fx; fade.fw = 0.05; fade.amt = 1;
      for (const slab of slabs) { slab.material.opacity = 0; slab.userData.edges.material.opacity = 0; }
      for (let l = 0; l < L; l++) labels[l].material.opacity = l === layerMode ? labels[l].material.userData.base : 0;
      for (const m of fadeLines) m.opacity = 0;
      return;
    }
    // continuous in zoom: eased on log-distance from the overview (no fade) to a sheet filling the view (full fade)
    const dist = camera.position.distanceTo(controls.target);
    const far = Math.log(homeDist * 0.7), near = Math.log(homeDist * 0.07);
    const amt = THREE.MathUtils.smoothstep(THREE.MathUtils.clamp((far - Math.log(dist)) / (far - near), 0, 1), 0, 1);
    // the focus is the layer whose sheet centre lands closest to the middle of the view on screen
    const tv = controls.target.clone().project(camera), pv = new THREE.Vector3();
    let fx = controls.target.x, best = Infinity;
    for (let l = 0; l < L; l++) {
      pv.set(layerX(l) + MLP_DX, SHEET_Y0 + SHEET / 2, 0).project(camera);
      const dd = Math.hypot((pv.x - tv.x) * camera.aspect, pv.y - tv.y);
      if (pv.z < 1 && dd < best) { best = dd; fx = layerX(l) + MLP_DX; }
    }
    const fw = GAP * (0.85 + 2.5 * (1 - amt));
    const side = Math.sign(camera.position.x - fx) || 1;
    fade.fx = fx; fade.fw = fw; fade.amt = amt; fade.side = side;
    for (const slab of slabs) {
      const d = (slab.position.x - fx) / fw, f = 1 - amt * (1 - Math.exp(-d * d));
      slab.material.opacity = slab.userData.baseOpacity * f; slab.userData.edges.material.opacity = f;
    }
    for (let l = 0; l < L; l++) {
      const d = (layerX(l) + MLP_DX - fx) / fw;
      labels[l].material.opacity = labels[l].material.userData.base * (1 - 0.5 * amt * (1 - Math.exp(-d * d)));  // other layers keep half
    }
    for (const m of fadeLines) m.opacity = m.userData.base * (1 - 0.9 * amt);
  }

  function applyLayerLighting() {
    const selectedLayer = selected && (layerMode === null || selected.layer === layerMode) ? selected.layer : null;
    const hasSelection = selectedLayer !== null;
    const showTiers = hasSelection || activeLayers.size > 0;
    for (let l = 0; l < L; l++) {
      const focused = l === selectedLayer, active = activeLayers.has(l);
      const tier = focused ? LAYER_BRIGHTNESS.selected
        : !active ? LAYER_BRIGHTNESS.idle : hasSelection ? LAYER_BRIGHTNESS.otherActive : LAYER_BRIGHTNESS.active;
      const visible = layerMode === null || l === layerMode;
      const shade = showTiers || layerMode !== null ? 1 : layerVisibility(l);
      layerBrightness[l] = visible ? tier * shade : 0;

      const label = labels[l].material, outline = outlines[l], slab = slabs[l];
      const emphasis = focused ? SELECTED_LAYER_COLOR : active ? ACTIVE_LAYER_COLOR : null;
      const hasData = dataLayers.has(l);
      label.opacity = focused ? 1 : label.opacity * tier;
      label.color.set(emphasis ?? (hasData ? 0xffffff : 0xc8d4da));
      outline.visible = visible;
      outline.material.opacity = layerBrightness[l];
      outline.material.color.set(emphasis ?? (hasData ? PAL.wireData : PAL.wire));
      outline.material.depthTest = !focused;
      outline.renderOrder = focused ? 4 : 0;
      slab.material.opacity = focused && layerMode === null ? slab.userData.baseOpacity : slab.material.opacity * tier;
      const edges = slab.userData.edges.material;
      edges.opacity = focused && layerMode === null ? 1 : edges.opacity * tier;
    }
    conLines.material.opacity = webMat.opacity = hasSelection ? LAYER_BRIGHTNESS.otherActive : 1;
    if (hasSelection) {
      for (const m of fadeLines) m.opacity *= LAYER_BRIGHTNESS.otherActive;
    }
  }

  function flyTo(target, dist) {
    const from = controls.target.clone(), to = target.clone();
    const dir = camera.position.clone().sub(controls.target), d0 = dir.length();
    dir.normalize();
    const t0 = performance.now();
    (function step(t) {
      const k = Math.min(1, (t - t0) / 450), e = k * k * (3 - 2 * k);
      controls.target.lerpVectors(from, to, e);
      camera.position.copy(controls.target).addScaledVector(dir, d0 + (dist - d0) * e);
      controls.update(); render();
      if (k < 1) requestAnimationFrame(step);
    })(t0);
  }
  const curDist = () => camera.position.distanceTo(controls.target);
  function neuronInView(point) {
    camera.updateMatrixWorld();
    const v = point.clone().project(camera), w = canvas.clientWidth;
    const x = (v.x + 1) / 2 * w;
    const left = w > 760 ? insets.left : 0, right = w > 760 ? insets.right : 0;
    return Math.abs(v.z) <= 1 && Math.abs(v.y) <= 0.9 && x >= left + 14 && x <= w - right - 14;
  }
  function locateNeuron(layer, n, center = false) {
    const target = neuronPos(layer, n);
    if (center || !neuronInView(target)) {
      const dist = curDist();
      flyTo(target, layerMode === null ? Math.min(dist, homeDist * 0.35) : dist);
    }
  }

  // face a layer's neuron sheet head-on, close enough that the neighbours fade away
  function viewLayer(layer) {
    const c = new THREE.Vector3(layerX(layer) + MLP_DX, SHEET_Y0 + SHEET / 2, 0);
    const dist = (SHEET / 2) / Math.tan(THREE.MathUtils.degToRad(camera.fov / 2)) * 1.25;
    const from = controls.target.clone(), p0 = camera.position.clone();
    const p1 = c.clone().add(new THREE.Vector3(dist, 0.0001, 0)), t0 = performance.now();
    (function step(t) {
      const k = Math.min(1, (t - t0) / 600), e = k * k * (3 - 2 * k);
      controls.target.lerpVectors(from, c, e); camera.position.lerpVectors(p0, p1, e);
      controls.update(); render();
      if (k < 1) requestAnimationFrame(step);
    })(t0);
  }

  return {
    neuronPos,
    setResidualTrace,
    focusResidual(layer, stage) {
      if (layerMode !== null) exitLayer();
      flyTo(residualPos(layer, stage), Math.min(curDist(), 12));
    },
    // heads lit by level (0..1 per head, per layer; null clears), and the selected head
    setHeadLevels(levels) { headLevels = levels; paintHeads(); render(); },
    selectHead(layer, head) {
      headSel = layer == null ? null : { layer, head };
      if (!headSel && headSrc) { headSrc = null; connections = []; refreshHighlights(); }
      paintHeads(); render();
    },
    // an open head's strongest neuron links, drawn from the middle of its line like a neuron's direct links
    setHeadLinks(layer, head, list) {
      const i = (headStart[layer] + head) * 6;
      headSrc = headMid(layer, head);
      connections = list.map(k => ({ ...k, sign: 0 }));
      refreshHighlights();
    },
    // a hovered head's links, previewed like a hovered neuron's (null clears)
    setHoverHead(layer, head, list) {
      hoverCon = layer == null ? null : { from: headMid(layer, head), list: list.map(k => ({ ...k, sign: 0 })) };
      refreshHighlights();
    },
    // mark a head as hovered from outside the map (null clears)
    hoverHead(layer, head) {
      const hh = layer == null ? null : { layer, head };
      if (JSON.stringify(hh) !== JSON.stringify(headHover)) { headHover = hh; paintHeads(); render(); }
    },
    // the open neuron's strongest linked heads, drawn from the neuron to each head's line
    setNeuronHeads(list) { neuronHeads = list || []; refreshHighlights(); },
    // the swept neurons' colours show on one layer at a time (the current one); the others go back to idle
    setUnits(layer, units) {
      dataLayers.add(layer);
      labels[layer].material.userData.base = 1;
      const zs = units.filter(u => u.kind === "neuron").map(u => u.z);
      const rnd = units.filter(u => u.kind === "random").map(u => u.z);
      const floor = rnd.length ? Math.max(...rnd) : Math.min(...zs), top = Math.max(...zs, floor + 1e-6);
      const colors = new Map();
      for (const u of units) {
        if (u.kind !== "neuron") continue;
        let c = PAL.swept;
        if (u.name) c = PAL.named;
        else if (u.z > floor) c = PAL.lo.clone().lerp(PAL.hi, Math.min(1, (u.z - floor) / (top - floor)));
        colors.set(layer * N + u.n, c);
      }
      setPointColors("units", colors);
      render();
    },
    // fly: bring the neuron into view without ever zooming out from where the user is
    select(layer, n, fly = true) {
      selected = layer == null ? null : { layer, n };
      connections = []; headSrc = null; neuronHeads = [];
      refreshHighlights();
      // off screen (or behind the camera): fly to it, never zooming out from where the user is
      if (fly && selected) locateNeuron(layer, n);
    },
    setConnections(con) {
      connections = [...con.down.map(c => ({ ...c, side: "down" })), ...con.up.map(c => ({ ...c, side: "up" }))];
      refreshHighlights();
    },
    setHoverConnections(src, con) {
      hoverCon = src ? { src, list: [...con.down, ...con.up] } : null;
      refreshHighlights();
    },
    focusLayer() {},
    setLayout(cells) {
      layoutCells = cells;
      for (let l = 0; l < L; l++) for (let n = 0; n < N; n++) {
        const p = neuronPos(l, n), k = (l * N + n) * 3;
        pos[k] = p.x; pos[k + 1] = p.y; pos[k + 2] = p.z;
      }
      geo.attributes.position.needsUpdate = true;
      geo.computeBoundingSphere();
      if (residualData) setResidualTrace(residualData);
      buildWeb();
      refreshHighlights();
    },
    // edges: Int16Array, 6 per link (src layer, src n, dst layer, dst n, 10 * z * sign, hop), or null to hide
    setWeb(edges) { webEdges = edges; buildWeb(); refreshHighlights(); },
    enterLayer, exitLayer, viewLayer, setNotes,
    // In a single layer, keep the sheet in place while the neuron is already visible.
    focusNeuron(layer, n) { locateNeuron(layer, n, layerMode === null); },
    // fly close to one neuron, keeping the current viewing direction
    flyToNeuron(layer, n, closeness = 0.22) { flyTo(neuronPos(layer, n), homeDist * closeness); },
    get layerMode() { return layerMode; },
    home() { flyTo(home.target, homeDist); },
    setMarks(marks, firedLayers = new Set()) {
      activeLayers = firedLayers;
      const colors = new Map(), down = new THREE.Color(1.0, 0.31, 0.6);
      for (const { layer, n, sign, w } of marks) {
        if (!Number.isInteger(layer) || layer < 0 || layer >= L || !Number.isInteger(n) || n < 0 || n >= N) continue;
        const tint = sign === "-" ? down : PAL.hi;
        colors.set(layer * N + n, PAL.idle.clone().lerp(tint, w ?? 1));
      }
      setPointColors("marks", colors);
      render();
    },
    setInsets(left, right) { insets.left = left; insets.right = right; resize(); },
    hover(layer, n) { hovered = layer == null ? null : { layer, n }; refreshHighlights(); },
    debug() { const tv = controls.target.clone().project(camera); return { tgtScreenX: +((tv.x + 1) / 2).toFixed(3), cam: camera.position.toArray().map(v => +v.toFixed(2)), tgt: controls.target.toArray().map(v => +v.toFixed(2)), selected, hovered, n: connections.length, hover: hoverCon ? hoverCon.list.length : 0, dist: curDist(), homeDist, w: canvas.clientWidth, h: canvas.clientHeight, insets }; },
    render,
  };
}
