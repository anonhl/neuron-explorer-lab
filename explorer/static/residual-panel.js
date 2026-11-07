// One shared capture: changing layer, token or neuron only reads it; it never reruns inference.
export function initResidualPanel(root, { api, t, esc, map, arch, onToken, onPick, onCompare, onIntervene, onRun, errorText }) {
  let capture = null, data = null, layer = 0, neuron = -1, stage = 'post', generation = 0, error = null, busy = false;
  let mapOn = true;
  let dimension = 0, windowStart = 0;
  const candidates = new Map();
  const num = v => Number.isFinite(v) ? (v > 0 ? '+' : '') + Number(v.toPrecision(4)) : '—';
  const tone = v => v < 0 ? 'res-negative' : 'res-positive';
  root.innerHTML = `<div class="res-intro" data-res="intro"></div>
    <div class="res-empty" data-res="empty"></div>
    <div data-res="controls" hidden>
      <label><span data-res="token-label"></span><select data-res="token"></select></label>
      <div class="res-controls"><label><span data-res="layer-label"></span><select data-res="layer">${Array.from({ length: arch.n_layers }, (_, l) => `<option value="${l}">L${l}</option>`).join('')}</select></label>
      <label><span data-res="stage-label"></span><select data-res="stage"><option value="pre"></option><option value="mid"></option><option value="post"></option></select></label></div>
      <label class="res-toggle"><input type="checkbox" data-res="map" checked><span data-res="map-label"></span></label>
      <div class="res-window"><label><span data-res="window-label"></span><input data-res="window" type="range" min="0" max="0" step="1" value="0"></label>
      <div><span data-res="window-range" class="mono"></span><button data-action="expand">${t('res.expand')}</button></div></div>
    </div><div data-res="status" role="status"></div>
    <div data-res="overview" class="res-overview"></div>
    <div data-res="detail"></div><div data-res="candidates" class="res-candidates"></div>`;
  const el = key => root.querySelector(`[data-res="${key}"]`);
  el('token').onchange = () => onToken(+el('token').value);
  el('layer').onchange = () => selectLayer(+el('layer').value);
  el('stage').onchange = () => { stage = el('stage').value; render(); };
  el('map').onchange = () => { mapOn = el('map').checked; syncMap(); };
  el('window').oninput = () => {
    windowStart = +el('window').value; dimension = windowStart;
    render(); map.focusResidual(layer, stage);
  };
  root.addEventListener('click', async e => {
    const b = e.target.closest('button'); if (!b) return;
    if (b.dataset.resLayer != null) selectLayer(+b.dataset.resLayer);
    if (b.dataset.resPick != null) onPick(+b.dataset.layer, +b.dataset.resPick);
    if (b.dataset.resKeep != null) {
      const l = +b.dataset.layer, n = +b.dataset.resKeep;
      candidates.set(`${l}:${n}`, { layer: l, n }); renderCandidates();
    }
    if (b.dataset.resRemove) { candidates.delete(b.dataset.resRemove); renderCandidates(); }
    if (b.dataset.action === 'run') onRun();
    if (b.dataset.action === 'intervene' && data) onIntervene?.({ layer, n: data.selected.n, stage, pos: data.pos });
    if (b.dataset.action === 'expand') map.focusResidual(layer, stage);
    if (b.dataset.action === 'compare') onCompare([...candidates.values()]);
    if (b.dataset.action === 'keep' && data) { const s = data.selected; candidates.set(`${s.layer}:${s.n}`, { layer: s.layer, n: s.n }); renderCandidates(); }
    if (b.dataset.action === 'export' && data) {
      try {
        const out = await api(`/api/infer/residual/export?${new URLSearchParams({ capture: capture.capture, pos: data.pos })}`);
        const url = URL.createObjectURL(new Blob([JSON.stringify(out, null, 2)], { type: 'application/json' }));
        const a = document.createElement('a'); a.href = url; a.download = `residual-token-${data.pos}-all-layers.json`; a.click(); URL.revokeObjectURL(url);
      } catch (err) { error = err; render(); }
    }
  });
  function selectLayer(l, nextStage = stage) { layer = l; neuron = -1; stage = nextStage; load(); }
  function syncMap() {
    map.setResidualTrace(mapOn && data && capture ? { ...data, stage, dimension, windowStart } : null);
    if (data && root.classList.contains('on')) {
      const marks = [...data.writers, ...data.readers].map(x => ({ layer: x.layer, n: x.n, sign: x.act < 0 ? '-' : '+', w: 0.9 }));
      map.setMarks(marks, new Set(marks.map(x => x.layer)));
    }
    const legend = document.querySelector('#res-map-key');
    if (legend) { legend.hidden = !(mapOn && data && capture); legend.textContent = t('res.mapKey'); }
  }
  function renderCandidates() {
    el('candidates').innerHTML = `<div class="lab">${t('res.candidates')} · ${candidates.size}</div>
      <div class="res-saved">${[...candidates].map(([key, x]) => `<span><button data-res-pick="${x.n}" data-layer="${x.layer}">L${x.layer} N${x.n}</button><button data-res-remove="${key}" aria-label="${esc(t('res.remove'))}">×</button></span>`).join('')}</div>
      <button class="res-action" data-action="compare" ${candidates.size ? '' : 'disabled'}>${t('res.compare')}</button>`;
  }
  function dimensions(v) {
    const rows = [...v.positive, ...v.negative];
    const max = Math.max(...rows.map(x => Math.abs(x.value)), 1e-12);
    return `<div class="res-dimensions">${rows.map(x => `<div class="res-dim"><span>d${x.dim}</span><span class="res-dim-bar"><i class="${tone(x.value)}" style="width:${Math.abs(x.value) / max * 48}%;${x.value < 0 ? 'right' : 'left'}:50%"></i></span><b class="${tone(x.value)}">${num(x.value)}</b></div>`).join('')}</div>`;
  }
  function tokens(rows) { return rows.map(x => `<span class="res-token" title="${esc(t('res.tokenScore'))}: ${num(x.score)}">${esc(x.token)} <small>${num(x.score)}</small></span>`).join(''); }
  function neuronRows(rows, readers = false) {
    return `<table class="res-table"><thead><tr><th>${t('res.neuron')}</th><th>${t('res.activation')}</th><th>${t(readers ? 'res.localRead' : 'res.writeNorm')}</th><th></th></tr></thead><tbody>${rows.map(x => `<tr>
      <td><button data-res-pick="${x.n}" data-layer="${x.layer}">L${x.layer} N${x.n}</button></td>
      <td class="${tone(x.act)}" title="${esc(t('res.normalized'))}: ${num(x.score)}×">${num(x.act)}</td>
      <td class="${readers ? tone(x.local_delta) : ''}">${readers ? num(x.local_delta) : num(x.write_norm)}</td>
      <td><button data-res-keep="${x.n}" data-layer="${x.layer}" aria-label="${esc(t('res.keep'))}">＋</button></td></tr>`).join('')}</tbody></table>`;
  }
  function render() {
    el('intro').textContent = t('res.intro');
    el('empty').hidden = !!capture;
    el('empty').innerHTML = `<p>${t('res.empty')}</p><button class="res-action" data-action="run">${t('res.goRun')}</button>`;
    el('controls').hidden = !capture;
    for (const key of ['token', 'layer', 'stage', 'map']) el(`${key}-label`).textContent = t(`res.${key}`);
    el('window-label').textContent = t('res.window');
    const hidden = data?.hidden || arch.hidden || 1;
    dimension = Math.min(Math.max(0, dimension), hidden - 1);
    windowStart = Math.min(Math.max(0, windowStart), Math.max(0, hidden - 64));
    el('window').max = Math.max(0, hidden - 64); el('window').value = windowStart;
    el('window-range').textContent = `d${windowStart} — d${Math.min(hidden - 1, windowStart + 63)}`;
    root.querySelector('[data-action="expand"]').textContent = t('res.expand');
    for (const option of el('stage').options) option.textContent = t(`res.${option.value}`);
    el('stage').value = stage; el('layer').value = layer;
    el('status').textContent = error ? errorText(error) : busy ? t('res.loading') : '';
    el('overview').innerHTML = data ? `<div class="lab">${t('res.allLayers', { n: data.layers.length })}</div><div class="res-layer-grid">${data.layers.map(r => `<button data-res-layer="${r.layer}" class="${r.layer === layer ? 'on' : ''}" title="${esc(t('res.post'))} ‖h‖=${r.post.toPrecision(4)}; ΔMLP=${r.mlp.toPrecision(4)}">L${r.layer}<small>${r.post.toPrecision(3)}</small></button>`).join('')}</div>` : '';
    const d = data;
    if (!d) { el('detail').replaceChildren(); renderCandidates(); syncMap(); return; }
    const s = d.selected, v = d.stages[stage];
    el('detail').innerHTML = `<div class="res-section"><div class="lab">L${d.layer} · ${esc(d.token)} · ${t(`res.${stage}`)}</div>
      <div class="res-metrics">${d.hidden} ${t('res.dimensions')} · ‖h‖ ${v.norm.toPrecision(5)}</div>
      <div class="res-tokens">${tokens(v.tokens)}</div><p class="dim res-note">${t('res.lensHint')}</p>
      <details><summary>${t('res.signedDimensions')}</summary>${dimensions(v)}
      <label>${t('res.inspectDim')} <input data-res-dim type="number" min="0" max="${d.hidden - 1}" value="${dimension}"></label><div data-res-values class="mono"></div></details>
      <button class="res-action" data-action="export">${t('res.export')}</button>
      <button class="res-action" data-action="intervene">${t('lab.interveneHere')}</button></div>
      <div class="res-section"><div class="lab">${t('res.updates')}</div>
      <p>${t('res.attentionUpdate')} ‖Δr‖ ${d.updates.attention.norm.toPrecision(5)}</p>
      <details><summary>${t('res.signedDimensions')}</summary>${dimensions(d.updates.attention)}</details>
      <p>${t('res.mlpUpdate')} ‖Δr‖ ${d.updates.mlp.norm.toPrecision(5)}</p>
      <details><summary>${t('res.signedDimensions')}</summary>${dimensions(d.updates.mlp)}</details></div>
      <div class="res-section"><div class="lab">L${s.layer} N${s.n} → ${t('res.write')}</div>
      <p>${t('res.activation')} <b class="${tone(s.act)}">${num(s.act)}</b> · ${t('res.normalized')} ${num(s.score)}×</p>
      <div class="res-metrics">‖Δr‖ ${s.write.norm.toPrecision(5)}</div>${dimensions(s.write)}
      <div class="lab">${t('res.localTokens')}</div><div class="res-tokens res-positive">${tokens(s.write.tokens.positive)}</div><div class="res-tokens res-negative">${tokens(s.write.tokens.negative)}</div>
      <p class="dim res-note">${t('res.localHint')}</p><button class="res-action" data-action="keep">${t('res.keep')}</button></div>
      <div class="res-section"><div class="lab">${t('res.writers')}</div>${neuronRows(d.writers)}</div>
      <div class="res-section"><div class="lab">${t('res.readers')}</div>${d.readers.length ? neuronRows(d.readers, true) : `<p class="dim">${t('res.lastLayer')}</p>`}<p class="dim res-note">${t('res.readerHint')}</p></div>
      ${d.batch ? `<p class="res-note">${t('res.batch', { name: esc(d.batch.name) })}</p>` : ''}`;
    const input = root.querySelector('[data-res-dim]'), values = root.querySelector('[data-res-values]');
    input.oninput = () => {
      const i = Number(input.value);
      if (Number.isInteger(i) && i >= 0 && i < d.hidden) {
        dimension = i;
        if (i < windowStart || i >= windowStart + 64) windowStart = Math.min(Math.floor(i / 64) * 64, Math.max(0, d.hidden - 64));
        el('window').value = windowStart;
        el('window-range').textContent = `d${windowStart} — d${Math.min(d.hidden - 1, windowStart + 63)}`;
      }
      values.textContent = Number.isInteger(i) && i >= 0 && i < d.hidden
        ? ['pre', 'mid', 'post'].map(key => `${t(`res.${key}`)} ${num(d.stages[key].vector[i])}`).join(' · ') + ` · Δr ${num(s.write.vector[i])}` : '—';
    };
    input.onchange = () => { input.oninput(); syncMap(); };
    input.oninput(); renderCandidates(); syncMap();
  }
  async function load() {
    const mine = ++generation;
    if (!capture) { data = null; busy = false; render(); return; }
    busy = true; error = null; data = null; render();
    const pos = capture.pos >= 0 ? capture.pos : capture.positions.at(-1);
    try {
      const result = await api(`/api/infer/residual?${new URLSearchParams({ capture: capture.capture, pos, layer, n: neuron })}`);
      if (mine !== generation) return;
      data = result;
    } catch (err) { if (mine !== generation) return; error = err; }
    busy = false; render();
  }
  function update(inf) {
    const old = capture;
    capture = inf?.capture && inf.text ? { capture: inf.capture, pos: inf.pos, text: inf.text, tokens: inf.tokens, positions: inf.positions } : null;
    if (capture) {
      el('token').innerHTML = capture.tokens.map((text, i) => `<option value="${capture.positions[i]}">${i}: ${esc(text)}</option>`).join('');
      el('token').value = capture.pos >= 0 ? capture.pos : capture.positions.at(-1);
    }
    if (old?.capture !== capture?.capture || old?.pos !== capture?.pos) load();
  }
  render();
  return { update, render, selectLayer,
    selectDimension(l, nextStage, dim) {
      if (Number.isInteger(dim)) {
        dimension = dim; windowStart = Math.floor(dim / 64) * 64;
      }
      if (data && layer === l) { stage = nextStage; render(); }
      else selectLayer(l, nextStage);
    },
    focus(l, n) { layer = l; neuron = n; if (capture) load(); },
    clear() { capture = null; data = null; error = null; busy = false; ++generation; render(); },
    retry() { load(); },
  };
}
