// Inputs stay independent. Only the active text uses the large neuron-analysis capture;
// experiments pass immutable text snapshots and tokenize each text in its own coordinate system.
export function parseResidualVector(text, hidden) {
  const values = JSON.parse(text);
  if (!Array.isArray(values) || values.length !== hidden || values.some(v => typeof v !== 'number' || !Number.isFinite(v))) {
    throw new Error(`Custom direction must contain ${hidden} finite numbers.`);
  }
  return values;
}

export async function readResidualEvents(response, onEvent) {
  if (!response.ok) {
    const body = await response.text();
    let message = body;
    try { message = JSON.parse(body).error || body; } catch {}
    throw new Error(message);
  }
  const reader = response.body.getReader(), decoder = new TextDecoder();
  let pending = '', doneEvent = false;
  try {
    while (true) {
      const { value, done } = await reader.read();
      pending += decoder.decode(value, { stream: !done }).replace(/\r\n/g, '\n');
      let end;
      while ((end = pending.indexOf('\n\n')) >= 0) {
        const frame = pending.slice(0, end); pending = pending.slice(end + 2);
        const data = frame.split('\n').filter(l => l.startsWith('data:')).map(l => l.slice(5).trimStart()).join('\n');
        if (data) {
          const event = JSON.parse(data);
          if (event.type === 'done') doneEvent = true;
          onEvent(event);
        }
      }
      if (done) break;
    }
    if (!doneEvent) throw new Error('Experiment connection ended before completion. Completed rows are preserved.');
  } finally {
    if (!doneEvent) { try { await reader.cancel(); } catch {} }
    reader.releaseLock();
  }
}

export function initResidualLab(root, { api, t, esc, arch, errorText, onActivate, onOpen, getSelected, onNeuron }) {
  const $ = s => document.querySelector(s), storeKey = 'ne.residual-inputs.v1';
  let texts = [{ id: 'text-1', label: 'A', text: '' }], serial = 1, active = 'text-1';
  try {
    const saved = JSON.parse(localStorage.getItem(storeKey));
    if (saved?.texts?.length && saved.texts.length <= 8 && saved.texts.every(r => typeof r.id === 'string' && typeof r.label === 'string' && typeof r.text === 'string')) {
      texts = saved.texts; serial = saved.serial || texts.length; active = texts[0].id;
    }
  } catch {}
  let tokens = new Map(), tokenChat = null, tokenVersion = 0, busy = false, preparing = false, ctl = null;
  let error = null, results = [], runSnapshot = null, progress = null, stale = false, stopped = false;
  const form = { method: 'add', layer: 0, stage: 'post', scope: 'last', strength: 0.1, direction: 'neuron',
    neuron: 0, component: 'neuron', head: 0, source: texts[0].id, reference: texts[1]?.id || '',
    source_pos: -1, reference_pos: -1, positions: {}, vector: '', chat: false, max_new: 48, targets: '' };
  const label = row => t('lab.text', { name: row.label });
  const save = () => { try { localStorage.setItem(storeKey, JSON.stringify({ texts, serial })); } catch {} };
  const current = () => texts.find(r => r.id === active);
  const option = (value, title, chosen) => `<option value="${esc(value)}"${String(value) === String(chosen) ? ' selected' : ''}>${esc(title)}</option>`;
  const choices = (name, values) => `<select name="${name}">${values.map(([v, key]) => option(v, t(key), form[name])).join('')}</select>`;
  const field = (key, content, extra = '') => `<label ${extra}><span>${t(key)}</span>${content}</label>`;
  const number = (name, min, max, step = 1) => `<input name="${name}" type="number" value="${form[name]}" min="${min}" max="${max}" step="${step}" required>`;
  function dirty(invalidateTokens = false) {
    if (runSnapshot) stale = true;
    if (invalidateTokens) { tokens.clear(); ++tokenVersion; }
    error = null;
  }
  function setActive(id, analyze = true) {
    if (busy || !texts.some(r => r.id === id)) return;
    active = id; $('#inf-text').value = current().text;
    renderInputs();
    if (analyze) onActivate(current().text);
  }
  function renderInputs() {
    $('#inf-text-tabs').innerHTML = texts.map(row => `<button type="button" data-text-id="${esc(row.id)}" class="${row.id === active ? 'on' : ''}" aria-pressed="${row.id === active}" ${busy ? 'disabled' : ''}>${esc(label(row))}</button>`).join('');
    $('#inf-add-text').disabled = busy || texts.length >= 8;
    $('#inf-remove-text').disabled = busy || texts.length === 1;
    $('#inf-text').disabled = busy;
    $('#inf-text-caption').textContent = t('lab.inputCaption', { name: label(current()), count: texts.length });
  }
  $('#inf-text').value = current().text;
  $('#inf-text-tabs').onclick = e => { const b = e.target.closest('[data-text-id]'); if (b) setActive(b.dataset.textId); };
  $('#inf-add-text').onclick = () => {
    if (busy || texts.length >= 8) return;
    ++serial;
    const row = { id: `text-${serial}`, label: serial <= 26 ? String.fromCharCode(64 + serial) : String(serial), text: '' };
    texts.push(row);
    if (!form.reference) form.reference = row.id;
    dirty(true); setActive(row.id); save(); render(); $('#inf-text').focus();
  };
  $('#inf-remove-text').onclick = () => {
    if (busy || texts.length === 1) return;
    texts = texts.filter(r => r.id !== active);
    delete form.positions[active];
    if (!texts.some(r => r.id === form.source)) { form.source = texts[0].id; form.source_pos = -1; }
    if (!texts.some(r => r.id === form.reference)) { form.reference = texts.find(r => r.id !== form.source)?.id || ''; form.reference_pos = -1; }
    dirty(true); setActive(texts[0].id); save(); render();
  };
  $('#inf-text').addEventListener('input', () => {
    if (busy) return;
    current().text = $('#inf-text').value;
    dirty(true); save(); render();
  });
  $('#inf-residual-open').onclick = () => { onOpen(); render(); };
  function tokenOptions(id, selected, allowLast = true) {
    const row = tokens.get(id);
    const options = allowLast ? [option(-1, t('lab.lastToken'), selected)] : [];
    if (row) for (const unit of row.tokens) for (const pos of unit.positions) {
      options.push(option(pos, `#${pos} ${unit.text}`, selected));
    }
    if (!options.length) options.push(option('', t('lab.prepareFirst'), ''));
    return options.join('');
  }
  const textSelect = name => `<select name="${name}">${texts.map(row => option(row.id, label(row), form[name])).join('')}</select>`;
  function sourceFields(name, key) {
    return field(key, textSelect(name)) + field('lab.sourceToken', `<select name="${name}_pos">${tokenOptions(form[name], form[`${name}_pos`])}</select>`);
  }
  function render() {
    renderInputs();
    const isRemove = form.method === 'remove', isPatch = form.method === 'patch';
    const usesNeuron = (isRemove && form.component === 'neuron') || (!isRemove && !isPatch && form.direction === 'neuron');
    const usesSource = isPatch || (!isRemove && form.direction === 'difference');
    root.innerHTML = `<p class="lab-intro">${t('lab.intro')}</p>
      <form id="rl-form"><fieldset ${busy || preparing ? 'disabled' : ''}>
      <div class="rl-section-heading"><strong>${t('lab.inputs')}</strong><button type="button" data-action="prepare">${t('lab.prepare')}</button></div>
      <p class="dim rl-hint">${t('lab.inputHint')}</p>
      <label class="rl-check"><input type="checkbox" name="chat" ${form.chat ? 'checked' : ''}>${t('lab.chat')}</label>
      <div class="rl-input-list">${texts.map(row => {
        const tok = tokens.get(row.id), pos = form.positions[row.id] ?? (tok ? tok.count - 1 : '');
        return `<div class="rl-input-row"><button type="button" data-analyze="${esc(row.id)}">${esc(label(row))} ↗</button><span>${esc(row.text.slice(0, 100) || t('lab.emptyInput'))}</span><small>${tok ? t('lab.tokenCount', { n: tok.count }) : t('lab.notPrepared')}</small>${form.scope === 'token' ? field('lab.targetToken', `<select data-target="${esc(row.id)}">${tokenOptions(row.id, pos, false)}</select>`) : ''}</div>`;
      }).join('')}</div>
      <div class="rl-section-heading"><strong>${t('lab.configuration')}</strong><button type="button" data-action="selected">${t('lab.useSelected')}</button></div>
      <div class="rl-fields">
      ${field('lab.method', choices('method', [['add', 'lab.add'], ['remove', 'lab.remove'], ['erase', 'lab.erase'], ['patch', 'lab.patch']]), 'class="rl-wide"')}
      ${field('batch.layer', number('layer', 0, arch.n_layers - 1))}
      ${field('res.stage', choices('stage', [['pre', 'res.pre'], ['mid', 'res.mid'], ['post', 'res.post']]))}
      ${field('lab.scope', choices('scope', [['token', 'lab.scopeToken'], ['last', 'lab.scopeLast'], ['prompt', 'lab.scopePrompt'], ['continuation', 'lab.scopeContinuation'], ['all', 'lab.scopeAll']]), 'class="rl-wide"')}
      ${isRemove ? field('lab.component', choices('component', [['neuron', 'lab.componentNeuron'], ['attention', 'lab.componentAttention'], ['mlp', 'lab.componentMLP'], ['head', 'lab.componentHead']]), 'class="rl-wide"') : ''}
      ${!isRemove && !isPatch ? field('lab.direction', choices('direction', [['neuron', 'lab.directionNeuron'], ['difference', 'lab.directionDifference'], ['custom', 'lab.directionCustom']]), 'class="rl-wide"') : ''}
      ${usesNeuron ? field('batch.neuron', number('neuron', 0, arch.inter - 1)) : ''}
      ${isRemove && form.component === 'head' ? field('lab.head', number('head', 0, (arch.layer_types[form.layer] === 'full_attention' ? arch.attn_heads : arch.linear_v_heads) - 1)) : ''}
      ${usesSource ? sourceFields('source', isPatch ? 'lab.donor' : 'lab.positiveSource') : ''}
      ${!isPatch && !isRemove && form.direction === 'difference' ? sourceFields('reference', 'lab.negativeSource') : ''}
      ${!isPatch && !isRemove && form.direction === 'custom' ? field('lab.vector', `<textarea name="vector" rows="3" spellcheck="false" placeholder="[0.1, -0.2, …]">${esc(form.vector)}</textarea>`, 'class="rl-wide"') : ''}
      ${field('lab.strength', number('strength', isPatch || form.method === 'erase' ? 0 : -10, isPatch || form.method === 'erase' ? 1 : 10, 'any'))}
      ${field('lab.maxNew', number('max_new', 0, 128))}
      ${field('lab.targets', `<textarea name="targets" rows="2" placeholder="${esc(t('lab.targetsPlaceholder'))}">${esc(form.targets)}</textarea>`, 'class="rl-wide"')}
      </div><p class="rl-method-note">${t(`lab.hint.${form.method}`)}</p>
      ${usesSource ? `<p class="dim rl-hint">${t('lab.alignmentHint')}</p>` : ''}
      <p class="dim rl-hint">${t('lab.scopeHint')}</p>
      <button class="rl-primary" type="submit">${t('lab.runAll', { n: texts.length })}</button>
      </fieldset></form>
      <div class="rl-run-bar"><span role="status" id="rl-progress"></span>${busy ? `<button type="button" data-action="stop">${t('chat.stop')}</button>` : ''}<button type="button" data-action="export" ${runSnapshot ? '' : 'disabled'}>${t('lab.export')}</button></div>
      <div id="rl-error" role="alert"></div><div id="rl-results"></div>`;
    root.querySelector('#rl-form').onsubmit = e => { e.preventDefault(); run(); };
    renderResults();
  }
  root.addEventListener('input', e => {
    if (e.target.dataset.target) { form.positions[e.target.dataset.target] = Number(e.target.value); dirty(); renderResults(); return; }
    const name = e.target.name;
    if (!name || e.target.tagName === 'SELECT' || e.target.type === 'checkbox') return;
    form[name] = e.target.type === 'number' ? Number(e.target.value) : e.target.value;
    dirty(); renderResults();
  });
  root.addEventListener('change', e => {
    const name = e.target.name;
    if (!name) return;
    if (e.target.type === 'checkbox') { form[name] = e.target.checked; dirty(true); render(); return; }
    if (name === 'layer' && Number.isInteger(form.layer) && form.layer >= 0 && form.layer < arch.n_layers) {
      const heads = arch.layer_types[form.layer] === 'full_attention' ? arch.attn_heads : arch.linear_v_heads;
      if (form.head >= heads) form.head = 0;
      render(); return;
    }
    if (e.target.tagName !== 'SELECT') return;
    form[name] = name.endsWith('_pos') ? Number(e.target.value) : e.target.value;
    if (name === 'source' || name === 'reference') form[`${name}_pos`] = -1;
    if (name === 'method') {
      form.strength = form.method === 'add' ? .1 : 1;
      if (form.method === 'remove') form.stage = 'post';
    }
    if (form.method === 'remove' && (form.stage === 'pre' || ['neuron', 'mlp'].includes(form.component))) form.stage = 'post';
    dirty(); render();
  });
  root.addEventListener('click', e => {
    const b = e.target.closest('button'); if (!b) return;
    if (b.dataset.analyze) { setActive(b.dataset.analyze); return; }
    if (b.dataset.neuron) { onNeuron(Number(b.dataset.layer), Number(b.dataset.neuron)); return; }
    if (b.dataset.action === 'stop') { stopped = true; ctl?.abort(); return; }
    if (b.dataset.action === 'prepare') prepare().catch(showError);
    if (b.dataset.action === 'selected') {
      const selected = getSelected();
      if (!selected) { error = t('lab.selectNeuron'); renderResults(); return; }
      form.layer = selected.layer; form.neuron = selected.n; dirty(); render();
    }
    if (b.dataset.action === 'export' && runSnapshot) {
      const blob = new Blob([JSON.stringify({ ...runSnapshot, results, stopped }, null, 2)], { type: 'application/json' });
      const url = URL.createObjectURL(blob), a = document.createElement('a');
      a.href = url; a.download = 'residual-experiment.json'; a.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
    }
  });
  function showError(err) { error = errorText(err); renderResults(); }
  function inputSnapshot() {
    if (texts.some(row => !row.text.trim())) throw new Error(t('lab.fillInputs'));
    return texts.map(row => ({ ...row }));
  }
  async function prepare() {
    const rows = inputSnapshot(), mine = ++tokenVersion;
    preparing = true; error = null; render();
    try {
      const out = await api('/api/residual/tokens', { texts: rows, chat: form.chat });
      if (mine !== tokenVersion) throw new Error(t('lab.inputsChanged'));
      tokens = new Map(out.texts.map(row => [row.id, row])); tokenChat = form.chat;
      for (const row of out.texts) {
        if (form.positions[row.id] == null || form.positions[row.id] >= row.count) form.positions[row.id] = row.count - 1;
      }
      for (const name of ['source', 'reference']) if (form[`${name}_pos`] >= (tokens.get(form[name])?.count || 0)) form[`${name}_pos`] = -1;
    } finally { preparing = false; render(); }
  }
  async function run() {
    if (busy || preparing) return;
    try {
      inputSnapshot();
      if (tokens.size !== texts.length || tokenChat !== form.chat) await prepare();
      const plan = { ...form, positions: { ...form.positions } };
      delete plan.chat; delete plan.targets; delete plan.max_new;
      if (['add', 'erase'].includes(form.method) && form.direction === 'custom') plan.vector = parseResidualVector(form.vector, arch.hidden);
      else delete plan.vector;
      if (['patch', 'erase'].includes(form.method) && !(form.strength >= 0 && form.strength <= 1)) throw new Error(t('lab.fractionError'));
      if (form.method !== 'patch' && form.method !== 'remove' && form.direction === 'difference' && form.source === form.reference) throw new Error(t('lab.distinctSources'));
      const payload = { texts: inputSnapshot(), chat: form.chat, plan, max_new: form.max_new,
        targets: form.targets.split('\n').filter(s => s.trim()) };
      runSnapshot = { request: structuredClone(payload), created_at: new Date().toISOString() };
      results = []; progress = null; stale = false; stopped = false; error = null; busy = true;
      ctl = new AbortController(); render();
      const response = await fetch('/api/residual/experiment', { method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload), signal: ctl.signal });
      await readResidualEvents(response, event => {
        if (event.type === 'start') runSnapshot.context = event;
        if (event.type === 'progress') progress = event;
        if (event.type === 'result') results.push(event);
        if (event.type === 'error') throw new Error(event.error);
        renderResults();
      });
    } catch (err) { if (err.name !== 'AbortError') error = errorText(err); }
    finally { busy = false; ctl = null; progress = null; render(); }
  }
  function renderResults() {
    const target = root.querySelector('#rl-results'); if (!target) return;
    root.querySelector('#rl-error').textContent = error || '';
    const row = progress && texts.find(r => r.id === progress.id);
    root.querySelector('#rl-progress').textContent = preparing ? t('lab.preparing') : busy
      ? progress ? t('lab.progress', { text: row ? label(row) : progress.id, phase: t(`lab.${progress.phase}`), done: results.length, total: texts.length }) : t('lab.starting')
      : runSnapshot ? t(stopped ? 'lab.stopped' : error ? 'lab.failed' : 'lab.completed', { n: results.length }) : '';
    if (!runSnapshot) { target.innerHTML = `<p class="dim rl-hint">${t('lab.noResults')}</p>`; return; }
    const p = runSnapshot.request.plan;
    const route = [t(`lab.${p.method}`), `L${p.layer} ${p.stage}`, t(`lab.scope${{ token: 'Token', last: 'Last', prompt: 'Prompt', continuation: 'Continuation', all: 'All' }[p.scope]}`), `α=${p.strength}`];
    const inputName = id => { const row = runSnapshot.request.texts.find(r => r.id === id); return row ? t('lab.text', { name: row.label }) : id; };
    const posName = pos => pos === -1 ? t('lab.lastToken') : `#${pos}`;
    if (p.method === 'remove') route.push(t(`lab.component${{ neuron: 'Neuron', mlp: 'MLP', attention: 'Attention', head: 'Head' }[p.component]}`));
    if ((p.method === 'remove' && p.component === 'neuron') || (['add', 'erase'].includes(p.method) && p.direction === 'neuron')) route.push(`N${p.neuron}`);
    if (p.method === 'remove' && p.component === 'head') route.push(`h${p.head}`);
    if (p.method === 'patch' || (['add', 'erase'].includes(p.method) && p.direction === 'difference')) route.push(`${inputName(p.source)} ${posName(p.source_pos)}${p.method === 'patch' ? ' →' : ` − ${inputName(p.reference)} ${posName(p.reference_pos)}`}`);
    const ctx = runSnapshot.context;
    target.innerHTML = `<div class="rl-snapshot"><strong>${t('lab.runSnapshot')}</strong><span>${esc(route.join(' · '))}</span><small>${esc(new Date(runSnapshot.created_at).toLocaleString())}${ctx?.batch ? ` · ${esc(t('batch.chatBase', { name: ctx.batch.name }))}` : ''}</small></div>
      ${stale ? `<p class="rl-stale">${t('lab.stale')}</p>` : ''}
      ${results.length ? `<div class="rl-summary-wrap"><table class="rl-summary"><thead><tr><th>${t('lab.inputs')}</th><th>${t('lab.kl')}</th><th>${t('lab.topBase')}</th><th>${t('lab.topAlt')}</th></tr></thead><tbody>${results.map(r => `<tr><td><a href="#rl-result-${esc(r.id)}">${esc(t('lab.text', { name: r.label }))}</a></td><td>${r.kl.toPrecision(4)}</td><td>${esc(r.baseline.top[0]?.token || '')}</td><td>${esc(r.intervened.top[0]?.token || '')}</td></tr>`).join('')}</tbody></table></div>` : ''}
      ${results.map(resultHTML).join('')}`;
  }
  const percent = n => `${(n * 100).toPrecision(4)}%`;
  function resultHTML(r) {
    const variant = (v, key) => `<div class="rl-variant"><strong>${t(key)}</strong><div class="rl-output">${esc(v.output || t('lab.noGeneration'))}</div><div class="rl-top">${v.top.map(x => `<span>${esc(x.token)} <small>${percent(x.prob)}</small></span>`).join('')}</div></div>`;
    return `<article class="rl-result" id="rl-result-${esc(r.id)}"><div class="rl-section-heading"><strong>${esc(t('lab.text', { name: r.label }))}</strong><span class="mono">KL ${r.kl.toPrecision(4)}</span></div>
      <p class="rl-prompt">${esc(r.text)}</p><p class="dim rl-hint">${t('lab.measurement', { pos: r.position, count: r.input_tokens })}</p>
      <div class="rl-pair">${variant(r.baseline, 'lab.baseline')}${variant(r.intervened, 'lab.intervened')}</div>
      ${r.baseline.targets.length ? `<div class="rl-summary-wrap"><table class="rl-summary"><thead><tr><th>${t('lab.target')}</th><th>${t('lab.baseLogprob')}</th><th>${t('lab.altLogprob')}</th><th>Δ log p</th></tr></thead><tbody>${r.baseline.targets.map((x, i) => `<tr><td>${esc(x.text)}</td><td>${x.logprob.toFixed(4)}</td><td>${r.intervened.targets[i].logprob.toFixed(4)}</td><td>${(r.intervened.targets[i].logprob - x.logprob).toFixed(4)}</td></tr>`).join('')}</tbody></table></div>` : ''}
      <details><summary>${t('lab.distribution')}</summary><table class="rl-summary"><thead><tr><th>Token</th><th>${t('lab.baseline')}</th><th>${t('lab.intervened')}</th><th>Δ p</th></tr></thead><tbody>${r.changes.map(x => `<tr><td>${esc(x.token)}</td><td>${percent(x.baseline)}</td><td>${percent(x.intervened)}</td><td class="${x.delta < 0 ? 'res-negative' : 'res-positive'}">${x.delta >= 0 ? '+' : ''}${percent(x.delta)}</td></tr>`).join('')}</tbody></table></details>
      <details><summary>${t('lab.downstream')}</summary><p class="dim rl-hint">${t('lab.downstreamHint')}</p><table class="rl-summary"><thead><tr><th>${t('lab.componentNeuron')}</th><th>${t('lab.baseline')}</th><th>${t('lab.intervened')}</th><th>Δ</th></tr></thead><tbody>${r.neurons.map(x => `<tr><td><button type="button" data-neuron="${x.n}" data-layer="${x.layer}">L${x.layer} N${x.n}</button></td><td>${x.baseline.toPrecision(4)}</td><td>${x.intervened.toPrecision(4)}</td><td>${x.delta.toPrecision(4)}</td></tr>`).join('')}</tbody></table>
      <table class="rl-summary"><thead><tr><th>L</th><th>${t('lab.baseNorm')}</th><th>${t('lab.altNorm')}</th></tr></thead><tbody>${r.layers.map(x => `<tr><td>${x.layer}</td><td>${x.baseline.toPrecision(5)}</td><td>${x.intervened.toPrecision(5)}</td></tr>`).join('')}</tbody></table></details></article>`;
  }
  render();
  return { render, setActive,
    configure({ layer, n, stage, pos }) {
      if (busy) return;
      Object.assign(form, { layer, neuron: n, stage, scope: 'token', method: 'add', direction: 'neuron', strength: .1 });
      form.positions[active] = pos; dirty(); render();
    },
    externalText(text) { if (!busy && current().text !== text) { current().text = text; $('#inf-text').value = text; dirty(true); save(); render(); } },
    invalidate() { ctl?.abort(); stopped = busy; dirty(true); render(); },
    opened() { render(); },
  };
}
