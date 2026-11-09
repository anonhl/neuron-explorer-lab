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

export function initResidualLab(root, { api, t, esc, arch, errorText, onActivate, onOpen, getSelected, onNeuron, onSelectionChange = () => {} }) {
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
  let mode = 'joint', resultGroup = 'all';
  let form = { method: 'add', layer: 0, stage: 'post', scope: 'last', strength: 0.1, direction: 'neuron',
    neuron: 0, component: 'neuron', head: 0, source: texts[0].id, reference: texts[1]?.id || '',
    source_pos: -1, reference_pos: -1, positions: {}, chat_position: 0, vector: '', chat: false, max_new: 48, targets: '' };
  let configurations = [], activeConfiguration = -1, held = [], holdVersion = 0, heldStale = false;
  const selectedConfigurations = () => configurations.filter(c => c.selected);
  const selection = () => selectedConfigurations().map(c => ({ layer: c.layer, n: c.neuron }));
  const notifySelection = () => onSelectionChange(selection());
  const notifyHeld = () => document.dispatchEvent(new CustomEvent('residualholdchange', { detail: { count: held.length } }));
  function configurationPlan(config) {
    const plan = { ...config, positions: { ...config.positions } };
    for (const key of ['chat', 'targets', 'max_new', 'selected', 'chat_position']) delete plan[key];
    if (plan.scope !== 'token') plan.positions = {};
    if (['add', 'erase'].includes(plan.method) && plan.direction === 'custom') plan.vector = parseResidualVector(config.vector, arch.hidden);
    else delete plan.vector;
    return plan;
  }
  function configurationPayload(config) {
    const plan = configurationPlan(config);
    if (plan.scope === 'token') plan.chat_position = config.chat_position;
    const rows = texts.filter(row => row.text.trim()).map(row => ({ ...row }));
    const needsSource = plan.method === 'patch' || (['add', 'erase'].includes(plan.method) && plan.direction === 'difference');
    if (needsSource && !rows.length) throw new Error(t('lab.fillInputs'));
    if (!rows.length) rows.push({ id: active, text: 'Chat', label: 'A' });
    if (plan.scope === 'token') plan.positions = Object.fromEntries(rows.map(row => [row.id, config.chat_position]));
    return { texts: rows, chat: config.chat, plan, max_new: 0, targets: [] };
  }
  const label = row => t('lab.text', { name: row.label });
  const save = () => { try { localStorage.setItem(storeKey, JSON.stringify({ texts, serial })); } catch {} };
  const current = () => texts.find(r => r.id === active);
  const option = (value, title, chosen) => `<option value="${esc(value)}"${String(value) === String(chosen) ? ' selected' : ''}>${esc(title)}</option>`;
  const choices = (name, values) => `<select name="${name}">${values.map(([v, key]) => option(v, t(key), form[name])).join('')}</select>`;
  const field = (key, content, extra = '') => `<label ${extra}><span>${t(key)}</span>${content}</label>`;
  const number = (name, min, max, step = 1) => `<input name="${name}" type="number" value="${form[name]}" min="${min}" max="${max}" step="${step}" required>`;
  function dirty(invalidateTokens = false) {
    if (runSnapshot) stale = true;
    if (held.length) heldStale = true;
    if (invalidateTokens) { tokens.clear(); ++tokenVersion; }
    error = null;
  }
  function addNeurons(addresses) {
    const additions = addresses.filter((a, i) => !configurations.some(c => c.layer === a.layer && c.neuron === a.n)
      && addresses.findIndex(b => b.layer === a.layer && b.n === a.n) === i);
    if (configurations.length + additions.length > 32) throw new Error(t('lab.invalidIds'));
    for (const a of additions) configurations.push({ ...structuredClone(form), layer: a.layer, neuron: a.n, selected: true });
    for (const a of addresses) {
      const index = configurations.findIndex(c => c.layer === a.layer && c.neuron === a.n);
      configurations[index].selected = true;
      activeConfiguration = index;
    }
    form = configurations[activeConfiguration]; dirty(); notifySelection(); render();
  }
  function renderConfigurations() {
    root.querySelector('#rl-configurations').innerHTML = configurations.map((c, i) => `<div class="rl-config-row${i === activeConfiguration ? ' on' : ''}">
      <input type="checkbox" data-enable="${i}" aria-label="${esc(t('lab.selectConfiguration', { layer: c.layer, n: c.neuron }))}" ${c.selected ? 'checked' : ''} ${busy || preparing ? 'disabled' : ''}>
      <button type="button" data-config="${i}" aria-pressed="${i === activeConfiguration}" ${busy || preparing ? 'disabled' : ''}>${i + 1}. L${c.layer} N${c.neuron} · ${t(`lab.${c.method}`)} · α=${c.strength} · ${c.stage}</button></div>`).join('');
    root.querySelector('#rl-selected-neurons').textContent = t('lab.selectedNeurons', { selected: selection().length, total: configurations.length })
      + (selection().length ? ` ${selection().map(c => `L${c.layer} N${c.n}`).join(', ')}` : '');
    root.querySelector('#rl-editing').textContent = activeConfiguration < 0 ? t('lab.noConfiguration') : t('lab.editingNeuron', { layer: form.layer, n: form.neuron });
    root.querySelector('#rl-held-edited').textContent = heldStale ? t('lab.heldEdited') : '';
  }
  function setActive(id, analyze = true) {
    if (busy || !texts.some(r => r.id === id)) return;
    active = id; $('#inf-text').value = current().text;
    renderInputs();
    if (analyze) onActivate(current().text);
  }
  function renderInputs() {
    // Labels follow the current input order; IDs remain stable for cached results.
    texts.forEach((row, index) => { row.label = String.fromCharCode(65 + index); });
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
    const row = { id: `text-${serial}`, label: String.fromCharCode(65 + texts.length), text: '' };
    texts.push(row);
    if (!form.reference) form.reference = row.id;
    dirty(true); setActive(row.id); save(); render(); $('#inf-text').focus();
  };
  $('#inf-remove-text').onclick = () => {
    if (busy || texts.length === 1) return;
    texts = texts.filter(r => r.id !== active);
    for (const config of [form, ...configurations.filter(c => c !== form)]) {
      delete config.positions[active];
      if (!texts.some(r => r.id === config.source)) { config.source = texts[0].id; config.source_pos = -1; }
      if (!texts.some(r => r.id === config.reference)) { config.reference = texts.find(r => r.id !== config.source)?.id || ''; config.reference_pos = -1; }
    }
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
  function experimentGroups(request) {
    return request.mode === 'independent'
      ? request.plans.map((plan, i) => ({ id: `setting-${i + 1}`, plans: [plan] }))
      : [{ id: 'joint', plans: request.plans }];
  }
  const groupTitle = group => group.id === 'joint' ? t('lab.modeJoint') : `L${group.plans[0].layer} N${group.plans[0].neuron}`;
  const resultId = r => `rl-result-${r.group_id || 'joint'}-${r.id}`;
  function render() {
    renderInputs();
    const isRemove = form.method === 'remove', isPatch = form.method === 'patch';
    const usesNeuron = (isRemove && form.component === 'neuron') || (!isRemove && !isPatch && form.direction === 'neuron');
    const usesSource = isPatch || (!isRemove && form.direction === 'difference');
    root.innerHTML = `<p class="lab-intro">${t('lab.intro')}</p>
      <div class="rl-section-heading"><strong>${t('lab.configurations')}</strong><button type="button" data-action="selected" ${busy || preparing ? 'disabled' : ''}>${t('lab.useSelected')}</button></div>
      <p class="dim rl-hint">${t('lab.selectionHint')}</p><p id="rl-selected-neurons" class="mono rl-hint" role="status"></p>
      <div id="rl-configurations" class="rl-input-list"></div>
      <div class="rl-add-neurons"><label for="rl-neuron-address">${t('lab.addById')}</label><input id="rl-neuron-address" placeholder="L22 N2742, L23 N100" ${busy || preparing ? 'disabled' : ''}><button type="button" data-action="add-address" ${busy || preparing ? 'disabled' : ''}>${t('lab.addConfiguration')}</button></div>
      <div class="rl-run-bar"><button type="button" data-action="select-all" ${busy || preparing ? 'disabled' : ''}>${t('lab.selectAll')}</button><button type="button" data-action="select-none" ${busy || preparing ? 'disabled' : ''}>${t('lab.selectNone')}</button><button type="button" data-action="remove-config" ${!configurations.length || busy || preparing ? 'disabled' : ''}>${t('lab.removeConfiguration')}</button></div>
      <div class="rl-run-bar"><button type="button" data-action="hold" aria-pressed="${!!held.length}" ${busy || preparing || (!held.length && !selectedConfigurations().length) ? 'disabled' : ''}>${t(held.length ? 'lab.release' : 'lab.hold')}</button><span role="status">${held.length ? t('lab.holding', { n: held.length }) : t('lab.notHolding')}</span></div>
      <p id="rl-held-edited" class="rl-stale" role="status"></p>
      <p class="dim rl-hint">${t('lab.holdHint')}</p>
      <div class="rl-section-heading"><strong>${t('lab.experimentMode')}</strong></div>
      <div class="rl-mode-picker" role="group" aria-label="${esc(t('lab.experimentMode'))}">${['joint', 'independent'].map(value => `<button type="button" data-mode="${value}" aria-pressed="${mode === value}" ${busy || preparing ? 'disabled' : ''}>${t(value === 'joint' ? 'lab.modeJoint' : 'lab.modeIndependent')}</button>`).join('')}</div>
      <p class="dim rl-hint">${t(mode === 'joint' ? 'lab.modeJointHint' : 'lab.modeIndependentHint')}</p>
      <form id="rl-form"><fieldset ${busy || preparing || !configurations.length ? 'disabled' : ''}>
      <div class="rl-section-heading"><strong>${t('lab.inputs')}</strong><button type="button" data-action="prepare">${t('lab.prepare')}</button></div>
      <p class="dim rl-hint">${t('lab.inputHint')}</p>
      <label class="rl-check"><input type="checkbox" name="chat" ${form.chat ? 'checked' : ''}>${t('lab.chat')}</label>
      <div class="rl-input-list">${texts.map(row => {
        const tok = tokens.get(row.id), pos = form.positions[row.id] ?? (tok ? tok.count - 1 : '');
        return `<div class="rl-input-row"><button type="button" data-analyze="${esc(row.id)}">${esc(label(row))} ↗</button><span>${esc(row.text.slice(0, 100) || t('lab.emptyInput'))}</span><small>${tok ? t('lab.tokenCount', { n: tok.count }) : t('lab.notPrepared')}</small>${form.scope === 'token' ? field('lab.targetToken', `<select data-target="${esc(row.id)}">${tokenOptions(row.id, pos, false)}</select>`) : ''}</div>`;
      }).join('')}</div>
      <div class="rl-section-heading"><strong id="rl-editing"></strong></div>
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
      ${form.scope === 'token' ? field('lab.chatPosition', number('chat_position', 0, 100000)) : ''}
      ${field('lab.maxNew', number('max_new', 0, 128))}
      ${field('lab.targets', `<textarea name="targets" rows="2" placeholder="${esc(t('lab.targetsPlaceholder'))}">${esc(form.targets)}</textarea>`, 'class="rl-wide"')}
      </div><p class="rl-method-note">${t(`lab.hint.${form.method}`)}</p>
      ${usesSource ? `<p class="dim rl-hint">${t('lab.alignmentHint')}</p>` : ''}
      <p class="dim rl-hint">${t('lab.scopeHint')}</p>
      <button class="rl-primary" type="submit" ${!selectedConfigurations().length ? 'disabled' : ''}>${t(mode === 'joint' ? 'lab.runSelected' : 'lab.runIndependent', { n: texts.length, settings: selectedConfigurations().length, total: texts.length * selectedConfigurations().length })}</button>
      </fieldset></form>
      <div class="rl-run-bar"><span role="status" id="rl-progress"></span>${busy ? `<button type="button" data-action="stop">${t('chat.stop')}</button>` : ''}<button type="button" data-action="export" ${runSnapshot ? '' : 'disabled'}>${t('lab.export')}</button></div>
      <div id="rl-error" role="alert"></div><div id="rl-results"></div>`;
    root.querySelector('#rl-form').onsubmit = e => { e.preventDefault(); return run(); };
    renderConfigurations();
    renderResults();
  }
  root.addEventListener('input', e => {
    if (busy || preparing) return;
    if (e.target.dataset.target) { form.positions[e.target.dataset.target] = Number(e.target.value); dirty(); renderConfigurations(); renderResults(); return; }
    const name = e.target.name;
    if (!name || e.target.tagName === 'SELECT' || e.target.type === 'checkbox') return;
    form[name] = e.target.type === 'number' ? Number(e.target.value) : e.target.value;
    if (['max_new', 'targets'].includes(name)) configurations.forEach(c => { c[name] = form[name]; });
    dirty(); renderConfigurations(); notifySelection(); renderResults();
  });
  root.addEventListener('change', e => {
    if (e.target.id === 'rl-result-group') { resultGroup = e.target.value; renderResults(); return; }
    if (busy || preparing) return;
    if (e.target.dataset.enable !== undefined) {
      configurations[Number(e.target.dataset.enable)].selected = e.target.checked;
      dirty(); notifySelection(); render(); return;
    }
    const name = e.target.name;
    if (!name) return;
    if (e.target.type === 'checkbox') { configurations.forEach(c => { c[name] = e.target.checked; }); dirty(true); render(); return; }
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
  root.addEventListener('click', async e => {
    const b = e.target.closest('button'); if (!b || preparing) return;
    if (busy && !['stop', 'export'].includes(b.dataset.action)) return;
    if (b.dataset.mode) {
      if (mode === b.dataset.mode) return;
      mode = b.dataset.mode;
      if (runSnapshot) stale = true;
      error = null; render(); return;
    }
    if (b.dataset.config !== undefined && !busy) {
      activeConfiguration = Number(b.dataset.config); form = configurations[activeConfiguration]; render(); return;
    }
    if (b.dataset.action === 'hold') {
      if (held.length) { held = []; heldStale = false; notifyHeld(); render(); return; }
      try {
        if (!selectedConfigurations().length) throw new Error(t('lab.noSelection'));
        preparing = true; render();
        const payloads = structuredClone(selectedConfigurations().map(configurationPayload));
        const version = ++holdVersion;
        const checked = await api('/api/residual/validate-held', { residuals: payloads });
        if (version === holdVersion) { held = checked.residuals; heldStale = false; notifyHeld(); }
      } catch (err) { showError(err); }
      finally { preparing = false; render(); }
      return;
    }
    if (['select-all', 'select-none'].includes(b.dataset.action)) {
      configurations.forEach(c => { c.selected = b.dataset.action === 'select-all'; });
      dirty(); notifySelection(); render(); return;
    }
    if (b.dataset.action === 'remove-config' && configurations.length) {
      configurations.splice(activeConfiguration, 1); activeConfiguration = Math.min(activeConfiguration, configurations.length - 1);
      form = configurations[activeConfiguration] || { ...structuredClone(form), selected: false };
      dirty(); notifySelection(); render(); return;
    }
    if (b.dataset.action === 'add-address' && !busy) {
      const raw = root.querySelector('#rl-neuron-address').value.trim();
      const matches = [...raw.matchAll(/L(\d+)\s*(?:N|[/:])\s*(\d+)/gi)];
      const remainder = raw.replace(/L\d+\s*(?:N|[/:])\s*\d+/gi, '').replace(/[,，;；\s]+/g, '');
      if (!matches.length || remainder || matches.some(m => +m[1] >= arch.n_layers || +m[2] >= arch.inter)) {
        showError(new Error(t('lab.invalidIds'))); return;
      }
      try { addNeurons(matches.map(m => ({ layer: +m[1], n: +m[2] }))); } catch (err) { showError(err); }
      return;
    }
    if (b.dataset.analyze) { setActive(b.dataset.analyze); return; }
    if (b.dataset.neuron !== undefined) { onNeuron(Number(b.dataset.layer), Number(b.dataset.neuron)); return; }
    if (b.dataset.action === 'stop') { stopped = true; ctl?.abort(); return; }
    if (b.dataset.action === 'prepare') prepare().catch(showError);
    if (b.dataset.action === 'selected') {
      const selected = getSelected();
      if (!selected) { error = t('lab.selectNeuron'); renderResults(); return; }
      try { addNeurons([selected]); } catch (err) { showError(err); }
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
      for (const config of configurations) {
        for (const row of out.texts) {
          if (config.positions[row.id] == null || config.positions[row.id] >= row.count) config.positions[row.id] = row.count - 1;
        }
        for (const name of ['source', 'reference']) if (config[`${name}_pos`] >= (tokens.get(config[name])?.count || 0)) config[`${name}_pos`] = -1;
      }
    } finally { preparing = false; render(); }
  }
  async function run() {
    if (busy || preparing) return;
    try {
      if (!selectedConfigurations().length) throw new Error(t('lab.noSelection'));
      inputSnapshot();
      if (tokens.size !== texts.length || tokenChat !== form.chat) await prepare();
      const plans = selectedConfigurations().map(configurationPlan);
      for (const plan of plans) {
        if (['patch', 'erase'].includes(plan.method) && !(plan.strength >= 0 && plan.strength <= 1)) throw new Error(t('lab.fractionError'));
        if (['add', 'erase'].includes(plan.method) && plan.direction === 'difference' && plan.source === plan.reference) throw new Error(t('lab.distinctSources'));
      }
      const payload = { mode, texts: inputSnapshot(), chat: form.chat, plan: plans[0], plans, max_new: form.max_new,
        targets: form.targets.split('\n').filter(s => s.trim()) };
      runSnapshot = { request: structuredClone(payload), created_at: new Date().toISOString() };
      results = []; resultGroup = 'all'; progress = null; stale = false; stopped = false; error = null; busy = true;
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
    const groups = runSnapshot ? experimentGroups(runSnapshot.request) : [];
    const independent = runSnapshot?.request.mode === 'independent';
    const total = runSnapshot ? runSnapshot.request.texts.length * groups.length : 0;
    const row = progress && texts.find(r => r.id === progress.id);
    const currentGroup = progress && groups.find(group => group.id === progress.group_id);
    const progressName = `${independent && currentGroup ? `${groupTitle(currentGroup)} · ` : ''}${row ? label(row) : progress?.id}`;
    root.querySelector('#rl-progress').textContent = preparing ? t('lab.preparing') : busy
      ? progress ? t('lab.progress', { text: progressName, phase: t(`lab.${progress.phase}`), done: results.length, total }) : t('lab.starting')
      : runSnapshot ? t(stopped ? 'lab.stopped' : error ? 'lab.failed' : 'lab.completed', { n: results.length }) : '';
    if (!runSnapshot) { target.innerHTML = `<p class="dim rl-hint">${t('lab.noResults')}</p>`; return; }
    const describePlan = p => {
      const route = [t(`lab.${p.method}`), `L${p.layer} ${p.stage}`, t(`lab.scope${{ token: 'Token', last: 'Last', prompt: 'Prompt', continuation: 'Continuation', all: 'All' }[p.scope]}`), `α=${p.strength}`];
      const inputName = id => { const row = runSnapshot.request.texts.find(r => r.id === id); return row ? t('lab.text', { name: row.label }) : id; };
      const posName = pos => pos === -1 ? t('lab.lastToken') : `#${pos}`;
      if (p.method === 'remove') route.push(t(`lab.component${{ neuron: 'Neuron', mlp: 'MLP', attention: 'Attention', head: 'Head' }[p.component]}`));
      if ((p.method === 'remove' && p.component === 'neuron') || (['add', 'erase'].includes(p.method) && p.direction === 'neuron')) route.push(`N${p.neuron}`);
      if (p.method === 'remove' && p.component === 'head') route.push(`h${p.head}`);
      if (p.method === 'patch' || (['add', 'erase'].includes(p.method) && p.direction === 'difference')) route.push(`${inputName(p.source)} ${posName(p.source_pos)}${p.method === 'patch' ? ' →' : ` − ${inputName(p.reference)} ${posName(p.reference_pos)}`}`);
      return `<span>${esc(route.join(' · '))}</span>`;
    };
    const ctx = runSnapshot.context;
    const groupRows = group => results.filter(r => (r.group_id || 'joint') === group.id);
    const summaryHTML = rows => rows.length ? `<div class="rl-summary-wrap"><table class="rl-summary"><thead><tr><th>${t('lab.inputs')}</th><th>${t('lab.kl')}</th><th>${t('lab.topBase')}</th><th>${t('lab.topAlt')}</th></tr></thead><tbody>${rows.map(r => `<tr><td><a href="#${esc(resultId(r))}">${esc(t('lab.text', { name: r.label }))}</a></td><td>${r.kl.toPrecision(4)}</td><td>${esc(r.baseline.top[0]?.token || '')}</td><td>${esc(r.intervened.top[0]?.token || '')}</td></tr>`).join('')}</tbody></table></div>` : '';
    target.innerHTML = `<div class="rl-snapshot"><strong>${t('lab.runSnapshot')} · ${t(independent ? 'lab.modeIndependent' : 'lab.modeJoint')}</strong>${runSnapshot.request.plans.map(describePlan).join('')}<small>${esc(new Date(runSnapshot.created_at).toLocaleString())}${ctx?.batch ? ` · ${esc(t('batch.chatBase', { name: ctx.batch.name }))}` : ''}</small></div>
      ${stale ? `<p class="rl-stale">${t('lab.stale')}</p>` : ''}
      <p class="rl-hint" role="status">${t('lab.resultOverview', { groups: groups.length, texts: runSnapshot.request.texts.length, done: results.length, total })}</p>
      ${independent ? `<label class="rl-result-filter">${t('lab.viewResults')}<select id="rl-result-group">${option('all', t('lab.allNeuronResults'), resultGroup)}${groups.map(group => option(group.id, `${groupTitle(group)} · ${groupRows(group).length}/${runSnapshot.request.texts.length}`, resultGroup)).join('')}</select></label>` : ''}
      ${groups.filter(group => !independent || resultGroup === 'all' || group.id === resultGroup).map(group => {
        const rows = groupRows(group);
        return `<section class="rl-result-group">${independent ? `<div class="rl-section-heading"><strong>${esc(groupTitle(group))}</strong><span>${t('lab.groupProgress', { done: rows.length, total: runSnapshot.request.texts.length })}</span></div><div class="rl-group-plan">${group.plans.map(describePlan).join('')}</div>` : ''}${summaryHTML(rows)}${rows.map(resultHTML).join('')}${!rows.length ? `<p class="dim rl-hint">${t('lab.noCompletedResults')}</p>` : ''}</section>`;
      }).join('')}`;
  }
  const percent = n => `${(n * 100).toPrecision(4)}%`;
  function resultHTML(r) {
    const variant = (v, key) => `<div class="rl-variant"><strong>${t(key)}</strong><div class="rl-output">${esc(v.output || t('lab.noGeneration'))}</div><div class="rl-top">${v.top.map(x => `<span>${esc(x.token)} <small>${percent(x.prob)}</small></span>`).join('')}</div></div>`;
    return `<article class="rl-result" id="${esc(resultId(r))}"><div class="rl-section-heading"><strong>${esc(t('lab.text', { name: r.label }))}</strong><span class="mono">KL ${r.kl.toPrecision(4)}</span></div>
      <p class="rl-prompt">${esc(r.text)}</p><p class="dim rl-hint">${t('lab.measurement', { pos: r.position, count: r.input_tokens })}</p>
      <div class="rl-pair">${variant(r.baseline, 'lab.baseline')}${variant(r.intervened, 'lab.intervened')}</div>
      ${r.baseline.targets.length ? `<div class="rl-summary-wrap"><table class="rl-summary"><thead><tr><th>${t('lab.target')}</th><th>${t('lab.baseLogprob')}</th><th>${t('lab.altLogprob')}</th><th>Δ log p</th></tr></thead><tbody>${r.baseline.targets.map((x, i) => `<tr><td>${esc(x.text)}</td><td>${x.logprob.toFixed(4)}</td><td>${r.intervened.targets[i].logprob.toFixed(4)}</td><td>${(r.intervened.targets[i].logprob - x.logprob).toFixed(4)}</td></tr>`).join('')}</tbody></table></div>` : ''}
      <details><summary>${t('lab.distribution')}</summary><table class="rl-summary"><thead><tr><th>Token</th><th>${t('lab.baseline')}</th><th>${t('lab.intervened')}</th><th>Δ p</th></tr></thead><tbody>${r.changes.map(x => `<tr><td>${esc(x.token)}</td><td>${percent(x.baseline)}</td><td>${percent(x.intervened)}</td><td class="${x.delta < 0 ? 'res-negative' : 'res-positive'}">${x.delta >= 0 ? '+' : ''}${percent(x.delta)}</td></tr>`).join('')}</tbody></table></details>
      <details><summary>${t('lab.downstream')}</summary><p class="dim rl-hint">${t('lab.downstreamHint')}</p><table class="rl-summary"><thead><tr><th>${t('lab.componentNeuron')}</th><th>${t('lab.baseline')}</th><th>${t('lab.intervened')}</th><th>Δ</th></tr></thead><tbody>${r.neurons.map(x => `<tr><td><button type="button" data-neuron="${x.n}" data-layer="${x.layer}">L${x.layer} N${x.n}</button></td><td>${x.baseline.toPrecision(4)}</td><td>${x.intervened.toPrecision(4)}</td><td>${x.delta.toPrecision(4)}</td></tr>`).join('')}</tbody></table>
      <table class="rl-summary"><thead><tr><th>L</th><th>${t('lab.baseNorm')}</th><th>${t('lab.altNorm')}</th></tr></thead><tbody>${r.layers.map(x => `<tr><td>${x.layer}</td><td>${x.baseline.toPrecision(5)}</td><td>${x.intervened.toPrecision(5)}</td></tr>`).join('')}</tbody></table></details></article>`;
  }
  render();
  return { render, setActive,
    getHeld() { return structuredClone(held); },
    getSelection() { return selection(); },
    getInputs() { return texts.map(({ id, label, text }) => ({ id, label, text })); },
    getTexts() { return texts.map(row => row.text); },
    configure({ layer, n, stage, pos }) {
      if (busy || preparing) return;
      try {
        addNeurons([{ layer, n }]);
        Object.assign(form, { stage, scope: 'token', method: 'add', direction: 'neuron', strength: .1, chat_position: pos });
        form.positions[active] = pos; dirty(); render();
      } catch (err) { showError(err); }
    },
    externalText(text) { if (!busy && current().text !== text) { current().text = text; $('#inf-text').value = text; dirty(true); save(); render(); } },
    invalidate() { ++holdVersion; held = []; heldStale = false; notifyHeld(); ctl?.abort(); stopped = busy; dirty(true); render(); },
    opened() { render(); },
  };
}
