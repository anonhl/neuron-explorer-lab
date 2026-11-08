import { t, getLang, localizeError, applyTranslations } from "./i18n.js?v=27";

export function initNeuronBatches({ getSelected, onApplied }) {
  const $ = id => document.getElementById(id);
  const dialog = $("batch-dialog");
  const history = $("batch-history"), menu = $("batch-menu");
  let info = null, active = null, busy = false;
  let sourceBatch = null;
  const esc = s => String(s ?? "").replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  const format = (id, dtype) => {
    const precision = dtype === "float16" ? "FP16" : "BF16";
    return id === "none" ? precision : `${id} · ${precision}`;
  };
  async function request(path, body, method = body === undefined ? "GET" : "POST") {
    const options = body === undefined ? { method } : { method, headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) };
    const response = await fetch(path, options);
    const data = await response.json();
    if (!response.ok) {
      if (response.status === 422) throw new Error(t("batch.invalid"));
      throw new Error(data.error || response.statusText);
    }
    return data;
  }
  function activeLabel() {
    $("batch-settings").textContent = active ? t("batch.headerActive", { name: active.name }) : t("batch.open");
    $("batch-settings").classList.toggle("on", !!active);
    $("batch-active").textContent = active ? t("batch.active", { name: active.name }) : t("batch.inactive");
    $("batch-stop").hidden = !active;
  }
  function matches(batch) {
    return batch.model === info.model && batch.quantization === info.quantization && batch.model_revision === info.model_revision
      && (batch.compute_dtype || "bfloat16") === info.compute_dtype;
  }
  function renderHistory() {
    if (!info?.groups.length) { history.innerHTML = `<p class="dim">${t("batch.noGroups")}</p>`; return; }
    history.innerHTML = info.groups.map(group => {
      const batches = info.batches.filter(b => b.group_id === group.id);
      return `<section class="batch-group-section"><h4>${esc(group.name)} <span class="dim">${batches.length}</span></h4>${batches.length ? batches.map(batch => {
        const used = active?.id === batch.id, compatible = matches(batch);
        const when = new Date(batch.created_at).toLocaleString(getLang() === "zh-CN" ? "zh-CN" : "en-US", { timeZoneName: "short" });
        const total = batch.operations.filter(op => op.enabled).length;
        const lines = batch.operations.map(op => `${op.enabled ? "✓" : "○"} L${op.layer} N${op.n} · ${op.mode === "mul" ? "×" : "+"} ${op.dose}`).join("\n");
        return `<article class="batch-card${used ? " on" : ""}" data-batch-id="${esc(batch.id)}" tabindex="0"><div class="batch-card-heading"><strong>${esc(batch.name)}</strong><div class="batch-card-actions"><button type="button" data-copy="${esc(batch.id)}">${t("batch.copy")}</button><button type="button" class="batch-use" data-use="${esc(batch.id)}" ${!compatible || !info.live_model || used ? "disabled" : ""}>${t(used ? "batch.using" : "batch.use")}</button></div></div>
          <p class="dim mono">${esc(batch.model)} · ${esc(format(batch.quantization, batch.compute_dtype))}${batch.model_revision ? ` · ${esc(batch.model_revision.slice(0, 12))}` : ""}</p>
          <p class="dim">${esc(when)} · ${t("batch.operationCount", { n: total })}</p>${batch.notes ? `<p>${esc(batch.notes)}</p>` : ""}
          ${!compatible ? `<p class="batch-mismatch">${t("batch.mismatch")}</p>` : ""}<details><summary>${t("batch.details")}</summary><pre>${esc(lines)}</pre></details></article>`;
      }).join("") : `<p class="dim">${t("batch.noBatches")}</p>`}</section>`;
    }).join("");
  }
  function renderInfo() {
    activeLabel();
    if (!info) return;
    $("batch-model").textContent = `${info.model} · ${format(info.quantization, info.compute_dtype)}`;
    const selected = $("batch-group").value;
    $("batch-group").replaceChildren(new Option(t("batch.chooseGroup"), ""));
    for (const group of info.groups) $("batch-group").add(new Option(group.name, group.id));
    $("batch-group").value = info.groups.some(g => g.id === selected) ? selected : "";
    renderHistory();
    setBusy(busy);
  }
  function setBusy(value) {
    busy = value;
    for (const el of dialog.querySelectorAll("button, input, select")) {
      if (el.id === "batch-close") continue;
      el.disabled = value || !info;
    }
    if (!value && info) {
      for (const button of dialog.querySelectorAll("[data-use]")) {
        const batch = info.batches.find(b => b.id === button.dataset.use);
        button.disabled = !matches(batch) || !info.live_model || active?.id === batch.id;
      }
      $("batch-add-current").disabled = !getSelected();
    }
    syncSave();
  }
  async function action(work) {
    if (busy) return;
    $("batch-error").textContent = "";
    $("batch-message").textContent = "";
    setBusy(true);
    try { await work(); }
    catch (error) { $("batch-error").textContent = localizeError(error); }
    finally { setBusy(false); }
  }
  async function refresh() {
    info = await request("/api/neuron-batches");
    active = info.active_batch;
    if (sourceBatch) sourceBatch = info.batches.find(batch => batch.id === sourceBatch.id) ?? null;
    renderInfo();
  }
  function addOperation(op = {}) {
    const row = document.createElement("tr");
    row.innerHTML = `<td><input type="checkbox" data-field="enabled" ${op.enabled !== false ? "checked" : ""} data-i18n-aria-label="batch.enabled"></td>
      <td><input type="number" data-field="layer" required min="0" max="${(info?.n_layers || 1) - 1}" step="1" value="${esc(op.layer ?? 0)}" data-i18n-aria-label="batch.layer"></td>
      <td><input type="number" data-field="n" required min="0" max="${(info?.intermediate_size || 1) - 1}" step="1" value="${esc(op.n ?? 0)}" data-i18n-aria-label="batch.neuron"></td>
      <td><select data-field="mode" data-i18n-aria-label="batch.mode"><option value="add" data-i18n="batch.modeAdd"></option><option value="mul" data-i18n="batch.modeMul"></option></select></td>
      <td><input type="number" data-field="dose" required step="any" value="${esc(op.dose ?? 0.8)}" data-i18n-aria-label="batch.dose"></td>
      <td><button type="button" data-remove data-i18n-aria-label="batch.remove">×</button></td>`;
    applyTranslations(row);
    row.querySelector("select").value = op.mode || "add";
    row.querySelector("[data-remove]").onclick = () => { row.remove(); syncSave(); };
    $("batch-operations").append(row);
  }
  function operations() {
    return [...$("batch-operations").rows].map(row => {
      const field = key => row.querySelector(`[data-field="${key}"]`);
      return { layer: Number(field("layer").value), n: Number(field("n").value), mode: field("mode").value,
        dose: Number(field("dose").value), enabled: field("enabled").checked };
    });
  }
  function syncSave() {
    const source = sourceBatch ?? active;
    let unchanged = false;
    if (info && source && matches(source)) {
      const edited = operations();
      unchanged = edited.length === source.operations.length && edited.every((op, i) => {
        const original = source.operations[i];
        return op.layer === original.layer && op.n === original.n && op.mode === original.mode
          && op.dose === original.dose && op.enabled === original.enabled;
      });
    }
    $("batch-save").disabled = busy || !info?.groups.length || unchanged;
    $("batch-unchanged").hidden = !unchanged;
  }
  function open(recipe) {
    dialog.showModal();
    action(async () => {
      await refresh();
      const table = $("batch-operations");
      if (recipe) {
        sourceBatch = null;
        $("batch-name").value = "";
        $("batch-notes").value = "";
        table.replaceChildren();
        for (const op of recipe) addOperation(op);
        dialog.scrollTop = 0;
      } else if (!table.rows.length) addOperation(getSelected() || {});
    });
  }
  $("batch-settings").onclick = () => open();
  $("batch-close").onclick = () => dialog.close();
  $("batch-group-form").onsubmit = event => {
    event.preventDefault();
    action(async () => {
      const group = await request("/api/neuron-batches/groups", { name: $("batch-group-name").value });
      $("batch-group-name").value = "";
      await refresh();
      $("batch-group").value = group.id;
    });
  };
  $("batch-operations").addEventListener("input", syncSave);
  $("batch-add").onclick = () => { addOperation(); syncSave(); };
  $("batch-add-current").onclick = () => {
    const selected = getSelected();
    if (selected) { addOperation(selected); syncSave(); }
  };
  $("batch-form").onsubmit = event => {
    event.preventDefault();
    if ($("batch-save").disabled) return;
    action(async () => {
      const batch = await request("/api/neuron-batches", { name: $("batch-name").value, notes: $("batch-notes").value,
        group_id: $("batch-group").value, model: info.model, model_revision: info.model_revision,
        quantization: info.quantization, compute_dtype: info.compute_dtype, operations: operations(),
        source_batch_id: sourceBatch?.id });
      sourceBatch = batch;
      await refresh();
      $("batch-message").textContent = t("batch.saved", { name: batch.name });
    });
  };
  history.onclick = event => {
    const button = event.target.closest("[data-use], [data-copy]");
    if (!button || button.disabled) return;
    if (button.dataset.copy) {
      const batch = info.batches.find(b => b.id === button.dataset.copy);
      sourceBatch = batch;
      $("batch-group").value = batch.group_id;
      $("batch-name").value = t("batch.copyName", { name: batch.name });
      $("batch-notes").value = batch.notes;
      $("batch-operations").replaceChildren();
      for (const op of batch.operations) addOperation(op);
      syncSave();
      $("batch-edit-title").scrollIntoView({ behavior: "smooth", block: "start" });
      return;
    }
    action(async () => {
      const result = await request(`/api/neuron-batches/${encodeURIComponent(button.dataset.use)}/use`, {});
      active = result.active_batch;
      renderInfo();
      onApplied(active);
      $("batch-message").textContent = t("batch.applied", { name: active.name });
    });
  };
  function closeMenu(restoreFocus = false) {
    if (menu.hidden) return;
    menu.hidden = true;
    if (restoreFocus) history.querySelector(`[data-batch-id="${CSS.escape(menu.dataset.targetId)}"]`)?.focus({ preventScroll: true });
  }
  history.addEventListener("contextmenu", event => {
    const card = event.target.closest("[data-batch-id]");
    if (!card) return;
    event.preventDefault();
    if (busy) return;
    menu.dataset.targetId = card.dataset.batchId;
    menu.hidden = false;
    const rect = card.getBoundingClientRect();
    const x = event.clientX || rect.left, y = event.clientY || rect.top;
    menu.style.left = `${Math.max(4, Math.min(x, window.innerWidth - menu.offsetWidth - 4))}px`;
    menu.style.top = `${Math.max(4, Math.min(y, window.innerHeight - menu.offsetHeight - 4))}px`;
    menu.querySelector("button").focus({ preventScroll: true });
  });
  dialog.addEventListener("pointerdown", event => { if (!menu.contains(event.target)) closeMenu(); });
  dialog.addEventListener("wheel", () => closeMenu(), { passive: true });
  dialog.addEventListener("close", () => closeMenu());
  dialog.addEventListener("keydown", event => {
    if (event.key !== "Escape" || menu.hidden) return;
    event.preventDefault();
    closeMenu(true);
  });
  menu.onclick = event => {
    const command = event.target.closest("[data-batch-action]")?.dataset.batchAction;
    if (!command || busy) return;
    const batch = info.batches.find(saved => saved.id === menu.dataset.targetId);
    closeMenu(true);
    if (!batch) return;
    const renaming = command === "rename";
    let name;
    if (renaming) {
      name = window.prompt(t("batch.renamePrompt"), batch.name);
      if (name === null || name.trim() === batch.name) return;
    } else {
      const message = active?.id === batch.id ? "batch.deleteActiveConfirm" : "batch.deleteConfirm";
      if (!window.confirm(t(message, { name: batch.name }))) return;
    }
    action(async () => {
      const path = `/api/neuron-batches/${encodeURIComponent(batch.id)}`;
      const result = renaming ? await request(path, { name }, "PATCH") : await request(path, undefined, "DELETE");
      if (renaming) info.batches = info.batches.map(saved => saved.id === batch.id ? result.batch : saved);
      else info.batches = info.batches.filter(saved => saved.id !== batch.id);
      if (sourceBatch?.id === batch.id) sourceBatch = renaming ? result.batch : null;
      const previousActiveId = active?.id;
      active = result.active_batch;
      renderInfo();
      if (previousActiveId !== active?.id) onApplied(active);
      $("batch-message").textContent = t(renaming ? "batch.renamed" : "batch.deleted", { name: result.batch.name });
    });
  };
  $("batch-stop").onclick = () => action(async () => {
    await request("/api/neuron-batches/stop", {});
    active = null;
    renderInfo();
    onApplied(null);
    $("batch-message").textContent = t("batch.stopped");
  });
  document.addEventListener("i18nchange", renderInfo);
  setBusy(false);
  return { getActive: () => active, openRecipe: open,
    syncActive(batch) {
      const changed = active?.id !== batch?.id || active?.applied_at !== batch?.applied_at;
      const renamed = !changed && active?.name !== batch?.name;
      active = batch;
      if (renamed && info) {
        const saved = info.batches.find(item => item.id === batch.id);
        if (saved) saved.name = batch.name;
      }
      activeLabel();
      if (dialog.open) {
        if (changed || renamed) renderHistory();
        setBusy(busy);
      }
      if (changed) onApplied(active);
    } };
}
