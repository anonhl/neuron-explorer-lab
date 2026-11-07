import { t, localizeError } from "./i18n.js?v=24";

const $ = id => document.getElementById(id);
const phaseLabels = {
  setup: "model.progress.setup",
  tokenizer: "model.progress.tokenizer",
  weights: "model.loading",
  lora: "model.progress.lora",
  finalizing: "model.progress.finalizing",
  lens: "model.progress.lens",
  vocab: "model.progress.vocab",
  ranges: "model.progress.ranges",
};

export function initModelStatus() {
  const notice = $("model-status"), error = $("model-status-error"), title = $("model-status-title");
  const stage = $("model-progress-stage"), count = $("model-progress-count"), bar = $("model-progress");
  const host = notice.parentElement;
  const dialogs = [$("model-dialog"), $("batch-dialog")];
  let status = null, pulse;

  // Keep the same notice visible in the page and in native modal dialogs.
  function placeNotice() {
    const dialog = dialogs.find(el => el.open);
    const target = dialog?.querySelector(".model-heading") || host;
    const parent = dialog ? target.parentElement : host;
    if (notice.parentElement === parent) return;
    if (dialog) target.after(notice);
    else host.prepend(notice);
  }
  const observer = new MutationObserver(placeNotice);
  for (const dialog of dialogs) observer.observe(dialog, { attributes: true, attributeFilter: ["open"] });

  function setText(el, text) {
    if (el.textContent !== text) el.textContent = text;
  }

  function render() {
    if (!status) return;
    const failed = status.status === "error";
    const progress = status.loading?.progress;
    const loading = !failed && (status.switching || Boolean(progress) || status.status === "loading");
    notice.hidden = !failed && !loading;
    error.hidden = !failed;
    $("model-progress-body").hidden = !loading;
    if (failed) {
      setText(title, t("model.failed"));
      setText($("model-status-detail"), status.error || "");
    }
    if (loading) {
      setText(stage, t(phaseLabels[progress?.phase] || "model.progress.setup"));
      if (progress?.total > 0 && Number.isFinite(progress.completed)) {
        bar.max = progress.total;
        bar.value = progress.completed;
        const percent = Math.floor(100 * progress.completed / progress.total);
        setText(count, t("model.progress.count", { percent, ...progress }));
      } else {
        bar.removeAttribute("value");
        setText(count, "");
      }
    }
  }

  document.addEventListener("i18nchange", render);
  document.addEventListener("click", event => {
    if (status?.status !== "error" || !event.target.closest?.("button, [role=button], input[type=submit], input[type=button]")) return;
    // Announce and highlight the existing error while the original action runs.
    title.replaceChildren(t("model.failed"));
    pulse?.cancel();
    pulse = notice.animate([
      { backgroundColor: "#29171e", outlineColor: "#ff8899" },
      { backgroundColor: "#0e1115", outlineColor: "transparent" },
    ], { duration: 800 });
  }, { capture: true });

  async function poll() {
    try {
      const response = await fetch("/api/status", { cache: "no-store" });
      if (!response.ok) throw new Error("server unreachable");
      status = await response.json();
      render();
      document.dispatchEvent(new CustomEvent("modelstatuschange", { detail: status }));
    } catch {
      // A brief disconnect should not erase the last loading failure.
      document.dispatchEvent(new CustomEvent("modelstatuschange", { detail: { status: "server unreachable" } }));
    }
    setTimeout(poll, notice.hidden && !dialogs[0].open ? 2000 : 750);
  }
  poll();
}

export function initModelLoading() {
  const dialog = $("model-dialog"), mode = $("model-mode");
  const submit = $("model-load"), error = $("model-error"), dtype = $("model-compute-dtype");
  const lora = $("model-lora-path");
  const branches = Array.from(dialog.querySelectorAll(".model-branch"), button => ({
    button,
    list: $(button.getAttribute("aria-controls")),
    summary: button.querySelector(".model-branch-selection"),
  }));
  let info = null, busy = false, requestError = "", modelError = "";
  let expandedBackend = "", selectedFormat = null;
  const label = fmt => {
    const name = fmt.quant_type?.toUpperCase() || `INT${fmt.bits}`;
    const format = t("model.bitFormat", { name, bits: fmt.bits });
    return fmt.double_quant ? `${format} · ${t("model.doubleQuant")}` : format;
  };
  function renderPrecisions() {
    for (const { button, list } of branches) {
      const items = (info?.options || []).filter(fmt => fmt.backend === button.value).map(fmt => {
        const item = document.createElement("li");
        const option = document.createElement("button");
        option.type = "button";
        option.value = fmt.id;
        option.textContent = label(fmt);
        option.onclick = () => {
          selectedFormat = fmt;
          requestError = "";
          render();
        };
        item.append(option);
        return item;
      });
      list.replaceChildren(...items);
    }
  }
  function render() {
    const quantized = mode.value === "quantized";
    $("model-quant-fields").hidden = !quantized;
    const backend = selectedFormat?.backend || expandedBackend;
    const support = info?.backends[backend];
    let hint = "";
    if (info && !info.live_model) hint = t("error.noModel");
    else if (quantized && lora.value.trim()) hint = t("model.loraFull");
    else if (quantized && support) {
      if (!support.supported_device) hint = t("model.cudaRequired");
      else if (!support.installed) hint = t("model.install", { backend });
    }
    $("model-availability").textContent = hint;
    const quantizationReady = selectedFormat && support?.installed && support?.supported_device;
    submit.disabled = busy || !info?.live_model || (quantized && (!quantizationReady || Boolean(lora.value.trim())));
    submit.textContent = t(busy ? "model.loading" : "model.load");
    $("model-close").disabled = busy;
    mode.disabled = dtype.disabled = lora.disabled = busy || !info;
    for (const { button, list, summary } of branches) {
      const expanded = quantized && button.value === expandedBackend;
      button.disabled = mode.disabled || !quantized;
      button.setAttribute("aria-expanded", String(expanded));
      list.hidden = !expanded;
      summary.textContent = !expanded && button.value === selectedFormat?.backend ? label(selectedFormat) : "";
      for (const option of list.querySelectorAll("button")) {
        option.disabled = mode.disabled;
        option.setAttribute("aria-pressed", String(option.value === selectedFormat?.id));
      }
    }
    error.textContent = requestError && requestError !== modelError ? localizeError(requestError) : "";
  }
  async function open() {
    if (dialog.open) return;
    requestError = "";
    info = null;
    lora.value = "";
    expandedBackend = "";
    selectedFormat = null;
    dialog.showModal();
    render();
    try {
      const response = await fetch("/api/model/loading");
      if (!response.ok) throw new Error(t("model.failed"));
      info = await response.json();
      lora.value = info.lora_path || "";
      dtype.value = info.compute_dtype || "bfloat16";
      const fmt = info.options.find(f => f.id === info.quantization);
      selectedFormat = fmt && fmt.backend !== "none" ? fmt : null;
      mode.value = selectedFormat ? "quantized" : "full";
      expandedBackend = selectedFormat?.backend || "";
      renderPrecisions();
    } catch (err) {
      requestError = err.message;
    }
    render();
  }
  $("model-settings").onclick = open;
  $("model-close").onclick = () => dialog.close();
  dialog.addEventListener("cancel", event => { if (busy) event.preventDefault(); });
  mode.onchange = dtype.onchange = lora.oninput = () => { requestError = ""; render(); };
  for (const { button } of branches) {
    button.onclick = () => {
      expandedBackend = expandedBackend === button.value ? "" : button.value;
      render();
    };
  }
  $("model-loading-form").onsubmit = async event => {
    event.preventDefault();
    if (submit.disabled) return;
    busy = true;
    requestError = "";
    render();
    const quantized = mode.value === "quantized";
    try {
      const response = await fetch("/api/model/loading", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          quantization: quantized ? selectedFormat.id : "none",
          compute_dtype: dtype.value,
          lora_path: lora.value.trim() || null,
        }),
      });
      const result = await response.json();
      if (!response.ok) throw new Error(result.error || t("model.failed"));
      // Clear all browser-side hover/activation caches after a successful model change.
      location.reload();
    } catch (err) {
      requestError = err.message;
      busy = false;
      render();
    }
  };
  document.addEventListener("modelstatuschange", ({ detail }) => {
    if (detail.status !== "server unreachable") modelError = detail.error || "";
    if (requestError && dialog.open) render();
  });
  document.addEventListener("i18nchange", () => {
    renderPrecisions();
    render();
  });
  return open;
}
