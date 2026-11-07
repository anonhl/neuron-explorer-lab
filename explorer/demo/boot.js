import { initI18n, t, localizeError } from "demo-i18n";
import { initModelLoading, initModelStatus } from "demo-model-loading";

try {
  if (!localStorage.getItem('ne.lang')) localStorage.setItem('ne.lang', 'zh-CN');
  if (!localStorage.getItem('ne.residual-inputs.v1')) localStorage.setItem('ne.residual-inputs.v1', JSON.stringify({
    serial: 2, texts: [{ id: 'text-1', label: 'A', text: '法国的首都是' }, { id: 'text-2', label: 'B', text: '德国的首都是' }],
  }));
} catch {}
initI18n(document.querySelector("#lang"));
const help = document.querySelector("#demo-help");
const helpToggle = document.querySelector("#demo-help-toggle");
function showHelp(show) {
  help.hidden = !show;
  helpToggle.setAttribute("aria-expanded", String(show));
}
helpToggle.onclick = () => showHelp(help.hidden);
document.querySelector("#demo-help-close").onclick = () => { showHelp(false); helpToggle.focus(); };
help.addEventListener("keydown", event => {
  if (event.key === "Escape") { showHelp(false); helpToggle.focus(); }
});
document.querySelector("#demo-open-example").onclick = () => {
  const jump = document.querySelector("#jump");
  jump.value = "L22 N2742";
  jump.dispatchEvent(new KeyboardEvent("keydown", { key: "Enter", bubbles: true }));
  showHelp(false);
};
document.querySelector("#demo-search-example").onclick = () => {
  document.querySelector('#q-mode [data-q="lens"]').click();
  const query = document.querySelector("#lens-q");
  query.value = "cat";
  query.dispatchEvent(new Event("input"));
  query.focus();
  showHelp(false);
};
function openResidualExample() {
  document.querySelector('#ltabs [data-lt="run"]').click();
  document.querySelector('#inf-residual-open').click();
  document.querySelector('#tab-residual [data-action="prepare"]').click();
  showHelp(false);
}
document.querySelector('#demo-open-residual').onclick = async () => {
  await import('demo-app');
  openResidualExample();
};
const openModelSettings = initModelLoading();
function startExplorer() {
  document.body.classList.remove("model-setup");
  import("demo-app").then(() => {
    if (new URLSearchParams(location.search).get('preview') === 'residual') openResidualExample();
  }).catch(error => {
    const status = document.querySelector("#status");
    const render = () => {
      status.textContent = t("error.label", { label: t("error.startup"), error: localizeError(error) });
    };
    render();
    document.addEventListener("i18nchange", render);
    console.error(error);
  });
}
function startPage({ detail }) {
  if (detail.switching) return;
  document.removeEventListener("modelstatuschange", startPage);
  if (detail.status === "idle" && detail.loading?.live_model) {
    document.querySelector("#model-dialog").addEventListener("close", startExplorer, { once: true });
    openModelSettings();
  } else {
    startExplorer();
  }
}
document.addEventListener("modelstatuschange", startPage);
initModelStatus();
