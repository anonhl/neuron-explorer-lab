const wordKey = text => text.trim().normalize("NFC").toLowerCase().normalize("NFC");

export function countWords(replies, lang) {
  const segmenter = new Intl.Segmenter(lang, { granularity: "word" });
  const words = new Map();
  let total = 0;
  for (const { text, prompt, source, condition } of replies) {
    const origin = JSON.stringify([prompt, source, condition]);
    for (const { segment, isWordLike } of segmenter.segment(text)) {
      if (!isWordLike) continue;
      const key = wordKey(segment);
      if (!words.has(key)) words.set(key, { key, text: segment, count: 0, sources: new Map() });
      const word = words.get(key);
      if (!word.sources.has(origin)) word.sources.set(origin, { prompt, source, condition, count: 0 });
      word.sources.get(origin).count++;
      word.count++;
      total++;
    }
  }
  return { total, rows: [...words.values()].sort((a, b) => b.count - a.count || a.text.localeCompare(b.text, lang)) };
}

export function countLens(connections) {
  const tokens = new Map();
  const totals = { up: 0, down: 0 };
  const neuronTotals = { up: new Map(), down: new Map() };
  const direction = () => ({ count: 0, positive: 0, sources: new Map() });
  for (const side of ["up", "down"]) {
    for (const { layer, n, sign, lens, lens_neg, lens_ids, lens_neg_ids } of connections[side] ?? []) {
      const positive = sign > 0;
      const values = (positive ? lens : lens_neg) ?? [];
      const ids = (positive ? lens_ids : lens_neg_ids) ?? [];
      const address = `${layer}:${n}`;
      neuronTotals[side].set(address, (neuronTotals[side].get(address) ?? 0) + values.length);
      totals[side] += values.length;
      values.forEach((text, index) => {
        const id = ids[index];
        const key = Number.isInteger(id) ? id : text;
        if (!tokens.has(key)) tokens.set(key, { key, id, text, up: direction(), down: direction() });
        const entry = tokens.get(key)[side];
        if (!entry.sources.has(address)) entry.sources.set(address, { layer, n, count: 0, positive: 0 });
        const source = entry.sources.get(address);
        source.count++;
        source.positive += Number(positive);
        entry.count++;
        entry.positive += Number(positive);
      });
    }
  }
  const rows = [...tokens.values()].sort((a, b) => (b.up.count + b.down.count) - (a.up.count + a.down.count));
  return { totals, neuronTotals, rows };
}

export function matchingTokens(rows, key, ids) {
  return rows.filter(row => wordKey(row.text) === key || ids?.includes(row.id));
}

function frequencyMarkup(t, lang, esc) {
  const formatter = new Intl.NumberFormat(lang, { style: "percent", maximumFractionDigits: 2 });
  const share = (count, total) => {
    if (!total) return "—";
    const ratio = count / total;
    return ratio > 0 && ratio < 0.0001 ? `<${formatter.format(0.0001)}` : formatter.format(ratio);
  };
  const label = key => esc(t(`frequency.${key}`));
  const tokenLabel = row => `<bdi class="mono frequency-token">${esc(JSON.stringify(row.text))}</bdi>${row.id == null ? "" : ` <small class="dim mono">#${row.id}</small>`}`;
  const message = key => `<p class="frequency-note">${label(key)}</p>`;
  const countShare = (count, total, showTotal) => `${count}${showTotal ? ` / ${total}` : ""} <small class="dim">(${esc(share(count, total))})</small>`;
  const countCell = (count, total, showTotal = false) => `<td class="num">${countShare(count, total, showTotal)}</td>`;
  const pole = (positive, value = "", detail = "") => `<span class="frequency-pole ${positive ? "up" : "down"}" title="${label(positive ? "positiveLink" : "negativeLink")}${detail}">${positive ? "↑" : "↓"}${value}</span>`;
  const poleShares = (count, positive) => {
    const negative = count - positive;
    const part = (isPositive, n) => pole(isPositive, ` ${esc(share(n, count))}`, ` · ${n} / ${count}`);
    return `<small class="frequency-poles mono">${part(true, positive)} ${part(false, negative)}</small>`;
  };
  const sourceToken = (row, count, positive) => {
    const negative = count - positive;
    const marker = (isPositive, n) => n ? ` ${pole(isPositive, n > 1 ? ` ×${n}` : "")}` : "";
    return `${tokenLabel(row)}${marker(true, positive)}${marker(false, negative)}`;
  };
  const table = (head, rows) => `<div class="frequency-scroll"><table class="dtable frequency-table"><thead><tr>${head}</tr></thead><tbody>${rows}</tbody></table></div>`;
  const th = (key, numeric = false) => `<th scope="col"${numeric ? ' class="num"' : ""}>${label(key)}</th>`;
  const lensSection = (lens, error, { renderToken = tokenLabel, showTotal = false } = {}) => {
    const lensCell = ({ count, positive }, total) => `<td class="num">${countShare(count, total, showTotal)}${poleShares(count, positive)}</td>`;
    let tokens;
    if (lens?.rows.length) {
      const ranked = side => lens.rows.map((row, index) => ({ row, index }))
        .filter(({ row }) => row[side].count > 0)
        .sort((a, b) => b.row[side].count - a.row[side].count);
      const input = ranked("up"), output = ranked("down");
      const cells = (entry, side) => entry
        ? `<td>${renderToken(entry.row, entry.index)}</td>${lensCell(entry.row[side], lens.totals[side])}`
        : '<td></td><td></td>';
      tokens = table(th("token") + th("input", true) + th("token") + th("output", true),
        Array.from({ length: Math.max(input.length, output.length) }, (_, i) =>
          `<tr>${cells(input[i], "up")}${cells(output[i], "down")}</tr>`).join(""));
    } else {
      tokens = message(error ? "lensError" : lens ? "noLens" : "loading");
    }
    return `<h2>${label("lens")}</h2>${message("lensHint")}${tokens}`;
  };
  return { share, tokenLabel, label, message, countCell, poleShares, sourceToken, table, th, lensSection };
}

export function lensFrequencyHTML({ connections, error }, { t, lang, esc }) {
  const { lensSection } = frequencyMarkup(t, lang, esc);
  return lensSection(connections ? countLens(connections) : null, error, { showTotal: true });
}

export function initFrequencyPanel(root, { t, getLang, esc, lookupTokens, openNeuron, copyNeuronButtonHTML, copyNeuronId }) {
  let focus, selected, words, lens, connectionError, markup;
  let lookup = new Map();
  const button = (kind, index, row, text) => `<button type="button" class="frequency-link" data-frequency-kind="${kind}" data-index="${index}" aria-pressed="${selected?.kind === kind && selected.key === row.key}">${text}</button>`;

  function renderSelection() {
    const target = root.querySelector("#frequency-selection");
    if (!target) return;
    const { share, tokenLabel, label, message, countCell, poleShares, sourceToken, table, th } = markup;
    const kind = selected?.kind;
    const word = kind === "word" ? words.rows.find(row => row.key === selected.key) : null;
    const token = kind === "token" ? lens?.rows.find(row => row.key === selected.key) : null;
    if (!word && !token) {
      selected = null;
      target.innerHTML = message("pick");
      return;
    }
    const result = word ? lookup.get(word.key) : null;
    const matches = word ? matchingTokens(lens?.rows ?? [], word.key, result?.ids) : [token];
    let html = `<h3>${word ? `<bdi>${esc(word.text)}</bdi>` : tokenLabel(token)}</h3>`;
    if (word) {
      const sources = [...word.sources.values()].sort((a, b) => b.count - a.count);
      html += `<h4>${label("replySources")}</h4>` + table(th("prompt") + th("source") + th("count", true),
        sources.map(({ prompt, source, condition, count }) => `<tr><td><bdi>${esc(prompt)}</bdi></td>
          <td>${label(source)} · ${label(condition)}</td><td class="num">${count}</td></tr>`).join(""));
    }
    if (connectionError) {
      html += message("lensError");
    } else if (!lens) {
      html += message("loading");
    } else {
      if (word && result?.pending) html += message("matching");
      if (word && result?.ids === null) html += message("exactOnly");
      if (word && matches.length) html += `<p class="frequency-note">${label("matchedTokens")} ${matches.map(tokenLabel).join(" · ")}</p>`;
      for (const [side, heading] of [["up", "input"], ["down", "output"]]) {
        const count = matches.reduce((sum, row) => sum + row[side].count, 0);
        const positive = matches.reduce((sum, row) => sum + row[side].positive, 0);
        const total = lens.totals[side];
        html += `<h4>${label(heading)} <span class="dim mono">${count} / ${total} · ${esc(share(count, total))}</span>${poleShares(count, positive)}</h4>`;
        const sources = new Map();
        for (const row of matches) {
          for (const [address, { layer, n, count, positive }] of row[side].sources) {
            if (!sources.has(address)) sources.set(address, {
              layer, n, count: 0, total: lens.neuronTotals[side].get(address), tokens: [],
            });
            const source = sources.get(address);
            source.tokens.push(sourceToken(row, count, positive));
            source.count += count;
          }
        }
        const rows = [...sources.values()].sort((a, b) => b.count - a.count || a.layer - b.layer || a.n - b.n);
        html += rows.length ? table(th("neuron") + th("token") + th("neuronShare", true), rows.map(({ tokens, layer, n, count, total }) =>
          `<tr><td><span class="neuron-id" data-layer="${layer}" data-neuron="${n}">
              <button type="button" class="frequency-link mono">L${layer} N${n}</button>${copyNeuronButtonHTML()}</span></td>
            <td>${tokens.join(" · ")}</td>${countCell(count, total, true)}</tr>`).join("")) : message(total ? "noMatch" : "noLens");
      }
    }
    target.innerHTML = html;
  }

  root.addEventListener("click", async event => {
    const neuron = event.target.closest(".neuron-id");
    if (neuron) {
      const { layer, neuron: n } = neuron.dataset;
      const copy = event.target.closest(".copy-neuron");
      if (copy) return copyNeuronId(copy, `L${layer} N${n}`);
      if (event.target.closest(".frequency-link")) return openNeuron(+layer, +n);
    }
    const button = event.target.closest("button[data-frequency-kind]");
    if (!button) return;
    const kind = button.dataset.frequencyKind;
    const row = (kind === "word" ? words.rows : lens.rows)[+button.dataset.index];
    selected = { kind, key: row.key };
    const cache = lookup;
    const fetchIds = kind === "word" && !cache.has(row.key);
    if (fetchIds) cache.set(row.key, { pending: true });
    root.querySelectorAll('[aria-pressed="true"]').forEach(el => el.setAttribute("aria-pressed", "false"));
    button.setAttribute("aria-pressed", "true");
    renderSelection();
    root.querySelector("#frequency-selection").scrollIntoView({ block: "nearest" });
    if (fetchIds) {
      const { ids } = await lookupTokens(row.text).catch(() => ({ ids: null }));
      cache.set(row.key, { ids });
      if (lookup === cache && selected?.kind === kind && selected.key === row.key) renderSelection();
    }
  });

  return {
    render({ neuron, replies, connections, error }) {
      if (focus !== neuron) {
        focus = neuron;
        selected = null;
        lookup = new Map();
        root.scrollTop = 0;
      }
      const lang = getLang();
      markup = frequencyMarkup(t, lang, esc);
      const { tokenLabel, label, message, countCell, table, th, lensSection } = markup;
      words = countWords(replies, lang);
      lens = connections ? countLens(connections) : null;
      connectionError = error;
      const wordTable = words.rows.length ? table(th("word") + th("countShare", true), words.rows.map((row, i) =>
        `<tr><td>${button("word", i, row, `<bdi>${esc(row.text)}</bdi>`)}</td>${countCell(row.count, words.total)}</tr>`).join("")) : message("noText");
      root.innerHTML = `<section><h2>${label("text")}</h2>${message("textHint")}${wordTable}</section>
        <section id="frequency-selection" aria-live="polite"></section>
        <section>${lensSection(lens, error, { renderToken: (row, i) => button("token", i, row, tokenLabel(row)) })}</section>`;
      renderSelection();
    },
  };
}
