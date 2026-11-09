(() => {
  "use strict";
  const model = "Qwen3.5-4B · 模拟";
  const nLayers = 32;
  const inter = 9216;
  const layerTypes = Array.from({ length: nLayers }, (_, i) => (i + 1) % 4 === 0 ? "full_attention" : "linear_attention");
  const arch = {
    n_layers: nLayers, hidden: 2560, inter, layer_types: layerTypes,
    attn_heads: 16, kv_heads: 4, linear_k_heads: 16, linear_v_heads: 32,
  };
  const topics = [
    { name: "猫与宠物", up: ["kitten", "feline", "cat", "猫咪", "purr", "宠物", "喵喵"], down: ["machine", "engine", "vehicle", "发动机"] },
    { name: "动物分类", up: ["species", "mammal", "dog", "动物", "wildlife"], down: ["category", "cat", "taxonomy", "分类", "标签"] },
    { name: "城市与交通", up: ["北京", "城市", "Beijing", "地铁", "street", "train"], down: ["forest", "野外", "river", "森林"] },
    { name: "科学与观测", up: ["research", "实验", "observe", "model", "data", "数据"], down: ["fiction", "fantasy", "魔法", "story"] },
    { name: "语气与对话", up: ["hello", "你好", "assistant", "对话", "question", "回答"], down: ["silence", "end", "独白", "none"] },
    { name: "音乐与节奏", up: ["music", "音乐", "melody", "rhythm", "节奏", "song"], down: ["quiet", "silence", "安静", "noise"] },
  ];
  const featured = [2742, 5203, 1180, 6341, 4077, 7526];
  const prompts = { A: "Describe what you see.", B: "Tell me a short story.", C: "What happens next?" };
  const modelOptions = [
    { id: "none", backend: "none" },
    { id: "bnb-nf4", backend: "bitsandbytes", quant_type: "nf4", bits: 4 },
    { id: "bnb-int8", backend: "bitsandbytes", quant_type: "int8", bits: 8 },
    { id: "hqq-int4", backend: "hqq", quant_type: "int4", bits: 4 },
  ];
  const storeKey = "neuron-demo.storage.v1";
  const defaultSettings = { quantization: "none", compute_dtype: "bfloat16", lora_path: null };
  const seed = {
    groups: [{ id: "practice", name: "练习示例" }],
    batches: [{
      id: "practice-batch", group_id: "practice", name: "猫与宠物 · 正向偏移", notes: "可先点「使用」，也可以复制后调整剂量。",
      model, model_revision: null, ...defaultSettings, created_at: "2026-10-01T00:00:00Z",
      operations: [{ layer: 22, n: 2742, mode: "add", dose: 0.2, enabled: true }],
    }],
    names: {}, settings: defaultSettings, chats: {},
  };
  let saved = structuredClone(seed);
  try {
    const stored = JSON.parse(localStorage.getItem(storeKey) || "null");
    if (stored) saved = stored;
  } catch {}
  let active = null;
  let loading = false;
  let inputPieces = [];
  let residualCapture = "";
  const cachedInputs = new Map();


  function simulatedResidual(pos, layer, neuron) {
    const norm = a => Math.sqrt(a.reduce((s, x) => s + x*x, 0));
    const vec = a => {
      const ids = a.map((value, dim) => ({dim, value}));
      return {vector:a, norm:norm(a), positive:ids.filter(x=>x.value>0).sort((a,b)=>b.value-a.value).slice(0,12), negative:ids.filter(x=>x.value<0).sort((a,b)=>a.value-b.value).slice(0,12)};
    };
    const bins = a => Array.from({length:64},(_,i)=>a.slice(i*40,(i+1)*40).reduce((s,v)=>s+v,0)/40);
    let pre = Array.from({length:2560},(_,i)=>Math.sin(i*.19+pos)*.15);
    const all = Array.from({length:nLayers},(_,l)=>{
      const mid=pre.map((v,i)=>v+Math.sin(i*.11+l+pos)*.018);
      const post=mid.map((v,i)=>v+Math.cos(i*.07+l-pos)*.025);
      const row={layer:l,pre,mid,post,attention:mid.map((v,i)=>v-pre[i]),mlp:post.map((v,i)=>v-mid[i])};pre=post;return row;
    });
    const n=neuron>=0?neuron:featured[0], act=rawActivation(layer,n,inputPieces[pos]||'',pos);
    const lens=lensFor(layer,n), row=all[layer];
    const tokenRows=(words,sg=1)=>words.slice(0,8).map((token,i)=>({id:hash(token),token,score:sg*(.42-i*.04)}));
    const write=vec(Array.from({length:2560},(_,i)=>act*Math.sin(i*.17+n)*.012));
    write.tokens={positive:tokenRows(lens.up),negative:tokenRows(lens.down,-1)};
    return {ready:true,capture:residualCapture,pos,token:inputPieces[pos],layer,hidden:2560,batch:selectedBatch(),
      map_vectors:{vectors:[all[0].pre,...all.flatMap(r=>[r.mid,r.post])]},
      layers:all.map(r=>({layer:r.layer,...Object.fromEntries(['pre','mid','post','attention','mlp'].map(k=>[k,norm(r[k])])),bins:Object.fromEntries(['pre','mid','post'].map(k=>[k,bins(r[k])]))})),
      stages:Object.fromEntries(['pre','mid','post'].map(k=>[k,{...vec(row[k]),tokens:tokenRows(lens.up)}])),
      updates:{attention:vec(row.attention),mlp:vec(row.mlp)},selected:{layer,n,act,score:act/2,write},
      writers:featured.map(n=>{const act=rawActivation(layer,n,inputPieces[pos]||'',pos);return {layer,n,act,score:act/2,write_norm:Math.abs(act)*.43}}),
      readers:layer===nLayers-1?[]:featured.map((n,i)=>({layer:layer+1,n,act:rawActivation(layer+1,n,inputPieces[pos]||'',pos),score:.4,local_delta:(i%2?-1:1)*(.4-i*.04)})),
      exportLayers:all};
  }

  function persist() {
    try { localStorage.setItem(storeKey, JSON.stringify(saved)); } catch {}
  }
  function uid() {
    return globalThis.crypto?.randomUUID?.() ?? Math.random().toString(36).slice(2);
  }
  function hash(value) {
    let sum = 2166136261;
    for (const char of String(value)) sum = Math.imul(sum ^ char.codePointAt(0), 16777619);
    return sum >>> 0;
  }
  function noise(value) { return hash(value) / 4294967296; }
  function wrap(value, size) { return (value % size + size) % size; }
  function topicFor(layer, n) {
    const i = featured.indexOf(n);
    return topics[i >= 0 ? i : hash(layer + "/" + n) % topics.length];
  }
  function nameFor(layer, n) {
    return saved.names[layer + "/" + n] ?? (featured.includes(n) ? topicFor(layer, n).name : "");
  }
  function lensFor(layer, n) {
    const { up, down } = topicFor(layer, n);
    const rotate = a => [...a.slice(1), a[0]];
    return {
      up, down, steer_up: up, steer_down: down,
      r_up: rotate(up), r_down: rotate(down), jpp_up: up.slice(0, 4), jpp_down: down.slice(0, 4),
      s_up: up.slice(1, 5), s_down: down.slice(1, 5), kl_up: 0.18, kl_down: 0.12,
    };
  }
  function neuronsFor(layer) {
    return [...featured, ...Array.from({ length: 14 }, (_, i) => 99 + ((layer * 97 + i * 401) % 8800))]
      .filter((value, i, all) => all.indexOf(value) === i);
  }
  function unitsFor(layer) {
    return [...neuronsFor(layer).map((n, i) => ({
      kind: "neuron", n, name: nameFor(layer, n), words: topicFor(layer, n).up.slice(0, 5),
      z: 26.5 - i * 0.91, null_pct: 0.04 + i * 0.008,
    })), { kind: "random", n: 0, z: 7.6, null_pct: 0.51, words: ["sample", "control", "随机"], name: "" }];
  }
  function exampleText(topic, prompt, dose) {
    const phrases = [
      "窗边的猫咪听见脚步声，轻轻摆了摆尾巴。The cat watched from the windowsill.",
      "研究员给不同的动物做分类，并记录每个物种的特征。Species belong to different groups.",
      "列车穿过城市，下一站是北京。The train arrived just as the city lights came on.",
      "团队重复实验，记录数据并核对结果。The research team repeated the experiment.",
      "你好，我在这里。你想先聊聊哪一件事？Hello, how can I help?",
      "安静的房间里响起一段旋律，节奏渐渐加快。The melody grew louder.",
    ];
    if (dose === 0) return { A: "窗外有些动静，我停下来仔细看了看。", B: "故事从一个普通的早晨开始。", C: "门被轻轻推开，有人走了进来。" }[prompt];
    if (dose < 0) return "画面慢慢改变，叙述转向" + topic.down[0] + "，接着出现了新的细节。";
    return phrases[topics.indexOf(topic)];
  }
  function rolloutRows(layer) {
    const rows = [];
    const add = (kind, n, doses) => {
      const topic = topicFor(layer, n);
      for (const prompt of Object.keys(prompts)) {
        for (const dose of doses) rows.push({
          kind, neuron: n, prompt, dose, rel_push: Math.abs(dose), finish: "stop",
          text: kind === "random" ? "随机方向对照。观察者记下了眼前的景象，然后继续往前走。" : exampleText(topic, prompt, dose),
        });
      }
    };
    add("baseline", -1, [0]);
    for (const n of neuronsFor(layer)) add("neuron", n, [-0.8, -0.2, 0.2, 0.8]);
    add("random", 0, [-0.8, 0.8]);
    return rows;
  }
  function unit(layer, n, kind) {
    const rows = rolloutRows(layer);
    const ranked = unitsFor(layer).filter(u => u.kind === kind);
    const rank = ranked.findIndex(u => u.n === n);
    const topic = topicFor(layer, n);
    return {
      layer, n, kind, name: kind === "neuron" ? nameFor(layer, n) : "",
      prompts, rollouts: rows.filter(r => r.kind === kind && r.neuron === n),
      baselines: rows.filter(r => r.kind === "baseline"),
      stats: rank < 0 ? null : {
        rank: rank + 1, mass: ranked[rank].z, null_pct: ranked[rank].null_pct, per_prompt: {},
        words: topic.up.map((word, i) => [word, 8 - i, 5.7 - i * 0.4]),
        dose: Object.fromEntries([-0.8, -0.2, 0.2, 0.8].map(d => [d, {
          rel_push: d, distinct4: 0.81 - Math.abs(d) * 0.07, capped: 0, logprob: -1.16 - Math.abs(d) * 0.18,
        }])),
      },
      rank: rank + 1, n_ranked: ranked.length,
    };
  }
  function embeddings(layer) {
    const rows = rolloutRows(layer);
    const x = [], y = [];
    for (const [i, row] of rows.entries()) {
      const topic = topics.indexOf(topicFor(layer, row.neuron));
      const angle = (topic / topics.length) * Math.PI * 2;
      const radius = row.kind === "neuron" ? (row.dose > 0 ? 0.31 : 0.17) : 0.035;
      x.push(0.5 + Math.cos(angle) * radius + (noise(i + "x") - 0.5) * 0.18);
      y.push(0.5 + Math.sin(angle) * radius + (noise(i + "y") - 0.5) * 0.18);
    }
    return {
      x, y, row: rows.map((_, i) => i), kind: rows.map(r => r.kind), neuron: rows.map(r => r.neuron),
      prompt: rows.map(r => r.prompt), dose: rows.map(r => r.dose),
    };
  }
  function base64(bytes) {
    const chunks = [];
    for (let i = 0; i < bytes.length; i += 8192) chunks.push(String.fromCharCode(...bytes.subarray(i, i + 8192)));
    return btoa(chunks.join(""));
  }
  let layout;
  function writesLayout() {
    if (layout) return layout;
    const cells = new Int16Array(nLayers * inter);
    for (let l = 0; l < nLayers; l++) {
      const byTopic = topics.map(() => []);
      for (let n = 0; n < inter; n++) byTopic[topics.indexOf(topicFor(l, n))].push(n);
      const ordered = byTopic.flat();
      for (let i = 0; i < inter; i++) cells[l * inter + ordered[i]] = (i % 96) * 96 + Math.floor(i / 96);
    }
    layout = base64(new Uint8Array(cells.buffer));
    return layout;
  }
  function links(layer, n, delta, count) {
    const target = layer + delta;
    if (target < 0 || target >= nLayers) return [];
    return Array.from({ length: count }, (_, i) => {
      const id = i < featured.length ? featured[wrap(n + i, featured.length)] : wrap(n + i * 613, inter);
      const L = lensFor(target, id);
      return {
        layer: target, n: id, z: Math.max(5, 59 - i * 2), sign: i % 3 === 1 ? -1 : 1,
        lens: L.up, lens_neg: L.down, logit: L.up.slice(0, 4), logit_neg: L.down.slice(0, 4),
        lens_ids: L.up.map(hash), lens_neg_ids: L.down.map(hash), name: nameFor(target, id),
      };
    });
  }
  function linkRows(layer, n, count) {
    const rows = [];
    for (const delta of [-1, 1]) {
      for (const c of links(layer, n, delta, count)) {
        const endpoints = delta === 1 ? [layer, n, c.layer, c.n] : [c.layer, c.n, layer, n];
        rows.push([...endpoints, c.z * 10 * c.sign, delta]);
        for (const d of links(c.layer, c.n, delta, 2)) {
          const next = delta === 1 ? [c.layer, c.n, d.layer, d.n] : [d.layer, d.n, c.layer, c.n];
          rows.push([...next, d.z * 7 * d.sign, delta * 2]);
        }
      }
    }
    return rows;
  }
  function chips(tokens) { return tokens.slice(0, 5).map((tok, i) => ({ tok, n: 6 - i, w: 1.3 - 0.2 * i })); }
  function search(q) {
    const query = q.toLowerCase().trim();
    const matches = [];
    const tokens = new Set();
    if (query.length >= 2) {
      for (const layer of [22, 23, 18, 26]) {
        for (const n of neuronsFor(layer).slice(0, layer === 22 ? 12 : 6)) {
          const lens = lensFor(layer, n);
          for (const sign of ["+", "-"]) {
            const list = sign === "+" ? lens.up : lens.down;
            const hits = list.flatMap((token, rank) => {
              const lower = token.toLowerCase();
              if (!lower.includes(query)) return [];
              tokens.add(lower);
              const tier = lower === query ? 0 : lower.startsWith(query) ? 1 : 2;
              return [{ tier, rank }];
            }).sort((a, b) => a.tier - b.tier || a.rank - b.rank);
            if (hits.length) matches.push({ layer, n, sign, ...hits[0], src: "steer", name: nameFor(layer, n), swept: true, lens: list.slice(0, 6) });
          }
        }
      }
    }
    matches.sort((a, b) => a.tier - b.tier || a.rank - b.rank || (a.layer === 22 ? -1 : 0) - (b.layer === 22 ? -1 : 0) || a.layer - b.layer);
    return { ready: true, results: matches, tokens: tokens.size, total: matches.length };
  }
  function pieces(text) { return (String(text).match(/[\u3400-\u9fff]|[A-Za-z0-9]+|[^\s]/gu) || []).slice(0, 140); }
  function rawActivation(layer, n, word, position) {
    const term = String(word).toLowerCase();
    const topic = topicFor(layer, n);
    const polarity = topic.up.some(t => t.toLowerCase().includes(term)) ? 1
      : topic.down.some(t => t.toLowerCase().includes(term)) ? -1 : 0;
    let act = (noise(term + ":" + layer + ":" + n + ":" + position) - 0.5) * 3.2 + 1.85 * polarity;
    for (const op of selectedBatch()?.operations || []) {
      if (op.layer === layer && op.n === n && op.enabled) act = op.mode === "mul" ? act * op.dose : act + op.dose;
    }
    return +act.toFixed(3);
  }
  function activationView(params, limit = 80) {
    const pos = Number(params.get("pos") ?? -1);
    const sign = params.get("sign") || "both";
    const lo = Number(params.get("lo") || 0);
    const hi = Number(params.get("hi") || 31);
    const min = Number(params.get("min") || 0);
    const q = (params.get("q") || "").toLowerCase();
    const addresses = params.get("addresses");
    const candidates = addresses ? addresses.split(",").map(value => value.split(":").map(Number))
      : Array.from({ length: nLayers }, (_, l) => neuronsFor(l).slice(0, 8).map(n => [l, n])).flat();
    const rows = [];
    if (!inputPieces.length) return { results: rows, total: 0 };
    for (const [layer, n] of candidates) {
      if (!addresses && (layer < lo || layer > hi)) continue;
      const positions = pos < 0 ? inputPieces.map((_, i) => i) : [pos];
      const rank = a => sign === "-" ? -a : sign === "+" ? a : Math.abs(a);
      const best = positions.map(i => ({ act: rawActivation(layer, n, inputPieces[i], i), token: inputPieces[i], pos: i }))
        .reduce((a, b) => rank(a.act) >= rank(b.act) ? a : b);
      const side = best.act < 0 ? "-" : "+";
      const score = +Math.abs(best.act).toFixed(2);
      const lens = lensFor(layer, n)[side === "+" ? "up" : "down"];
      if (!addresses && (score < min || (sign !== "both" && sign !== side) || (q && !lens.some(t => t.toLowerCase().includes(q))))) continue;
      rows.push({ layer, n, sign: side, score, lens: lens.slice(0, 6), name: nameFor(layer, n), ...best });
    }
    if (!addresses) rows.sort((a, b) => b.score - a.score);
    return { results: rows.slice(0, limit), total: rows.length };
  }
  function float16(value) {
    if (value === 0) return 0;
    const magnitude = Math.abs(value);
    const exp = Math.floor(Math.log2(magnitude));
    return (value < 0 ? 32768 : 0) + (exp + 15) * 1024 + Math.floor((magnitude / 2 ** exp - 1) * 1024);
  }
  function headInfo(layer, head) {
    const reads = links(layer, head, -1, 4);
    const writes = links(layer, head, 1, 4);
    return {
      layer, head, full: layerTypes[layer] === "full_attention", src: "steer", reads, writes,
      q: reads.slice(0, 2), k: reads.slice(2), read_toks: chips(topics[2].up), write_toks: chips(topics[3].up),
      q_toks: chips(topics[4].up), k_toks: chips(topics[0].up),
    };
  }
  function neuronHeads(layer) {
    const roles = {};
    for (const [i, role] of ["q", "k", "v", "o"].entries()) {
      const target = layer + (role === "o" ? -1 : i + 1);
      const h = i + 1;
      roles[role] = target < 0 || target >= nLayers ? [] : [{ layer: target, head: h, heads: [h], z: 6 - i * 0.6, sign: i % 2 ? -1 : 1 }];
    }
    const layers = {};
    for (const entries of Object.values(roles)) for (const h of entries) {
      layers[h.layer] = Array.from({ length: layerTypes[h.layer] === "full_attention" ? 16 : 32 }, (_, i) => i === h.head ? h.z : 0);
    }
    return { roles, layers };
  }
  function exemplar(tokens, polarity, layer, n) {
    return Array.from({ length: 2 }, (_, i) => {
      const words = ["在", "这段", "示例", "中", tokens[i], "出现", "在这里", "。"];
      const trace = words.map((word, pos) => pos === 4 ? polarity * (2.8 - i * 0.3) : 0.14 * rawActivation(layer, n, word, pos));
      return { tokens: words, trace, peak: 4, act: trace[4], domain: "demo_text", url: "" };
    });
  }

  const json = (data, status = 200) => new Response(JSON.stringify(data), { status, headers: { "Content-Type": "application/json" } });
  const binary = rows => new Response(new Int16Array(rows.flat()), { headers: { "Content-Type": "application/octet-stream", "X-Web-Total": rows.length } });
  const pause = ms => new Promise(resolve => setTimeout(resolve, ms));
  const selectedBatch = () => saved.batches.find(b => b.id === active) ?? null;
  const settings = () => ({
    ...saved.settings, live_model: true, options: modelOptions,
    backends: { bitsandbytes: { installed: true, supported_device: true }, hqq: { installed: true, supported_device: true } },
  });
  function batchApi(path, method, body) {
    const root = "/api/neuron-batches";
    if (path === root && method === "GET") return json({
      model, model_revision: null, ...saved.settings, n_layers: nLayers, intermediate_size: inter, live_model: true,
      groups: saved.groups, batches: saved.batches, active_batch: selectedBatch(),
    });
    if (path === root + "/groups") {
      const name = body.name.trim();
      if (!name || saved.groups.some(g => g.name === name)) return json({ error: "请输入不同的分组名称。" }, 400);
      const group = { id: uid(), name };
      saved.groups.push(group);
      persist();
      return json(group);
    }
    if (path === root && method === "POST") {
      if (!body.name.trim() || !saved.groups.some(g => g.id === body.group_id) || !body.operations.some(op => op.enabled)) {
        return json({ error: "请检查批次名称、分组和操作。" }, 422);
      }
      const batch = { ...body, name: body.name.trim(), id: uid(), created_at: new Date().toISOString() };
      saved.batches.push(batch);
      persist();
      return json(batch);
    }
    if (path === root + "/stop") {
      active = null;
      return json({ active_batch: null });
    }
    const [, , , id, action] = path.split("/");
    const batch = saved.batches.find(b => b.id === id);
    if (!batch) return json({ error: "找不到该批次。" }, 404);
    if (action === "use") active = id;
    else if (method === "PATCH") {
      if (!body.name.trim()) return json({ error: "请输入批次名称。" }, 400);
      batch.name = body.name.trim();
    } else if (method === "DELETE") {
      saved.batches = saved.batches.filter(b => b.id !== id);
      if (active === id) active = null;
    }
    persist();
    return json({ batch, active_batch: selectedBatch() });
  }
  function steer(body, signal) {
    const single = body.group === "conversation";
    const batchOps = selectedBatch()?.operations.filter(op => op.enabled) || [];
    const chatOps = body.operations.filter(op => op.enabled);
    const lead = chatOps[0] || batchOps[0];
    const topic = lead ? topicFor(lead.layer, lead.n) : topics[4];
    const subject = String(body.prompt || "").replace(/\s+/g, " ").slice(0, 45);
    const reply = side => {
      const adjusted = side === 0 ? [] : chatOps.map(op => ({ ...op, dose: side * op.dose }));
      const operations = [...batchOps, ...adjusted];
      const describe = op => "L" + op.layer + " N" + op.n + " " + (op.mode === "mul" ? "×" : "偏移 ") + op.dose;
      const residuals = side === 1 ? body.residuals || [] : [];
      const recipe = [...operations.slice(0, 3).map(describe), ...residuals.map(({ plan: p }) => `残差 L${p.layer} N${p.neuron} α=${p.strength}`)].join("、");
      const profile = operations.reduce((value, op) => op.mode === "mul" ? value * op.dose : value + op.dose, 1);
      const effect = profile - 1;
      const direction = Math.abs(effect) < 1e-6 ? 0 : Math.sign(effect);
      return "模拟回复" + (recipe ? "（" + recipe + "）" : "（基准）")
        + "。对于「" + subject + "」，" + exampleText(topic, side === 0 ? "B" : "A", direction);
    };
    const replies = single ? [reply(1)] : [reply(1), reply(0), reply(-1)];
    const events = replies.flatMap((reply, row) => (reply.match(/.{1,10}/gu) || []).map(t => ({ row, t })));
    events.push({ row: null });
    const encoder = new TextEncoder();
    return new Response(new ReadableStream({
      async pull(controller) {
        await pause(38);
        if (signal?.aborted) { controller.error(new DOMException("Aborted", "AbortError")); return; }
        const event = events.shift();
        if (event) controller.enqueue(encoder.encode("data: " + JSON.stringify(event) + "\n\n"));
        else controller.close();
      },
    }), { headers: { "Content-Type": "text/event-stream" } });
  }

  const streams = new Map();
  const replyTimers = new Map();
  const encoder = new TextEncoder();
  function emit(chat, event) {
    try { streams.get(chat)?.enqueue(encoder.encode(JSON.stringify(event) + "\n")); } catch {}
  }
  function agentApi(path, p, body, signal) {
    const chat = p.get("chat") || body.chat;
    switch (path) {
      case "/api/agent/events":
        return new Response(new ReadableStream({
          start(controller) {
            streams.set(chat, controller);
            emit(chat, { t: "state", busy: replyTimers.has(chat) });
            signal?.addEventListener("abort", () => {
              if (streams.get(chat) === controller) streams.delete(chat);
              try { controller.close(); } catch {}
            }, { once: true });
          },
        }));
      case "/api/agent/send": {
        const { message, layer, n } = body;
        const record = saved.chats[chat] ??= { chat, agent: body.agent, title: message.slice(0, 42), events: [] };
        record.t = Date.now() / 1000;
        record.events.push({ t: "user", text: message });
        const reply = n == null
          ? "这是代理对话的模拟回复。试试在顶部搜索 **cat**，或者输入 **L22 N2742** 打开一个神经元，再点这里的「解释」。"
          : "这是对 **L" + layer + " N" + n + "** 的模拟解释。这个神经元的样例主题是「" + topicFor(layer, n).name
            + "」。你可以先看正负透镜词元，再到「对话 → 评测」比较三列输出。";
        replyTimers.set(chat, setTimeout(() => {
          replyTimers.delete(chat);
          const event = { t: "text", text: reply };
          record.events.push(event);
          persist();
          emit(chat, event);
          emit(chat, { t: "done" });
        }, 240));
        return json({ ok: true });
      }
      case "/api/agent/interrupt":
        clearTimeout(replyTimers.get(chat));
        replyTimers.delete(chat);
        return json({ running: false });
      case "/api/agent/chats":
        return json(Object.values(saved.chats).map(c => ({
          chat: c.chat, agent: c.agent, title: c.title, t: c.t, turns: c.events.filter(e => e.t === "user").length,
        })).sort((a, b) => b.t - a.t));
      case "/api/agent/history":
        return json({ events: saved.chats[chat]?.events || [] });
      case "/api/agent/upload":
        return json({ id: uid(), name: body.name });
      case "/api/agent/rewind": {
        const events = saved.chats[chat]?.events;
        const i = events?.findLastIndex(e => e.t === "user" && e.text === body.text) ?? -1;
        if (i >= 0) events.splice(i);
        persist();
        return json({ ok: true });
      }
      default: return json({ error: "练习页未提供此代理操作。" }, 404);
    }
  }

  function residualTextRows(body) {
    if (!Array.isArray(body.texts) || body.texts.length < 1 || body.texts.length > 8) throw new Error('请提供 1–8 条独立文本。');
    return body.texts.map(row => {
      const raw = String(row.text).match(/[\u3400-\u9fff]|[A-Za-z0-9]+|[^\s]/gu) || [];
      const words = body.chat ? ['<用户>', ...raw, '<助手>'] : raw;
      if (!raw.length || words.length > 1024) throw new Error('请填写每条文本，并保持在 1–1,024 个模拟 token 内。');
      return { ...row, count: words.length, tokens: words.map((text, i) => ({ text, positions: [i] })) };
    });
  }
  // A deterministic toy residual calculation, only for practicing the UI. No model runs here.
  function residualExample(row, rows, plans, maxNew, targets) {
    const dot = (a, b) => a.reduce((sum, v, i) => sum + v * b[i], 0);
    const norm = a => Math.sqrt(dot(a, a));
    const unit = a => {
      const n = norm(a);
      if (!n || !Number.isFinite(n)) throw new Error('Direction is zero or invalid; choose a different direction or source pair.');
      return a.map(v => v / n);
    };
    let plan = plans[0];
    const position = row.count - 1;
    const state = (source, pos) => {
      const stageOffset = { pre: 0, mid: .31, post: .67 }[plan.stage];
      const phase = noise(source.text) * 4 + (pos < 0 ? source.count - 1 : pos) * .13 + plan.layer * .09 + stageOffset;
      return Array.from({ length: arch.hidden }, (_, i) => Math.sin(i * .19 + phase) * (.15 + plan.layer * .004) + Math.cos(i * .041 - phase) * .04);
    };
    const before = state(row, position);
    const beforeNorm = norm(before);
    const find = id => { const source = rows.find(r => r.id === id); if (!source) throw new Error('请选择来源文本。'); return source; };
    let after = before;
    for (plan of plans) {
      const target = plan.scope === 'token' ? plan.positions[row.id] : position;
      if (!Number.isInteger(target) || target < 0 || target >= row.count) throw new Error('Target token is outside the selected text.');
      let direction;
      if (plan.method === 'patch') direction = state(find(plan.source), plan.source_pos);
      else if (plan.method !== 'remove') {
        if (plan.direction === 'custom') {
          if (!Array.isArray(plan.vector) || plan.vector.length !== arch.hidden) throw new Error('Custom direction must contain 2560 finite numbers.');
          direction = unit(plan.vector);
        } else if (plan.direction === 'difference') {
          const a = state(find(plan.source), plan.source_pos), b = state(find(plan.reference), plan.reference_pos);
          direction = unit(a.map((v, i) => v - b[i]));
        } else direction = unit(before.map((_, i) => Math.sin(i * .17 + plan.neuron + plan.layer * .09)));
      }
      const alpha = plan.strength;
      const currentNorm = norm(after);
      if (plan.method === 'add') after = after.map((v, i) => v + alpha * currentNorm * direction[i]);
      else if (plan.method === 'erase') {
        const projection = dot(after, direction);
        after = after.map((v, i) => v - alpha * projection * direction[i]);
      } else if (plan.method === 'patch') after = after.map((v, i) => (1 - alpha) * v + alpha * direction[i]);
      else {
        const phase = plan.component === 'head' ? plan.head : plan.neuron;
        const scale = plan.component === 'mlp' || plan.component === 'attention' ? .045 : .014;
        const activation = .5 + noise(row.text + ':' + phase) * 2;
        after = after.map((v, i) => v - alpha * activation * Math.sin(i * .17 + phase) * scale);
      }
    }
    plan = plans.reduce((a, b) => a.layer < b.layer ? a : b);
    const delta = after.map((v, i) => v - before[i]);
    const deltaNorm = norm(delta), changed = deltaNorm > 1e-12;
    const words = ['巴黎', '柏林', '伦敦', '北京', '猫咪', '科学', '因此', '。'];
    const propagation = (plan.scope === 'token' ? .35 + .65 * (position + 1) / row.count : 1) * (1 + (32 - plan.layer) / 32);
    const scores = words.map((word, k) => {
      const semantic = row.text.includes('法国') && word === '巴黎' || row.text.includes('德国') && word === '柏林' ? 1.7 : 0;
      return (noise(row.text + ':' + word) - .5) * 1.5 + semantic;
    });
    const adjusted = scores.map((v, k) => v + propagation * delta.reduce((sum, x, i) => sum + x * Math.sin(i * .19 + k * .61), 0) / Math.sqrt(arch.hidden) * 1.7);
    const softmax = values => { const clipped = values.map(v => Math.max(-20, Math.min(20, v))); const e = clipped.map(v => Math.exp(v - Math.max(...clipped))); const z = e.reduce((a, b) => a + b, 0); return e.map(v => v / z); };
    const bp = softmax(scores), ap = softmax(adjusted);
    const output = probs => {
      const top = words.map((token, i) => ({ id: i, token, prob: probs[i] })).sort((a, b) => b.prob - a.prob);
      const reply = `模拟回复：根据「${row.text}」，当前示例最倾向续写「${top[0].token}」。这些文字仅用于练习对照操作。`;
      const text = maxNew ? Array.from(reply).slice(0, maxNew).join('') : '';
      return { output: text, generated_tokens: Array.from(text).length, top,
        targets: targets.map(answer => {
          const count = Math.max(1, (answer.match(/[\u3400-\u9fff]|[A-Za-z0-9]+|[^\s]/gu) || []).length);
          const index = words.findIndex(w => answer.includes(w));
          return { text: answer, tokens: count, logprob: Math.log(index < 0 ? .03 : probs[index]) - (count - 1) * 1.8 };
        }) };
    };
    const baseline = output(bp), intervened = output(ap);
    const signal = changed ? deltaNorm * propagation : 0;
    return { type: 'result', simulated: true, id: row.id, label: row.label, text: row.text, position, input_tokens: row.count,
      baseline, intervened, kl: Math.max(0, bp.reduce((sum, v, i) => sum + v * Math.log(v / ap[i]), 0)),
      changes: words.map((token, i) => ({ token, baseline: bp[i], intervened: ap[i], delta: ap[i] - bp[i] })).sort((a, b) => Math.abs(b.delta) - Math.abs(a.delta)),
      layers: layerTypes.map((_, layer) => {
        const baseline = beforeNorm * (.7 + layer / 40);
        return { layer, baseline, intervened: baseline + (layer >= plan.layer ? signal * (layer - plan.layer + 1) / 32 * .2 : 0) };
      }),
      neurons: changed ? Array.from({ length: Math.min(12, 32 - plan.layer) }, (_, i) => {
        const layer = Math.min(31, plan.layer + i + (plan.stage === 'post' ? 1 : 0)), n = featured[i % featured.length];
        const baseline = rawActivation(layer, n, row.tokens[position].text, position), change = signal * Math.sin(i + 1) * .17;
        return { layer, n, baseline, intervened: baseline + change, delta: change };
      }) : [],
    };
  }
  function residualExperiment(body, signal) {
    let rows;
    try { rows = residualTextRows(body); } catch (error) { return json({ error: error.message }, 400); }
    const mode = body.mode === undefined ? 'joint' : body.mode;
    if (!['joint', 'independent'].includes(mode)) return json({ error: 'Choose joint or independent residual interventions.' }, 400);
    const plans = body.plans || [body.plan];
    const groups = mode === 'independent'
      ? plans.map((_, i) => ({ id: `setting-${i + 1}`, plan_indices: [i] }))
      : [{ id: 'joint', plan_indices: plans.map((_, i) => i) }];
    let cancelled = false;
    async function* events() {
      try {
        yield { type: 'start', simulated: true, total: rows.length * groups.length, text_count: rows.length,
          mode, groups, model_key: 'static-demo-residual-v2',
          compute_dtype: saved.settings.compute_dtype, quantization: saved.settings.quantization,
          plan: plans[0], plans, batch: selectedBatch() };
        for (const group of groups) {
          for (const row of rows) {
            if (cancelled || signal?.aborted) return;
            yield { type: 'progress', group_id: group.id, id: row.id, phase: 'baseline' };
            await pause(250);
            yield { type: 'progress', group_id: group.id, id: row.id, phase: 'intervened' };
            await pause(300);
            if (cancelled || signal?.aborted) return;
            yield { ...residualExample(row, rows, group.plan_indices.map(i => plans[i]), body.max_new, body.targets || []),
              group_id: group.id, plan_indices: group.plan_indices };
          }
        }
      } catch (error) { yield { type: 'error', error: error.message }; }
      yield { type: 'done', simulated: true, cancelled };
    }
    const iterator = events(), encoder = new TextEncoder();
    return new Response(new ReadableStream({
      async pull(controller) {
        if (signal?.aborted) { controller.error(new DOMException('Aborted', 'AbortError')); return; }
        const next = await iterator.next();
        if (cancelled) return;
        if (next.done) controller.close();
        else controller.enqueue(encoder.encode('data: ' + JSON.stringify(next.value) + '\n\n'));
      },
      cancel() { cancelled = true; iterator.return(); },
    }), { headers: { 'Content-Type': 'text/event-stream' } });
  }

  window.fetch = async (url, options = {}) => {
    const { pathname: path, searchParams: p } = new URL(String(url), "https://neuron-demo.invalid");
    const body = options.body ? JSON.parse(options.body) : {};
    const layer = Number(p.get("layer") || 0);
    const n = Number(p.get("n") || 0);
    const pos = Number(p.get("pos") || 0);
    if (path.startsWith("/api/neuron-batches")) return batchApi(path, options.method || "GET", body);
    if (path.startsWith("/api/agent/")) return agentApi(path, p, body, options.signal);
    switch (path) {
      case '/api/residual/validate-held': return json({ residuals: body.residuals, simulated: true });
      case '/api/residual/tokens':
        try { return json({ texts: residualTextRows(body), model_key: 'static-demo-residual-v2', simulated: true }); }
        catch (error) { return json({ error: error.message }, 400); }
      case '/api/residual/experiment': return residualExperiment(body, options.signal);
      case "/api/index":
        return json({ model, arch, status: "ready", loading: settings(), layers: Object.fromEntries(layerTypes.map((_, i) => [i, "demo-L" + i])) });
      case "/api/status":
        return json({ status: loading ? "loading" : "ready", switching: loading, active_batch: selectedBatch(), loading: settings() });
      case "/api/model/loading":
        if (options.method === "POST") {
          loading = true;
          await pause(550);
          saved.settings = { ...defaultSettings, ...body };
          active = null;
          inputPieces = [];
          persist();
          loading = false;
        }
        return json(settings());
      case "/api/units": return json(unitsFor(layer));
      case "/api/unit": return json(unit(layer, n, p.get("kind") || "neuron"));
      case "/api/embeddings": return json(embeddings(layer));
      case "/api/rollout": return json(rolloutRows(layer)[Number(p.get("i"))] || { text: "模拟生成样例。" });
      case "/api/layout": return json({ available: true, shape: [nLayers, inter], cell: writesLayout() });
      case "/api/name":
        saved.names[body.layer + "/" + body.n] = body.name;
        persist();
        return json({ ok: true });
      case "/api/lens": return json(lensFor(layer, n));
      case "/api/lens/rjpp": return json(lensFor(layer, n));
      case "/api/search": return json(search(p.get("q") || ""));
      case "/api/word-tokens": return json({ ids: [hash(p.get("word"))] });
      case "/api/connections": return json({ up: links(layer, n, -1, 9), down: links(layer, n, 1, 9) });
      case "/api/hover": {
        const L = lensFor(layer, n);
        return json({
          lens: L.up, lens_neg: L.down, r: L.r_up, r_neg: L.r_down, steer: L.steer_up, steer_neg: L.steer_down,
          con: { up: links(layer, n, -1, 3), down: links(layer, n, 1, 3) },
        });
      }
      case "/api/web": return binary(linkRows(layer, n, Number(p.get("first")) || 5));
      case "/api/web/summary": {
        const L = lensFor(layer, n);
        const count = Number(p.get("first")) || 5;
        return json({
          src: "steer",
          up: { neurons: layer > 0 ? count * 2 : 0, fed_by: layer > 0 ? chips(L.up) : [], held_by: layer > 0 ? chips(L.down) : [] },
          down: { neurons: layer < nLayers - 1 ? count * 2 : 0, toward: layer < nLayers - 1 ? chips(L.up) : [], away: layer < nLayers - 1 ? chips(L.down) : [] },
        });
      }
      case "/api/head": return json(headInfo(layer, Number(p.get("head"))));
      case "/api/neuron/heads": return json(neuronHeads(layer));
      case "/api/exemplars": {
        const L = lensFor(layer, n);
        return json({
          range: { act_max: 3.2, act_min: -3.2 }, "+": exemplar(L.up, 1, layer, n), "-": exemplar(L.down, -1, layer, n),
          "domains+": [{ domain: "demo_text", lift: 2.2 }], "domains-": [{ domain: "demo_text", lift: 1.7 }],
        });
      }
      case "/api/infer": {
        if (!cachedInputs.has(body.text)) cachedInputs.set(body.text, { tokens: pieces(body.text), capture: uid() });
        const cached = cachedInputs.get(body.text);
        inputPieces = cached.tokens; residualCapture = cached.capture;
        if (body.texts) for (const text of cachedInputs.keys()) if (!body.texts.includes(text)) cachedInputs.delete(text);
        return json({ ready: true, capture: residualCapture, tokens: inputPieces, pos: inputPieces.map((_, i) => i) });
      }
      case "/api/infer/cached-texts": return json({ texts: body.texts.map((text, index) => {
        const cached = cachedInputs.get(text);
        return { text, index, ready: !!cached, tokens: cached?.tokens ?? [], positions: cached?.tokens.map((_, i) => i) ?? [] };
      }) });
      case "/api/infer/contexts":
      case "/api/infer/contexts/export": {
        const texts = body.texts || [body.text];
        if (texts.some(text => !cachedInputs.has(text))) return json({ ready: false, contexts: [], message: 'results.commonPending' });
        const savedPieces = inputPieces, savedCapture = residualCapture;
        try {
          const contexts = texts.map((text, index) => {
            const cached = cachedInputs.get(text); inputPieces = cached.tokens; residualCapture = cached.capture;
            const layer = Number(body.layer || 0), n = Number(body.n ?? -1);
            const pos = Number(body.pos ?? -1) >= 0 ? Number(body.pos) : n < 0 ? inputPieces.length - 1
              : inputPieces.reduce((best, token, i) => Math.abs(rawActivation(layer, n, token, i)) > Math.abs(rawActivation(layer, n, inputPieces[best], best)) ? i : best, 0);
            const residual = simulatedResidual(pos, layer, n);
            if (path.endsWith('/export')) return { text, pos, layers: residual.exportLayers };
            delete residual.exportLayers;
            return { text, index, capture: residualCapture, pos, tokens: inputPieces, positions: inputPieces.map((_, i) => i), residual };
          });
          return json(path.endsWith('/export') ? contexts[0] : { ready: true, contexts });
        } finally { inputPieces = savedPieces; residualCapture = savedCapture; }
      }
      case "/api/infer/residual": {
        if (p.get('capture') !== residualCapture) return json({error:'请重新运行文本。'},409);
        const result=simulatedResidual(pos,layer,p.has('n')?Number(p.get('n')):-1);delete result.exportLayers;return json(result);
      }
      case "/api/infer/residual/export": return json({simulated:true,pos,layers:simulatedResidual(pos,0,-1).exportLayers});
      case "/api/infer/common": {
        const saved = inputPieces;
        let common = null;
        try {
          for (const text of body.texts) {
            inputPieces = pieces(text);
            const params = new URLSearchParams({ sign: body.sign, lo: body.lo, hi: body.hi, min: body.min, q: body.q });
            const rows = new Map(activationView(params, Infinity).results.map(row => [`${row.layer}:${row.n}`, row]));
            common = common === null ? rows : new Map([...common].filter(([key]) => rows.has(key))
              .map(([key, row]) => [key, row.score <= rows.get(key).score ? row : rows.get(key)]));
          }
          let rows = [...common.values()];
          if (body.addresses?.length) {
            const allowed = new Set(body.addresses.map(row => `${row.layer}:${row.n}`));
            rows = rows.filter(row => allowed.has(`${row.layer}:${row.n}`));
          }
          rows.sort((a, b) => b.score - a.score);
          return json({ ready: true, results: rows.slice(0, 300), total: rows.length, texts: body.texts.length });
        } finally { inputPieces = saved; }
      }
      case "/api/infer/view": {
        const view = activationView(p);
        if (!p.has("addresses")) {
          const map = new Uint16Array(nLayers * inter);
          for (const row of view.results) map[row.layer * inter + row.n] = float16(row.act);
          view.map = base64(new Uint8Array(map.buffer));
          view.shape = [nLayers, inter];
        }
        return json(view);
      }
      case "/api/infer/heads": return json({
        layers: layerTypes.map((type, l) => Array.from({ length: type === "full_attention" ? 16 : 32 },
          (_, i) => noise(l + ":" + i + ":" + pos + ":" + inputPieces[pos]) ** 4)),
      });
      case "/api/head/row": return json({
        units: inputPieces.map((word, i) => i <= pos ? noise(word + ":" + i + ":" + layer) - 0.42 : 0),
      });
      case "/api/infer/paths": {
        const top = activationView(p).results.slice(0, 14).sort((a, b) => a.layer - b.layer);
        return binary(top.slice(1).map((r, i) => [top[i].layer, top[i].n, r.layer, r.n, (r.sign === "+" ? 1 : -1) * Math.round(r.score * 140), 1]));
      }
      case "/api/steer": return steer(body, options.signal);
      case "/api/reads": return json({
        results: body.neurons.map(({ layer, n }) => ({
          layer, n, units: pieces(body.text).map((text, i) => {
            const act = rawActivation(layer, n, text, i);
            return { text, act, pieces: [act] };
          }),
        })),
      });
      default: return json({ error: "这个操作没有模拟数据。" }, 404);
    }
  };
})();
