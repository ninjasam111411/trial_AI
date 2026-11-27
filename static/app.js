// Neural chat app
const $ = (sel) => document.querySelector(sel);
const thread = $("#thread");
const messagesEl = $("#messages");
const input = $("#input");
const send = $("#send");

const SOURCES = {
  knowledge: "knowledge",
  neural: "neural net",
  calculator: "calculator",
  spelling: "spelling",
  learned: "learned from you",
  fallback: "doesn't know yet",
};

let currentId = null;
let busy = false;

// ---------- small helpers ----------
const store = {
  get(k) { try { return localStorage.getItem(k); } catch { return null; } },
  set(k, v) { try { localStorage.setItem(k, v); } catch {} },
};

function el(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined) e.textContent = text;
  return e;
}

async function api(path, options = {}) {
  const headers = { "Content-Type": "application/json" };
  const key = store.get("neural-admin-key");
  if (key) headers["X-Admin-Key"] = key;
  const res = await fetch(path, { ...options, headers });
  const data = await res.json().catch(() => ({}));
  if (res.status === 403 && /password/i.test(data.error || "")) {
    const entered = prompt("This Neural is protected. Enter the admin password:");
    if (entered) { store.set("neural-admin-key", entered); return api(path, options); }
  }
  if (!res.ok) throw new Error(data.error || "Request failed");
  return data;
}

// Safe, tiny markdown: paragraphs, lists, **bold**, `code`, ```blocks```.
function escapeHtml(s) {
  return s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}
function inline(s) {
  return s.replace(/`([^`]+)`/g, "<code>$1</code>").replace(/\*\*([^*]+)\*\*/g, "<b>$1</b>");
}
// Code blocks: language label, simple colors, Copy and Save-as-file buttons.
const CODE_EXT = { python: "py", py: "py", javascript: "js", js: "js", html: "html", css: "css",
  lua: "lua", luau: "lua", json: "json", bash: "sh", sh: "sh", bat: "bat", java: "java",
  c: "c", cpp: "cpp", sql: "sql", text: "txt", txt: "txt" };
const KEYWORDS = new Set(("def return if elif else for while in import from as class try except finally " +
  "with lambda pass break continue and or not is None True False print local function end then do " +
  "repeat until nil const let var new this async await switch case default typeof true false null " +
  "public private static void int float string bool").split(" "));

function guessLang(code) {
  if (/^\s*(def |import |print\()/m.test(code)) return "python";
  if (/\b(local |game:GetService|\bend\b)/.test(code)) return "lua";
  if (/<\/?[a-z][\s\S]*>/i.test(code)) return "html";
  if (/\b(const|let|function)\b/.test(code)) return "javascript";
  return "text";
}

function highlight(code, lang) {
  if (lang === "text" || lang === "html") return escapeHtml(code);
  const comment = lang === "lua" ? "--[^\\n]*" : lang === "python" ? "#[^\\n]*" : "\\/\\/[^\\n]*";
  const re = new RegExp(`(${comment})|("(?:[^"\\\\\\n]|\\\\.)*"|'(?:[^'\\\\\\n]|\\\\.)*')|(\\b\\d+(?:\\.\\d+)?\\b)|([A-Za-z_]\\w*)`, "g");
  let out = "", last = 0, m;
  while ((m = re.exec(code))) {
    out += escapeHtml(code.slice(last, m.index));
    const [tok, c, s, n, w] = m;
    if (c) out += `<span class="tk-c">${escapeHtml(tok)}</span>`;
    else if (s) out += `<span class="tk-s">${escapeHtml(tok)}</span>`;
    else if (n) out += `<span class="tk-n">${tok}</span>`;
    else if (KEYWORDS.has(w)) out += `<span class="tk-k">${tok}</span>`;
    else out += escapeHtml(tok);
    last = re.lastIndex;
  }
  return out + escapeHtml(code.slice(last));
}

function codeBlock(lang, code) {
  code = code.replace(/\n$/, "");
  lang = (lang || guessLang(code)).toLowerCase();
  return `<div class="code"><div class="code-head"><span>${escapeHtml(lang)}</span>` +
    `<button type="button" data-code="copy">⧉ Copy</button>` +
    `<button type="button" data-code="save" data-ext="${CODE_EXT[lang] || "txt"}">⬇ Save file</button></div>` +
    `<pre><code>${highlight(code, lang)}</code></pre></div>`;
}

function renderMarkdown(text) {
  let html = "", last = 0, m;
  const fence = /```(\w*)\n?([\s\S]*?)```/g;
  while ((m = fence.exec(text))) {
    html += renderText(text.slice(last, m.index)) + codeBlock(m[1], m[2]);
    last = fence.lastIndex;
  }
  return html + renderText(text.slice(last));
}

function renderText(part) {
  let html = "", list = null;
  for (const line of escapeHtml(part).split("\n")) {
    const ul = line.match(/^\s*[-*•]\s+(.*)/), ol = line.match(/^\s*\d+[.)]\s+(.*)/);
    const kind = ul ? "ul" : ol ? "ol" : null;
    if (list && kind !== list) { html += `</${list}>`; list = null; }
    if (kind) {
      if (!list) { html += `<${kind}>`; list = kind; }
      html += `<li>${inline((ul || ol)[1])}</li>`;
    } else if (line.trim()) html += `<p>${inline(line)}</p>`;
  }
  if (list) html += `</${list}>`;
  return html;
}

function downloadText(name, content) {
  const url = URL.createObjectURL(new Blob([content], { type: "text/plain" }));
  const a = el("a"); a.href = url; a.download = name;
  document.body.append(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 2000);
}

document.addEventListener("click", (e) => {
  const btn = e.target.closest("[data-code]");
  if (!btn) return;
  const code = btn.closest(".code").querySelector("code").textContent;
  if (btn.dataset.code === "copy") {
    navigator.clipboard?.writeText(code);
    toast("Code copied");
  } else {
    downloadText(`neural-code.${btn.dataset.ext}`, code);
    toast(`Saved neural-code.${btn.dataset.ext}`, "good");
  }
});

// Little message that pops up at the bottom of the screen.
function toast(text, kind = "") {
  let holder = document.getElementById("toasts");
  if (!holder) { holder = el("div"); holder.id = "toasts"; document.body.append(holder); }
  const t = el("div", "toast " + kind, text);
  holder.append(t);
  setTimeout(() => t.classList.add("out"), 2600);
  setTimeout(() => t.remove(), 3000);
}

function scrollDown() { messagesEl.scrollTop = messagesEl.scrollHeight; }

// ---------- thinking orbs ----------
const orbs = new Set();
function makeOrb(canvas, state, size) {
  if (!window.ThinkingOrbs) return null;
  const orb = ThinkingOrbs.createOrb(canvas, {
    state, size, dark: document.documentElement.dataset.theme !== "light",
  });
  orbs.add(orb);
  return orb;
}
function dropOrb(orb) { if (orb) { orb.destroy(); orbs.delete(orb); } }

// ---------- theme ----------
function applyTheme(t) {
  document.documentElement.dataset.theme = t;
  $("#theme").textContent = t === "dark" ? "☀️ Light mode" : "🌙 Dark mode";
  store.set("neural-theme", t);
  orbs.forEach((o) => o.setDark(t !== "light"));
}
applyTheme(store.get("neural-theme") || "dark");
$("#theme").onclick = () =>
  applyTheme(document.documentElement.dataset.theme === "dark" ? "light" : "dark");

// ---------- sidebar ----------
$("#open-side").onclick = () => document.body.classList.add("side-open");
$("#close-side").onclick = $("#scrim").onclick = () => document.body.classList.remove("side-open");

async function loadChatList() {
  const chats = await api("/api/conversations");
  const list = $("#chat-list");
  list.innerHTML = "";
  if (!chats.length) list.append(el("div", "muted small", "  No chats yet"));
  for (const c of chats) {
    const item = el("div", "chat-item" + (c.id === currentId ? " active" : ""));
    item.append(el("span", "", c.title));
    const del = el("button", "del", "🗑");
    del.title = "Delete chat";
    del.onclick = async (e) => {
      e.stopPropagation();
      if (!confirm("Delete this chat?")) return;
      await api(`/api/conversations/${c.id}`, { method: "DELETE" });
      if (c.id === currentId) newChat();
      loadChatList();
    };
    item.append(del);
    item.onclick = () => openChat(c.id);
    list.append(item);
  }
}

function newChat() {
  currentId = null;
  thread.innerHTML = "";
  showAttached(null, true);
  $("#empty").hidden = false;
  $("#title").textContent = "New chat";
  document.body.classList.remove("side-open");
  loadChatList();
  input.focus();
}
$("#new-chat").onclick = newChat;

async function openChat(id) {
  const convo = await api(`/api/conversations/${id}`);
  currentId = id;
  thread.innerHTML = "";
  showAttached(null, true);
  (convo.file_names || []).forEach((n) => showAttached(n));
  $("#empty").hidden = convo.messages.length > 0;
  $("#title").textContent = convo.title;
  let lastUser = "";
  for (const m of convo.messages) {
    if (m.role === "User") { addUser(m.message, m.corrected); lastUser = m.message; }
    else addBot(m.message, m.source, m.id, lastUser, m.rating, false,
                { agent: m.agent, options: m.options, file: m.file });
  }
  document.body.classList.remove("side-open");
  loadChatList();
  scrollDown();
}

// ---------- messages ----------
function addReadAs(msg, corrected) {
  if (!corrected) return;
  const note = el("div", "read-as", "✎ Read as: ");
  note.append(el("b", "", corrected));
  msg.append(note);
}

function addUser(text, corrected) {
  const msg = el("div", "msg user");
  msg.append(el("div", "bubble", text));
  addReadAs(msg, corrected);
  thread.append(msg);
  scrollDown();
  return msg;
}

function agentChip(agentId, consulted) {
  const a = agentsById[agentId];
  if (!a) return null;
  const chip = el("span", "chip agent", `${a.emoji} ${a.name}`);
  chip.style.color = a.color;
  chip.style.borderColor = a.color + "66";
  const asked = (consulted || []).map((c) => agentsById[c.id] ?
    `${agentsById[c.id].name} ${Math.round(c.score * 100)}%` : "").filter(Boolean);
  chip.title = asked.length ? `Neural asked: ${asked.join(", ")}` : `${a.name} agent`;
  chip.onclick = () => openAgents(agentId);
  return chip;
}

function addBot(text, source, id, question, rating, animate = false, extra = {}) {
  const msg = el("div", "msg bot");
  msg.append(el("div", "logo"));
  const body = el("div", "body");
  const bubble = el("div", "bubble");
  body.append(bubble);
  msg.append(body);
  thread.append(msg);

  const meta = el("div", "meta");
  const chip = agentChip(extra.agent, extra.consulted);
  if (chip) meta.append(chip);
  if (SOURCES[source]) meta.append(el("span", "chip " + source, SOURCES[source]));

  // Question card: tap an option to answer.
  if (extra.options && extra.options.length) {
    const card = el("div", "options");
    for (const opt of extra.options) {
      const b = el("button", "option", opt.label);
      b.type = "button";
      b.onclick = () => {
        card.querySelectorAll("button").forEach((x) => (x.disabled = true));
        b.classList.add("picked");
        if (opt.send) sendMessage(opt.send); else input.focus();
      };
      card.append(b);
    }
    body.append(card);
  }
  // A file Neural made for you.
  if (extra.file) {
    const f = el("div", "file-card");
    f.append(el("span", "file-icon", "📄"), el("span", "file-name", extra.file.name));
    const dl = el("button", "primary", "⬇ Download");
    dl.type = "button";
    dl.onclick = () => { downloadText(extra.file.name, extra.file.content); toast(`Saved ${extra.file.name}`, "good"); };
    f.append(dl);
    body.append(f);
  }

  if (id) {
    const up = el("button", "act good", "👍");
    const down = el("button", "act bad", "👎");
    up.title = "Good answer"; down.title = "Bad answer - teach me";
    const copy = el("button", "act", "⧉");
    copy.title = "Copy";

    const note = el("span", "rate-note");
    const setRating = (r) => {
      up.classList.toggle("on", r === "good");
      down.classList.toggle("on", r === "bad");
    };
    setRating(rating);
    const pop = (btn) => { btn.classList.remove("pop"); void btn.offsetWidth; btn.classList.add("pop"); };
    const flash = (text, cls) => {
      note.textContent = text;
      note.className = "rate-note show " + cls;
      clearTimeout(note._t);
      note._t = setTimeout(() => note.classList.remove("show"), 2500);
    };
    const send = async (r) => {
      try {
        await api("/api/feedback", { method: "POST", body: JSON.stringify({ id, rating: r }) });
        return true;
      } catch (err) { toast("Couldn't save that: " + err.message, "bad"); return false; }
    };

    up.onclick = async () => {
      const r = up.classList.contains("on") ? null : "good";
      setRating(r);
      pop(up);
      body.querySelector(".teach")?.remove();
      if (await send(r)) {
        if (r) { flash("Thanks! I'll learn from this one.", "good"); toast("👍 Saved. This answer goes into my next training.", "good"); }
        else flash("Rating removed", "");
      }
    };
    down.onclick = async () => {
      setRating("bad");
      pop(down);
      if (await send("bad")) {
        flash("Got it. Teach me the right answer below.", "bad");
        if (!body.querySelector(".teach")) showTeach(body, question, id);
        else body.querySelector(".teach textarea")?.focus();
      }
    };
    copy.onclick = () => {
      navigator.clipboard?.writeText(text);
      pop(copy);
      toast("Copied to clipboard");
    };
    const speak = el("button", "act", "🔊");
    speak.title = "Read out loud";
    speak.onclick = () => { pop(speak); readAloud(text); };
    meta.append(up, down, copy, speak, note);
  }
  body.append(meta);

  if (animate) typeOut(bubble, text); else bubble.innerHTML = renderMarkdown(text);
  if (animate && store.get("neural-read-aloud") === "on") readAloud(text);
  scrollDown();
}

function typeOut(node, text) {
  let i = 0;
  const step = Math.max(1, Math.round(text.length / 60));
  const tick = () => {
    i = Math.min(text.length, i + step);
    node.textContent = text.slice(0, i);
    scrollDown();
    if (i < text.length) requestAnimationFrame(tick);
    else node.innerHTML = renderMarkdown(text);
  };
  tick();
}

function showTeach(body, question, id) {
  const box = el("div", "teach");
  const label = el("label", "", `What should I have said to "${question}"? I'll learn it right away.`);
  const area = el("textarea");
  area.placeholder = "Type the right answer and I'll learn it...";
  const row = el("div", "row");
  const cancel = el("button", "ghost", "Cancel");
  const save = el("button", "primary", "Teach Neural");
  cancel.type = save.type = "button";
  cancel.onclick = () => box.remove();
  save.onclick = async () => {
    const answer = area.value.trim();
    if (!answer) return area.focus();
    save.disabled = true;
    try {
      await api("/api/teach", { method: "POST", body: JSON.stringify({ question, answer, id }) });
      box.innerHTML = "";
      box.classList.add("learned");
      box.append(el("div", "done", "✓ Learned! Ask me again and I'll give that answer."));
      toast("🧠 New answer learned!", "good");
    } catch (err) {
      save.disabled = false;
      alert(err.message);
    }
  };
  row.append(cancel, save);
  box.append(label, area, row);
  body.append(box);
  area.focus();
  scrollDown();
}

// While Neural thinks, an orb animates and changes as it works:
// searching its knowledge first, then composing a reply.
function showTyping(firstLabel) {
  const msg = el("div", "msg bot");
  const box = el("div", "thinking");
  const canvas = el("canvas");
  const a = targetAgent && agentsById[targetAgent];
  const label = el("span", "label", firstLabel ||
    (a ? `Asking the ${a.name} agent...` : "Neural is asking the agents..."));
  box.append(canvas, label);
  msg.append(box);
  thread.append(msg);
  const orb = makeOrb(canvas, "searching", 34);
  const steps = [[700, "composing", "Writing a reply..."], [2500, "weaving", "Thinking it through..."],
                 [6000, "breathing", "Almost there..."]];
  const timers = steps.map(([ms, state, text]) => setTimeout(() => {
    orb?.setState(state);
    label.textContent = text;
    label.style.animation = "none"; void label.offsetWidth; label.style.animation = "";
  }, ms));
  msg.stop = () => { timers.forEach(clearTimeout); dropOrb(orb); msg.remove(); };
  scrollDown();
  return msg;
}

// ---------- sending ----------
async function sendMessage(text) {
  if (busy || !text.trim()) return;
  busy = true;
  send.disabled = true;
  $("#empty").hidden = true;
  input.value = "";
  autoGrow();

  const userMsg = addUser(text);  // show your message right away
  const typing = showTyping();
  const started = Date.now();

  try {
    const data = await api("/api/chat", {
      method: "POST",
      body: JSON.stringify({ message: text, conversation_id: currentId, agent: targetAgent,
                             style: store.get("neural-style") === "plain" ? "plain" : "casual" }),
    });
    await new Promise((r) => setTimeout(r, Math.max(0, 650 - (Date.now() - started))));
    typing.stop();
    addReadAs(userMsg, data.corrected);
    addBot(data.answer, data.source, data.id, text, null, true,
           { agent: data.agent, consulted: data.consulted, options: data.options, file: data.file });
    pulseAgents(data.agent, data.consulted);
    if (!currentId) { currentId = data.conversation_id; }
    loadChatList().then(() => {
      const active = document.querySelector(".chat-item.active span");
      if (active) $("#title").textContent = active.textContent;
    });
  } catch (err) {
    typing.stop();

    addBot("⚠ " + (err.message || "Could not reach Neural. Is app.py running?"), "fallback");
  } finally {
    busy = false;
    send.disabled = !input.value.trim();
    input.focus();
  }
}

function autoGrow() {
  input.style.height = "auto";
  input.style.height = Math.min(input.scrollHeight, 180) + "px";
  send.disabled = busy || !input.value.trim();
}
input.addEventListener("input", autoGrow);
input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    sendMessage(input.value);
  }
});
$("#composer").onsubmit = (e) => { e.preventDefault(); sendMessage(input.value); };
document.querySelectorAll(".suggestions button").forEach((b) => {
  b.onclick = () => sendMessage(b.textContent);
});
send.disabled = true;

// ---------- brain panel ----------
const modal = $("#brain-modal");
let trainOrb = null;
let pollTimer = null;

function fmt(n) { return typeof n === "number" ? n.toLocaleString() : n; }

async function refreshStats() {
  const s = await api("/api/stats");
  const stats = [
    [s.topics, "built-in topics"],
    [s.taught_answers, "answers taught"],
    [s.messages_logged, "messages seen"],
    [s.rated_good, "rated 👍"],
    [s.typos_learned, "typos learned"],
    [s.words_from_chats, "new words from chats"],
  ];
  $("#stats").innerHTML = "";
  for (const [n, label] of stats) {
    const box = el("div", "stat");
    box.append(el("b", "", fmt(n)), el("span", "", label));
    $("#stats").append(box);
  }

  const card = $("#model-card");
  card.innerHTML = "";
  if (s.model) {
    const n = s.model.params;
    const size = n >= 1e9 ? (n / 1e9).toFixed(1) + "B" : (n / 1e6).toFixed(1) + "M";
    const num = el("div", "big-num", size);
    const txt = el("div", "");
    txt.append(el("div", "", "parameters, trained from scratch"));
    txt.append(el("div", "muted", `${s.model.layers} layers · width ${s.model.dim} · reads ${s.model.context} tokens · vocab ${fmt(s.model.vocab)}`));
    card.append(num, txt);
  } else if (s.model_error) {
    card.append(el("div", "", "The neural network couldn't load: " + s.model_error +
                   ". Try Train a new model, or send this message to get it fixed."));
  } else if (s.training.running) {
    card.append(el("div", "", "Training your first neural network... You can keep chatting meanwhile."));
  } else {
    card.append(el("div", "", "No neural network yet. Train one below."));
  }

  const lt = s.last_training;
  $("#train-info").textContent = !s.model_ready
    ? "The neural network isn't trained yet. It still answers from its knowledge, but training it lets it handle new questions."
    : lt
      ? `Last trained ${new Date(lt.time * 1000).toLocaleString()} on ${fmt(lt.tokens || 0)} tokens${lt.examples ? ` from ${fmt(lt.examples)} chats` : ""} (loss ${lt.val_loss}, lower is better).`
      : "Neural network is trained and ready.";
  $("#retrain").textContent = s.model_ready ? "Retrain with what I've learned" : "Train the neural network";
  $("#fresh-box").hidden = !s.model_ready;

  const t = s.training;
  const status = $("#model-status");
  status.className = "status " + (t.running ? "busy" : s.model_ready ? "ready" : "");
  status.textContent = t.running ? "Training..." : s.model_ready ? "Model ready" : "Knowledge only";

  $("#retrain").disabled = $("#fresh").disabled = t.running;
  $("#train-live").hidden = !t.running;
  if (t.running && !trainOrb) trainOrb = makeOrb($("#train-orb"), "weaving", 28);
  if (!t.running && trainOrb) { dropOrb(trainOrb); trainOrb = null; }
  $("#train-progress div").style.width = Math.round((t.progress || 0) * 100) + "%";
  $("#train-stage").textContent = t.error
    ? "Training failed: " + t.error
    : t.running ? `${t.stage}... ${t.log.at(-1) || ""}`
    : t.stage === "Done" ? "✓ Training finished. Neural just got smarter." : "";

  clearTimeout(pollTimer);
  if (t.running) pollTimer = setTimeout(refreshStats, 2000);
}

$("#open-brain").onclick = () => {
  modal.hidden = false;
  document.body.classList.remove("side-open");
  refreshStats();
  refreshUpdate();
};
modal.addEventListener("click", (e) => {
  if (e.target === modal || e.target.closest("[data-close]")) modal.hidden = true;
});
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape") { modal.hidden = true; $("#agents-modal").hidden = true; graph.running = false; }
});

async function startTraining(fresh) {
  const preset = $("#preset").value;
  if (fresh && !confirm(`Replace the current neural network with a new ${preset} one trained from scratch?`)) return;
  try {
    await api("/api/retrain", { method: "POST", body: JSON.stringify({ fresh, preset }) });
  } catch (err) { alert(err.message); }
  refreshStats();
}
$("#retrain").onclick = () => startTraining(false);
$("#fresh").onclick = () => startTraining(true);

// ---------- updates from GitHub ----------
const updateBar = $("#update-bar");
let updateOrb = null;
let updateState = null;

function timeAgo(t) {
  const s = Math.round(Date.now() / 1000 - t);
  if (s < 60) return "just now";
  if (s < 3600) return Math.round(s / 60) + " min ago";
  if (s < 86400) return Math.round(s / 3600) + " hours ago";
  return Math.round(s / 86400) + " days ago";
}

function showUpdate(u) {
  updateState = u;
  const dismissed = store.get("neural-update-dismissed");
  updateBar.hidden = !(u.applying || (u.available && dismissed !== u.commit));
  const n = u.changed.length;
  $("#update-msg").textContent = u.message ? `"${u.message}" · ${n} file${n === 1 ? "" : "s"} changed` : "";
  $("#update-stage").textContent = u.applying ? `${u.stage}...` : (u.error && u.available ? u.error : "");
  $("#update-now").disabled = $("#update-later").disabled = u.applying;
  $("#update-icon").hidden = !!u.applying;
  $("#update-orb").hidden = !u.applying;
  if (u.applying && !updateOrb) updateOrb = makeOrb($("#update-orb"), "connecting", 28);

  // Brain panel
  let info;
  if (!u.enabled) info = u.is_git ? "This copy uses Git: update it with git pull." : "Automatic updates are turned off.";
  else if (u.checking) info = "Checking GitHub...";
  else if (u.available) info = `Update available: "${u.message}".`;
  else if (u.online === false) info = u.error || "Offline. Neural works fine without the internet.";
  else if (u.last_check) info = `Up to date. Checked ${timeAgo(u.last_check)}.`;
  else info = "Will check for updates when you're online.";
  if (u.installed) info += ` Installed version: ${u.installed}.`;
  $("#version-info").textContent = info;
  $("#check-updates").hidden = !u.enabled;
  $("#panel-update").hidden = !(u.available && !u.applying);
  $("#undo-update").hidden = !u.can_undo;
}

async function refreshUpdate() {
  try { showUpdate(await api("/api/update")); } catch {}
}

async function waitForRestart(bootId) {
  $("#update-stage").textContent = "Restarting...";
  for (let i = 0; i < 120; i++) {
    await new Promise((r) => setTimeout(r, 1500));
    try {
      const u = await (await fetch("/api/update")).json();
      if (u.boot_id !== bootId) { location.reload(); return; }
    } catch {}  // server is restarting
  }
  $("#update-stage").textContent = "Neural didn't come back. Start it again with start.bat / start.sh.";
}

async function followUpdate() {
  const bootId = updateState.boot_id;
  while (true) {
    await new Promise((r) => setTimeout(r, 1000));
    let u;
    try { u = await (await fetch("/api/update")).json(); } catch { return waitForRestart(bootId); }
    showUpdate(u);
    if (u.boot_id !== bootId) return location.reload();
    if (u.stage === "Restarting") return waitForRestart(bootId);
    if (!u.applying) return;  // failed; message is shown
  }
}

async function applyUpdate() {
  try {
    await api("/api/update/apply", { method: "POST" });
    modal.hidden = true;
    store.set("neural-update-dismissed", "");
    updateBar.hidden = false;
    toast("Updating Neural...");
    followUpdate();
  } catch (err) {
    updateBar.hidden = false;
    $("#update-stage").textContent = err.message;
    toast("Update failed: " + err.message, "bad");
  }
}
$("#update-now").onclick = applyUpdate;
$("#panel-update").onclick = applyUpdate;
$("#update-later").onclick = () => {
  store.set("neural-update-dismissed", updateState.commit);
  updateBar.hidden = true;
};
$("#check-updates").onclick = async () => {
  $("#version-info").textContent = "Checking GitHub...";
  store.set("neural-update-dismissed", "");
  showUpdate(await api("/api/update/check", { method: "POST" }));
};
$("#undo-update").onclick = async () => {
  if (!confirm("Go back to the version from before the last update?")) return;
  try {
    await api("/api/update/undo", { method: "POST" });
    modal.hidden = true;
    updateBar.hidden = false;
    $("#update-msg").textContent = "Undoing the last update.";
    waitForRestart(updateState.boot_id);
  } catch (err) { alert(err.message); }
};

// ---------- answer style ----------
$("#casual").checked = store.get("neural-style") !== "plain";
$("#casual").onchange = (e) => {
  store.set("neural-style", e.target.checked ? "casual" : "plain");
  toast(e.target.checked ? "Casual answers on" : "Textbook-style answers on");
};

// ---------- agents ----------
let agents = [];
let agentsById = {};
let targetAgent = null;  // null = Neural picks

async function loadAgents() {
  try {
    const s = await api("/api/stats");
    agents = s.agents || [];
    agentsById = Object.fromEntries(agents.map((a) => [a.id, a]));
  } catch {}
}

function setTarget(id) {
  targetAgent = id && id !== "neural" ? id : null;
  const a = targetAgent && agentsById[targetAgent];
  const pill = $("#target");
  pill.textContent = a ? `${a.emoji} Asking ${a.name} directly ✕` : "🧠 Neural · picks the best agent ▾";
  pill.classList.toggle("set", !!a);
  if (a) pill.style.setProperty("--agent", a.color); else pill.style.removeProperty("--agent");
  input.placeholder = a ? `Ask the ${a.name} agent...` : "Message Neural...";
  $("#target-menu").hidden = true;
}

$("#target").onclick = () => {
  if (targetAgent) return setTarget(null);
  const menu = $("#target-menu");
  menu.innerHTML = "";
  for (const a of agents) {
    if (a.id === "files" || a.id === "memory") continue;
    const b = el("button", "", `${a.emoji} ${a.id === "neural" ? "Neural (picks for you)" : a.name}`);
    b.type = "button";
    b.title = a.about;
    b.onclick = () => { setTarget(a.id); input.focus(); };
    menu.append(b);
  }
  menu.hidden = !menu.hidden;
};
document.addEventListener("click", (e) => {
  if (!e.target.closest("#target, #target-menu")) $("#target-menu").hidden = true;
});

// ---- Agent graph (bubbles, like Obsidian's graph view) ----
const graphModal = $("#agents-modal");
const graph = { nodes: [], running: false, drag: null, hover: null, pulse: null, selected: null };

function buildGraph() {
  const cv = $("#agent-graph");
  const w = cv.clientWidth, h = cv.clientHeight;
  const old = Object.fromEntries(graph.nodes.map((n) => [n.id, n]));
  graph.nodes = agents.map((a, i) => {
    const angle = (i / Math.max(1, agents.length - 1)) * Math.PI * 2;
    const prev = old[a.id];
    return {
      ...a,
      r: a.id === "neural" ? 34 : Math.min(44, 15 + 2.6 * Math.sqrt(a.topics || 1)),
      x: prev ? prev.x : a.id === "neural" ? w / 2 : w / 2 + Math.cos(angle) * 160,
      y: prev ? prev.y : a.id === "neural" ? h / 2 : h / 2 + Math.sin(angle) * 130,
      vx: 0, vy: 0,
    };
  });
}

function stepGraph(w, h) {
  const ns = graph.nodes;
  const center = ns.find((n) => n.id === "neural");
  for (const a of ns) {
    for (const b of ns) {
      if (a === b) continue;
      const dx = a.x - b.x, dy = a.y - b.y, d2 = dx * dx + dy * dy + 0.01, d = Math.sqrt(d2);
      const push = 2600 / d2 + (d < a.r + b.r + 18 ? (a.r + b.r + 18 - d) * 0.08 : 0);
      a.vx += (dx / d) * push; a.vy += (dy / d) * push;
    }
    if (center && a !== center) {  // spring to Neural
      const dx = center.x - a.x, dy = center.y - a.y, d = Math.hypot(dx, dy) || 1;
      const pull = (d - (110 + a.r * 1.5)) * 0.012;
      a.vx += (dx / d) * pull; a.vy += (dy / d) * pull;
    }
    a.vx += (w / 2 - a.x) * 0.002; a.vy += (h / 2 - a.y) * 0.002;  // stay centered
  }
  for (const n of ns) {
    if (n === graph.drag) { n.vx = n.vy = 0; continue; }
    n.vx *= 0.82; n.vy *= 0.82;
    n.x = Math.max(n.r + 4, Math.min(w - n.r - 4, n.x + n.vx));
    n.y = Math.max(n.r + 4, Math.min(h - n.r - 22, n.y + n.vy));
  }
}

function drawGraph() {
  const cv = $("#agent-graph");
  const dpr = Math.min(2, devicePixelRatio || 1);
  const w = cv.clientWidth, h = cv.clientHeight;
  if (cv.width !== Math.round(w * dpr)) { cv.width = Math.round(w * dpr); cv.height = Math.round(h * dpr); }
  const ctx = cv.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  stepGraph(w, h);
  const css = getComputedStyle(document.documentElement);
  const text = css.getPropertyValue("--text").trim(), muted = css.getPropertyValue("--muted").trim();
  const center = graph.nodes.find((n) => n.id === "neural");
  const now = performance.now();
  const pulse = graph.pulse && now - graph.pulse.start < 2600 ? graph.pulse : null;

  for (const n of graph.nodes) {  // links
    if (!center || n === center) continue;
    const active = pulse && pulse.targets.includes(n.id);
    const focus = graph.hover === n || graph.selected === n.id;
    ctx.strokeStyle = active || focus ? n.color : muted;
    ctx.globalAlpha = active || focus ? 0.9 : 0.25;
    ctx.lineWidth = active ? 2.5 : 1.2;
    ctx.beginPath(); ctx.moveTo(center.x, center.y); ctx.lineTo(n.x, n.y); ctx.stroke();
    if (active) {  // a dot traveling from Neural to the agent it called
      const t = ((now - pulse.start) % 900) / 900;
      ctx.globalAlpha = 1; ctx.fillStyle = n.color;
      ctx.beginPath(); ctx.arc(center.x + (n.x - center.x) * t, center.y + (n.y - center.y) * t, 4, 0, 7); ctx.fill();
    }
  }
  ctx.globalAlpha = 1;
  for (const n of graph.nodes) {  // bubbles
    const glow = graph.hover === n || graph.selected === n.id || (pulse && pulse.answered === n.id);
    const breathe = n.id === "neural" ? Math.sin(now / 700) * 1.5 : 0;
    ctx.shadowColor = n.color; ctx.shadowBlur = glow ? 28 : 10;
    const grad = ctx.createRadialGradient(n.x - n.r / 3, n.y - n.r / 3, 2, n.x, n.y, n.r + breathe);
    grad.addColorStop(0, n.color + "ff"); grad.addColorStop(1, n.color + "88");
    ctx.fillStyle = grad;
    ctx.beginPath(); ctx.arc(n.x, n.y, n.r + breathe, 0, 7); ctx.fill();
    ctx.shadowBlur = 0;
    ctx.font = `${Math.round(n.r * 0.8)}px system-ui, sans-serif`;
    ctx.textAlign = "center"; ctx.textBaseline = "middle";
    ctx.fillText(n.emoji, n.x, n.y + 1);
    ctx.font = "600 12.5px system-ui, sans-serif"; ctx.fillStyle = glow ? text : muted;
    ctx.fillText(n.name, n.x, n.y + n.r + 12);
  }
  if (graph.running) requestAnimationFrame(drawGraph);
}

function nodeAt(e) {
  const r = $("#agent-graph").getBoundingClientRect();
  const x = e.clientX - r.left, y = e.clientY - r.top;
  return { x, y, node: [...graph.nodes].reverse().find((n) => Math.hypot(n.x - x, n.y - y) <= n.r + 4) };
}

(() => {
  const cv = $("#agent-graph");
  let moved = false, start = null;
  cv.addEventListener("pointerdown", (e) => {
    const hit = nodeAt(e);
    if (hit.node) { graph.drag = hit.node; moved = false; start = hit; cv.setPointerCapture(e.pointerId); }
  });
  cv.addEventListener("pointermove", (e) => {
    const hit = nodeAt(e);
    if (graph.drag) {
      if (Math.hypot(hit.x - start.x, hit.y - start.y) > 3) moved = true;
      graph.drag.x = hit.x; graph.drag.y = hit.y;
    }
    graph.hover = hit.node || null;
    cv.style.cursor = hit.node ? "pointer" : "default";
  });
  cv.addEventListener("pointerup", () => {
    if (graph.drag && !moved) showAgentInfo(graph.drag.id);
    graph.drag = null;
  });
  cv.addEventListener("pointerleave", () => { graph.hover = null; });
})();

function showAgentInfo(id) {
  const a = agentsById[id];
  if (!a) return;
  graph.selected = id;
  const box = $("#agent-info");
  box.innerHTML = "";
  box.style.borderColor = a.color + "88";
  const head = el("div", "agent-head");
  head.append(el("span", "agent-emoji", a.emoji));
  const t = el("div");
  t.append(el("b", "", a.id === "neural" ? "Neural (main agent)" : `${a.name} agent`));
  t.append(el("div", "muted small", a.id === "neural" ? "Calls the other agents" :
    a.id === "files" ? "Knows the files you attach" :
    a.id === "memory" ? `${a.topics} things people taught it` : `${a.topics} topics`));
  head.append(t);
  box.append(head, el("p", "", a.about));
  if (!["files", "memory"].includes(id)) {
    const ask = el("button", "primary", id === "neural" ? "Let Neural pick (default)" : `Ask ${a.name} directly`);
    ask.onclick = () => { setTarget(id); graphModal.hidden = true; graph.running = false; input.focus(); };
    box.append(ask);
  }
}

async function openAgents(focusId) {
  await loadAgents();
  graphModal.hidden = false;
  document.body.classList.remove("side-open");
  buildGraph();
  graph.selected = null;
  $("#agent-info").innerHTML = "<span class='muted small'>Click a bubble to learn about that agent.</span>";
  $("#agent-info").style.borderColor = "";
  if (focusId) showAgentInfo(focusId);
  if (!graph.running) { graph.running = true; requestAnimationFrame(drawGraph); }
}
$("#open-agents").onclick = () => openAgents();
graphModal.addEventListener("click", (e) => {
  if (e.target === graphModal || e.target.closest("[data-close]")) { graphModal.hidden = true; graph.running = false; }
});

function pulseAgents(answered, consulted) {
  graph.pulse = { start: performance.now(), answered,
                  targets: [...new Set([answered, ...(consulted || []).map((c) => c.id)])].filter(Boolean) };
}

// ---------- ➕ attach files ----------
$("#attach-btn").onclick = () => $("#attach-input").click();
$("#attach-input").onchange = async (e) => {
  const file = e.target.files[0];
  e.target.value = "";
  if (!file) return;
  if (file.size > 20 * 1024 * 1024) return toast("That file is over 20 MB", "bad");
  $("#empty").hidden = true;
  addUser(`📎 ${file.name}`);
  const typing = showTyping("Reading your file...");
  const form = new FormData();
  form.append("file", file);
  if (currentId) form.append("conversation_id", currentId);
  try {
    const res = await fetch("/api/attach", { method: "POST", body: form });
    const data = await res.json().catch(() => ({}));
    typing.stop();
    if (!res.ok) throw new Error(data.error || "Couldn't read that file");
    currentId = data.conversation_id;
    addBot(data.answer, data.source, null, "", null, true, { agent: "files" });
    showAttached(data.file.name);
    loadChatList();
  } catch (err) {
    typing.stop();
    addBot("⚠ " + err.message, "fallback", null, "", null, false, { agent: "files" });
  }
};

function showAttached(name, reset = false) {
  const box = $("#attached");
  if (reset) box.innerHTML = "";
  if (name) box.append(el("span", "attached-chip", `📎 ${name}`));
  box.hidden = !box.children.length;
}

// ---------- 🎤 talk to text ----------
const Recognition = window.SpeechRecognition || window.webkitSpeechRecognition;
let listening = null, micOrb = null;

function stopListening() {
  if (listening) { try { listening.stop(); } catch {} }
}
$("#mic-orb").onclick = stopListening;
$("#mic-btn").onclick = () => {
  if (!Recognition) {
    return toast("Talk-to-text needs Chrome or Edge.", "bad");
  }
  const rec = new Recognition();
  rec.lang = navigator.language || "en-US";
  rec.interimResults = true;
  rec.continuous = false;
  const before = input.value ? input.value.trim() + " " : "";
  let finalText = "";
  rec.onresult = (e) => {
    let interim = "";
    for (let i = e.resultIndex; i < e.results.length; i++) {
      if (e.results[i].isFinal) finalText += e.results[i][0].transcript;
      else interim += e.results[i][0].transcript;
    }
    input.value = before + finalText + interim;
    autoGrow();
  };
  rec.onerror = (e) => {
    toast(e.error === "not-allowed" ? "Allow microphone access to talk to Neural." :
          e.error === "network" ? "Talk-to-text needs internet in this browser." :
          e.error === "no-speech" ? "I didn't hear anything." : "Talk-to-text stopped: " + e.error, "bad");
  };
  rec.onend = () => {
    listening = null;
    $("#mic-btn").hidden = false; $("#mic-orb").hidden = true;
    dropOrb(micOrb); micOrb = null;
    if (finalText.trim()) sendMessage(input.value);
  };
  listening = rec;
  $("#mic-btn").hidden = true; $("#mic-orb").hidden = false;
  micOrb = makeOrb($("#mic-orb"), "listening", 34);
  rec.start();
};

// ---------- 🔊 read out loud ----------
function readAloud(text) {
  if (!("speechSynthesis" in window)) return toast("This browser can't read out loud.", "bad");
  speechSynthesis.cancel();
  const plain = text.replace(/```[\s\S]*?```/g, " (code) ").replace(/[*`#>]/g, "").replace(/[\u{1F300}-\u{1FAFF}]/gu, "");
  const u = new SpeechSynthesisUtterance(plain);
  u.rate = 1.03;
  speechSynthesis.speak(u);
}
$("#read-aloud").checked = store.get("neural-read-aloud") === "on";
$("#read-aloud").onchange = (e) => {
  store.set("neural-read-aloud", e.target.checked ? "on" : "off");
  toast(e.target.checked ? "I'll read answers out loud" : "Read-aloud off");
  if (!e.target.checked) speechSynthesis?.cancel();
};

// ---------- start ----------
makeOrb($("#hero-orb"), "working", 96);
loadChatList();
loadAgents();
refreshStats().catch(() => {});
refreshUpdate();
setInterval(refreshUpdate, 60 * 1000);
