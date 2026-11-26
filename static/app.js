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
  fallback: "",
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
function renderMarkdown(text) {
  const parts = escapeHtml(text).split(/```(?:\w*)\n?/);
  let html = "";
  parts.forEach((part, i) => {
    if (i % 2) { html += `<pre><code>${part.replace(/\n$/, "")}</code></pre>`; return; }
    let list = null;
    for (const line of part.split("\n")) {
      const ul = line.match(/^\s*[-*•]\s+(.*)/), ol = line.match(/^\s*\d+[.)]\s+(.*)/);
      const kind = ul ? "ul" : ol ? "ol" : null;
      if (list && kind !== list) { html += `</${list}>`; list = null; }
      if (kind) {
        if (!list) { html += `<${kind}>`; list = kind; }
        html += `<li>${inline((ul || ol)[1])}</li>`;
      } else if (line.trim()) html += `<p>${inline(line)}</p>`;
    }
    if (list) html += `</${list}>`;
  });
  return html;
}

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
  $("#empty").hidden = convo.messages.length > 0;
  $("#title").textContent = convo.title;
  let lastUser = "";
  for (const m of convo.messages) {
    if (m.role === "User") { addUser(m.message, m.corrected); lastUser = m.message; }
    else addBot(m.message, m.source, m.id, lastUser, m.rating);
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

function addBot(text, source, id, question, rating, animate = false) {
  const msg = el("div", "msg bot");
  msg.append(el("div", "logo"));
  const body = el("div", "body");
  const bubble = el("div", "bubble");
  body.append(bubble);
  msg.append(body);
  thread.append(msg);

  const meta = el("div", "meta");
  if (SOURCES[source]) meta.append(el("span", "chip " + source, SOURCES[source]));

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
    meta.append(up, down, copy, note);
  }
  body.append(meta);

  if (animate) typeOut(bubble, text); else bubble.innerHTML = renderMarkdown(text);
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
function showTyping() {
  const msg = el("div", "msg bot");
  const box = el("div", "thinking");
  const canvas = el("canvas");
  const label = el("span", "label", "Searching my knowledge...");
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
      body: JSON.stringify({ message: text, conversation_id: currentId,
                             style: store.get("neural-style") === "plain" ? "plain" : "casual" }),
    });
    await new Promise((r) => setTimeout(r, Math.max(0, 650 - (Date.now() - started))));
    typing.stop();
    addReadAs(userMsg, data.corrected);
    addBot(data.answer, data.source, data.id, text, null, true);
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
document.addEventListener("keydown", (e) => { if (e.key === "Escape") modal.hidden = true; });

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

$("#update-now").onclick = async () => {
  try {
    await api("/api/update/apply", { method: "POST" });
    updateBar.hidden = false;
    followUpdate();
  } catch (err) { $("#update-stage").textContent = err.message; }
};
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

// ---------- start ----------
makeOrb($("#hero-orb"), "working", 96);
loadChatList();
refreshStats().catch(() => {});
refreshUpdate();
setInterval(refreshUpdate, 60 * 1000);
