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

function scrollDown() { messagesEl.scrollTop = messagesEl.scrollHeight; }

// ---------- theme ----------
function applyTheme(t) {
  document.documentElement.dataset.theme = t;
  $("#theme").textContent = t === "dark" ? "☀️ Light mode" : "🌙 Dark mode";
  store.set("neural-theme", t);
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
function addUser(text, corrected) {
  const msg = el("div", "msg user");
  msg.append(el("div", "bubble", text));
  if (corrected) {
    const note = el("div", "read-as", "✎ Read as: ");
    note.append(el("b", "", corrected));
    msg.append(note);
  }
  thread.append(msg);
  scrollDown();
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
    copy.onclick = () => navigator.clipboard?.writeText(text);

    const setRating = (r) => {
      up.classList.toggle("on", r === "good");
      down.classList.toggle("on", r === "bad");
    };
    setRating(rating);

    up.onclick = async () => {
      const r = up.classList.contains("on") ? null : "good";
      setRating(r);
      body.querySelector(".teach")?.remove();
      await api("/api/feedback", { method: "POST", body: JSON.stringify({ id, rating: r }) });
    };
    down.onclick = async () => {
      setRating("bad");
      await api("/api/feedback", { method: "POST", body: JSON.stringify({ id, rating: "bad" }) });
      if (!body.querySelector(".teach")) showTeach(body, question, id);
    };
    meta.append(up, down, copy);
  }
  body.append(meta);

  if (animate) typeOut(bubble, text); else bubble.textContent = text;
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
  };
  tick();
}

function showTeach(body, question, id) {
  const box = el("div", "teach");
  const label = el("label", "", `What should I have said to "${question}"?`);
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
      box.append(el("div", "done", "✓ Thanks! I learned that. Ask me again anytime."));
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

function showTyping() {
  const msg = el("div", "msg bot");
  msg.append(el("div", "logo"));
  const dots = el("div", "typing");
  dots.innerHTML = "<span></span><span></span><span></span>";
  msg.append(dots);
  thread.append(msg);
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

  const userMsg = el("div");  // placeholder so the order stays right
  thread.append(userMsg);
  const typing = showTyping();
  const started = Date.now();

  try {
    const data = await api("/api/chat", {
      method: "POST",
      body: JSON.stringify({ message: text, conversation_id: currentId }),
    });
    await new Promise((r) => setTimeout(r, Math.max(0, 350 - (Date.now() - started))));
    typing.remove();
    userMsg.remove();
    addUser(text, data.corrected);
    addBot(data.answer, data.source, data.id, text, null, true);
    if (!currentId) { currentId = data.conversation_id; }
    loadChatList().then(() => {
      const active = document.querySelector(".chat-item.active span");
      if (active) $("#title").textContent = active.textContent;
    });
  } catch (err) {
    typing.remove();
    userMsg.remove();
    addUser(text);
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

  const lt = s.last_training;
  $("#train-info").textContent = !s.model_ready
    ? "The neural network isn't trained yet. It still answers from its knowledge, but training it lets it handle new questions."
    : lt
      ? `Last trained ${new Date(lt.time * 1000).toLocaleString()} on ${fmt(lt.examples)} examples (loss ${lt.val_loss}, lower is better).`
      : "Neural network is trained and ready.";
  $("#retrain").textContent = s.model_ready ? "Retrain with what I've learned" : "Train the neural network";
  $("#fresh").hidden = !s.model_ready;

  const t = s.training;
  const status = $("#model-status");
  status.className = "status " + (t.running ? "busy" : s.model_ready ? "ready" : "");
  status.textContent = t.running ? "Training..." : s.model_ready ? "Model ready" : "Knowledge only";

  $("#retrain").disabled = $("#fresh").disabled = t.running;
  $("#train-progress").hidden = !t.running;
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
};
modal.addEventListener("click", (e) => {
  if (e.target === modal || e.target.closest("[data-close]")) modal.hidden = true;
});
document.addEventListener("keydown", (e) => { if (e.key === "Escape") modal.hidden = true; });

async function startTraining(fresh) {
  if (fresh && !confirm("Start the neural network over from scratch? This takes longer.")) return;
  try {
    await api("/api/retrain", { method: "POST", body: JSON.stringify({ fresh }) });
  } catch (err) { alert(err.message); }
  refreshStats();
}
$("#retrain").onclick = () => startTraining(false);
$("#fresh").onclick = () => startTraining(true);

// ---------- start ----------
loadChatList();
refreshStats().catch(() => {});
