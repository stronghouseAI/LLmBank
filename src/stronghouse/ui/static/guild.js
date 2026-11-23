// Guild tab: create agents (character sheets), run quests, approve their actions.
// Uses helpers from app.js: $, api, withErrors, toast, escapeHtml.
"use strict";

const QUEST_POLL_MS = 1200;
const ACTIVE_QUEST_STATES = ["running", "waiting_approval"];
const SKILL_CATEGORY_NAMES = {
  files: "📁 Files", web: "🌐 Web", documents: "📄 Documents", desktop: "🖥️ Desktop", utility: "🧮 Utility",
};
const guild = { catalog: null, agents: [], editingId: null, questId: null };

async function loadGuild() {
  if (!guild.catalog) {
    guild.catalog = await withErrors(() => api("/agents/catalog"));
    if (!guild.catalog) return;
    renderCatalogOptions();
  }
  await loadAgents();
  loadQuests();
}

document.querySelector('.tabs button[data-tab="guild"]').addEventListener("click", loadGuild);

// ---------- agents ----------

async function loadAgents() {
  guild.agents = (await withErrors(() => api("/agents"))) || [];
  $("#agent-list").innerHTML = guild.agents.map(agentCard).join("")
    || '<p class="muted">No agents yet. Create one or recruit a starter below.</p>';
  const options = guild.agents.map((a) => `<option value="${a.id}">${escapeHtml(a.portrait)} ${escapeHtml(a.name)}</option>`).join("");
  const [worker, reviewer] = [$("#quest-worker").value, $("#quest-reviewer").value];
  $("#quest-worker").innerHTML = options;
  $("#quest-reviewer").innerHTML = `<option value="">No reviewer</option>${options}`;
  if (guild.agents.some((a) => String(a.id) === worker)) $("#quest-worker").value = worker;
  $("#quest-reviewer").value = guild.agents.some((a) => String(a.id) === reviewer) ? reviewer : "";
  updateQuestHint();
}

// Tasks that clearly need a kind of skill, so we can warn before a quest is wasted.
const TASK_NEEDS = [
  { pattern: /https?:\/\/|\brss\b|\bfeed\b|\bwebsite\b|\bweb\b|\bonline\b|\binternet\b|\bheadlines?\b|\bnews\b|\bsearch the\b/i,
    skills: ["fetch_web_page", "read_news_feed", "web_search"], label: "🌐 Web" },
  { pattern: /\.pdf\b|\bpdf\b/i, skills: ["read_pdf"], label: "Read PDF" },
];

function updateQuestHint() {
  const hint = $("#quest-skill-hint");
  const agent = guild.agents.find((a) => String(a.id) === $("#quest-worker").value);
  if (!agent) { hint.textContent = ""; return; }
  const skills = agent.skills.map(skillTitle).join(", ") || "no skills";
  const task = $("#quest-task").value;
  const missing = TASK_NEEDS.find((need) => need.pattern.test(task) && !need.skills.some((s) => agent.skills.includes(s)));
  if (!missing) {
    hint.innerHTML = `${escapeHtml(agent.name)} can: ${escapeHtml(skills)}`;
    return;
  }
  const able = guild.agents.filter((a) => missing.skills.some((s) => a.skills.includes(s))).map((a) => a.name);
  const suggestion = able.length ? `Pick ${escapeHtml(able.join(" or "))}` : "Recruit Herald or edit an agent";
  hint.innerHTML = `<span class="warning">⚠ ${escapeHtml(agent.name)} has no ${escapeHtml(missing.label)} skill for this task
    (can: ${escapeHtml(skills)}). ${suggestion}, or press Edit to add it.</span>`;
}

$("#quest-worker").addEventListener("change", updateQuestHint);
$("#quest-task").addEventListener("input", updateQuestHint);

function agentCard(agent) {
  const xpIntoLevel = agent.xp % 50;
  return `
    <div class="card">
      <div class="agent-head">
        <span class="portrait">${escapeHtml(agent.portrait || "👤")}</span>
        <div><h4>${escapeHtml(agent.name)}</h4>
          <div class="meta">Level ${agent.level} ${escapeHtml(agent.agent_class)} · ${agent.xp} XP</div></div>
      </div>
      <div class="xp"><div style="width:${(xpIntoLevel / 50) * 100}%"></div></div>
      <div>${agent.skills.map((s) => `<span class="badge">${escapeHtml(skillTitle(s))}</span>`).join("")}</div>
      <div class="row">
        <button class="secondary" data-edit="${agent.id}">Edit</button>
        <button class="secondary" data-journal="${agent.id}">Journal</button>
        <button class="secondary" data-export="${agent.id}">Export</button>
        <button class="danger" data-delete="${agent.id}">Dismiss</button>
      </div>
    </div>`;
}

function skillTitle(name) {
  return guild.catalog?.skills.find((s) => s.name === name)?.title ?? name;
}

function renderCatalogOptions() {
  const { classes, skills, starters } = guild.catalog;
  $("#agent-class").innerHTML = classes.map((c) => `<option value="${c.id}">${escapeHtml(c.id)}</option>`).join("");
  const categories = [...new Set(skills.map((s) => s.category))];
  $("#agent-skills").innerHTML = categories.map((category) => `
    <div class="skill-category">${escapeHtml(SKILL_CATEGORY_NAMES[category] ?? category)}</div>
    ${skills.filter((s) => s.category === category).map((s) => `
    <label><input type="checkbox" value="${escapeHtml(s.name)}">
      <span>${escapeHtml(s.title)}${s.needs_approval ? " 🔒" : ""}<small>${escapeHtml(s.description)}</small></span>
    </label>`).join("")}`).join("");
  $("#starter-list").innerHTML = starters.map((card, index) => `
    <div class="item"><span>${escapeHtml(card.portrait)} <strong>${escapeHtml(card.name)}</strong>
      <span class="muted">${escapeHtml(card.agent_class)} · ${escapeHtml(card.personality)}</span></span>
      <button class="secondary" data-recruit="${index}">Recruit</button></div>`).join("");
}

function openSheet(agent) {
  guild.editingId = agent?.id ?? null;
  $("#agent-form-title").textContent = agent ? `Edit ${agent.name}` : "New agent";
  $("#agent-name").value = agent?.name ?? "";
  $("#agent-portrait").value = agent?.portrait ?? "";
  $("#agent-class").value = agent?.agent_class ?? "scholar";
  $("#agent-personality").value = agent?.personality ?? "";
  $("#agent-creativity").value = agent?.creativity ?? 3;
  $("#agent-diligence").value = agent?.diligence ?? 5;
  document.querySelectorAll("#agent-skills input").forEach((box) => {
    box.checked = agent ? agent.skills.includes(box.value) : ["read_file", "take_note"].includes(box.value);
  });
  updateSheetHints();
  $("#agent-form").hidden = false;
  $("#agent-name").focus();
}

function updateSheetHints() {
  const max = guild.catalog.max_skill_slots;
  const boxes = [...document.querySelectorAll("#agent-skills input")];
  const used = boxes.filter((b) => b.checked).length;
  boxes.forEach((b) => { b.disabled = !b.checked && used >= max; });
  $("#skill-slots").textContent = `${used} of ${max} slots`;
  $("#creativity-value").textContent = $("#agent-creativity").value;
  $("#diligence-value").textContent = $("#agent-diligence").value;
  const chosen = guild.catalog.classes.find((c) => c.id === $("#agent-class").value);
  $("#agent-class-hint").textContent = chosen ? chosen.description : "";
}

["#agent-skills", "#agent-creativity", "#agent-diligence", "#agent-class"].forEach((sel) =>
  $(sel).addEventListener("input", updateSheetHints));

$("#agent-new").addEventListener("click", () => openSheet(null));
$("#agent-cancel").addEventListener("click", () => { $("#agent-form").hidden = true; });

$("#agent-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const body = {
    name: $("#agent-name").value,
    portrait: $("#agent-portrait").value,
    agent_class: $("#agent-class").value,
    personality: $("#agent-personality").value,
    skills: [...document.querySelectorAll("#agent-skills input:checked")].map((b) => b.value),
    creativity: Number($("#agent-creativity").value),
    diligence: Number($("#agent-diligence").value),
  };
  const path = guild.editingId ? `/agents/${guild.editingId}` : "/agents";
  const saved = await withErrors(() => api(path, { method: guild.editingId ? "PUT" : "POST", body }));
  if (!saved) return;
  toast(`${saved.name} is ready.`);
  $("#agent-form").hidden = true;
  loadAgents();
});

$("#agent-list").addEventListener("click", async (event) => {
  const { edit, journal, delete: dismiss } = event.target.dataset;
  const exportId = event.target.dataset.export;
  if (edit) openSheet(guild.agents.find((a) => a.id === Number(edit)));
  if (journal) openJournal(Number(journal));
  if (exportId) {
    const card = await withErrors(() => api(`/agents/${exportId}/card`));
    if (card) downloadJson(card, `${card.name.replace(/\W+/g, "_")}.agent.json`);
  }
  if (dismiss) {
    const agent = guild.agents.find((a) => a.id === Number(dismiss));
    if (!confirm(`Dismiss ${agent.name}? Their journal is deleted too.`)) return;
    await withErrors(() => api(`/agents/${dismiss}`, { method: "DELETE" }));
    loadAgents();
  }
});

$("#starter-list").addEventListener("click", async (event) => {
  const index = event.target.dataset.recruit;
  if (index === undefined) return;
  const agent = await withErrors(() => api("/agents/import", { method: "POST", body: guild.catalog.starters[index] }));
  if (agent) { toast(`${agent.name} joined your guild.`); loadAgents(); }
});

$("#agent-import").addEventListener("change", async (event) => {
  const file = event.target.files[0];
  event.target.value = "";
  if (!file) return;
  let card;
  try { card = JSON.parse(await file.text()); } catch { toast("That file is not valid JSON."); return; }
  const agent = await withErrors(() => api("/agents/import", { method: "POST", body: card }));
  if (agent) { toast(`${agent.name} joined your guild.`); loadAgents(); }
});

function downloadJson(data, filename) {
  const url = URL.createObjectURL(new Blob([JSON.stringify(data, null, 2)], { type: "application/json" }));
  const link = Object.assign(document.createElement("a"), { href: url, download: filename });
  link.click();
  URL.revokeObjectURL(url);
}

// ---------- journal ----------

async function openJournal(agentId) {
  const agent = guild.agents.find((a) => a.id === agentId);
  $("#journal-title").textContent = `${agent.name}'s journal`;
  await renderJournal(agentId);
  $("#journal-dialog").showModal();
}

async function renderJournal(agentId) {
  const notes = (await withErrors(() => api(`/agents/${agentId}/journal`))) || [];
  $("#journal-list").innerHTML = notes.map((n) => `
    <div class="item"><span>${escapeHtml(n.note)}</span>
      <button class="danger" data-agent="${agentId}" data-note="${n.id}">Forget</button></div>`).join("")
    || '<p class="muted">Nothing written yet. Agents with the Take note skill fill this in.</p>';
}

$("#journal-list").addEventListener("click", async (event) => {
  const { agent, note } = event.target.dataset;
  if (!note) return;
  await withErrors(() => api(`/agents/${agent}/journal/${note}`, { method: "DELETE" }));
  renderJournal(Number(agent));
});
$("#journal-close").addEventListener("click", () => $("#journal-dialog").close());

// ---------- quests ----------

$("#quest-start").addEventListener("click", async () => {
  const task = $("#quest-task").value.trim();
  const worker = Number($("#quest-worker").value);
  const reviewer = Number($("#quest-reviewer").value);
  if (!task || !worker) { toast("Pick an agent and describe the task."); return; }
  if ($("#quest-skill-hint .warning") && !confirm("This agent seems to lack a skill this task needs. Start anyway?")) return;
  const agentIds = reviewer && reviewer !== worker ? [worker, reviewer] : [worker];
  const quest = await withErrors(() => api("/quests", { method: "POST", body: { task, agent_ids: agentIds } }));
  if (!quest) return;
  $("#quest-task").value = "";
  showQuest(quest.id);
});

async function loadQuests() {
  const quests = (await withErrors(() => api("/quests"))) || [];
  $("#quest-list").innerHTML = quests.map((q) => `
    <div class="item selectable" data-quest="${q.id}">
      <span class="name">${escapeHtml(q.task.slice(0, 80))}</span>
      <span class="muted">${escapeHtml(q.state.replace("_", " "))}</span></div>`).join("")
    || '<p class="muted">No quests yet.</p>';
}

$("#quest-list").addEventListener("click", (event) => {
  const row = event.target.closest("[data-quest]");
  if (row) showQuest(Number(row.dataset.quest));
});

function showQuest(questId) {
  guild.questId = questId;
  $("#quest-view").hidden = false;
  pollQuest();
}

async function pollQuest() {
  clearTimeout(pollQuest.timer);
  if (!guild.questId) return;
  const quest = await api(`/quests/${guild.questId}`).catch(() => null);
  if (!quest) return;
  renderQuest(quest);
  if (ACTIVE_QUEST_STATES.includes(quest.state)) {
    pollQuest.timer = setTimeout(pollQuest, QUEST_POLL_MS);
  } else {
    loadQuests();
    loadAgents();
  }
}

function renderQuest(quest) {
  $("#quest-title").textContent = `Quest #${quest.id} · ${quest.state.replace("_", " ")}`;
  $("#quest-cancel").hidden = !ACTIVE_QUEST_STATES.includes(quest.state);
  const log = $("#quest-log");
  const atBottom = log.scrollTop + log.clientHeight >= log.scrollHeight - 20;
  log.innerHTML = quest.steps.map((s) => `
    <div class="step ${escapeHtml(s.kind)}"><span class="who">${escapeHtml(s.agent_name)}</span>${escapeHtml(s.text)}</div>`).join("")
    + (quest.result && quest.state === "done" ? `<div class="step review"><span class="who">Result</span>${escapeHtml(quest.result)}</div>` : "");
  if (atBottom) log.scrollTop = log.scrollHeight;

  const box = $("#quest-approval");
  const pending = quest.pending;
  box.hidden = !pending;
  if (pending && box.dataset.shown !== JSON.stringify(pending)) {
    box.dataset.shown = JSON.stringify(pending);
    box.innerHTML = `
      <strong>${escapeHtml(pending.agent_name)} wants to ${escapeHtml(skillTitle(pending.skill).toLowerCase())}</strong>
      <pre>${escapeHtml(describeAction(pending))}</pre>
      <div class="row"><button data-decide="yes">Allow once</button>
        <button class="danger" data-decide="no">Deny</button></div>`;
  }
  if (!pending) box.dataset.shown = "";
}

function describeAction({ skill, arguments: args }) {
  if (skill === "run_command") return `$ ${args.command ?? ""}`;
  if (skill === "write_file") return `File: ${args.path ?? ""}\n\n${args.content ?? ""}`;
  return Object.entries(args).map(([k, v]) => `${k}: ${v}`).join("\n");
}

$("#quest-approval").addEventListener("click", async (event) => {
  const choice = event.target.dataset.decide;
  if (!choice) return;
  await withErrors(() => api(`/quests/${guild.questId}/decision`, { method: "POST", body: { approve: choice === "yes" } }));
  pollQuest();
});

$("#quest-cancel").addEventListener("click", async () => {
  await withErrors(() => api(`/quests/${guild.questId}/cancel`, { method: "POST" }));
  pollQuest();
});
