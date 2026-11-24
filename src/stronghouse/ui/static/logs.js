// Logs page: read the app's logs (secrets hidden by the server) and copy them in one click.
// Uses helpers from app.js: $, api, withErrors, whileBusy, toast, escapeHtml, formatBytes,
// copyWithToast, setLogText.
"use strict";

const LOGS_POLL_MS = 3000;
const logs = { sources: [], current: "app", copyText: "" };

async function loadLogs() {
  if (!logs.sources.length) {
    const sources = await withErrors(() => api("/logs"));
    if (!sources) return;
    logs.sources = sources;
    $("#logs-sources").innerHTML = sources.map((s) => `
      <button type="button" data-log="${escapeHtml(s.key)}" title="${escapeHtml(s.description)}">${escapeHtml(s.title)}</button>`).join("");
  }
  showLogSource(logs.current);
}
TAB_LOADERS.logs = loadLogs;

function logsTabOpen() {
  return $("#tab-logs").classList.contains("active");
}

function showLogSource(key) {
  logs.current = key;
  document.querySelectorAll("#logs-sources button").forEach((b) => {
    b.classList.toggle("active", b.dataset.log === key);
    b.setAttribute("aria-pressed", String(b.dataset.log === key));
  });
  $("#logs-view").textContent = "";
  pollLogs();
}

function logQuery() {
  return `lines=${$("#logs-lines").value}&problems=${$("#logs-problems").checked}`;
}

async function pollLogs() {
  clearTimeout(pollLogs.timer);
  const key = logs.current;
  const view = await api(`/logs/${encodeURIComponent(key)}?${logQuery()}`).catch((error) => {
    toast(error.message);
    return null;
  });
  if (view && key === logs.current) {
    logs.copyText = view.copy_text;
    const facts = [escapeHtml(view.description), `<code>${escapeHtml(view.location)}</code>`];
    facts.push(view.exists ? escapeHtml(formatBytes(view.size_bytes)) : "not written yet");
    $("#logs-meta").innerHTML = facts.join(" · ");
    const empty = view.exists ? "Nothing to show." : "This log has not been written yet.";
    setLogText($("#logs-view"), view.lines.length ? view.lines.join("\n") : empty);
  }
  if (logsTabOpen() && $("#logs-follow").checked) pollLogs.timer = setTimeout(pollLogs, LOGS_POLL_MS);
}

$("#logs-sources").addEventListener("click", (event) => {
  const button = event.target.closest("[data-log]");
  if (button) showLogSource(button.dataset.log);
});
["#logs-lines", "#logs-problems", "#logs-follow"].forEach((selector) => {
  $(selector).addEventListener("change", () => pollLogs());
});
$("#logs-refresh").addEventListener("click", () => pollLogs());

$("#logs-copy").addEventListener("click", (event) => whileBusy(event.currentTarget, async () => {
  if (!logs.copyText) { toast("Nothing to copy yet."); return; }
  const title = logs.sources.find((s) => s.key === logs.current)?.title ?? "";
  await copyWithToast(logs.copyText, `${title} log`.trim());
}));

$("#logs-copy-all").addEventListener("click", (event) => whileBusy(event.currentTarget, async () => {
  const report = await withErrors(() => api(`/logs/report?problems=${$("#logs-problems").checked}`));
  if (report) await copyWithToast(report.copy_text, "All logs");
}));
