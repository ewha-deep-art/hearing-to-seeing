// The conversions this browser asked for, shown like a download bubble in the
// top right of every page until each is dismissed.
import { STEPS, api, el, icon, isPending, statusLabel, stepIndex, youtubeThumb } from "/static/common.js";

const STORAGE_KEY = "h2s.conversions";
const POLL_MS = 2000;

let memory = []; // used when localStorage is unavailable
function loadIds() {
  try { return JSON.parse(localStorage.getItem(STORAGE_KEY)) ?? []; } catch { return memory; }
}
function saveIds(ids) {
  memory = ids;
  try { localStorage.setItem(STORAGE_KEY, JSON.stringify(ids)); } catch {}
}

const videos = new Map(); // id → latest meta
let tray = null;
let collapsed = false;
let pollTimer = null;

function forget(id) {
  saveIds(loadIds().filter((x) => x !== id));
  videos.delete(id);
}

function item(id, video) {
  if (!video) {
    return el("li", { class: "tray-item" }, el("div", { class: "tray-text" }, el("div", { class: "tray-status" }, "불러오는 중…")));
  }

  const close = el("button", {
    class: "icon-btn tray-close", type: "button", "aria-label": "목록에서 지우기",
    hidden: isPending(video.status), onclick: () => { forget(id); render(); },
  }, icon("close"));

  const step = stepIndex(video.status);
  let status, bar = null;
  if (isPending(video.status)) {
    const transcribing = video.status === "transcribing";
    status = `${statusLabel(video.status)} (${step + 1}/${STEPS.length})` +
      (transcribing && video.duration ? ` · 약 ${Math.max(1, Math.round(video.duration / 60))}분 걸려요` : "");
    bar = el("div", { class: transcribing ? "tray-bar running" : "tray-bar" },
      el("span", { style: `width:${((step + 1) / STEPS.length) * 100}%` }));
  } else if (video.status === "ready") {
    status = el("a", { class: "tray-link", href: `/watch/${id}` }, "완성 · 보러 가기");
  } else {
    status = el("span", { class: "tray-error" }, "자막을 만들지 못했어요 · 다시 요청해 주세요");
  }

  const thumb = el("img", { class: "tray-thumb", alt: "", src: youtubeThumb(video.youtube_id, "mqdefault") });
  thumb.onerror = () => thumb.removeAttribute("src");

  return el("li", { class: "tray-item" },
    el("a", { href: `/watch/${id}`, tabindex: "-1" }, thumb),
    el("div", { class: "tray-text" },
      el("a", { class: "tray-title", href: `/watch/${id}`, title: video.title ?? "" }, video.title || "제목 없음"),
      el("div", { class: "tray-status" }, status),
      bar),
    close);
}

function render() {
  const ids = loadIds();
  tray.hidden = ids.length === 0;
  if (!ids.length) return;

  const running = ids.filter((id) => isPending(videos.get(id)?.status ?? "queued")).length;
  const title = running ? `자막 ${running}건 만드는 중` : "자막 완성";
  tray.classList.toggle("collapsed", collapsed);

  const toggle = el("button", {
    class: "icon-btn tray-toggle", type: "button",
    "aria-label": collapsed ? "펼치기" : "접기", "aria-expanded": String(!collapsed),
    onclick: () => { collapsed = !collapsed; render(); },
  }, icon("chevron"));

  tray.replaceChildren(
    el("div", { class: "tray-head" },
      running ? el("div", { class: "spinner small" }) : null,
      el("strong", {}, title),
      toggle),
    // replaceChildren would print a null as the text "null".
    ...(collapsed ? [] : [el("ol", { class: "tray-list" }, ids.map((id) => item(id, videos.get(id))))]));
}

async function poll() {
  clearTimeout(pollTimer);
  const ids = loadIds();
  await Promise.all(ids.map(async (id) => {
    const known = videos.get(id);
    if (known && !isPending(known.status)) return; // finished: nothing changes now
    try {
      const video = await api(`/api/videos/${id}`);
      const { cues, speakers, ...meta } = video;
      videos.set(id, meta);
      if (meta.status === "error" && known?.status !== "error") console.error(`[${id}] ${meta.error}`);
      if (known && isPending(known.status) && meta.status === "ready") {
        document.dispatchEvent(new CustomEvent("conversion-ready", { detail: { id } }));
      }
    } catch (err) {
      if (err.status === 404) forget(id); // deleted meanwhile
    }
  }));
  render();
  if (loadIds().some((id) => isPending(videos.get(id)?.status ?? "queued"))) {
    pollTimer = setTimeout(poll, POLL_MS);
  }
}

export function trackConversion(id) {
  const ids = loadIds().filter((x) => x !== id);
  saveIds([id, ...ids]);
  videos.delete(id);
  collapsed = false;
  createTray();
  poll();
}

function createTray() {
  if (tray) return false;
  tray = el("aside", { class: "tray", "aria-label": "자막 만들기 진행 상황", "aria-live": "polite", hidden: true });
  document.body.append(tray);
  return true;
}

export function mountTray() {
  if (createTray()) poll();
  return tray;
}
