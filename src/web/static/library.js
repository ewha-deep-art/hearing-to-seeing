import { api, el, formatTime, icon, isPending, statusLabel, timeAgo } from "/static/common.js";

const grid = document.getElementById("grid");
const empty = document.getElementById("empty");
let pollTimer = null;

// --- library grid ----------------------------------------------------------

function card(video) {
  const pending = isPending(video.status);
  const ready = video.status === "ready";

  const thumb = el("div", { class: "thumb" });
  if (!pending || video.status === "transcribing" || video.status === "finishing") {
    const img = el("img", { src: `/api/videos/${video.id}/thumb`, alt: "", loading: "lazy" });
    img.onerror = () => img.remove();
    thumb.append(img);
  }
  if (ready && video.duration) thumb.append(el("span", { class: "duration" }, formatTime(video.duration)));
  if (pending) {
    thumb.append(el("div", { class: "status" }, el("div", { class: "spinner" }), statusLabel(video.status)));
  } else if (video.status === "error") {
    console.error(`[${video.id}] ${video.error ?? "처리 실패"}`);
    thumb.append(el("div", { class: "status error" }, "처리 실패"));
  }

  const title = video.title || video.source_url || "제목 없음";
  const meta = ["YouTube", timeAgo(video.created_at)].join(" · ");

  const remove = el("button", {
    class: "icon-btn", type: "button", title: "삭제", "aria-label": "삭제",
    hidden: pending,
    onclick: async (e) => {
      e.preventDefault();
      if (!confirm(`'${title}'을(를) 삭제할까요?`)) return;
      try { await api(`/api/videos/${video.id}`, { method: "DELETE" }); } catch (err) { console.error(err); }
      refresh();
    },
  }, icon("trash"));

  return el("article", { class: "card" },
    el("a", { class: "cover", href: `/watch/${video.id}` }, thumb),
    el("div", { class: "card-body" },
      el("a", { class: "card-text", href: `/watch/${video.id}` },
        el("div", { class: "card-title" }, title),
        el("div", { class: "card-meta" }, meta)),
      remove));
}

async function refresh() {
  clearTimeout(pollTimer);
  let videos = [];
  try { videos = await api("/api/videos"); } catch (err) { console.error(err); }
  grid.replaceChildren(...videos.map(card));
  empty.hidden = videos.length > 0;
  if (videos.some((v) => isPending(v.status))) pollTimer = setTimeout(refresh, 2000);
}

// --- add dialog -------------------------------------------------------------

const dialog = document.getElementById("add-dialog");
const form = document.getElementById("add-form");
const formError = document.getElementById("form-error");
const submit = document.getElementById("add-submit");

const openButton = document.getElementById("add-open");
openButton.append(icon("plus"), "영상 추가");
openButton.onclick = () => { form.reset(); setError(null); submit.disabled = false; dialog.showModal(); };
document.getElementById("add-cancel").onclick = () => dialog.close();

function setError(message) {
  formError.hidden = !message;
  formError.textContent = message ?? "";
}

form.onsubmit = async (e) => {
  e.preventDefault();
  setError(null);
  const url = form.url.value.trim();
  if (!url) return setError("링크를 입력하세요.");

  const data = new FormData();
  data.append("url", url);
  data.append("title", form.title.value);

  submit.disabled = true;
  try {
    await api("/api/videos", { method: "POST", body: data });
    dialog.close();
    refresh();
  } catch (err) {
    console.error(err);
    setError("영상을 추가하지 못했습니다.");
  } finally {
    submit.disabled = false;
  }
};

refresh();
