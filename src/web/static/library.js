import { api, el, formatTime, icon, timeAgo, youtubeThumb } from "/static/common.js";
import { mountTray, trackConversion } from "/static/tray.js";

const GENRES = ["드라마", "영화", "애니메이션", "기타"];

const $ = (id) => document.getElementById(id);
const grid = $("grid");
const searchInput = $("search-input");

const params = new URLSearchParams(location.search);
let query = params.get("q") ?? "";
let genre = GENRES.includes(params.get("genre")) ? params.get("genre") : "";
searchInput.value = query;

// --- catalogue grid -----------------------------------------------------------

function card(video) {
  const thumb = el("div", { class: "thumb" });
  const img = el("img", { src: youtubeThumb(video.youtube_id), alt: "", loading: "lazy" });
  img.onerror = () => img.remove();
  thumb.append(img);
  if (video.duration) thumb.append(el("span", { class: "duration" }, formatTime(video.duration)));

  const title = video.title || "제목 없음";
  const meta = [video.channel, video.genre, timeAgo(video.created_at)].filter(Boolean).join(" · ");
  const href = `/watch/${video.id}` + (query ? `?q=${encodeURIComponent(query)}` : "");

  const remove = el("button", {
    class: "icon-btn", type: "button", title: "삭제", "aria-label": "삭제",
    onclick: async (e) => {
      e.preventDefault();
      if (!confirm(`'${title}'을(를) 삭제할까요?`)) return;
      try { await api(`/api/videos/${video.id}`, { method: "DELETE" }); } catch (err) { console.error(err); }
      refresh();
    },
  }, icon("trash"));

  return el("article", { class: "card" },
    el("a", { class: "cover", href }, thumb),
    el("div", { class: "card-body" },
      el("a", { class: "card-text", href },
        el("div", { class: "card-title" }, title),
        el("div", { class: "card-meta" }, meta)),
      remove));
}

function renderGenres() {
  $("genres").replaceChildren(...["", ...GENRES].map((g) =>
    el("button", {
      class: "genre-chip", type: "button", "aria-pressed": String(g === genre),
      onclick: () => { genre = g; renderGenres(); refresh(); },
    }, g || "전체")));
}

function filterParams() {
  const params = new URLSearchParams();
  if (query) params.set("q", query);
  if (genre) params.set("genre", genre);
  return params.toString();
}

let listRequest = 0;
async function refresh() {
  const search = filterParams();
  history.replaceState(null, "", search ? `/?${search}` : "/");
  const request = ++listRequest;
  let videos = [];
  try { videos = await api(`/api/videos?${search}`); } catch (err) { console.error(err); }
  if (request !== listRequest) return; // a newer search has gone out

  grid.replaceChildren(...videos.map((v) => card(v)));
  $("results-title").hidden = !query;
  $("results-title").textContent = `'${query}' 검색 결과 ${videos.length}개`;

  $("empty").hidden = videos.length > 0;
  if (query) {
    $("empty-title").textContent = "검색 결과가 없어요";
    $("empty-text").textContent = "다른 제목이나 인물 이름, 대사 한 구절로 찾아보세요.";
  } else if (genre) {
    $("empty-title").textContent = `아직 ${genre} 영상이 없어요`;
    $("empty-text").textContent = "YouTube 링크로 자막을 만들어 보세요.";
  } else {
    $("empty-title").textContent = "아직 영상이 없어요";
    $("empty-text").textContent = "YouTube 링크를 넣으면 누가, 언제, 얼마나 크게 말하는지 보이는 자막을 만들어 모두 함께 볼 수 있어요.";
  }
}

// --- conversions in progress (the tray) ----------------------------------------

let quota = null;

async function refreshQuota() {
  try { quota = await api("/api/queue"); } catch (err) { console.error(err); }
}

// A conversion finished: it now belongs in the catalogue.
document.addEventListener("conversion-ready", () => refresh());

// --- search -------------------------------------------------------------------

let searchTimer;
function setQuery(value) {
  clearTimeout(searchTimer);
  if (value.trim() === query) return;
  query = value.trim();
  refresh();
}
searchInput.addEventListener("input", () => {
  clearTimeout(searchTimer);
  searchTimer = setTimeout(() => setQuery(searchInput.value), 300);
});
$("search").onsubmit = (e) => { e.preventDefault(); setQuery(searchInput.value); };

// --- conversion request dialog --------------------------------------------------

const dialog = $("add-dialog");
const form = $("add-form");
const formError = $("form-error");
const submit = $("add-submit");

function setError(message) {
  formError.hidden = !message;
  formError.textContent = message ?? "";
}

function showQuota() {
  if (!quota) return;
  const left = Math.max(quota.limit - quota.used, 0);
  $("quota-note").textContent = left
    ? `${Math.floor(quota.max_duration / 60)}분 이하 영상만 만들 수 있어요 · 오늘 ${left}건 더 만들 수 있어요`
    : "오늘은 더 만들 수 없어요. 내일 다시 요청해 주세요.";
  submit.disabled = left === 0;
}

const openButton = $("add-open");
openButton.append(icon("plus"), "자막 만들기");
openButton.onclick = async () => {
  form.reset(); setError(null); submit.disabled = false;
  showQuota();
  dialog.showModal();
  await refreshQuota();
  showQuota();
};
$("add-cancel").onclick = () => dialog.close();

form.onsubmit = async (e) => {
  e.preventDefault();
  setError(null);
  const url = form.url.value.trim();
  if (!url) return setError("YouTube 링크를 입력해 주세요.");

  const data = new FormData();
  data.append("url", url);

  submit.disabled = true;
  submit.textContent = "영상 확인 중…";
  try {
    const res = await api("/api/videos", { method: "POST", body: data });
    if (res.existing) {
      location.href = `/watch/${res.id}`;
      return;
    }
    dialog.close();
    trackConversion(res.id);
  } catch (err) {
    console.error(err);
    setError(err.detail ?? "자막을 만들지 못했어요. 잠시 후 다시 시도해 주세요.");
  } finally {
    submit.textContent = "만들기";
    submit.disabled = false;
    await refreshQuota(); // this request may have used today's last one
    showQuota();
  }
};

renderGenres();
refresh();
mountTray();
