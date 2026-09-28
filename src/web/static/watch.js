import { STEPS, api, el, formatTime, icon, isPending, statusLabel } from "/static/common.js";

// Modules come from esm.sh, which keeps abslink a single shared instance
// across JASSUB's entry points (jsDelivr's +esm inlines a second copy into
// abslink/w3c, and the worker handshake then fails). Static assets stay on jsDelivr.
const JASSUB_MODULE = "https://esm.sh/jassub@2.5.16";
const JASSUB_ASSETS = "https://cdn.jsdelivr.net/npm/jassub@2.5.16/dist";
// One Hangul-capable face for every subtitle. The ASS style names Arial,
// which has no Hangul; with local font lookup off, libass falls back to this
// default for every glyph. The name must be the font's own family name, and
// the font is preloaded because every line needs it — loaded lazily, the
// first frame after a paused seek renders blank.
const SUBTITLE_FONT_NAME = "Pretendard SemiBold";
const SUBTITLE_FONT = "https://cdn.jsdelivr.net/npm/pretendard@1.3.9/dist/web/static/woff2/Pretendard-SemiBold.woff2";
const SUB_MODES = ["kinetic", "plain", "off"];
const SUB_MODE_LABELS = { kinetic: "키네틱 자막", plain: "일반 자막", off: "자막 끔" };

const videoId = location.pathname.split("/").pop();
const $ = (id) => document.getElementById(id);
const player = $("player");
const video = $("video");

// --- loading / processing state --------------------------------------------

async function load() {
  let meta;
  try {
    meta = await api(`/api/videos/${videoId}`);
  } catch (err) {
    console.error(err);
    return showState("error", "영상을 불러올 수 없습니다");
  }
  document.title = `${meta.title || "제목 없음"} · Hearing to Seeing`;
  $("title").textContent = meta.title || meta.source_url || "제목 없음";

  if (isPending(meta.status)) {
    showProgress(meta);
    setTimeout(load, 2000);
  } else if (meta.status === "error") {
    console.error(meta.error);
    showState("error", "처리에 실패했습니다");
  } else {
    setup(meta);
  }
}

function showState(kind, title) {
  const state = $("state");
  state.className = `player-state ${kind}`;
  // replaceChildren takes only Nodes/strings — a bare null argument would
  // stringify to the text "null" instead of being dropped.
  state.replaceChildren(el("strong", {}, title));
  state.hidden = false;
}

function showProgress(meta) {
  const current = STEPS.findIndex(([key]) => key === meta.status);
  const state = $("state");
  state.className = "player-state";
  // replaceChildren takes only Nodes/strings — a bare null argument would
  // stringify to the text "null" instead of being dropped, so filter first.
  state.replaceChildren(...[
    el("div", { class: "spinner" }),
    el("strong", {}, `${statusLabel(meta.status)} 중…`),
    el("div", { class: "steps" }, STEPS.map(([, label], i) =>
      el("span", { class: i < current ? "done" : i === current ? "current" : null }, label))),
    meta.status === "transcribing" ? el("span", {}, "원격 서버에서 영상 길이만큼 걸립니다.") : null,
  ].filter((c) => c != null));
}

// --- ready: player, legend, script -----------------------------------------

function setup(meta) {
  $("state").hidden = true;
  $("controls").hidden = false;
  video.src = `/api/videos/${videoId}/media`;

  const colors = Object.fromEntries(meta.speakers.map((s) => [s.label, s]));
  $("legend").replaceChildren(...meta.speakers.map((s) =>
    el("span", { class: "chip" }, el("span", { class: "swatch", style: `background:${s.color}` }), s.name)));

  const parts = ["YouTube"];
  if (meta.duration) parts.push(formatTime(meta.duration));
  parts.push(`화자 ${meta.speakers.length}명`, `자막 ${meta.cues.length}줄`);
  $("meta-sub").textContent = parts.join(" · ");

  for (const link of document.querySelectorAll("#downloads a")) {
    link.href = `/api/videos/${videoId}/files/${link.dataset.kind}?download=1`;
  }
  $("downloads-button").prepend(icon("download"));
  $("downloads").hidden = false;

  setupScript(meta.cues, colors);
  setupControls();
  setupSubtitles();
}

// --- subtitles (JASSUB) ------------------------------------------------------

let jassub = null;
let loadedTrack = null; // "kinetic" | "plain": the track JASSUB currently holds
let subMode = "kinetic";
try { subMode = SUB_MODES.includes(localStorage.subMode) ? localStorage.subMode : "kinetic"; } catch {}

const trackUrl = (mode) =>
  new URL(`/api/videos/${videoId}/files/${mode === "plain" ? "plain" : "ass"}`, location.href).href;

async function setupSubtitles() {
  try {
    const { default: JASSUB } = await import(JASSUB_MODULE);
    loadedTrack = subMode === "plain" ? "plain" : "kinetic";
    jassub = new JASSUB({
      video,
      subUrl: trackUrl(loadedTrack),
      workerUrl: new URL("/static/jassub-worker.js", location.href).href,
      wasmUrl: `${JASSUB_ASSETS}/wasm/jassub-worker.wasm`,
      modernWasmUrl: `${JASSUB_ASSETS}/wasm/jassub-worker-modern.wasm`,
      fonts: [SUBTITLE_FONT],
      defaultFont: SUBTITLE_FONT_NAME,
      queryFonts: false,
    });
    await jassub.ready;
  } catch (err) {
    console.error(err);
    toast("자막 렌더러를 불러오지 못했습니다");
  }
  applySubMode(false);
}

async function setSubMode(mode) {
  subMode = mode;
  try { localStorage.subMode = mode; } catch {}
  applySubMode(true);
  if (jassub && mode !== "off" && mode !== loadedTrack) {
    loadedTrack = mode;
    await jassub.ready;
    await jassub.renderer.setTrackByUrl(trackUrl(mode));
    if (video.paused) jassub.resize(true);
  }
}

function applySubMode(announce) {
  for (const b of document.querySelectorAll("#sub-mode button")) b.setAttribute("aria-pressed", b.dataset.mode === subMode);
  if (jassub) {
    jassub._canvas.style.visibility = subMode === "off" ? "hidden" : "visible";
  }
  if (announce) toast(SUB_MODE_LABELS[subMode]);
}

// --- script panel -------------------------------------------------------------

let cueItems = [];
let cueStarts = [];
let activeCue = -1;

function setupScript(cues, colors) {
  const list = $("script-list");
  if (!cues.length) {
    list.replaceChildren(el("li", { class: "script-empty" }, "인식된 대사가 없습니다."));
  }
  cueStarts = cues.map((c) => c.start);
  cueItems = cues.map((cue, i) => {
    const speaker = colors[cue.speaker] ?? { name: cue.speaker, color: "#888" };
    const showName = i === 0 || cues[i - 1].speaker !== cue.speaker;
    return el("li", {
      class: "cue",
      onclick: () => { video.currentTime = cue.start; video.play(); },
    },
      el("span", { class: "cue-time" }, formatTime(cue.start)),
      el("span", { class: "cue-bar", style: `background:${speaker.color}` }),
      el("span", { class: "cue-text" },
        showName ? el("span", { class: "cue-speaker", style: `color:${speaker.color}` }, speaker.name) : null,
        cue.text));
  });
  if (cues.length) list.replaceChildren(...cueItems);
  $("script").hidden = false;
  video.addEventListener("timeupdate", syncScript);
  video.addEventListener("seeked", syncScript);
}

function syncScript() {
  // Last cue that has started; binary search since this runs several times a second.
  const t = video.currentTime;
  let lo = 0, hi = cueStarts.length - 1, found = -1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (cueStarts[mid] <= t) { found = mid; lo = mid + 1; } else hi = mid - 1;
  }
  if (found === activeCue) return;
  cueItems[activeCue]?.classList.remove("active");
  activeCue = found;
  const item = cueItems[found];
  if (!item) return;
  item.classList.add("active");
  if ($("autoscroll").checked) {
    const list = $("script-list");
    list.scrollTo({ top: item.offsetTop - list.offsetTop - list.clientHeight / 3, behavior: "smooth" });
  }
}

// --- controls -------------------------------------------------------------------

function setupControls() {
  const playBtn = $("play"), bigPlay = $("big-play"), muteBtn = $("mute"), volume = $("volume");
  const fsBtn = $("fullscreen"), seek = $("seek"), tip = $("seek-tip");
  bigPlay.append(icon("play"));

  const togglePlay = () => (video.paused || video.ended ? video.play() : video.pause());
  const setIcon = (btn, name) => btn.replaceChildren(icon(name));

  const syncPlay = () => {
    setIcon(playBtn, video.paused ? "play" : "pause");
    bigPlay.hidden = !video.paused;
    wake();
  };
  const syncVolume = () => {
    setIcon(muteBtn, video.muted || video.volume === 0 ? "muted" : "volume");
    volume.value = video.muted ? 0 : video.volume;
  };
  const syncTime = () => {
    const d = video.duration || 0;
    $("time").textContent = `${formatTime(video.currentTime)} / ${formatTime(d)}`;
    const pct = d ? (video.currentTime / d) * 100 : 0;
    $("seek-played").style.width = `${pct}%`;
    $("seek-thumb").style.left = `${pct}%`;
    if (d && video.buffered.length) {
      $("seek-buffered").style.width = `${(video.buffered.end(video.buffered.length - 1) / d) * 100}%`;
    }
  };
  const syncFullscreen = () => setIcon(fsBtn, document.fullscreenElement ? "exitFullscreen" : "fullscreen");

  playBtn.onclick = togglePlay;
  bigPlay.onclick = togglePlay;
  video.onclick = togglePlay;
  video.ondblclick = () => toggleFullscreen();
  muteBtn.onclick = () => { video.muted = !video.muted; };
  volume.oninput = () => { video.volume = Number(volume.value); video.muted = video.volume === 0; };
  fsBtn.onclick = () => toggleFullscreen();
  for (const b of document.querySelectorAll("#sub-mode button")) b.onclick = () => setSubMode(b.dataset.mode);

  for (const ev of ["play", "pause", "ended"]) video.addEventListener(ev, syncPlay);
  video.addEventListener("volumechange", syncVolume);
  for (const ev of ["timeupdate", "durationchange", "progress", "seeked"]) video.addEventListener(ev, syncTime);
  document.addEventListener("fullscreenchange", syncFullscreen);

  // Seek bar: click or drag anywhere along it.
  const timeAt = (e) => {
    const rect = seek.getBoundingClientRect();
    return Math.min(Math.max((e.clientX - rect.left) / rect.width, 0), 1) * (video.duration || 0);
  };
  seek.onpointerdown = (e) => {
    seek.setPointerCapture(e.pointerId);
    video.currentTime = timeAt(e);
    seek.onpointermove = (ev) => { video.currentTime = timeAt(ev); showTip(ev); };
  };
  seek.onpointerup = () => { seek.onpointermove = showTip; };
  const showTip = (e) => {
    const rect = seek.getBoundingClientRect();
    tip.hidden = false;
    tip.textContent = formatTime(timeAt(e));
    tip.style.left = `${Math.min(Math.max(e.clientX - rect.left, 24), rect.width - 24)}px`;
  };
  seek.onpointermove = showTip;
  seek.onpointerleave = () => { tip.hidden = true; };

  // Controls fade while playing and the pointer is still.
  let idleTimer;
  function wake() {
    player.classList.remove("idle", "hide-cursor");
    clearTimeout(idleTimer);
    if (!video.paused) idleTimer = setTimeout(() => player.classList.add("idle", "hide-cursor"), 2500);
  }
  player.addEventListener("pointermove", wake);
  player.addEventListener("pointerleave", () => { if (!video.paused) player.classList.add("idle"); });

  document.addEventListener("keydown", (e) => {
    if (e.target.closest("input, textarea, select, [contenteditable]") && e.target.type !== "range") return;
    if (e.ctrlKey || e.metaKey || e.altKey) return;
    const actions = {
      " ": togglePlay, k: togglePlay,
      ArrowLeft: () => { video.currentTime -= 5; },
      ArrowRight: () => { video.currentTime += 5; },
      j: () => { video.currentTime -= 10; },
      l: () => { video.currentTime += 10; },
      f: toggleFullscreen,
      m: () => { video.muted = !video.muted; },
      c: () => setSubMode(SUB_MODES[(SUB_MODES.indexOf(subMode) + 1) % SUB_MODES.length]),
    };
    const action = actions[e.key.length === 1 ? e.key.toLowerCase() : e.key];
    if (!action) return;
    e.preventDefault();
    action();
    wake();
  });

  syncPlay(); syncVolume(); syncTime(); syncFullscreen();
}

function toggleFullscreen() {
  // The wrapper, not the <video>, goes fullscreen so the subtitle canvas and
  // controls come along.
  if (document.fullscreenElement) document.exitFullscreen();
  else player.requestFullscreen();
}

let toastTimer;
function toast(message) {
  const t = $("toast");
  t.textContent = message;
  t.classList.add("show");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => t.classList.remove("show"), 1200);
}

load();
