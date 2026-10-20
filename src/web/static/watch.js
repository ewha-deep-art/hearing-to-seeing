import { STEPS, api, el, formatTime, icon, isPending, statusLabel, stepIndex } from "/static/common.js";
import { mountTray } from "/static/tray.js";

// Modules come from esm.sh, which keeps abslink a single shared instance
// across JASSUB's entry points (jsDelivr's +esm inlines a second copy into
// abslink/w3c, and the worker handshake then fails). Static assets stay on jsDelivr.
const JASSUB_MODULE = "https://esm.sh/jassub@2.5.16";
const JASSUB_ASSETS = "https://cdn.jsdelivr.net/npm/jassub@2.5.16/dist";
// The font the ASS styles name, served from the bundled file
// (design/layout.py), so subtitles look exactly as the pipeline made them.
// libass finds it by the family name inside the file, which must equal
// layout.FONT_NAME; it is also the default, so nothing falls back to another
// face. Preloaded because every line needs it — loaded lazily, the first frame
// after a paused seek renders blank.
const SUBTITLE_FONT_NAME = "KoPubDotum Bold";
const SUBTITLE_FONT = "/api/subtitle-font";
// The ASS PlayRes (converter/ass.py); the stage is always this shape.
const FRAME = { width: 1920, height: 1080 };
const SUB_MODES = ["kinetic", "plain", "off"];
const SUB_MODE_LABELS = { kinetic: "키네틱 자막", plain: "일반 자막", off: "자막 끔" };
// Subtitle size: the canvas scales about the lines' bottom-centre anchor (app.css).
const SUB_SCALES = [0.75, 1, 1.25, 1.5];

const videoId = location.pathname.split("/").pop();
const searchQuery = (new URLSearchParams(location.search).get("q") ?? "").trim();
const $ = (id) => document.getElementById(id);
const player = $("player");
let media = null; // YouTubeMedia, once the player is up

// --- YouTube player, behind the HTMLMediaElement surface the page uses ---------

let youtubeApi = null;
function loadYouTubeApi() {
  youtubeApi ??= new Promise((resolve) => {
    window.onYouTubeIframeAPIReady = () => resolve(window.YT);
    document.head.append(el("script", { src: "https://www.youtube.com/iframe_api" }));
  });
  return youtubeApi;
}

// The IFrame API only reports its time when asked; between polls the clock
// runs on, so the subtitles move every frame rather than four times a second.
const POLL_MS = 250;
const MAX_DRIFT = 0.12;

class YouTubeMedia extends EventTarget {
  constructor(YT, target, youtubeId, start) {
    super();
    this.YT = YT;
    this._time = start;
    this._at = performance.now();
    this._paused = true;
    this._running = false; // playing and not buffering: the clock advances
    this._ended = false;
    this._duration = 0;
    this._muted = false;
    this._volume = 1;
    this.ready = new Promise((resolve) => {
      this.player = new YT.Player(target, {
        videoId: youtubeId,
        width: "100%",
        height: "100%",
        playerVars: {
          controls: 0, disablekb: 1, fs: 0, rel: 0, playsinline: 1, iv_load_policy: 3,
          start: Math.floor(start), origin: location.origin, cc_load_policy: 0,
        },
        events: {
          onReady: () => {
            this._hideCaptions();
            this._muted = this.player.isMuted();
            this._volume = this.player.getVolume() / 100;
            this._syncDuration();
            setInterval(() => this._poll(), POLL_MS);
            resolve(this);
          },
          onStateChange: (e) => this._onState(e.data),
          // Fires when a module (captions among them) finishes loading — often
          // after PLAYING, which is why hiding there alone stopped working.
          onApiChange: () => this._hideCaptions(),
          onError: (e) => { this.error = e.data; this._emit("error"); },
        },
      });
    });
  }

  _emit(type) { this.dispatchEvent(new Event(type)); }

  // YouTube's own captions (often auto-generated) would sit under ours.
  // cc_load_policy alone doesn't stop them: a viewer's account setting or an
  // auto track turns them back on, and they reload once playback starts.
  _hideCaptions() {
    this.player.setOption?.("captions", "track", {});
    for (const module of ["captions", "cc"]) this.player.unloadModule?.(module);
  }

  _syncDuration() {
    const d = this.player.getDuration?.() || 0;
    if (d && d !== this._duration) { this._duration = d; this._emit("durationchange"); }
  }

  _rebase(t = this.player.getCurrentTime()) {
    this._time = t;
    this._at = performance.now();
  }

  _poll() {
    if (this._running) {
      const t = this.player.getCurrentTime();
      if (Math.abs(t - this.currentTime) > MAX_DRIFT) this._rebase(t);
    }
    this._syncDuration();
    this._emit("timeupdate");
  }

  _onState(state) {
    const S = this.YT.PlayerState;
    // Before the first play the player reports 0, not the `start` it will begin at.
    if (state !== S.UNSTARTED && state !== S.CUED) this._rebase();
    const wasPaused = this._paused;
    this._running = state === S.PLAYING;
    if (state === S.PLAYING) {
      this._hideCaptions();
      this._paused = false;
      this._ended = false;
      if (wasPaused) this._emit("play");
    } else if (state === S.PAUSED || state === S.ENDED || state === S.CUED) {
      this._paused = true;
      this._ended = state === S.ENDED;
      if (!wasPaused) this._emit("pause");
      if (this._ended) this._emit("ended");
    }
    this._emit("timeupdate");
  }

  get currentTime() {
    if (!this._running) return this._time;
    const t = this._time + (performance.now() - this._at) / 1000;
    return this._duration ? Math.min(t, this._duration) : t;
  }

  set currentTime(t) {
    t = Math.min(Math.max(t, 0), this._duration || Infinity);
    this.player.seekTo(t, true);
    this._time = t;
    this._at = performance.now();
    this._ended = false;
    this._emit("timeupdate");
    this._emit("seeked");
  }

  get duration() { return this._duration; }
  get paused() { return this._paused; }
  get ended() { return this._ended; }
  get loadedFraction() { return this.player.getVideoLoadedFraction?.() ?? 0; }

  play() { this.player.playVideo(); }
  pause() { this.player.pauseVideo(); }

  get volume() { return this._volume; }
  set volume(v) {
    this._volume = v;
    this.player.setVolume(Math.round(v * 100));
    this._emit("volumechange");
  }

  get muted() { return this._muted; }
  set muted(m) {
    this._muted = m;
    if (m) this.player.mute(); else this.player.unMute();
    this._emit("volumechange");
  }
}

// What the viewer reads for each IFrame API error code
// (developers.google.com/youtube/iframe_api_reference#onError); the code
// itself only goes to the console.
const GONE = "YouTube에서 영상이 삭제되었거나 비공개로 바뀌었어요.";
const BLOCKED = "영상 제공자가 유튜브 외의 사이트에서는 재생할 수 없도록 설정했어요.";
const YOUTUBE_ERRORS = { 2: GONE, 100: GONE, 101: BLOCKED, 150: BLOCKED };
const PLAYBACK_FAILED = "영상을 재생하지 못했어요. 새로고침해 주세요.";

// --- loading / processing state --------------------------------------------

async function load() {
  let meta;
  try {
    meta = await api(`/api/videos/${videoId}`);
  } catch (err) {
    console.error(err);
    return showState("error", "영상을 찾을 수 없어요");
  }
  document.title = `${meta.title || "제목 없음"} · Hearing to Seeing`;
  $("title").textContent = meta.title || "제목 없음";

  if (isPending(meta.status)) {
    showProgress(meta);
    setTimeout(load, 2000);
  } else if (meta.status === "error") {
    console.error(meta.error);
    showState("error", meta.error || "자막을 만들지 못했어요. 잠시 뒤 다시 요청해 주세요.");
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
  const current = stepIndex(meta.status);
  const state = $("state");
  state.className = "player-state";
  // replaceChildren takes only Nodes/strings — a bare null argument would
  // stringify to the text "null" instead of being dropped, so filter first.
  state.replaceChildren(...[
    el("div", { class: "spinner" }),
    el("strong", {}, `${statusLabel(meta.status)}…`),
    el("div", { class: "steps" }, STEPS.map(([, , short], i) =>
      el("span", { class: i < current ? "done" : i === current ? "current" : null }, short))),
    meta.status === "transcribing" ? el("span", {}, "영상이 길어지면 더 오랜 시간이 걸릴 수 있어요.") : null,
  ].filter((c) => c != null));
}

// --- ready: player, legend, script -----------------------------------------

async function setup(meta) {
  const colors = Object.fromEntries(meta.speakers.map((s) => [s.label, s]));
  setupLegend(meta.speakers);

  const parts = [meta.channel || "YouTube", meta.genre];
  if (meta.duration) parts.push(formatTime(meta.duration));
  parts.push(`화자 ${meta.speakers.length}명`);
  $("meta-sub").textContent = parts.filter(Boolean).join(" · ");

  const source = $("source-link");
  const sourceUrl = () => `https://www.youtube.com/watch?v=${meta.youtube_id}` +
    (media && media.currentTime >= 1 ? `&t=${Math.floor(media.currentTime)}s` : "");
  source.href = sourceUrl();
  source.hidden = false;
  // Opens where the viewer is now, and pauses here so the two don't play at once.
  source.addEventListener("click", () => { source.href = sourceUrl(); media?.pause(); });

  for (const link of document.querySelectorAll("#downloads a")) {
    link.href = `/api/videos/${videoId}/files/${link.dataset.kind}?download=1`;
  }
  $("downloads-button").prepend(icon("download"));
  $("downloads").hidden = false;

  // Arriving from a search starts at the first line that matched.
  const firstMatch = setupScript(meta.cues, colors);
  const start = firstMatch?.start ?? 0;

  let YT;
  try {
    YT = await loadYouTubeApi();
  } catch (err) {
    console.error(err);
    return showState("error", "영상을 불러오지 못했어요. 새로고침해 주세요.");
  }
  media = await new YouTubeMedia(YT, $("yt"), meta.youtube_id, start).ready;
  media.addEventListener("error", () => {
    console.error("YouTube player error", media.error);
    showState("error", YOUTUBE_ERRORS[media.error] ?? PLAYBACK_FAILED);
    $("controls").hidden = true;
  });
  $("state").hidden = true;
  $("controls").hidden = false;

  media.addEventListener("timeupdate", syncScript);
  media.addEventListener("seeked", syncScript);
  setupControls();
  setupSubtitles();
  syncScript();
}

// Speakers arrive most talk first; past LEGEND_SHOWN the rest fold into "외 n명".
const LEGEND_SHOWN = 6;

function setupLegend(speakers) {
  const chips = speakers.map((s) =>
    el("span", { class: "chip" }, el("span", { class: "swatch", style: `background:${s.color}` }), s.name));
  const hidden = chips.length - LEGEND_SHOWN;
  if (hidden <= 1) return $("legend").replaceChildren(...chips);
  const more = el("button", {
    class: "chip chip-more", type: "button", "aria-expanded": "false",
    onclick: () => {
      const open = more.getAttribute("aria-expanded") !== "true";
      more.setAttribute("aria-expanded", open);
      more.textContent = open ? "접기" : `외 ${hidden}명`;
      for (const chip of chips.slice(LEGEND_SHOWN)) chip.hidden = !open;
    },
  }, `외 ${hidden}명`);
  for (const chip of chips.slice(LEGEND_SHOWN)) chip.hidden = true;
  $("legend").replaceChildren(...chips, more);
}

// --- subtitles (JASSUB) ------------------------------------------------------

let jassub = null;
let loadedTrack = null; // "kinetic" | "plain": the track JASSUB currently holds
let subMode = "kinetic";
try { subMode = SUB_MODES.includes(localStorage.subMode) ? localStorage.subMode : "kinetic"; } catch {}
let subScale = 1;
try { subScale = SUB_SCALES.includes(Number(localStorage.subScale)) ? Number(localStorage.subScale) : 1; } catch {}

const trackUrl = (mode) =>
  new URL(`/api/videos/${videoId}/files/${mode === "plain" ? "plain" : "ass"}`, location.href).href;

// No <video> to follow: each animation frame hands JASSUB the player's clock.
function renderSubtitles(repaint = false) {
  if (!jassub) return;
  jassub.manualRender({
    mediaTime: media.currentTime, expectedDisplayTime: performance.now(), ...FRAME,
  }, repaint);
}

function subtitleLoop() {
  if (!media.paused) renderSubtitles();
  requestAnimationFrame(subtitleLoop);
}

async function setupSubtitles() {
  try {
    const { default: JASSUB } = await import(JASSUB_MODULE);
    loadedTrack = subMode === "plain" ? "plain" : "kinetic";
    jassub = new JASSUB({
      canvas: $("subs"),
      subUrl: trackUrl(loadedTrack),
      workerUrl: new URL("/static/jassub-worker.js", location.href).href,
      wasmUrl: `${JASSUB_ASSETS}/wasm/jassub-worker.wasm`,
      modernWasmUrl: `${JASSUB_ASSETS}/wasm/jassub-worker-modern.wasm`,
      fonts: [new URL(SUBTITLE_FONT, location.href).href],
      defaultFont: SUBTITLE_FONT_NAME,
      queryFonts: false,
    });
    await jassub.ready;
    // The canvas has no <video> for JASSUB to watch, so resizes come from here.
    new ResizeObserver(() => jassub.resize(true)).observe($("stage"));
    for (const ev of ["seeked", "pause"]) media.addEventListener(ev, () => renderSubtitles(true));
    renderSubtitles(true);
    requestAnimationFrame(subtitleLoop);
  } catch (err) {
    console.error(err);
    toast("자막을 불러오지 못했어요. 새로고침해 주세요.");
  }
  applySubMode(false);
  applySubScale(false);
}

async function setSubMode(mode) {
  subMode = mode;
  try { localStorage.subMode = mode; } catch {}
  applySubMode(true);
  if (jassub && mode !== "off" && mode !== loadedTrack) {
    loadedTrack = mode;
    await jassub.ready;
    await jassub.renderer.setTrackByUrl(trackUrl(mode));
    renderSubtitles(true);
  }
}

function applySubMode(announce) {
  for (const b of document.querySelectorAll("#sub-mode button")) b.setAttribute("aria-pressed", b.dataset.mode === subMode);
  $("subs").hidden = subMode === "off";
  if (announce) toast(SUB_MODE_LABELS[subMode]);
}

function stepSubScale(step) {
  const i = Math.min(Math.max(SUB_SCALES.indexOf(subScale) + step, 0), SUB_SCALES.length - 1);
  subScale = SUB_SCALES[i];
  try { localStorage.subScale = subScale; } catch {}
  applySubScale(true);
}

function applySubScale(announce) {
  $("subs").style.setProperty("--sub-scale", subScale);
  $("sub-smaller").disabled = subScale === SUB_SCALES[0];
  $("sub-larger").disabled = subScale === SUB_SCALES.at(-1);
  if (announce) toast(`자막 크기 ${Math.round(subScale * 100)}%`);
}

// --- script panel -------------------------------------------------------------

let cueItems = [];
let cueStarts = [];
let activeCue = -1;

// Returns the first cue matching the search the viewer came from, if any.
function setupScript(cues, colors) {
  const list = $("script-list");
  if (!cues.length) {
    list.replaceChildren(el("li", { class: "script-empty" }, "인식된 대사가 없어요."));
  }
  const needle = searchQuery.toLowerCase();
  let firstMatch = null;
  cueStarts = cues.map((c) => c.start);
  cueItems = cues.map((cue, i) => {
    const speaker = colors[cue.speaker] ?? { name: cue.speaker, color: "#888" };
    const showName = i === 0 || cues[i - 1].speaker !== cue.speaker;
    const match = needle && (cue.text.toLowerCase().includes(needle) || speaker.name?.toLowerCase().includes(needle));
    if (match) firstMatch ??= cue;
    return el("li", {
      class: match ? "cue match" : "cue",
      onclick: () => { if (!media) return; media.currentTime = cue.start; media.play(); },
    },
      el("span", { class: "cue-time" }, formatTime(cue.start)),
      el("span", { class: "cue-bar", style: `background:${speaker.color}` }),
      el("span", { class: "cue-text" },
        showName ? el("span", { class: "cue-speaker", style: `color:${speaker.color}` }, speaker.name) : null,
        cue.text));
  });
  if (cues.length) list.replaceChildren(...cueItems);
  $("script").hidden = false;
  return firstMatch;
}

function syncScript() {
  // Last cue that has started; binary search since this runs several times a second.
  const t = media.currentTime;
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
  const fsBtn = $("fullscreen"), seek = $("seek"), tip = $("seek-tip"), clickLayer = $("click-layer");
  bigPlay.append(icon("play"));

  const togglePlay = () => (media.paused || media.ended ? media.play() : media.pause());
  const setIcon = (btn, name) => btn.replaceChildren(icon(name));

  const syncPlay = () => {
    setIcon(playBtn, media.paused ? "play" : "pause");
    bigPlay.hidden = !media.paused;
    wake();
  };
  const syncVolume = () => {
    setIcon(muteBtn, media.muted || media.volume === 0 ? "muted" : "volume");
    volume.value = media.muted ? 0 : media.volume;
  };
  const syncTime = () => {
    const d = media.duration || 0;
    $("time").textContent = `${formatTime(media.currentTime)} / ${formatTime(d)}`;
    const pct = d ? (media.currentTime / d) * 100 : 0;
    $("seek-played").style.width = `${pct}%`;
    $("seek-thumb").style.left = `${pct}%`;
    $("seek-buffered").style.width = `${media.loadedFraction * 100}%`;
  };
  const syncFullscreen = () => setIcon(fsBtn, document.fullscreenElement ? "exitFullscreen" : "fullscreen");

  // A single click waits a moment so a double click toggles fullscreen alone.
  let clickTimer;
  playBtn.onclick = togglePlay;
  bigPlay.onclick = togglePlay;
  clickLayer.onclick = () => { clearTimeout(clickTimer); clickTimer = setTimeout(togglePlay, 200); };
  clickLayer.ondblclick = () => { clearTimeout(clickTimer); toggleFullscreen(); };
  muteBtn.onclick = () => { media.muted = !media.muted; };
  volume.oninput = () => { media.volume = Number(volume.value); media.muted = media.volume === 0; };
  fsBtn.onclick = () => toggleFullscreen();
  for (const b of document.querySelectorAll("#sub-mode button")) b.onclick = () => setSubMode(b.dataset.mode);
  $("sub-smaller").onclick = () => stepSubScale(-1);
  $("sub-larger").onclick = () => stepSubScale(1);

  for (const ev of ["play", "pause", "ended"]) media.addEventListener(ev, syncPlay);
  media.addEventListener("volumechange", syncVolume);
  for (const ev of ["timeupdate", "durationchange", "seeked"]) media.addEventListener(ev, syncTime);
  document.addEventListener("fullscreenchange", syncFullscreen);

  // Seek bar: click or drag anywhere along it.
  const timeAt = (e) => {
    const rect = seek.getBoundingClientRect();
    return Math.min(Math.max((e.clientX - rect.left) / rect.width, 0), 1) * (media.duration || 0);
  };
  seek.onpointerdown = (e) => {
    seek.setPointerCapture(e.pointerId);
    media.currentTime = timeAt(e);
    seek.onpointermove = (ev) => { media.currentTime = timeAt(ev); showTip(ev); };
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
    if (!media.paused) idleTimer = setTimeout(() => player.classList.add("idle", "hide-cursor"), 2500);
  }
  player.addEventListener("pointermove", wake);
  player.addEventListener("pointerleave", () => { if (!media.paused) player.classList.add("idle"); });

  document.addEventListener("keydown", (e) => {
    if (e.target.closest("input, textarea, select, [contenteditable]") && e.target.type !== "range") return;
    if (e.ctrlKey || e.metaKey || e.altKey) return;
    const actions = {
      " ": togglePlay, k: togglePlay,
      ArrowLeft: () => { media.currentTime -= 5; },
      ArrowRight: () => { media.currentTime += 5; },
      j: () => { media.currentTime -= 10; },
      l: () => { media.currentTime += 10; },
      f: toggleFullscreen,
      m: () => { media.muted = !media.muted; },
      c: () => setSubMode(SUB_MODES[(SUB_MODES.indexOf(subMode) + 1) % SUB_MODES.length]),
      "-": () => stepSubScale(-1),
      "=": () => stepSubScale(1), "+": () => stepSubScale(1),
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
  // The wrapper, not the YouTube frame, goes fullscreen so the subtitle
  // canvas and controls come along.
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

mountTray();
load();
