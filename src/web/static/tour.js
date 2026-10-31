import { el } from "/static/common.js";

// A step-by-step tour over the page itself (driver.js): each step highlights a
// real control with a short note. It starts by itself the first time a page is
// opened in this browser, and the "?" button in the top bar replays it.
const DRIVER_MODULE = "https://cdn.jsdelivr.net/npm/driver.js@1.3.1/dist/driver.js.mjs";
const DRIVER_CSS = "https://cdn.jsdelivr.net/npm/driver.js@1.3.1/dist/driver.css";

const seenKey = (name) => `h2s.tour.${name}`;

function seen(name) {
  try { return Boolean(localStorage.getItem(seenKey(name))); } catch { return true; }
}

function markSeen(name) {
  try { localStorage.setItem(seenKey(name), "1"); } catch {}
}

const visible = (selector) => {
  const node = document.querySelector(selector);
  return node && node.getClientRects().length > 0;
};

let loading = null;
function loadDriver() {
  loading ??= (async () => {
    document.head.append(el("link", { rel: "stylesheet", href: DRIVER_CSS }));
    return (await import(DRIVER_MODULE)).driver;
  })();
  return loading;
}

// `steps` are driver.js steps; one whose element isn't on screen is skipped.
// A step with `advanceOnClick` also moves on when its element is clicked (the
// viewer tries the control), as well as with the "다음" button.
// `onStart` runs only when the tour does show.
export async function startTour(name, steps, { force = false, onStart } = {}) {
  if (!force && seen(name)) return;
  const shown = steps.filter((s) => !s.element || visible(s.element));
  if (!shown.length) return;
  let driver;
  try {
    driver = await loadDriver();
  } catch (err) {
    return console.error(err);
  }
  // driver.js 1.3 hands step hooks only {config, state}, not the tour itself.
  let tour = null;
  const next = () => tour?.moveNext();
  tour = driver({
    steps: shown.map((s) => (s.advanceOnClick ? clickToAdvance(s, next) : s)),
    showProgress: true,
    progressText: "{{current}} / {{total}}",
    nextBtnText: "다음",
    prevBtnText: "이전",
    doneBtnText: "시작하기",
    popoverClass: "h2s-tour",
    stagePadding: 6,
    stageRadius: 12,
    overlayOpacity: 0.8,
    onDestroyed: () => markSeen(name),
  });
  onStart?.();
  tour.drive();
}

function clickToAdvance(step, next) {
  const onClick = () => setTimeout(next, 600); // let the change show first
  let target = null;
  return {
    ...step,
    onHighlighted: (element) => {
      target = element;
      target?.addEventListener("click", onClick, { once: true });
    },
    onDeselected: () => target?.removeEventListener("click", onClick),
  };
}

// The "이용 안내" (?) button and "소개" link in the top bar.
export function mountHelp(replay) {
  const bar = document.querySelector(".topbar");
  const links = el("nav", { class: "topbar-links", "aria-label": "안내" },
    el("a", { class: "topbar-link", href: "/about" }, "소개"),
    el("button", {
      class: "icon-btn help-btn", type: "button", "aria-label": "이용 안내 다시 보기", title: "이용 안내",
      onclick: replay,
    }, "?"));
  // Before the page's own buttons (the home page's "자막 만들기"), after the search.
  const action = bar.querySelector(".btn-primary");
  bar.insertBefore(links, action);
}
