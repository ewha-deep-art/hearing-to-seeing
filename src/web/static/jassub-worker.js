// Module workers must be same-origin, so this file stands in for JASSUB's
// worker and pulls the real one from the CDN (esm.sh — see watch.js).
// The prebuilt es2022 file, not `dist/worker/worker.js`: that path makes esm.sh
// build on request, which fails whenever its build storage does (an HTTP 500
// "no space left on device" left the player without subtitles).
import "https://esm.sh/jassub@2.5.16/es2022/dist/worker/worker.mjs";
