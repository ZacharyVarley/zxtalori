#!/usr/bin/env node
// The web app end to end in headless Chrome (WebGPU), driven over the DevTools protocol with no
// npm packages (Node 22+: the built-in WebSocket).
//
//   node tests/test_web.mjs                        the repo's app (zig build wasm first)
//   node tests/test_web.mjs --dir dist/web         a built site
//   CHROME=/path/to/chrome node tests/test_web.mjs
//   node tests/test_web.mjs --dir dist/web --save-pipelines dist/web/engine/pipelines.json
//                                                  … and record the pipelines it compiled: the
//                                                  site's first-visit warm start (web/zxtalori.js)
//
// The startup sample (O, h=6) with its covering check, repulsion steps, covering optimization,
// the point colors following the objective, a million-point grid drawn on the GPU (discs, then
// its SO(3) images as a density, and a GPU pick), and a 25,000-point Haar sample relaxed with the
// near field. Fails on any error the page logs.
import { spawn } from "node:child_process";
import { existsSync, mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const args = process.argv.slice(2);
const dir = args.includes("--dir") ? args[args.indexOf("--dir") + 1] : null;
const savePipelines = args.includes("--save-pipelines") ? args[args.indexOf("--save-pipelines") + 1] : null;
const port = 4300 + Math.floor(Math.random() * 500);
const appUrl = `http://127.0.0.1:${port}/`;

function chromePath() {
  if (process.env.CHROME) return process.env.CHROME;
  const c = {
    darwin: ["/Applications/Google Chrome.app/Contents/MacOS/Google Chrome", "/Applications/Chromium.app/Contents/MacOS/Chromium"],
    win32: ["C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe", "C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe"],
    linux: ["/usr/bin/google-chrome", "/usr/bin/google-chrome-stable", "/usr/bin/chromium", "/usr/bin/chromium-browser"],
  }[process.platform] || [];
  const found = c.find((p) => existsSync(p));
  if (!found) throw new Error("no Chrome found; set CHROME");
  return found;
}

const failures = [];
// In GitHub Actions a failure is also an annotation, shown on the run's page without its log.
const annotate = (text) => String(text).replace(/%/g, "%25").replace(/\r/g, "%0D").replace(/\n/g, "%0A").slice(0, 4000);
const check = (name, ok, info = "") => {
  console.log(`${ok ? "ok  " : "FAIL"} ${name}${info ? "  " + info : ""}`);
  if (!ok) failures.push(name);
  if (!ok && process.env.GITHUB_ACTIONS) console.log(`::error title=web app::${annotate(`${name}  ${info}`)}`);
};

// ── the static server and the browser ──
const server = spawn(process.execPath, [join(ROOT, "scripts", "serve.mjs"), "--port", String(port), ...(dir ? ["--dir", dir] : [])], { stdio: "ignore" });
const profile = mkdtempSync(join(tmpdir(), "zxtalori-chrome-"));
const flags = [
  "--headless=new", "--remote-debugging-port=0", `--user-data-dir=${profile}`, "--no-first-run", "--no-default-browser-check",
  "--enable-unsafe-webgpu", "--ignore-gpu-blocklist", "--enable-features=Vulkan", "--window-size=1500,900",
];
const chrome = spawn(chromePath(), [...flags, "about:blank"], { stdio: ["ignore", "ignore", "pipe"] });
const cleanup = () => {
  try { chrome.kill(); } catch { /* gone */ }
  try { server.kill(); } catch { /* gone */ }
  try { rmSync(profile, { recursive: true, force: true }); } catch { /* locked */ }
};
process.on("exit", cleanup);

const wsUrl = await new Promise((res, rej) => {
  let err = "";
  const t = setTimeout(() => rej(new Error("Chrome did not start: " + err.slice(-500))), 30000);
  chrome.stderr.on("data", (d) => {
    err += d;
    const m = /DevTools listening on (ws:\/\/\S+)/.exec(err);
    if (m) { clearTimeout(t); res(m[1]); }
  });
});

// ── a minimal DevTools protocol client ──
const ws = new WebSocket(wsUrl);
await new Promise((r, j) => { ws.onopen = r; ws.onerror = j; });
let seq = 0;
const pending = new Map();
const pageErrors = [];
ws.onmessage = (ev) => {
  const m = JSON.parse(ev.data);
  if (m.id && pending.has(m.id)) {
    const { res, rej } = pending.get(m.id);
    pending.delete(m.id);
    if (m.error) rej(new Error(m.error.message)); else res(m.result);
  } else if (m.method === "Runtime.exceptionThrown") {
    pageErrors.push(m.params.exceptionDetails.exception?.description || m.params.exceptionDetails.text);
  } else if (m.method === "Runtime.consoleAPICalled" && m.params.type === "error") {
    pageErrors.push(m.params.args.map((a) => a.value ?? a.description).join(" "));
  }
};
const send = (method, params = {}, sessionId) => new Promise((res, rej) => {
  const id = ++seq;
  pending.set(id, { res, rej });
  ws.send(JSON.stringify({ id, method, params, ...(sessionId ? { sessionId } : {}) }));
});
const { targetId } = await send("Target.createTarget", { url: "about:blank" });
const { sessionId } = await send("Target.attachToTarget", { targetId, flatten: true });
const evaluate = async (expr, timeout = 600000) => {
  const r = await send("Runtime.evaluate", { expression: `(async () => { ${expr} })()`, awaitPromise: true, returnByValue: true, timeout }, sessionId);
  if (r.exceptionDetails) throw new Error(r.exceptionDetails.exception?.description || r.exceptionDetails.text);
  return r.result.value;
};
await send("Runtime.enable", {}, sessionId);
await send("Page.navigate", { url: appUrl }, sessionId);

// Page helpers: wait for the app to finish its current operation, read a measure, the log.
const helpers = `
  const $ = (id) => document.getElementById(id);
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  const idle = async () => { await sleep(300); for (let i = 0; i < 6000 && document.body.classList.contains("busy"); i++) await sleep(100); };
  const num = (id) => parseFloat(($(id).textContent || "").replace(/[^0-9.eE+-]/g, ""));
  const log = () => [...document.querySelectorAll("#log-list li")].map((li) => li.textContent);
  const errors = () => [...document.querySelectorAll('#log-list li[data-level="error"]')].map((li) => li.textContent);
  const pick = (name, value) => { const input = document.querySelector('input[name="' + name + '"][value="' + value + '"]'); input.click(); input.dispatchEvent(new Event("change", { bubbles: true })); };
  const run = async (iterations) => { $("iterations").value = String(iterations); $("relax-button").click(); await idle(); };`;

try {
  const ready = await evaluate(`${helpers}
    // the page may not have loaded yet; it generates a sample and checks its covering itself.
    // A file that fails to load (a module left out of the site) stops the app before it starts.
    const missing = () => performance.getEntriesByType("resource").filter((e) => e.responseStatus >= 400 && /\.(js|css|wasm|html)$/.test(new URL(e.name).pathname)).map((e) => e.name);
    for (let i = 0; i < 3000 && !missing().length && !(($("gpu-badge")?.dataset.state === "ready" && $("generate-button") && !$("generate-button").disabled) || $("gpu-badge")?.dataset.state === "error"); i++) await sleep(100);
    if (missing().length) return { missing: missing() };
    await idle();
    for (let i = 0; i < 1200 && $("covering-radius").textContent === "—" && !errors().length; i++) await sleep(100);
    return { gpu: $("gpu-badge").dataset.state, adapter: $("gpu-badge").textContent, count: $("sample-badge").textContent.split(" ")[0],
             separation: num("separation"), covering: num("covering-radius"), errors: errors() };`);
  if (ready.missing) throw new Error("the site is missing files: " + ready.missing.join(", "));
  check("the app starts on WebGPU", ready.gpu === "ready", ready.adapter);
  if (ready.gpu !== "ready") throw new Error("no WebGPU: " + ready.adapter);
  check("the startup sample is the O inverse Hopf grid nearest 1,000", ready.count === "1,048", ready.count);
  check("its separation and covering radius are measured", ready.separation > 6 && ready.separation < 8 && ready.covering > 6 && ready.covering < 9,
    `separation ${ready.separation}°, covering ${ready.covering}°`);

  const relaxed = await evaluate(`${helpers}
    pick("objective", "repulsion");
    const before = num("energy");
    await run(30);
    return { before, after: num("energy"), iteration: $("iteration-count").textContent, separation: num("separation"),
             key: $("color-name").textContent, log: log().slice(-3), errors: errors() };`);
  check("30 repulsion steps lower the energy", relaxed.after < relaxed.before && relaxed.iteration === "30",
    `${relaxed.before} → ${relaxed.after}, step ${relaxed.iteration}`);
  check("points are colored by nearest neighbour after repulsion", relaxed.key.startsWith("Nearest neighbour"), relaxed.key);

  const optimized = await evaluate(`${helpers}
    const before = num("covering-radius");
    pick("objective", "covering");
    await run(20);
    return { before, after: num("covering-radius"), key: $("color-name").textContent, log: log().slice(-2), errors: errors() };`);
  check("covering optimization does not raise the covering radius", optimized.after <= optimized.before + 1e-4,
    `${optimized.before}° → ${optimized.after}°`);
  check("points are colored by the largest adjacent cap after covering optimization", optimized.key.startsWith("Largest adjacent cap"), optimized.key);

  const cut = await evaluate(`${helpers}
    const shown = () => +$("display-count").textContent.split(" ")[0].replace(/,/g, "");
    await sleep(200);
    const all = shown();
    $("cut-axis").value = "0"; $("cut-axis").dispatchEvent(new Event("change"));
    await sleep(200);
    const half = shown(), controls = !$("cut-offset").hidden && !$("cut-flip").hidden;
    $("cut-offset").value = "1"; $("cut-offset").dispatchEvent(new Event("input"));
    await sleep(200);
    const none = shown();
    $("cut-flip").click();
    await sleep(200);
    const flipped = shown(), label = $("cut-flip").textContent;
    $("cut-offset").value = "0"; $("cut-axis").value = "none"; $("cut-axis").dispatchEvent(new Event("change"));
    await sleep(200);
    return { all, half, none, flipped, label, controls, back: shown(), errors: errors() };`);
  check("a cut plane hides one side and moves along its axis", cut.controls && cut.half > 0 && cut.half < cut.all && cut.none === 0 && cut.flipped === cut.all && cut.label === "keep −x" && cut.back === cut.all,
    `${cut.all} → ${cut.half} → ${cut.none} → ${cut.flipped} (${cut.label}) → ${cut.back}`);

  // A wide plot keeps its controls at its sides: view, projection and cut on the left, the
  // colour key (an upright bar) on the right, both clear of the plotted zone's middle.
  const rails = await evaluate(`${helpers}
    const box = e => e.getBoundingClientRect(), plot = box($("canvas-wrap")), mid = plot.left + plot.width / 2;
    const left = box(document.querySelector(".rail-left")), right = box(document.querySelector(".rail-right")), bar = box($("color-hist"));
    return { aspect: plot.width / plot.height, left: left.right < mid - plot.height * .25, right: right.left > mid + plot.height * .25,
             upright: bar.height > bar.width, inside: left.left >= plot.left && right.right <= plot.right, errors: errors() };`);
  check("a wide plot has its view controls on the left and an upright colour key on the right",
    rails.aspect > 1.35 && rails.left && rails.right && rails.upright && rails.inside, JSON.stringify(rails));

  const transport = await evaluate(`${helpers}
    pick("grid-kind", "transport");
    const shown = !$("h-field").hidden && $("count-field").hidden;
    $("tab-instantiate").click();
    $("generate-button").click();
    await idle();
    const optimizeTab = $("tab-optimize").getAttribute("aria-selected") === "true" && !$("optimize-panel").hidden;
    const count = $("sample-badge").textContent.split(" ")[0];
    pick("grid-kind", "hopf");
    pick("group", "D4");
    const hidden = $("grid-kind").hidden && $("h-field").hidden && !$("count-field").hidden;
    $("generate-button").click();
    await idle();
    const dihedral = $("sample-badge").textContent.split(" ")[0];
    pick("group", "O");
    return { shown, count, hidden, dihedral, optimizeTab, errors: errors() };`);
  check("the transport construction generates the FCC grid at h=6", transport.shown && transport.count === "1,043", transport.count);
  check("Generate opens the Optimize tab", transport.optimizeTab);
  check("D4 uses the inverse Hopf grid nearest 1,000", transport.hidden && transport.dihedral === "978", transport.dihedral);

  const identity = await evaluate(`${helpers}
    pick("group", "D3");
    $("generate-button").click();
    await idle();
    const base = $("sample-badge").textContent.split(" ")[0];
    $("add-identity").click();
    await idle();
    const added = $("sample-badge").textContent.split(" ")[0];
    $("fix-identity").click();
    await idle();
    $("add-identity").click();
    await idle();
    const removed = $("sample-badge").textContent.split(" ")[0], fixed = $("fix-identity").checked;
    pick("group", "O");
    return { base, added, removed, fixed, errors: errors() };`);
  check("the identity toggle adds and removes the point at once", identity.added === String(+identity.base.replace(",", "") + 1).replace(/(\d)(\d{3})$/, "$1,$2") && identity.removed === identity.base && !identity.fixed,
    `${identity.base} → ${identity.added} → ${identity.removed}`);

  const auto = await evaluate(`${helpers}
    $("target-count").value = "300";
    $("generate-button").click();
    await idle();
    pick("objective", "auto");
    const steps = +$("iterations").value;
    $("relax-button").click();
    await idle();
    const lines = log();
    pick("objective", "repulsion");
    $("target-count").value = "1000";
    return { steps, kept: num("separation"), relaxed: lines.filter((l) => l.includes("Measured")).at(-2) || "", line: lines.filter((l) => l.includes("Covering optimization")).pop() || "", errors: errors() };`);
  const taken = +(/Covering optimization ([\d,]+) steps/.exec(auto.line)?.[1] || "0").replace(",", "");
  check("Auto runs both stages and stops the covering stage at a plateau", taken > 0 && taken < auto.steps && !auto.errors.length, auto.line);
  const floor = +(/separation floor ([\d.]+)°/.exec(auto.line)?.[1] || "NaN");
  check("Auto keeps 99% of the separation repulsion reached", auto.kept >= floor * (1 - 1e-4), `${auto.kept}° against a floor of ${floor}°`);

  const large = await evaluate(`${helpers}
    $("target-count").value = "1000000";
    $("generate-button").click();
    await idle();
    const points = $("display-count").textContent, key = $("color-name").textContent, spread = $("color-spread").textContent;
    const manual = $("covering-mode").disabled && $("covering-mode").value === "manual" && /Check now/.test($("covering-mode").title);
    pick("view-domain", "so3");
    await sleep(400);
    const density = $("display-count").textContent;
    const c = $("view"), r = c.getBoundingClientRect(), x = r.left + r.width / 2 + 12, y = r.top + r.height / 2 + 6;
    c.dispatchEvent(new PointerEvent("pointerdown", { clientX: x, clientY: y, pointerId: 1, bubbles: true }));
    c.dispatchEvent(new PointerEvent("pointerup", { clientX: x, clientY: y, pointerId: 1, bubbles: true }));
    await sleep(800);
    const selected = $("selection-index").textContent;
    pick("view-domain", "fz");
    $("target-count").value = "1000";
    return { points, density, key, spread, selected, manual, errors: errors() };`);
  check("covering updates show Manual, locked, for a sample too large to check automatically", large.manual);
  check("a million orientations draw as GPU points, colored by their spacing", /GPU points/.test(large.points) && large.key.startsWith("Nearest neighbour") && /CV/.test(large.spread), `${large.points} · ${large.key} ${large.spread}`);
  check("their SO(3) images draw as a density", /density/.test(large.density), large.density);
  check("a click selects an orientation from the GPU view", /^#\d+/.test(large.selected), large.selected);

  const near = await evaluate(`${helpers}
    pick("count-mode", "random");
    $("target-count").value = "25000";
    $("generate-button").click();
    await idle();
    pick("objective", "repulsion");
    pick("forces", "auto");
    const before = num("energy");
    await run(10);
    return { count: $("sample-badge").textContent.split(" ")[0], line: log().filter((l) => l.includes("Relaxation")).pop() || "", errors: errors() };`);
  check("a 25,000-point Haar sample relaxes with the near field", near.count === "25,000" && /near field/.test(near.line), `${near.count}: ${near.line}`);
  check("the page logged no failures", !near.errors.length, near.errors.join(" | "));

  if (savePipelines) {
    const record = await evaluate(`await new Promise((r) => setTimeout(r, 2500)); return localStorage.getItem("zxtalori-warm-pipelines");`);
    const list = record ? JSON.parse(record).list : [];
    check("the compiled pipelines are recorded", list.length > 20, `${list.length} pipelines`);
    if (record) {
      mkdirSync(dirname(resolve(savePipelines)), { recursive: true });
      writeFileSync(savePipelines, record);
      console.log(`saved ${savePipelines} (${(record.length / 1024).toFixed(0)} KB)`);
    }
  }
} catch (e) {
  check("the run completes", false, e.message);
}
check("no uncaught errors or console errors", !pageErrors.length, pageErrors.slice(0, 3).join(" | "));
console.log(failures.length ? `web app FAILED: ${failures.join("; ")}` : "web app OK");
ws.close();
cleanup();
process.exit(failures.length ? 1 : 0);
