import { spawn } from "node:child_process";
import { mkdir, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";

const root = resolve("docs/ui-ux-audit-2026-06");
const screenshotDir = join(root, "screenshots");
const userDataDir = join(tmpdir(), "agent-business-ui-audit-edge-profile");
const port = 9223;
const edgePath = "C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe";

await mkdir(screenshotDir, { recursive: true });
await rm(userDataDir, { recursive: true, force: true });

const edge = spawn(edgePath, [
  "--headless=new",
  "--disable-gpu",
  "--no-first-run",
  "--no-default-browser-check",
  `--remote-debugging-port=${port}`,
  `--user-data-dir=${userDataDir}`,
  "--window-size=1440,1000",
  "about:blank",
], { stdio: "ignore" });

function delay(ms) {
  return new Promise((resolveDelay) => setTimeout(resolveDelay, ms));
}

async function waitForJson(url, attempts = 50) {
  let lastError;
  for (let idx = 0; idx < attempts; idx += 1) {
    try {
      const response = await fetch(url);
      if (response.ok) return response.json();
      lastError = new Error(`HTTP ${response.status}`);
    } catch (error) {
      lastError = error;
    }
    await delay(200);
  }
  throw lastError;
}

let nextId = 1;
const pending = new Map();
let socket;

function send(method, params = {}) {
  const id = nextId;
  nextId += 1;
  socket.send(JSON.stringify({ id, method, params }));
  return new Promise((resolveSend, rejectSend) => {
    pending.set(id, { resolve: resolveSend, reject: rejectSend });
  });
}

async function navigate(url) {
  let loaded = false;
  const previousMessage = socket.onmessage;
  socket.onmessage = (event) => {
    const message = JSON.parse(event.data);
    if (message.id && pending.has(message.id)) {
      const item = pending.get(message.id);
      pending.delete(message.id);
      if (message.error) item.reject(new Error(message.error.message));
      else item.resolve(message.result);
      return;
    }
    if (message.method === "Page.loadEventFired") loaded = true;
    if (previousMessage) previousMessage(event);
  };
  await send("Page.navigate", { url });
  for (let idx = 0; idx < 80 && !loaded; idx += 1) {
    await delay(100);
  }
  await delay(900);
}

async function evaluate(expression) {
  return send("Runtime.evaluate", {
    expression,
    awaitPromise: true,
    returnByValue: true,
  });
}

async function screenshot(name, fullPage = false) {
  if (fullPage) {
    const metrics = await evaluate(`(() => {
      return {
        width: Math.max(document.documentElement.scrollWidth, document.body.scrollWidth, 1440),
        height: Math.max(document.documentElement.scrollHeight, document.body.scrollHeight, 1000)
      };
    })()`);
    const value = metrics.result?.value || { width: 1440, height: 1000 };
    await send("Emulation.setDeviceMetricsOverride", {
      width: Math.min(value.width, 1800),
      height: Math.min(value.height, 5000),
      deviceScaleFactor: 1,
      mobile: false,
    });
    await delay(200);
  }
  const captured = await send("Page.captureScreenshot", {
    format: "png",
    fromSurface: true,
  });
  const file = join(screenshotDir, name);
  await writeFile(file, Buffer.from(captured.data, "base64"));
  return file;
}

try {
  await waitForJson(`http://127.0.0.1:${port}/json/version`);
  const target = await fetch(`http://127.0.0.1:${port}/json/new?about:blank`, { method: "PUT" }).then((r) => r.json());
  socket = new WebSocket(target.webSocketDebuggerUrl);
  await new Promise((resolveOpen) => {
    socket.onopen = resolveOpen;
  });
  socket.onmessage = (event) => {
    const message = JSON.parse(event.data);
    if (message.id && pending.has(message.id)) {
      const item = pending.get(message.id);
      pending.delete(message.id);
      if (message.error) item.reject(new Error(message.error.message));
      else item.resolve(message.result);
    }
  };

  await send("Page.enable");
  await send("Runtime.enable");
  await send("Emulation.setDeviceMetricsOverride", {
    width: 1440,
    height: 1000,
    deviceScaleFactor: 1,
    mobile: false,
  });

  const files = [];
  await navigate("http://127.0.0.1:8080/admin");
  files.push(await screenshot("01-admin-knowledge-workspace.png"));

  await evaluate(`document.querySelector('button[data-target="view-analytics"]')?.click()`);
  await delay(900);
  files.push(await screenshot("02-admin-analytics-evaluations.png"));

  await evaluate(`document.querySelector('button[data-target="view-system"]')?.click()`);
  await delay(900);
  files.push(await screenshot("03-admin-system-runtime.png"));

  await navigate("http://127.0.0.1:8080/chat");
  files.push(await screenshot("04-chat-rag-surface.png"));

  await navigate("http://127.0.0.1:8080/portal");
  files.push(await screenshot("05-internal-support-portal.png"));

  await send("Emulation.setDeviceMetricsOverride", {
    width: 390,
    height: 844,
    deviceScaleFactor: 2,
    mobile: true,
  });
  await navigate("http://127.0.0.1:8080/admin");
  files.push(await screenshot("06-admin-mobile-knowledge.png"));

  console.log(JSON.stringify({ files }, null, 2));
  socket.close();
} finally {
  edge.kill();
}
