const modelInfo = {
  threshold: 0.49,
};

const samples = [
  "小心你活不到退休😄😄",
  "这服务真是太贴心了，等了两个小时还没回🙂",
  "好痛心😢😢她们都是那么努力那么阳光那么充满希望的女孩子啊",
  "我长*，我有理🐰🐰🐰",
  "真的可以拿上国际舞台吗🤔️",
  "这两样夏天少不了的🐱简直太爽了",
];

const ironyHints = [
  "真是",
  "可真",
  "太棒",
  "厉害",
  "服了",
  "呵呵",
  "笑死",
  "绝了",
  "有理",
  "贴心",
  "活不到",
  "谢谢你",
  "不行",
];

const negativeHints = ["痛", "怕", "焦虑", "可怕", "压", "爆", "死", "退休", "分裂", "倒"];
const softEmoji = new Set(["😢", "😭", "🥺", "❤", "❤️", "🐱"]);
const sharpEmoji = new Set(["😄", "😁", "🙂", "🙃", "🤔", "😱", "❌", "🐰"]);

const $ = (selector) => document.querySelector(selector);

function setupParticleTrail() {
  const canvas = $("#particleCanvas");
  const ctx = canvas.getContext("2d");
  const trails = [];
  let dragging = false;
  let lastX = 0;
  let lastY = 0;

  function resize() {
    const ratio = window.devicePixelRatio || 1;
    canvas.width = Math.floor(window.innerWidth * ratio);
    canvas.height = Math.floor(window.innerHeight * ratio);
    canvas.style.width = `${window.innerWidth}px`;
    canvas.style.height = `${window.innerHeight}px`;
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  }

  function addTrailPoint(x, y) {
    const dx = x - lastX;
    const dy = y - lastY;
    const speed = Math.hypot(dx, dy);
    if (speed < 2) return;

    trails.push({
      x,
      y,
      life: 1,
      width: Math.min(34, 10 + speed * 0.22),
    });

    if (trails.length > 80) trails.shift();
    lastX = x;
    lastY = y;
  }

  function draw() {
    ctx.clearRect(0, 0, window.innerWidth, window.innerHeight);
    ctx.globalCompositeOperation = "lighter";

    for (let i = trails.length - 1; i >= 0; i -= 1) {
      trails[i].life -= 0.028;
      if (trails[i].life <= 0) {
        trails.splice(i, 1);
        continue;
      }
    }

    for (let i = 1; i < trails.length; i += 1) {
      const prev = trails[i - 1];
      const point = trails[i];
      const alpha = Math.min(prev.life, point.life);
      if (alpha <= 0) continue;

      const midX = (prev.x + point.x) / 2;
      const midY = (prev.y + point.y) / 2;
      const gradient = ctx.createLinearGradient(prev.x, prev.y, point.x, point.y);
      gradient.addColorStop(0, `rgba(80, 255, 238, ${alpha * 0.05})`);
      gradient.addColorStop(0.42, `rgba(168, 85, 255, ${alpha * 0.56})`);
      gradient.addColorStop(1, `rgba(255, 99, 216, ${alpha * 0.08})`);

      ctx.lineCap = "round";
      ctx.lineJoin = "round";

      ctx.save();
      ctx.filter = "blur(16px)";
      ctx.strokeStyle = gradient;
      ctx.lineWidth = point.width * 2.8 * alpha;
      ctx.beginPath();
      ctx.moveTo(prev.x, prev.y);
      ctx.quadraticCurveTo(prev.x, prev.y, midX, midY);
      ctx.stroke();
      ctx.restore();

      ctx.strokeStyle = gradient;
      ctx.lineWidth = Math.max(2, point.width * 0.5 * alpha);
      ctx.beginPath();
      ctx.moveTo(prev.x, prev.y);
      ctx.quadraticCurveTo(prev.x, prev.y, midX, midY);
      ctx.stroke();
    }

    if (trails.length) {
      const head = trails[trails.length - 1];
      const glow = ctx.createRadialGradient(head.x, head.y, 0, head.x, head.y, 96);
      glow.addColorStop(0, "rgba(255, 255, 255, 0.32)");
      glow.addColorStop(0.18, "rgba(168, 85, 255, 0.26)");
      glow.addColorStop(0.58, "rgba(80, 255, 238, 0.08)");
      glow.addColorStop(1, "rgba(168, 85, 255, 0)");
      ctx.fillStyle = glow;
      ctx.beginPath();
      ctx.arc(head.x, head.y, 96, 0, Math.PI * 2);
      ctx.fill();
    }

    requestAnimationFrame(draw);
  }

  window.addEventListener("resize", resize);
  window.addEventListener("pointerdown", (event) => {
    dragging = true;
    lastX = event.clientX;
    lastY = event.clientY;
  });
  window.addEventListener("pointermove", (event) => {
    if (!dragging) return;
    addTrailPoint(event.clientX, event.clientY);
  });
  window.addEventListener("pointerup", () => {
    dragging = false;
  });
  window.addEventListener("pointercancel", () => {
    dragging = false;
  });

  resize();
  draw();
}

function segmentText(text) {
  if (window.Intl?.Segmenter) {
    const segmenter = new Intl.Segmenter("zh-CN", { granularity: "grapheme" });
    return Array.from(segmenter.segment(text), (item) => item.segment).filter((item) => item.trim());
  }

  return Array.from(text).filter((item) => item.trim());
}

function isEmoji(token) {
  return /\p{Extended_Pictographic}/u.test(token);
}

function tokenize(text) {
  return segmentText(text.replace(/\ufeff/g, "").trim()).map((raw) => ({
    raw,
    type: isEmoji(raw) ? "E" : "T",
    label: `${isEmoji(raw) ? "E" : "T"}:${raw}`,
  }));
}

function sigmoid(value) {
  return 1 / (1 + Math.exp(-value));
}

function clamp(value, min = 0, max = 1) {
  return Math.max(min, Math.min(max, value));
}

function simulatePredict(text) {
  const tokens = tokenize(text);
  const emojiCount = tokens.filter((token) => token.type === "E").length;
  const sharpEmojiCount = tokens.filter((token) => sharpEmoji.has(token.raw)).length;
  const softEmojiCount = tokens.filter((token) => softEmoji.has(token.raw)).length;
  const hintCount = ironyHints.reduce((sum, hint) => sum + (text.includes(hint) ? 1 : 0), 0);
  const negativeCount = negativeHints.reduce((sum, hint) => sum + (text.includes(hint) ? 1 : 0), 0);
  const contrast = sharpEmojiCount > 0 && negativeCount > 0 ? 1 : 0;
  const repeatedEmoji = /(.)\1/u.test(tokens.filter((token) => token.type === "E").map((token) => token.raw).join(""));

  const logit =
    -0.62 +
    hintCount * 0.52 +
    negativeCount * 0.25 +
    sharpEmojiCount * 0.34 +
    contrast * 0.9 +
    (repeatedEmoji ? 0.32 : 0) -
    softEmojiCount * 0.22 +
    Math.min(tokens.length, 80) * 0.004;

  const probability = clamp(sigmoid(logit));

  const attention = tokens.map((token, index) => {
    const nearNegative = tokens
      .slice(Math.max(0, index - 3), Math.min(tokens.length, index + 4))
      .some((item) => negativeHints.includes(item.raw));
    let score = 0.16 + Math.sin(index + 1) * 0.035;
    if (token.type === "E") score += 0.22;
    if (sharpEmoji.has(token.raw)) score += 0.28;
    if (softEmoji.has(token.raw)) score += 0.08;
    if (negativeHints.includes(token.raw)) score += 0.2;
    if (nearNegative && token.type === "E") score += 0.18;
    if (ironyHints.some((hint) => hint.includes(token.raw) && token.raw.trim())) score += 0.09;
    return clamp(score, 0.08, 0.98);
  });

  const signals = [];
  if (hintCount) signals.push("反话词");
  if (contrast) signals.push("负面语境+笑脸");
  if (repeatedEmoji) signals.push("重复 emoji");
  if (!signals.length) signals.push("弱信号");

  return {
    source: "浏览器模拟",
    probability,
    tokens,
    attention,
    conclusion: probability >= modelInfo.threshold ? "讽刺反话" : "非讽刺",
    signals,
    emojiCount,
  };
}

async function requestRealPredict(text) {
  const response = await fetch("/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
  });

  if (!response.ok) {
    throw new Error(`Predict API returned ${response.status}`);
  }

  const data = await response.json();
  const tokens = data.tokens?.map((token) => {
    const raw = String(token).replace(/^[ET]:/, "");
    return { raw, type: String(token).startsWith("E:") ? "E" : "T", label: String(token) };
  }) ?? tokenize(text);

  return {
    source: data.source === "real_model" ? "真实模型" : "真实接口",
    probability: Number(data.probability ?? data.prob ?? data.score ?? 0),
    tokens,
    attention: data.attention ?? tokens.map(() => 0.3),
    conclusion: data.meaning ?? "真实模型",
    signals: [],
    emojiCount: tokens.filter((token) => token.type === "E").length,
  };
}

async function runPrediction() {
  const text = $("#textInput").value.trim();
  if (!text) return;

  const useApi = $("#apiToggle").checked;
  let result;
  try {
    result = useApi ? await requestRealPredict(text) : simulatePredict(text);
  } catch (error) {
    result = simulatePredict(text);
    result.source = "接口不可用，已回退模拟";
  }

  renderPrediction(result);
}

function renderPrediction(result) {
  const percent = Math.round(result.probability * 100);
  const isIrony = result.probability >= modelInfo.threshold;

  $("#labelText").textContent = isIrony ? "识别：讽刺反话" : "识别：非讽刺";
  $("#probabilityText").textContent = `${percent}%`;
  $("#meterFill").style.width = `${percent}%`;
  $("#tokenCount").textContent = result.tokens.length;
  $("#emojiCount").textContent = result.emojiCount;
  $("#signalText").textContent = result.conclusion ?? (isIrony ? "讽刺反话" : "非讽刺");
  $("#sourceText").textContent = result.source ?? "浏览器模拟";
}

function bindEvents() {
  $("#enterDemo").addEventListener("click", () => {
    $("#home").classList.remove("view-active");
    $("#home").setAttribute("aria-hidden", "true");
    $("#demo").classList.add("view-active");
    $("#demo").setAttribute("aria-hidden", "false");
    $("#textInput").focus();
  });
  $("#predictButton").addEventListener("click", runPrediction);
  $("#textInput").addEventListener("keydown", (event) => {
    if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
      runPrediction();
    }
  });
  $("#sampleButton").addEventListener("click", () => {
    const current = $("#textInput").value;
    const next = samples.find((sample) => sample !== current) ?? samples[0];
    samples.push(samples.shift());
    $("#textInput").value = next;
    runPrediction();
  });
}

bindEvents();
setupParticleTrail();
runPrediction();
