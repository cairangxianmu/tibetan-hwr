/**
 * Tibetan HWR — Frontend logic
 *
 * Features:
 *   - HTML5 Canvas drawing (mouse + touch)
 *   - Undo (stroke-level history)
 *   - Image upload (click + drag-and-drop)
 *   - Mode switching (digit / letter)
 *   - POST /predict → display result + top-5 bar chart
 */

'use strict';

// ── State ────────────────────────────────────────────────────────────────────
let currentMode = 'digit';
let isDrawing = false;
let strokeHistory = [];        // Array of ImageData snapshots (one per stroke)
let currentStroke = null;      // ImageData snapshot at stroke start
let hasMark = false;           // Whether canvas has any content

// ── DOM refs ─────────────────────────────────────────────────────────────────
const canvas       = document.getElementById('canvas');
const ctx          = canvas.getContext('2d');
const canvasHint   = document.getElementById('canvasHint');
const penSlider    = document.getElementById('penSize');
const penSizeVal   = document.getElementById('penSizeVal');
const undoBtn      = document.getElementById('undoBtn');
const clearBtn     = document.getElementById('clearBtn');
const recognizeBtn = document.getElementById('recognizeBtn');
const uploadArea   = document.getElementById('uploadArea');
const fileInput    = document.getElementById('fileInput');
const tabs         = document.querySelectorAll('.tab');

const stateIdle    = document.getElementById('stateIdle');
const stateLoading = document.getElementById('stateLoading');
const stateError   = document.getElementById('stateError');
const stateResult  = document.getElementById('stateResult');
const errorMsg     = document.getElementById('errorMsg');

const resultChar   = document.getElementById('resultChar');
const resultLabel  = document.getElementById('resultLabel');
const resultMode   = document.getElementById('resultMode');
const confidencePct= document.getElementById('confidencePct');
const confidenceFill=document.getElementById('confidenceFill');
const top5List     = document.getElementById('top5List');

// ── Canvas setup ─────────────────────────────────────────────────────────────
function initCanvas() {
  ctx.fillStyle = '#ffffff';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.lineCap    = 'round';
  ctx.lineJoin   = 'round';
  ctx.strokeStyle = '#000000';
  updatePenSize();
}

function updatePenSize() {
  ctx.lineWidth = parseInt(penSlider.value, 10);
  penSizeVal.textContent = penSlider.value;
}

function getPos(e) {
  const rect = canvas.getBoundingClientRect();
  const scaleX = canvas.width  / rect.width;
  const scaleY = canvas.height / rect.height;
  const src = e.touches ? e.touches[0] : e;
  return {
    x: (src.clientX - rect.left) * scaleX,
    y: (src.clientY - rect.top)  * scaleY,
  };
}

// ── Drawing ───────────────────────────────────────────────────────────────────
function startDraw(e) {
  e.preventDefault();
  isDrawing = true;
  currentStroke = ctx.getImageData(0, 0, canvas.width, canvas.height);
  const { x, y } = getPos(e);
  ctx.beginPath();
  ctx.moveTo(x, y);
  hidHint();
}

function draw(e) {
  if (!isDrawing) return;
  e.preventDefault();
  const { x, y } = getPos(e);
  ctx.lineTo(x, y);
  ctx.stroke();
  hasMark = true;
}

function endDraw(e) {
  if (!isDrawing) return;
  isDrawing = false;
  if (currentStroke) {
    strokeHistory.push(currentStroke);
    currentStroke = null;
  }
  ctx.beginPath();
}

canvas.addEventListener('mousedown',  startDraw);
canvas.addEventListener('mousemove',  draw);
canvas.addEventListener('mouseup',    endDraw);
canvas.addEventListener('mouseleave', endDraw);
canvas.addEventListener('touchstart', startDraw, { passive: false });
canvas.addEventListener('touchmove',  draw,      { passive: false });
canvas.addEventListener('touchend',   endDraw);

// ── Hint ──────────────────────────────────────────────────────────────────────
function hidHint() {
  canvasHint.classList.add('hidden');
}

function showHint() {
  canvasHint.classList.remove('hidden');
}

// ── Undo ──────────────────────────────────────────────────────────────────────
undoBtn.addEventListener('click', () => {
  if (strokeHistory.length === 0) return;
  const prev = strokeHistory.pop();
  ctx.putImageData(prev, 0, 0);
  if (strokeHistory.length === 0) {
    hasMark = false;
    showHint();
  }
});

// ── Clear ─────────────────────────────────────────────────────────────────────
clearBtn.addEventListener('click', clearCanvas);

function clearCanvas() {
  ctx.fillStyle = '#ffffff';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  strokeHistory = [];
  hasMark = false;
  showHint();
  showState('idle');
}

// ── Pen size ──────────────────────────────────────────────────────────────────
penSlider.addEventListener('input', updatePenSize);

// ── Mode tabs ─────────────────────────────────────────────────────────────────
tabs.forEach(tab => {
  tab.addEventListener('click', () => {
    tabs.forEach(t => { t.classList.remove('active'); t.setAttribute('aria-selected', 'false'); });
    tab.classList.add('active');
    tab.setAttribute('aria-selected', 'true');
    currentMode = tab.dataset.mode;
    clearCanvas();
  });
});

// ── Keyboard shortcuts ────────────────────────────────────────────────────────
document.addEventListener('keydown', e => {
  if ((e.ctrlKey || e.metaKey) && e.key === 'z') {
    e.preventDefault();
    undoBtn.click();
  }
  if (e.key === 'Enter' && !e.ctrlKey) {
    recognizeBtn.click();
  }
});

// ── Upload area ───────────────────────────────────────────────────────────────
uploadArea.addEventListener('click', () => fileInput.click());

uploadArea.addEventListener('dragover', e => {
  e.preventDefault();
  uploadArea.classList.add('drag-over');
});

uploadArea.addEventListener('dragleave', () => {
  uploadArea.classList.remove('drag-over');
});

uploadArea.addEventListener('drop', e => {
  e.preventDefault();
  uploadArea.classList.remove('drag-over');
  const file = e.dataTransfer.files[0];
  if (file) loadImageFile(file);
});

fileInput.addEventListener('change', () => {
  if (fileInput.files[0]) loadImageFile(fileInput.files[0]);
});

function loadImageFile(file) {
  if (!file.type.startsWith('image/')) return;
  const reader = new FileReader();
  reader.onload = ev => {
    const img = new Image();
    img.onload = () => {
      ctx.fillStyle = '#ffffff';
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      // Scale to fit, center
      const scale = Math.min(canvas.width / img.width, canvas.height / img.height);
      const w = img.width  * scale;
      const h = img.height * scale;
      const x = (canvas.width  - w) / 2;
      const y = (canvas.height - h) / 2;
      ctx.drawImage(img, x, y, w, h);
      strokeHistory = [];
      hasMark = true;
      hidHint();
    };
    img.src = ev.target.result;
  };
  reader.readAsDataURL(file);
}

// ── Recognize ─────────────────────────────────────────────────────────────────
recognizeBtn.addEventListener('click', async () => {
  if (!hasMark) {
    showState('error', '请先在画板上书写字符，或上传图片。');
    return;
  }

  showState('loading');

  const imageBase64 = canvas.toDataURL('image/png');

  try {
    const res = await fetch('/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image: imageBase64, mode: currentMode }),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `服务器错误 (${res.status})`);
    }

    const data = await res.json();
    showResult(data);
  } catch (err) {
    showState('error', err.message || '网络错误，请检查服务是否运行。');
  }
});

// ── Result rendering ──────────────────────────────────────────────────────────
function showState(state, msg = '') {
  stateIdle.classList.add('hidden');
  stateLoading.classList.add('hidden');
  stateError.classList.add('hidden');
  stateResult.classList.add('hidden');

  if (state === 'idle')    stateIdle.classList.remove('hidden');
  if (state === 'loading') stateLoading.classList.remove('hidden');
  if (state === 'error')   { stateError.classList.remove('hidden'); errorMsg.textContent = msg; }
  if (state === 'result')  stateResult.classList.remove('hidden');
}

function showResult(data) {
  const modeLabel = currentMode === 'digit' ? '藏文数字' : '藏文字母';

  resultChar.textContent  = data.character;
  resultLabel.textContent = `类别 ${data.label}  ·  ${data.character}`;
  resultMode.textContent  = modeLabel;

  confidencePct.textContent  = `${data.confidence.toFixed(1)} %`;
  confidenceFill.style.width = `${data.confidence}%`;

  // Top 5
  top5List.innerHTML = '';
  data.top5.forEach((item, i) => {
    const li = document.createElement('li');
    li.className = 'top5-item';
    li.innerHTML = `
      <span class="top5-rank">${i + 1}</span>
      <span class="top5-char">${item.character}</span>
      <div class="top5-bar-track">
        <div class="top5-bar-fill" style="width:${item.confidence}%"></div>
      </div>
      <span class="top5-pct">${item.confidence.toFixed(1)}%</span>
    `;
    top5List.appendChild(li);
  });

  showState('result');
}

// ── Init ──────────────────────────────────────────────────────────────────────
initCanvas();
showState('idle');
