// 動作頁面：開啟鏡頭、同時播放示範影片，在瀏覽器裡用 MediaPipe 抓兩邊的骨架並評分。
// 全程在使用者的裝置上執行，影片不會上傳到任何伺服器。
import {
  FilesetResolver,
  PoseLandmarker,
  DrawingUtils,
} from 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs';
import { analyze, anglesFromResult, SAMPLE_FPS, MAX_SAMPLES } from './coach.js';

const WASM_URL = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm';
const MODEL_URL =
  'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task';

const exercise = document.body.dataset.exercise;
const standardVideo = document.getElementById('standardVideo');
const userCamera = document.getElementById('userCamera');
const overlay = document.getElementById('overlay');
const startBtn = document.getElementById('startBtn');
const statusEl = document.getElementById('status');
const resultEl = document.getElementById('similarityDisplay');

let vision;
let userLandmarker, refLandmarker;
let running = false;
let refRows = [], usrRows = [];
let lastSample = 0;
const drawer = new DrawingUtils(overlay.getContext('2d'));

const setStatus = (text) => { statusEl.textContent = text; };

async function createLandmarker() {
  const options = (delegate) => ({
    baseOptions: { modelAssetPath: MODEL_URL, delegate },
    runningMode: 'VIDEO',
    numPoses: 1,
    minPoseDetectionConfidence: 0.5,
    minTrackingConfidence: 0.5,
  });
  try {
    return await PoseLandmarker.createFromOptions(vision, options('GPU'));
  } catch {
    return await PoseLandmarker.createFromOptions(vision, options('CPU'));
  }
}

async function startCamera() {
  if (!navigator.mediaDevices?.getUserMedia) {
    throw new Error('這個瀏覽器不支援鏡頭，請改用 Chrome、Edge 或 Safari，並確認網址是 https 開頭');
  }
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { facingMode: 'user', width: { ideal: 640 }, height: { ideal: 480 } },
    audio: false,
  });
  userCamera.srcObject = stream;
  await new Promise((resolve) => (userCamera.onloadedmetadata = resolve));
  await userCamera.play();
  // 讓畫面比例跟鏡頭一致，骨架才會疊在正確的位置
  const ratio = `${userCamera.videoWidth} / ${userCamera.videoHeight}`;
  userCamera.style.aspectRatio = ratio;
  overlay.style.aspectRatio = ratio;
}

function drawSkeleton(result) {
  const ctx = overlay.getContext('2d');
  if (overlay.width !== userCamera.videoWidth) {
    overlay.width = userCamera.videoWidth;
    overlay.height = userCamera.videoHeight;
  }
  ctx.clearRect(0, 0, overlay.width, overlay.height);
  for (const lm of result.landmarks ?? []) {
    drawer.drawConnectors(lm, PoseLandmarker.POSE_CONNECTIONS, { color: '#FFEB99', lineWidth: 3 });
    drawer.drawLandmarks(lm, { color: '#FFD700', radius: 3 });
  }
}

// 鏡頭的即時骨架一直畫；錄製中每 1/SAMPLE_FPS 秒同時取樣使用者與示範影片
function loop() {
  const now = performance.now();
  if (userCamera.readyState >= 2) {
    const userResult = userLandmarker.detectForVideo(userCamera, now);
    drawSkeleton(userResult);

    if (running && now - lastSample >= 1000 / SAMPLE_FPS) {
      lastSample = now;
      usrRows.push(anglesFromResult(userResult));
      if (!standardVideo.paused && standardVideo.readyState >= 2) {
        refRows.push(anglesFromResult(refLandmarker.detectForVideo(standardVideo, now)));
      }
      if (usrRows.length >= MAX_SAMPLES) finish();
    }
  }
  requestAnimationFrame(loop);
}

function start() {
  refRows = [];
  usrRows = [];
  lastSample = 0;
  resultEl.innerHTML = '';
  setStatus('跟著示範影片做動作，影片結束後自動評分');
  startBtn.disabled = true;
  standardVideo.currentTime = 0;
  standardVideo.play();
  running = true;
}

function finish() {
  if (!running) return;
  running = false;
  standardVideo.pause();
  startBtn.disabled = false;
  setStatus('分析中…');

  // 讓畫面先更新再開始計算
  setTimeout(() => {
    try {
      const r = analyze(refRows, usrRows, exercise);
      resultEl.innerHTML = `
        <div class="score">相似度 ${r.similarity}%</div>
        <div class="breakdown">姿勢 ${r.accuracy} · 幅度 ${r.range} · 節奏 ${r.tempo} · 次數 ${r.reps} / ${r.reference_reps}</div>
        <ul class="feedback">${r.feedback.map((f) => `<li>${f}</li>`).join('')}</ul>`;
      saveHistory(r);
      setStatus('完成！可以再按一次「開始」重新挑戰');
    } catch (err) {
      resultEl.textContent = `分析失敗：${err.message}`;
      setStatus('');
    }
  }, 30);
}

// 把每次成績存在這台裝置上，之後可以做紀錄頁
function saveHistory(r) {
  try {
    const key = 'wwm-history';
    const history = JSON.parse(localStorage.getItem(key) || '[]');
    history.push({ exercise, similarity: r.similarity, reps: r.reps, time: new Date().toISOString() });
    localStorage.setItem(key, JSON.stringify(history.slice(-200)));
  } catch {
    // 無痕模式等情況存不了，不影響評分
  }
}

async function init() {
  startBtn.disabled = true;
  try {
    setStatus('載入 AI 模型中（第一次約需 10 秒）…');
    vision = await FilesetResolver.forVisionTasks(WASM_URL);
    // 使用者與示範影片各用一個偵測器，避免追蹤狀態互相干擾
    [userLandmarker, refLandmarker] = await Promise.all([createLandmarker(), createLandmarker()]);
    setStatus('開啟鏡頭中…');
    await startCamera();
    setStatus('站到鏡頭前，讓上半身（最好是全身）都在畫面內，然後按「開始」');
    startBtn.disabled = false;
    requestAnimationFrame(loop);
  } catch (err) {
    console.error(err);
    const denied = err?.name === 'NotAllowedError';
    setStatus(denied ? '需要鏡頭權限才能評分，請在網址列允許使用鏡頭後重新整理' : `無法啟動：${err.message}`);
  }
}

startBtn.addEventListener('click', start);
standardVideo.addEventListener('ended', finish);
init();
