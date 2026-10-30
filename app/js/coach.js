// 動作評分：app.py 的 JavaScript 版本，在瀏覽器裡直接執行，不需要伺服器。
//
// 計分方式（與 app.py 相同）:
// 1. 用 MediaPipe 抓出每一幀的 3D 骨架，算出各關節角度。
// 2. 用 DTW（動態時間校正）對齊兩段動作，所以速度不同、晚開始也能正確比對。
// 3. 每個動作只比它的關鍵關節，並依權重加總；角度差在容忍範圍內給滿分。
// 4. 總分 = 姿勢準確度 70% + 動作幅度 15% + 節奏穩定度 15%。

export const SAMPLE_FPS = 10;        // 每秒取樣幾幀
export const MAX_SAMPLES = 900;      // 最多分析 90 秒
export const MIN_VISIBILITY = 0.5;   // 低於這個可見度的點不採用
const TOL_FULL = 10.0;               // 角度差 <= 10 度給滿分
const TOL_ZERO = 45.0;               // 角度差 >= 45 度給 0 分

// 角度定義: 名稱 -> [a, b, c]，量的是 b 點的夾角
// MediaPipe 編號: 11/12 肩, 13/14 肘, 15/16 腕, 23/24 髖, 25/26 膝, 27/28 踝
export const ANGLES = {
  l_elbow: [11, 13, 15], r_elbow: [12, 14, 16],
  l_shoulder: [13, 11, 23], r_shoulder: [14, 12, 24],
  l_hip: [11, 23, 25], r_hip: [12, 24, 26],
  l_knee: [23, 25, 27], r_knee: [24, 26, 28],
};
export const ANGLE_NAMES = Object.keys(ANGLES);
const COL = Object.fromEntries(ANGLE_NAMES.map((n, i) => [n, i]));

const JOINT_LABEL = { elbow: '手肘', shoulder: '肩膀', hip: '髖部 / 軀幹', knee: '膝蓋' };

// 各動作的關鍵關節與權重（加總為 1）
export const EXERCISES = {
  shoulder_press: { shoulder: 0.4, elbow: 0.4, hip: 0.2 },
  side_lateral_raise: { shoulder: 0.6, elbow: 0.2, hip: 0.2 },
  bicep_curl: { elbow: 0.7, shoulder: 0.3 },
  hammer_curl: { elbow: 0.7, shoulder: 0.3 },
  flat_chest_press: { elbow: 0.5, shoulder: 0.5 },
  incline_chest_press: { elbow: 0.5, shoulder: 0.5 },
  dumbbell_chest_press: { elbow: 0.5, shoulder: 0.5 },
  cable_row: { elbow: 0.4, shoulder: 0.3, hip: 0.3 },
  cable_reverse_pull: { shoulder: 0.5, elbow: 0.3, hip: 0.2 },
  neutral_grip_pull_down: { elbow: 0.5, shoulder: 0.5 },
  single_leg_squat: { knee: 0.5, hip: 0.4, shoulder: 0.1 },
};
const DEFAULT_WEIGHTS = { elbow: 0.25, shoulder: 0.25, hip: 0.25, knee: 0.25 };

const clip = (x, lo, hi) => Math.min(hi, Math.max(lo, x));
const mean = (xs) => xs.reduce((s, x) => s + x, 0) / xs.length;

function angle3d(a, b, c) {
  const ba = [a.x - b.x, a.y - b.y, a.z - b.z];
  const bc = [c.x - b.x, c.y - b.y, c.z - b.z];
  const denom = Math.hypot(...ba) * Math.hypot(...bc);
  if (denom < 1e-9) return NaN;
  const cos = (ba[0] * bc[0] + ba[1] * bc[1] + ba[2] * bc[2]) / denom;
  return (Math.acos(clip(cos, -1, 1)) * 180) / Math.PI;
}

/** 一幀的 MediaPipe 結果 -> 8 個關節角度（偵測不到的是 NaN）。 */
export function anglesFromResult(result) {
  const row = new Array(ANGLE_NAMES.length).fill(NaN);
  const world = result?.worldLandmarks?.[0];
  const vis = result?.landmarks?.[0];
  if (!world || !vis) return row;
  ANGLE_NAMES.forEach((name, i) => {
    const [a, b, c] = ANGLES[name];
    const v = Math.min(vis[a].visibility ?? 1, vis[b].visibility ?? 1, vis[c].visibility ?? 1);
    if (v >= MIN_VISIBILITY) row[i] = angle3d(world[a], world[b], world[c]);
  });
  return row;
}

/** 補齊短暫偵測失敗的空洞，並做輕微平滑。偵測率太低的欄位整欄設為 NaN。 */
export function clean(seq) {
  const T = seq.length;
  const out = seq.map((r) => r.slice());
  for (let j = 0; j < ANGLE_NAMES.length; j++) {
    const okIdx = [];
    for (let t = 0; t < T; t++) if (!Number.isNaN(seq[t][j])) okIdx.push(t);
    if (okIdx.length / T < 0.5) {
      for (let t = 0; t < T; t++) out[t][j] = NaN;
      continue;
    }
    // 線性內插（端點外以最近的有效值補，等同 np.interp）
    const col = new Array(T);
    let k = 0;
    for (let t = 0; t < T; t++) {
      while (k < okIdx.length - 1 && okIdx[k + 1] <= t) k++;
      const t0 = okIdx[k];
      if (t <= okIdx[0]) col[t] = seq[okIdx[0]][j];
      else if (t >= okIdx[okIdx.length - 1]) col[t] = seq[okIdx[okIdx.length - 1]][j];
      else {
        const t1 = okIdx[k + 1];
        const v0 = seq[t0][j], v1 = seq[t1][j];
        col[t] = t === t0 ? v0 : v0 + ((v1 - v0) * (t - t0)) / (t1 - t0);
      }
    }
    // 3 點移動平均，邊界用 edge padding
    for (let t = 0; t < T; t++) {
      if (T >= 3) {
        const prev = col[Math.max(0, t - 1)], next = col[Math.min(T - 1, t + 1)];
        out[t][j] = (prev + col[t] + next) / 3;
      } else {
        out[t][j] = col[t];
      }
    }
  }
  return out;
}

/** 左右對調，處理使用者用另一邊做動作或畫面鏡像的情況。 */
function swapSides(seq) {
  return seq.map((row) => {
    const r = row.slice();
    for (const name of ANGLE_NAMES) {
      if (name.startsWith('l_')) {
        const l = COL[name], rr = COL['r_' + name.slice(2)];
        r[l] = row[rr];
        r[rr] = row[l];
      }
    }
    return r;
  });
}

const colAllNaN = (seq, j) => seq.every((r) => Number.isNaN(r[j]));

/** 每對 (參考幀, 使用者幀) 在某個關節的平均角度差，左右分開比。回傳 N×M 矩陣或 null。 */
function groupErrors(ref, usr, joint) {
  const sides = [];
  for (const side of ['l_', 'r_']) {
    const j = COL[side + joint];
    if (colAllNaN(ref, j) || colAllNaN(usr, j)) continue;
    sides.push(j);
  }
  if (!sides.length) return null;
  const n = ref.length, m = usr.length;
  const E = Array.from({ length: n }, () => new Float64Array(m));
  for (let i = 0; i < n; i++) {
    for (let k = 0; k < m; k++) {
      let s = 0;
      for (const j of sides) s += Math.abs(ref[i][j] - usr[k][j]);
      E[i][k] = s / sides.length;
    }
  }
  return E;
}

const toScore = (err) => clip((TOL_ZERO - err) / (TOL_ZERO - TOL_FULL), 0, 1) * 100;

function dtwPath(cost) {
  const n = cost.length, m = cost[0].length;
  const acc = Array.from({ length: n + 1 }, () => new Float64Array(m + 1).fill(Infinity));
  acc[0][0] = 0;
  for (let i = 1; i <= n; i++) {
    const rowCost = cost[i - 1], prev = acc[i - 1], cur = acc[i];
    for (let j = 1; j <= m; j++) {
      cur[j] = rowCost[j - 1] + Math.min(prev[j], prev[j - 1], cur[j - 1]);
    }
  }
  // 回溯最佳路徑（與 np.argmin 相同，平手時取前面的選項）
  let i = n, j = m;
  const path = [];
  while (i > 0 && j > 0) {
    path.push([i - 1, j - 1]);
    const opts = [acc[i - 1][j - 1], acc[i - 1][j], acc[i][j - 1]];
    let k = 0;
    if (opts[1] < opts[k]) k = 1;
    if (opts[2] < opts[k]) k = 2;
    if (k === 0) { i--; j--; } else if (k === 1) { i--; } else { j--; }
  }
  return path.reverse();
}

function compare(ref, usr, weights) {
  const errs = {};
  for (const joint of Object.keys(weights)) {
    const e = groupErrors(ref, usr, joint);
    if (e) errs[joint] = e;
  }
  const joints = Object.keys(errs);
  if (!joints.length) return null;
  const totalW = joints.reduce((s, j) => s + weights[j], 0);
  const w = Object.fromEntries(joints.map((j) => [j, weights[j] / totalW]));

  const n = ref.length, m = usr.length;
  const cost = Array.from({ length: n }, (_, i) => {
    const row = new Float64Array(m);
    for (const j of joints) for (let k = 0; k < m; k++) row[k] += w[j] * errs[j][i][k];
    return row;
  });
  const path = dtwPath(cost);

  const jointScores = {}, jointErr = {};
  for (const j of joints) {
    const e = path.map(([a, b]) => errs[j][a][b]);
    jointScores[j] = mean(e.map(toScore));
    jointErr[j] = mean(e);
  }
  const accuracy = joints.reduce((s, j) => s + w[j] * jointScores[j], 0);
  return { accuracy, path, jointScores, jointErr, w };
}

/** 與 np.percentile 相同的線性內插。 */
function percentile(values, p) {
  const v = [...values].sort((a, b) => a - b);
  const pos = ((v.length - 1) * p) / 100;
  const lo = Math.floor(pos), hi = Math.ceil(pos);
  return v[lo] + (v[hi] - v[lo]) * (pos - lo);
}

function sideColumns(seq, joint) {
  return ['l_', 'r_']
    .map((s) => COL[s + joint])
    .filter((j) => !colAllNaN(seq, j))
    .map((j) => seq.map((r) => r[j]));
}

function motionRange(seq, joint) {
  const cols = sideColumns(seq, joint);
  if (!cols.length) return null;
  // 取變化最大的那一側
  return Math.max(...cols.map((c) => percentile(c, 95) - percentile(c, 5)));
}

/** 用遲滯門檻數角度曲線的來回次數。 */
function countReps(seq, joint) {
  let best = 0;
  for (const raw of sideColumns(seq, joint)) {
    const lo = percentile(raw, 5), hi = percentile(raw, 95);
    if (hi - lo < 15) continue; // 幾乎沒動
    const x = raw.map((v) => (v - lo) / (hi - lo));
    let state = x[0] > 0.5, reps = 0;
    for (const v of x) {
      if (!state && v > 0.7) state = true;
      else if (state && v < 0.3) { state = false; reps++; }
    }
    best = Math.max(best, reps);
  }
  return best;
}

const round = (x, d) => Math.round(x * 10 ** d) / 10 ** d;

/**
 * 比對使用者與示範的角度序列（皆為 anglesFromResult 產生的列）。
 * 回傳格式與 app.py 的 JSON 相同；無法評分時丟出錯誤。
 */
export function analyze(refRaw, usrRaw, exercise) {
  const weights = EXERCISES[exercise] || DEFAULT_WEIGHTS;
  if (!refRaw.length) throw new Error('示範影片沒有可分析的畫面');
  if (!usrRaw.length) throw new Error('沒有錄到畫面，請再試一次');
  const ref = clean(refRaw);
  const emptyRows = usrRaw.filter((r) => r.every(Number.isNaN)).length;
  if (emptyRows / usrRaw.length > 0.8) {
    throw new Error('錄影中幾乎偵測不到人，請確認全身（至少上半身）都在畫面內');
  }
  const usr = clean(usrRaw);

  // 左右正常比一次、對調比一次，取較好的
  const candidates = [compare(ref, usr, weights), compare(ref, swapSides(usr), weights)].filter(Boolean);
  if (!candidates.length) throw new Error('關鍵關節偵測不到，請調整拍攝角度');
  const best = candidates.reduce((a, b) => (b.accuracy > a.accuracy ? b : a));
  const { accuracy, path, jointScores, jointErr, w } = best;

  const feedback = [];

  // 動作幅度：看權重最高的關節
  const keyJoint = Object.keys(w).reduce((a, b) => (w[b] > w[a] ? b : a));
  const romRef = motionRange(ref, keyJoint), romUsr = motionRange(usr, keyJoint);
  let romScore = 100;
  if (romRef && romUsr !== null && romRef > 5) {
    const ratio = romUsr / romRef;
    romScore = clip(1 - Math.abs(1 - ratio), 0, 1) * 100;
    if (ratio < 0.75) feedback.push(`${JOINT_LABEL[keyJoint]}動作幅度不足（約示範的 ${Math.round(ratio * 100)}%）`);
    else if (ratio > 1.3) feedback.push(`${JOINT_LABEL[keyJoint]}動作幅度過大`);
  }

  // 節奏：對齊路徑偏離對角線越多，代表速度越不穩定
  const n = ref.length, m = usr.length;
  let tempoScore = 100;
  if (n > 1 && m > 1) {
    const dev = mean(path.map(([i, j]) => Math.abs(i / (n - 1) - j / (m - 1))));
    tempoScore = clip(1 - dev / 0.25, 0, 1) * 100;
  }
  if (tempoScore < 60) feedback.push('節奏不穩定，盡量跟著示範影片的速度');

  const worst = Object.keys(jointScores).reduce((a, b) => (jointScores[b] < jointScores[a] ? b : a));
  if (jointScores[worst] < 70) {
    feedback.push(`${JOINT_LABEL[worst]}角度與示範差距較大（平均差 ${Math.round(jointErr[worst])}°）`);
  }
  if (!feedback.length) feedback.push('動作很標準，繼續保持！');

  const similarity = 0.7 * accuracy + 0.15 * romScore + 0.15 * tempoScore;
  return {
    similarity: round(similarity, 2),
    accuracy: round(accuracy, 2),
    range: round(romScore, 2),
    tempo: round(tempoScore, 2),
    joints: Object.fromEntries(Object.entries(jointScores).map(([j, s]) => [JOINT_LABEL[j], round(s, 1)])),
    reps: countReps(usr, keyJoint),
    reference_reps: countReps(ref, keyJoint),
    feedback,
  };
}
