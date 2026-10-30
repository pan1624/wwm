"""比對使用者錄影與示範影片的動作相似度。

用法: python app.py <使用者影片> <示範影片> [動作名稱]
結果以一行 JSON 輸出到 stdout（最後一行）。

計分方式:
1. 用 MediaPipe 抓出每一幀的 3D 骨架，算出各關節角度。
2. 用 DTW（動態時間校正）對齊兩段動作，所以速度不同、晚開始也能正確比對。
3. 每個動作只比它的關鍵關節，並依權重加總；角度差在容忍範圍內給滿分。
4. 總分 = 姿勢準確度 70% + 動作幅度 15% + 節奏穩定度 15%。
"""
import json
import os
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("GLOG_minloglevel", "2")

import cv2  # noqa: E402
import mediapipe as mp  # noqa: E402
import numpy as np  # noqa: E402

CACHE_VERSION = 1
SAMPLE_FPS = 10          # 每秒取樣幾幀
MAX_SAMPLES = 900        # 最多分析 90 秒
FRAME_WIDTH = 640        # 縮小畫面加快偵測
MIN_VISIBILITY = 0.5     # 低於這個可見度的點不採用
TOL_FULL = 10.0          # 角度差 <= 10 度給滿分
TOL_ZERO = 45.0          # 角度差 >= 45 度給 0 分

# 角度定義: 名稱 -> (a, b, c)，量的是 b 點的夾角
# MediaPipe 編號: 11/12 肩, 13/14 肘, 15/16 腕, 23/24 髖, 25/26 膝, 27/28 踝
ANGLES = {
    "l_elbow": (11, 13, 15), "r_elbow": (12, 14, 16),
    "l_shoulder": (13, 11, 23), "r_shoulder": (14, 12, 24),
    "l_hip": (11, 23, 25), "r_hip": (12, 24, 26),
    "l_knee": (23, 25, 27), "r_knee": (24, 26, 28),
}
ANGLE_NAMES = list(ANGLES)
COL = {name: i for i, name in enumerate(ANGLE_NAMES)}

JOINT_LABEL = {"elbow": "手肘", "shoulder": "肩膀", "hip": "髖部 / 軀幹", "knee": "膝蓋"}

# 各動作的關鍵關節與權重（加總為 1）
EXERCISES = {
    "shoulder_press":         {"shoulder": 0.4, "elbow": 0.4, "hip": 0.2},
    "side_lateral_raise":     {"shoulder": 0.6, "elbow": 0.2, "hip": 0.2},
    "bicep_curl":             {"elbow": 0.7, "shoulder": 0.3},
    "hammer_curl":            {"elbow": 0.7, "shoulder": 0.3},
    "flat_chest_press":       {"elbow": 0.5, "shoulder": 0.5},
    "incline_chest_press":    {"elbow": 0.5, "shoulder": 0.5},
    "dumbbell_chest_press":   {"elbow": 0.5, "shoulder": 0.5},
    "cable_row":              {"elbow": 0.4, "shoulder": 0.3, "hip": 0.3},
    "cable_reverse_pull":     {"shoulder": 0.5, "elbow": 0.3, "hip": 0.2},
    "neutral_grip_pull_down": {"elbow": 0.5, "shoulder": 0.5},
    "single_leg_squat":       {"knee": 0.5, "hip": 0.4, "shoulder": 0.1},
}
DEFAULT_WEIGHTS = {"elbow": 0.25, "shoulder": 0.25, "hip": 0.25, "knee": 0.25}


def angle_3d(a, b, c):
    ba, bc = a - b, c - b
    denom = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denom < 1e-9:
        return np.nan
    return float(np.degrees(np.arccos(np.clip(np.dot(ba, bc) / denom, -1.0, 1.0))))


def extract_angles(video_path):
    """回傳 (T, 8) 的角度序列，偵測不到的位置是 NaN。"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"無法開啟影片: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0 or fps > 120:  # 瀏覽器錄的 webm 常常沒有正確的 fps
        fps = 30.0
    step = max(1, round(fps / SAMPLE_FPS))

    rows = []
    idx = 0
    # 每支影片用自己的 Pose 物件，避免追蹤狀態互相干擾
    with mp.solutions.pose.Pose(model_complexity=1,
                                min_detection_confidence=0.5,
                                min_tracking_confidence=0.5) as pose:
        while len(rows) < MAX_SAMPLES:
            ok, frame = cap.read()
            if not ok:
                break
            idx += 1
            if (idx - 1) % step:
                continue

            h, w = frame.shape[:2]
            if w > FRAME_WIDTH:
                frame = cv2.resize(frame, (FRAME_WIDTH, int(h * FRAME_WIDTH / w)))
            res = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            row = np.full(len(ANGLE_NAMES), np.nan)
            if res.pose_world_landmarks and res.pose_landmarks:
                world = res.pose_world_landmarks.landmark
                vis = res.pose_landmarks.landmark
                pts = np.array([[p.x, p.y, p.z] for p in world])
                for i, (a, b, c) in enumerate(ANGLES.values()):
                    if min(vis[a].visibility, vis[b].visibility, vis[c].visibility) >= MIN_VISIBILITY:
                        row[i] = angle_3d(pts[a], pts[b], pts[c])
            rows.append(row)
    cap.release()

    if not rows:
        raise RuntimeError(f"影片沒有可讀取的畫面: {video_path}")
    return np.array(rows)


def clean(seq):
    """補齊短暫偵測失敗的空洞，並做輕微平滑。偵測率太低的欄位整欄設為 NaN。"""
    seq = seq.copy()
    t = np.arange(len(seq))
    for j in range(seq.shape[1]):
        col = seq[:, j]
        ok = ~np.isnan(col)
        if ok.mean() < 0.5:
            seq[:, j] = np.nan
            continue
        col = np.interp(t, t[ok], col[ok])
        if len(col) >= 3:
            col = np.convolve(np.pad(col, 1, mode="edge"), np.ones(3) / 3, mode="valid")
        seq[:, j] = col
    return seq


def load_reference(ref_path):
    """示範影片的角度序列會快取起來，影片沒變就不用重算。"""
    cache = f"{ref_path}.angles.v{CACHE_VERSION}.npy"
    if os.path.exists(cache) and os.path.getmtime(cache) >= os.path.getmtime(ref_path):
        return np.load(cache)
    seq = extract_angles(ref_path)
    try:
        np.save(cache, seq)
    except OSError:
        pass
    return seq


def swap_sides(seq):
    """左右對調，處理使用者用另一邊做動作或畫面鏡像的情況。"""
    out = seq.copy()
    for name in ANGLE_NAMES:
        if name.startswith("l_"):
            l, r = COL[name], COL["r_" + name[2:]]
            out[:, [l, r]] = seq[:, [r, l]]
    return out


def group_errors(ref, usr, joint):
    """每對 (參考幀, 使用者幀) 在某個關節的平均角度差，左右分開比。回傳 (N, M)。"""
    diffs = []
    for side in ("l_", "r_"):
        j = COL[side + joint]
        a, b = ref[:, j], usr[:, j]
        if np.isnan(a).all() or np.isnan(b).all():
            continue
        diffs.append(np.abs(a[:, None] - b[None, :]))
    if not diffs:
        return None
    return np.nanmean(np.stack(diffs), axis=0)


def to_score(err):
    return np.clip((TOL_ZERO - err) / (TOL_ZERO - TOL_FULL), 0.0, 1.0) * 100.0


def dtw_path(cost):
    n, m = cost.shape
    acc = np.full((n + 1, m + 1), np.inf)
    acc[0, 0] = 0.0
    for i in range(1, n + 1):
        row_cost = cost[i - 1]
        prev, cur = acc[i - 1], acc[i]
        for j in range(1, m + 1):
            cur[j] = row_cost[j - 1] + min(prev[j], prev[j - 1], cur[j - 1])
    # 回溯最佳路徑
    i, j, path = n, m, []
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        k = int(np.argmin((acc[i - 1, j - 1], acc[i - 1, j], acc[i, j - 1])))
        if k == 0:
            i, j = i - 1, j - 1
        elif k == 1:
            i -= 1
        else:
            j -= 1
    return path[::-1]


def compare(ref, usr, weights):
    """回傳 (準確度, 路徑, 各關節分數)。"""
    errs = {}
    for joint in weights:
        e = group_errors(ref, usr, joint)
        if e is not None:
            errs[joint] = e
    if not errs:
        return None
    total_w = sum(weights[j] for j in errs)
    w = {j: weights[j] / total_w for j in errs}

    cost = sum(w[j] * errs[j] for j in errs)
    path = dtw_path(cost)
    ii = np.array([p[0] for p in path])
    jj = np.array([p[1] for p in path])

    joint_scores = {j: float(to_score(errs[j][ii, jj]).mean()) for j in errs}
    joint_err = {j: float(errs[j][ii, jj].mean()) for j in errs}
    accuracy = sum(w[j] * joint_scores[j] for j in errs)
    return accuracy, path, joint_scores, joint_err, w


def motion_range(seq, joint):
    vals = seq[:, [COL["l_" + joint], COL["r_" + joint]]]
    vals = vals[:, ~np.isnan(vals).all(axis=0)]
    if vals.size == 0:
        return None
    # 取變化最大的那一側
    return float(max(np.percentile(c, 95) - np.percentile(c, 5) for c in vals.T))


def count_reps(seq, joint):
    """用遲滯門檻數角度曲線的來回次數。"""
    cols = [COL["l_" + joint], COL["r_" + joint]]
    best = 0
    for c in cols:
        x = seq[:, c]
        if np.isnan(x).all():
            continue
        lo, hi = np.percentile(x, 5), np.percentile(x, 95)
        if hi - lo < 15:  # 幾乎沒動
            continue
        x = (x - lo) / (hi - lo)
        state, reps = x[0] > 0.5, 0
        for v in x:
            if not state and v > 0.7:
                state = True
            elif state and v < 0.3:
                state = False
                reps += 1
        best = max(best, reps)
    return best


def analyze(user_path, ref_path, exercise):
    weights = EXERCISES.get(exercise, DEFAULT_WEIGHTS)
    ref = clean(load_reference(ref_path))
    usr_raw = extract_angles(user_path)
    if np.isnan(usr_raw).all(axis=1).mean() > 0.8:
        raise RuntimeError("錄影中幾乎偵測不到人，請確認全身（至少上半身）都在畫面內")
    usr = clean(usr_raw)

    # 左右正常比一次、對調比一次，取較好的
    candidates = [r for r in (compare(ref, usr, weights), compare(ref, swap_sides(usr), weights)) if r]
    if not candidates:
        raise RuntimeError("關鍵關節偵測不到，請調整拍攝角度")
    accuracy, path, joint_scores, joint_err, w = max(candidates, key=lambda r: r[0])

    feedback = []

    # 動作幅度：看權重最高的關節
    key_joint = max(w, key=w.get)
    rom_ref, rom_usr = motion_range(ref, key_joint), motion_range(usr, key_joint)
    if rom_ref and rom_usr is not None and rom_ref > 5:
        ratio = rom_usr / rom_ref
        rom_score = float(np.clip(1 - abs(1 - ratio), 0, 1) * 100)
        if ratio < 0.75:
            feedback.append(f"{JOINT_LABEL[key_joint]}動作幅度不足（約示範的 {ratio:.0%}）")
        elif ratio > 1.3:
            feedback.append(f"{JOINT_LABEL[key_joint]}動作幅度過大")
    else:
        rom_score = 100.0

    # 節奏：對齊路徑偏離對角線越多，代表速度越不穩定
    n, m = len(ref), len(usr)
    if n > 1 and m > 1:
        dev = np.mean([abs(i / (n - 1) - j / (m - 1)) for i, j in path])
        tempo_score = float(np.clip(1 - dev / 0.25, 0, 1) * 100)
    else:
        tempo_score = 100.0
    if tempo_score < 60:
        feedback.append("節奏不穩定，盡量跟著示範影片的速度")

    worst = min(joint_scores, key=joint_scores.get)
    if joint_scores[worst] < 70:
        feedback.append(f"{JOINT_LABEL[worst]}角度與示範差距較大（平均差 {joint_err[worst]:.0f}°）")
    if not feedback:
        feedback.append("動作很標準，繼續保持！")

    similarity = 0.7 * accuracy + 0.15 * rom_score + 0.15 * tempo_score
    return {
        "similarity": round(similarity, 2),
        "accuracy": round(accuracy, 2),
        "range": round(rom_score, 2),
        "tempo": round(tempo_score, 2),
        "joints": {JOINT_LABEL[j]: round(s, 1) for j, s in joint_scores.items()},
        "reps": count_reps(usr, key_joint),
        "reference_reps": count_reps(ref, key_joint),
        "feedback": feedback,
    }


def main():
    if len(sys.argv) not in (3, 4):
        print("用法: python app.py <使用者影片> <示範影片> [動作名稱]", file=sys.stderr)
        sys.exit(1)
    user_path, ref_path = sys.argv[1], sys.argv[2]
    exercise = sys.argv[3] if len(sys.argv) == 4 else os.path.splitext(os.path.basename(ref_path))[0]

    try:
        result = analyze(user_path, ref_path, exercise)
        code = 0
    except Exception as e:  # 把錯誤也用 JSON 回傳給 server
        result, code = {"error": str(e)}, 2
    sys.stdout.reconfigure(encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))
    sys.exit(code)


if __name__ == "__main__":
    main()
