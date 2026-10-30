# WWM 網頁版

不需要伺服器的版本：動作分析（MediaPipe 骨架偵測＋DTW 評分）全部在瀏覽器裡執行，鏡頭畫面不會上傳。

## 本機測試

鏡頭只能在 `https` 或 `localhost` 開啟，所以不能直接雙擊 html 檔，要用本機伺服器：

```bash
cd web
python -m http.server 8000
```

然後用 Chrome 或 Edge 打開 http://localhost:8000 。

## 檔案說明

| 檔案 | 說明 |
|---|---|
| `app/js/coach.js` | 評分邏輯，`app.py` 的 JavaScript 版本，計算結果與 `app.py` 相同 |
| `app/js/exercise.js` | 動作頁面：開鏡頭、即時畫骨架、取樣、呼叫 coach.js 評分 |
| `app/exercise.css` | 11 個動作頁面共用的樣式 |
| `app/*.mp4` | 示範影片 |
| `manifest.json`、`service-worker.js`、`icons/` | 讓網站可以「安裝」成 App（PWA） |

## 修改評分規則

`app.py` 與 `coach.js` 是同一套演算法。改其中一邊的參數（例如 `TOL_FULL`、各動作的關節權重 `EXERCISES`）時，記得另一邊也一起改。
