// 讓 WWM 可以安裝成 App，並在離線時開啟已經看過的頁面。
// 換新版本時把 CACHE_NAME 的數字加一，舊快取會自動清掉。
const CACHE_NAME = 'wwm-v2';

const APP_SHELL = [
  './',
  './index.html',
  './manifest.json',
  './icons/icon-192.png',
  './icons/icon-512.png',
  './app/interface.html',
  './app/arms.html',
  './app/back.html',
  './app/chest.html',
  './app/thigh.html',
  './app/timer.html',
  './app/track.html',
  './app/style.css',
  './app/exercise.css',
  './app/js/coach.js',
  './app/js/exercise.js',
];

self.addEventListener('install', (event) => {
  event.waitUntil(caches.open(CACHE_NAME).then((cache) => cache.addAll(APP_SHELL)));
  self.skipWaiting();
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then((names) =>
      Promise.all(names.filter((n) => n !== CACHE_NAME).map((n) => caches.delete(n)))
    )
  );
  self.clients.claim();
});

// 先用網路拿最新版本，失敗（離線）才用快取；拿到的東西順便存起來，
// 包含 AI 模型，第二次開啟會快很多。影片是分段（Range）下載的，交給瀏覽器處理。
self.addEventListener('fetch', (event) => {
  const { request } = event;
  if (request.method !== 'GET' || request.headers.has('range')) return;

  event.respondWith(
    fetch(request)
      .then((response) => {
        if (response.ok) {
          const copy = response.clone();
          caches.open(CACHE_NAME).then((cache) => cache.put(request, copy));
        }
        return response;
      })
      .catch(() => caches.match(request))
  );
});
