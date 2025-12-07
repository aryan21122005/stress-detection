(function(){
  const STATE = {
    active: false,
    inferBase: 'http://127.0.0.1:5001',
    timer: null,
    stream: null,
  };

  const els = {
    root: null,
    panel: null,
    emotion: null,
    attention: null,
    stress: null,
    latency: null,
    video: null,
    canvas: null,
    toggle: null,
  };
  let lastSave = 0;

  function createOverlay(){
    if (els.root) return;
    const root = document.createElement('div');
    root.id = 'stress-overlay-root';
    root.className = 'stress-overlay-root';
    root.innerHTML = `
      <div class="stress-overlay-panel">
        <div class="row header">
          <strong>Stress Monitor</strong>
          <button id="stress-overlay-toggle">Stop</button>
        </div>
        <div class="row"><span>Emotion:</span><span id="ov-emo">-</span></div>
        <div class="row"><span>Attention:</span><span id="ov-att">-</span></div>
        <div class="row"><span>Stress:</span><span id="ov-str">-</span></div>
        <div class="row small"><span>Latency:</span><span id="ov-lat">-</span></div>
      </div>
    `;
    const video = document.createElement('video');
    video.muted = true; video.playsInline = true; video.style.display = 'none';
    const canvas = document.createElement('canvas');
    canvas.width = 224; canvas.height = 224; canvas.style.display = 'none';
    document.documentElement.appendChild(root);
    document.documentElement.appendChild(video);
    document.documentElement.appendChild(canvas);
    els.root = root;
    els.panel = root.querySelector('.stress-overlay-panel');
    els.emotion = root.querySelector('#ov-emo');
    els.attention = root.querySelector('#ov-att');
    els.stress = root.querySelector('#ov-str');
    els.latency = root.querySelector('#ov-lat');
    els.toggle = root.querySelector('#stress-overlay-toggle');
    els.video = video; els.canvas = canvas;
    els.toggle.addEventListener('click', () => stop());
  }

  async function start(){
    if (STATE.active) return;
    createOverlay();
    try{
      STATE.stream = await navigator.mediaDevices.getUserMedia({ video: { width: 320, height: 240 }, audio: false });
      els.video.srcObject = STATE.stream;
      await els.video.play();
      STATE.active = true;
      loop();
    }catch(e){
      console.warn('overlay camera error', e);
      teardown();
    }
  }

  function stop(){
    if (!STATE.active) return;
    STATE.active = false;
    if (STATE.timer) { clearInterval(STATE.timer); STATE.timer = null; }
    if (STATE.stream) {
      STATE.stream.getTracks().forEach(t => t.stop());
      STATE.stream = null;
    }
    if (els.root) els.toggle.textContent = 'Start';
  }

  function teardown(){
    stop();
    if (els.root) { els.root.remove(); els.root = null; }
    if (els.video) { els.video.remove(); els.video = null; }
    if (els.canvas) { els.canvas.remove(); els.canvas = null; }
  }

  function toScore(probs, anchors){
    if (!probs || !probs.length) return 0;
    let s=0; for(let i=0;i<Math.min(probs.length, anchors.length);i++) s+=probs[i]*anchors[i];
    return Math.max(0, Math.min(100, s));
  }

  function loop(){
    const ctx = els.canvas.getContext('2d');
    STATE.timer = setInterval(() => {
      if (!STATE.active) return;
      try{
        ctx.drawImage(els.video, 0, 0, els.canvas.width, els.canvas.height);
        els.canvas.toBlob(async (blob) => {
          if (!blob) return;
          const t0 = performance.now();
          const form = new FormData();
          form.append('file', new File([blob], 'frame.jpg', { type: 'image/jpeg' }));
          const res = await fetch(`${STATE.inferBase}/infer`, { method:'POST', body: form });
          if (!res.ok) return;
          const j = await res.json();
          const att = toScore(j?.engagement?.probs||[], [25,50,75,95]);
          const str = toScore(j?.stress?.probs||[], [10,35,65,90]);
          els.emotion.textContent = `${j?.emotion?.label||'-'} (${Math.round((j?.emotion?.confidence||0)*100)}%)`;
          els.attention.textContent = `${Math.round(att)}`;
          els.stress.textContent = `${Math.round(str)}`;
          els.latency.textContent = `${Math.round(performance.now()-t0)} ms`;

          // Throttle saving to ~1 insert per 2s
          const now = Date.now();
          if (now - lastSave > 2000) {
            lastSave = now;
            chrome.storage.local.get(['saveMetrics'], (s) => {
              if (s?.saveMetrics) {
                chrome.runtime.sendMessage({ type: 'SAVE_METRIC', row: { emotion: j?.emotion?.label, attention: att, stress: str } });
              }
            });
          }
        }, 'image/jpeg', 0.8);
      }catch(e){ /* ignore transient */ }
    }, 250);
  }

  function updateInferBase(){
    chrome.storage.local.get(['inferBase'], s => {
      if (s?.inferBase) STATE.inferBase = s.inferBase;
    });
  }

  chrome.runtime.onMessage.addListener((msg) => {
    if (msg?.type === 'OVERLAY_TOGGLE'){
      if (msg.active) start(); else stop();
    }
  });

  chrome.storage.local.get(['overlayActive','inferBase'], s => {
    STATE.active = !!s.overlayActive;
    if (s?.inferBase) STATE.inferBase = s.inferBase;
    if (STATE.active) start();
  });

  updateInferBase();
})();
