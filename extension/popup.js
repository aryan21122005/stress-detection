document.addEventListener('DOMContentLoaded', () => {
  const inferBase = document.getElementById('inferBase');
  const toggle = document.getElementById('toggle');
  const save = document.getElementById('save');

  chrome.storage.local.get(['inferBase', 'overlayActive'], (s) => {
    if (s?.inferBase) inferBase.value = s.inferBase;
    toggle.textContent = s?.overlayActive ? 'Stop Overlay' : 'Start Overlay';
  });

  save.addEventListener('click', () => {
    const v = inferBase.value || 'http://127.0.0.1:5001';
    chrome.runtime.sendMessage({ type: 'SET_INFER_BASE', value: v }, () => {
      window.close();
    });
  });

  toggle.addEventListener('click', () => {
    chrome.storage.local.get(['overlayActive'], (s) => {
      const next = !s?.overlayActive;
      chrome.runtime.sendMessage({ type: 'SET_ACTIVE', active: next }, () => {
        toggle.textContent = next ? 'Stop Overlay' : 'Start Overlay';
      });
    });
  });
});
