document.addEventListener('DOMContentLoaded', () => {
  const inferBase = document.getElementById('inferBase');
  const toggle = document.getElementById('toggle');
  const save = document.getElementById('save');
  const sessionId = document.getElementById('sessionId');
  const relayUrl = document.getElementById('relayUrl');
  const saveMetricsChk = document.getElementById('saveMetricsChk');

  chrome.storage.local.get(['inferBase', 'overlayActive', 'relayUrl', 'sessionId', 'saveMetrics'], (s) => {
    if (s?.inferBase) inferBase.value = s.inferBase;
    toggle.textContent = s?.overlayActive ? 'Stop Overlay' : 'Start Overlay';
    if (s?.relayUrl) relayUrl.value = s.relayUrl;
    if (s?.sessionId) sessionId.value = s.sessionId;
    if (typeof s?.saveMetrics === 'boolean') saveMetricsChk.checked = s.saveMetrics;
  });

  save.addEventListener('click', () => {
    const v = inferBase.value || 'http://127.0.0.1:5001';
    const rurl = relayUrl.value || 'http://127.0.0.1:5001/metrics';
    const sid = sessionId.value || '';
    const saveEnabled = !!saveMetricsChk.checked;
    chrome.runtime.sendMessage({ type: 'SET_INFER_BASE', value: v }, () => {
      chrome.runtime.sendMessage({ type: 'SET_RELAY', url: rurl }, () => {
        chrome.runtime.sendMessage({ type: 'SET_SESSION', sessionId: sid }, () => {
          chrome.runtime.sendMessage({ type: 'SET_SAVE_METRICS', enabled: saveEnabled }, () => {
            window.close();
          });
        });
      });
    });
  });

  toggle.addEventListener('click', () => {
    chrome.storage.local.get(['overlayActive'], (s) => {
      const next = !s?.overlayActive;
      chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
        const tabId = tabs?.[0]?.id;
        chrome.runtime.sendMessage({ type: 'SET_ACTIVE', active: next, tabId }, () => {
          toggle.textContent = next ? 'Stop Overlay' : 'Start Overlay';
        });
      });
    });
  });
});
