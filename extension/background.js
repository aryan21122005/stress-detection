chrome.runtime.onInstalled.addListener(() => {
  chrome.storage.local.set({
    inferBase: 'http://127.0.0.1:5001',
    overlayActive: false,
    saveMetrics: false,
    sessionId: '',
    relayUrl: 'http://127.0.0.1:5001/metrics'
  });
});

chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  if (msg?.type === 'SET_ACTIVE') {
    chrome.storage.local.set({ overlayActive: !!msg.active }, () => {
      const targetTabId = msg.tabId ?? sender.tab?.id;
      if (targetTabId != null) chrome.tabs.sendMessage(targetTabId, { type: 'OVERLAY_TOGGLE', active: !!msg.active });
      sendResponse({ ok: true });
    });
    return true;
  }
  if (msg?.type === 'SET_INFER_BASE') {
    chrome.storage.local.set({ inferBase: msg.value || 'http://127.0.0.1:5001' }, () => sendResponse({ ok: true }));
    return true;
  }
  if (msg?.type === 'SET_RELAY') {
    chrome.storage.local.set({ relayUrl: msg.url || 'http://127.0.0.1:5001/metrics' }, () => sendResponse({ ok: true }));
    return true;
  }
  if (msg?.type === 'SET_SESSION') {
    chrome.storage.local.set({ sessionId: msg.sessionId || '' }, () => sendResponse({ ok: true }));
    return true;
  }
  if (msg?.type === 'SET_SAVE_METRICS') {
    chrome.storage.local.set({ saveMetrics: !!msg.enabled }, () => sendResponse({ ok: true }));
    return true;
  }
  if (msg?.type === 'SAVE_METRIC') {
    chrome.storage.local.get(['relayUrl','sessionId'], async (s) => {
      const sessionId = s?.sessionId;
      const relayUrl = s?.relayUrl || 'http://127.0.0.1:5001/metrics';
      if (!sessionId) { sendResponse({ ok: false, error: 'no-session' }); return; }
      try {
        const res = await fetch(relayUrl, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ session_id: sessionId, emotion: msg.row.emotion, stress: msg.row.stress, attention: msg.row.attention })
        });
        if (!res.ok) throw new Error('relay-failed');
        sendResponse({ ok: true, via: 'relay' });
      } catch (e) {
        sendResponse({ ok: false, error: String(e) });
      }
    });
    return true;
  }
  if (msg?.type === 'PING') {
    sendResponse({ ok: true });
  }
});
