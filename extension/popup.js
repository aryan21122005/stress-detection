document.addEventListener('DOMContentLoaded', () => {
  const inferBase = document.getElementById('inferBase');
  const toggle = document.getElementById('toggle');
  const save = document.getElementById('save');
  const sbUrl = document.getElementById('sbUrl');
  const sbAnon = document.getElementById('sbAnon');
  const email = document.getElementById('email');
  const password = document.getElementById('password');
  const loginBtn = document.getElementById('login');
  const loginStatus = document.getElementById('loginStatus');
  const hostKey = document.getElementById('hostKey');
  const sessionName = document.getElementById('sessionName');
  const joinBtn = document.getElementById('join');
  const joinStatus = document.getElementById('joinStatus');

  chrome.storage.local.get(['inferBase', 'overlayActive', 'sbUrl', 'sbAnon', 'userId', 'sessionId'], (s) => {
    if (s?.inferBase) inferBase.value = s.inferBase;
    toggle.textContent = s?.overlayActive ? 'Stop Overlay' : 'Start Overlay';
    if (s?.sbUrl) sbUrl.value = s.sbUrl;
    if (s?.sbAnon) sbAnon.value = s.sbAnon;
    if (s?.userId) loginStatus.textContent = `Logged in (${s.userId.slice(0,8)})`;
    if (s?.sessionId) joinStatus.textContent = `Session: ${s.sessionId.slice(0,8)}`;
  });

  save.addEventListener('click', () => {
    const v = inferBase.value || 'http://127.0.0.1:5001';
    chrome.runtime.sendMessage({ type: 'SET_INFER_BASE', value: v }, () => {
      window.close();
    });
  });

  loginBtn.addEventListener('click', () => {
    loginStatus.textContent = '...';
    chrome.runtime.sendMessage({ type: 'SET_SUPABASE', url: sbUrl.value, anon: sbAnon.value }, () => {
      chrome.runtime.sendMessage({ type: 'LOGIN_SUPABASE', email: email.value, password: password.value }, (resp) => {
        loginStatus.textContent = resp?.ok ? `Logged in (${resp.userId.slice(0,8)})` : `Error: ${resp?.error || 'login failed'}`;
      });
    });
  });

  joinBtn.addEventListener('click', () => {
    joinStatus.textContent = '...';
    chrome.runtime.sendMessage({ type: 'JOIN_SESSION', hostKey: hostKey.value, sessionName: sessionName.value }, (resp) => {
      joinStatus.textContent = resp?.ok ? `Session: ${resp.sessionId.slice(0,8)}` : `Error: ${resp?.error || 'join failed'}`;
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
