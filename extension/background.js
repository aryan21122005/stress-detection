chrome.runtime.onInstalled.addListener(() => {
  chrome.storage.local.set({
    inferBase: 'http://127.0.0.1:5001',
    overlayActive: false,
    saveMetrics: false,
    sbUrl: '', sbAnon: '', accessToken: '', userId: '', sessionId: '',
    relayMode: true, relayUrl: 'http://127.0.0.1:5001/metrics'
  });
});

async function sbFetch(path, { method='GET', headers={}, body } = {}){
  const { sbUrl, accessToken, sbAnon } = await chrome.storage.local.get(['sbUrl','accessToken','sbAnon']);
  if (!sbUrl) throw new Error('Supabase URL not set');
  const h = { 'apikey': sbAnon || '', ...headers };
  if (accessToken) h['Authorization'] = `Bearer ${accessToken}`;
  const res = await fetch(`${sbUrl}${path}`, { method, headers: h, body });
  return res;
}

async function loginSupabase(email, password){
  const { sbUrl, sbAnon } = await chrome.storage.local.get(['sbUrl','sbAnon']);
  if (!sbUrl || !sbAnon) throw new Error('Supabase URL/Anon key missing');
  const res = await fetch(`${sbUrl}/auth/v1/token?grant_type=password`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', 'apikey': sbAnon },
    body: JSON.stringify({ email, password })
  });
  if (!res.ok) throw new Error('Login failed');
  const j = await res.json();
  const accessToken = j.access_token;
  // fetch user to get id
  const ures = await fetch(`${sbUrl}/auth/v1/user`, { headers: { 'Authorization': `Bearer ${accessToken}`, 'apikey': sbAnon } });
  if (!ures.ok) throw new Error('User fetch failed');
  const uj = await ures.json();
  const userId = uj?.id;
  await chrome.storage.local.set({ accessToken, userId });
  return { accessToken, userId };
}

async function joinSession(hostKey, sessionName){
  const { sbUrl, sbAnon, userId } = await chrome.storage.local.get(['sbUrl','sbAnon','userId']);
  if (!userId) throw new Error('Not logged in');
  const key = (hostKey || '').trim().toUpperCase();
  // Verify host key
  const kres = await sbFetch(`/rest/v1/host_keys?select=id,is_active&key_code=eq.${encodeURIComponent(key)}`);
  if (!kres.ok) throw new Error('Host key query failed');
  const krows = await kres.json();
  if (!Array.isArray(krows) || krows.length === 0 || krows[0].is_active === false) throw new Error('Invalid or inactive host key');
  const host_key_id = krows[0].id;
  // Create analytics_sessions row
  const body = JSON.stringify({ participant_id: userId, host_key_id, session_name: sessionName || null });
  const sres = await sbFetch('/rest/v1/analytics_sessions', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', 'Prefer': 'return=representation' },
    body
  });
  if (!sres.ok) throw new Error('Session create failed');
  const srows = await sres.json();
  const sessionId = srows?.[0]?.id;
  if (!sessionId) throw new Error('No session id');
  await chrome.storage.local.set({ sessionId, saveMetrics: true });
  return { sessionId };
}

async function saveMetric(row){
  const { sessionId } = await chrome.storage.local.get(['sessionId']);
  if (!sessionId) throw new Error('No session');
  const body = JSON.stringify({
    session_id: sessionId,
    timestamp: new Date().toISOString(),
    emotion_state: row.emotion,
    stress_score: row.stress,
    attention_score: row.attention
  });
  const res = await sbFetch('/rest/v1/analytics_data', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', 'Prefer': 'return=minimal' },
    body
  });
  if (!res.ok) throw new Error('Insert failed');
  return true;
}

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
  if (msg?.type === 'LOGIN_SUPABASE') {
    loginSupabase(msg.email, msg.password).then((r) => sendResponse({ ok: true, userId: r.userId })).catch(e => sendResponse({ ok: false, error: e.message }));
    return true;
  }
  if (msg?.type === 'SET_SUPABASE') {
    chrome.storage.local.set({ sbUrl: msg.url, sbAnon: msg.anon }, () => sendResponse({ ok: true }));
    return true;
  }
  if (msg?.type === 'SET_RELAY') {
    chrome.storage.local.set({ relayMode: !!msg.enabled, relayUrl: msg.url || 'http://127.0.0.1:5001/metrics' }, () => sendResponse({ ok: true }));
    return true;
  }
  if (msg?.type === 'JOIN_SESSION') {
    joinSession(msg.hostKey, msg.sessionName).then((r) => sendResponse({ ok: true, sessionId: r.sessionId })).catch(e => sendResponse({ ok: false, error: e.message }));
    return true;
  }
  if (msg?.type === 'SAVE_METRIC') {
    // Prefer relay if enabled
    chrome.storage.local.get(['relayMode','relayUrl','sessionId'], async (s) => {
      const sessionId = s?.sessionId;
      if (!sessionId) { sendResponse({ ok: false, error: 'no-session' }); return; }
      const row = { session_id: sessionId, emotion: msg.row.emotion, stress: msg.row.stress, attention: msg.row.attention };
      if (s?.relayMode) {
        try {
          const res = await fetch(s?.relayUrl || 'http://127.0.0.1:5001/metrics', {
            method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(row)
          });
          if (!res.ok) throw new Error('relay-failed');
          sendResponse({ ok: true, via: 'relay' });
        } catch (e) {
          sendResponse({ ok: false, error: String(e) });
        }
      } else {
        saveMetric({ emotion: row.emotion, stress: row.stress, attention: row.attention })
          .then(() => sendResponse({ ok: true, via: 'direct' }))
          .catch(e => sendResponse({ ok: false, error: e.message }));
      }
    });
    return true;
  }
  if (msg?.type === 'PING') {
    sendResponse({ ok: true });
  }
});
