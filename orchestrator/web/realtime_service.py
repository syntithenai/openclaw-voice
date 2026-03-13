"""Embedded HTTP + WebSocket service for realtime voice UI telemetry."""

from __future__ import annotations

import asyncio
from collections import deque
import json
import logging
import math
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Awaitable, Callable

import websockets

logger = logging.getLogger("orchestrator.web.realtime")


def _build_ui_html(ws_port: int, mic_starts_disabled: bool = True, audio_authority: str = "native") -> str:
    mic_disabled_js = "true" if mic_starts_disabled else "false"
    return f"""<!doctype html>
<html lang="en" class="dark">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>OpenClaw Voice</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <script>
    tailwind.config = {{
      darkMode: 'class',
      theme: {{ extend: {{ colors: {{
        gray: {{ 950: '#0f1117', 900: '#171b26', 800: '#242a3b', 700: '#2f3548' }}
      }} }} }}
    }};
  </script>
  <style>
    .mic-btn {{ transition: border-width 80ms linear, background-color 200ms ease; }}
    .chat-msg {{ animation: fadeIn 0.18s ease; }}
    @keyframes fadeIn {{ from {{ opacity:0; transform:translateY(4px); }} to {{ opacity:1; transform:none; }} }}
    ::-webkit-scrollbar {{ width:6px; }} ::-webkit-scrollbar-track {{ background:transparent; }}
    ::-webkit-scrollbar-thumb {{ background:#3e4d8a; border-radius:999px; }}
  </style>
</head>
<body class="bg-gray-950 text-gray-100 h-screen flex flex-col overflow-hidden">

<!-- HEADER -->
<header class="flex items-center justify-between px-3 h-14 bg-gray-900 border-b border-gray-800 flex-none gap-2 z-10">
  <div class="relative flex-none" id="menuWrap">
    <button id="menuBtn" class="p-2 rounded-lg hover:bg-gray-700 transition-colors" title="Menu">
      <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <line x1="3" y1="6" x2="21" y2="6"/><line x1="3" y1="12" x2="21" y2="12"/><line x1="3" y1="18" x2="21" y2="18"/>
      </svg>
    </button>
    <div id="menuDropdown" class="hidden absolute left-0 top-11 w-44 bg-gray-800 border border-gray-700 rounded-xl shadow-xl z-50 py-1">
      <a href="#/home"  class="flex items-center gap-2 px-4 py-2.5 hover:bg-gray-700 text-sm" data-nav="home">🏠 Home</a>
      <a href="#/music" class="flex items-center gap-2 px-4 py-2.5 hover:bg-gray-700 text-sm" data-nav="music">🎵 Music</a>
    </div>
  </div>

  <div id="musicHeader" class="hidden flex-1 flex items-center gap-2 min-w-0 px-2">
    <button id="musicToggleBtn" class="flex-none w-8 h-8 flex items-center justify-center rounded-full bg-gray-700 hover:bg-gray-600 transition-colors text-xs" title="Play/Pause">&#9654;</button>
    <div class="min-w-0">
      <div id="musicTitle"  class="text-sm font-medium truncate text-white">&#8212;</div>
      <div id="musicArtist" class="text-xs text-gray-400 truncate">&#8212;</div>
    </div>
  </div>

  <div class="flex items-center gap-2 flex-none">
    <span id="wsDot" class="w-2 h-2 rounded-full bg-red-500 transition-colors" title="WebSocket"></span>
    <button id="micBtn" class="mic-btn w-11 h-11 rounded-full flex items-center justify-center font-bold text-xl border-4" title="Microphone">&#127908;</button>
  </div>
</header>

<main id="main" class="flex-1 overflow-y-auto min-h-0"></main>
<div id="timerBar" class="hidden flex-none px-3 py-2 bg-gray-900 border-t border-gray-800 flex gap-2 flex-wrap items-center text-sm"></div>

<script>
const WS_PORT = {ws_port};
const MIC_STARTS_DISABLED = {mic_disabled_js};
const AUDIO_AUTHORITY = '{audio_authority}';

const S = {{
  ws:null, wsConnected:false,
  micEnabled:!MIC_STARTS_DISABLED,
  voice_state:'idle', wake_state:'asleep', tts_playing:false, mic_rms:0,
  chat:[], music:{{state:'stop',title:'',artist:'',queue_length:0,elapsed:0,duration:0,position:-1}},
    chatThreads:[], activeChatId:'active', selectedChatId:'active', chatSidebarOpen:true,
  musicQueue:[], timers:[], page:'home',
    audioCtx:null, mediaStream:null, processor:null, lastLevel:0,
    pendingChatSends:new Set(), nextClientMsgId:1,
}};

function updateChatComposerState(){{
    const input=document.getElementById('chatInput');
    const sendBtn=document.getElementById('chatSendBtn');
    const isPending=S.pendingChatSends.size>0;
    if(input){{
        input.disabled=isPending;
        input.placeholder=isPending?'Sending...':'Type a message';
    }}
    if(sendBtn){{
        sendBtn.disabled=isPending;
        sendBtn.classList.toggle('opacity-60',isPending);
        sendBtn.classList.toggle('cursor-not-allowed',isPending);
        sendBtn.textContent=isPending?'Sending...':'Send';
    }}
}}

function getPage(){{ const h=location.hash.replace('#',''); return h==='/music'?'music':'home'; }}
function navigate(p){{ location.hash='#/'+p; }}
window.addEventListener('hashchange',()=>{{ S.page=getPage(); renderPage(); closeMenu(); }});

function closeMenu(){{ document.getElementById('menuDropdown').classList.add('hidden'); }}
document.getElementById('menuBtn').addEventListener('click',e=>{{ e.stopPropagation(); document.getElementById('menuDropdown').classList.toggle('hidden'); }});
document.addEventListener('click',closeMenu);
document.querySelectorAll('[data-nav]').forEach(el=>el.addEventListener('click',e=>{{ e.preventDefault(); navigate(el.dataset.nav); }}));
document.addEventListener('click', e => {{
    const newChatBtn = e.target.closest('[data-action="chat-new"]');
    if (newChatBtn) {{
        sendAction({{type:'chat_new'}});
        S.selectedChatId = 'active';
        renderPage();
        return;
    }}

    const toggleSidebarBtn = e.target.closest('[data-action="chat-sidebar-toggle"]');
    if (toggleSidebarBtn) {{
        S.chatSidebarOpen = !S.chatSidebarOpen;
        renderPage();
        return;
    }}

    const selectThreadBtn = e.target.closest('[data-action="chat-select"]');
    if (selectThreadBtn) {{
        const tid = selectThreadBtn.dataset.threadId || 'active';
        S.selectedChatId = tid;
        renderPage();
        return;
    }}

    const timerBtn = e.target.closest('[data-action="timer-cancel"]');
    if (timerBtn) {{
        sendAction({{type:'timer_cancel', timer_id: timerBtn.dataset.timerId}});
        return;
    }}

    const musicRow = e.target.closest('[data-action="music-play-track"]');
    if (musicRow) {{
        sendAction({{type:'music_play_track', position: Number(musicRow.dataset.position)}});
        return;
    }}

    const musicToggle = e.target.closest('[data-action="music-toggle"]');
    if (musicToggle) {{
        sendAction({{type: (S.music.state === 'play' ? 'music_stop' : 'music_toggle')}});
    }}
}});

document.addEventListener('submit', e => {{
    const form = e.target;
    if (!form || form.id !== 'chatComposer') return;
    e.preventDefault();
    const input = document.getElementById('chatInput');
    if (!input) return;
    const text = String(input.value || '').trim();
    if (!text) return;
    const clientMsgId='c'+(S.nextClientMsgId++);
    S.pendingChatSends.add(clientMsgId);
    updateChatComposerState();
    sendAction({{type:'chat_text', text, client_msg_id:clientMsgId}});
    input.value = '';
    if (S.selectedChatId !== 'active') {{
        S.selectedChatId = 'active';
        renderPage();
    }}
}});

function applyMicState(){{
  const btn=document.getElementById('micBtn');
  const rms=S.mic_rms||0;
  const bw=S.micEnabled?Math.round(2+Math.min(8,Math.pow(rms,0.55)*40)):4;
  btn.style.borderWidth=bw+'px';
  btn.classList.remove('bg-red-900','border-red-600','bg-green-900','border-green-500','bg-pink-900','border-pink-500');
  if(!S.micEnabled) btn.classList.add('bg-red-900','border-red-600');
  else if(S.wake_state==='awake') btn.classList.add('bg-green-900','border-green-500');
  else btn.classList.add('bg-pink-900','border-pink-500');
}}
document.getElementById('micBtn').addEventListener('click',()=>{{
  sendAction({{type:'mic_toggle'}});
  if(!S.micEnabled){{ S.micEnabled=true; S.wake_state='awake'; }}
  else if(S.wake_state==='awake'){{ S.wake_state='asleep'; }}
  else{{ S.wake_state='awake'; }}
  applyMicState();
}});

document.getElementById('musicToggleBtn').addEventListener('click',()=>sendAction({{type: (S.music.state === 'play' ? 'music_stop' : 'music_toggle')}}));
function applyMusicHeader(){{
  const m=S.music;
  const active=m.state==='play'||(m.state==='pause'&&(m.title||m.queue_length>0));
  document.getElementById('musicHeader').classList.toggle('hidden',!active);
  document.getElementById('musicTitle').textContent=m.title||'\u2014';
  document.getElementById('musicArtist').textContent=m.artist||'\u2014';
    document.getElementById('musicToggleBtn').textContent=m.state==='play'?'\u23f9':'\u25b6';
}}

function renderTimerBar(){{
  const bar=document.getElementById('timerBar');
  if(!S.timers.length){{ bar.classList.add('hidden'); bar.innerHTML=''; return; }}
  bar.classList.remove('hidden');
  bar.innerHTML=S.timers.map(t=>{{
    const rem=Math.max(0,Math.round(t.remaining_seconds));
    const mm=String(Math.floor(rem/60)).padStart(2,'0'),ss=String(rem%60).padStart(2,'0');
    return '<button class="flex items-center gap-1 px-3 py-1 rounded-full bg-amber-700 hover:bg-amber-600 text-xs transition-colors" data-action="timer-cancel" data-timer-id="'+esc(t.id)+'" title="Click to cancel">\u23f1 '+esc(t.label||'Timer')+' '+mm+':'+ss+'</button>';
  }}).join('');
}}

function renderPage(){{
  const main=document.getElementById('main');
  if(S.page==='music') renderMusicPage(main); else renderHomePage(main);
}}

function renderHomePage(main){{
    const sidebarClass = S.chatSidebarOpen ? 'w-72 border-r border-gray-800' : 'w-0 border-r-0';
    const sidebarInnerClass = S.chatSidebarOpen ? 'opacity-100' : 'opacity-0 pointer-events-none';
    const sidebarToggleText = S.chatSidebarOpen ? 'Hide chats' : 'Show chats';
    const selected = S.selectedChatId || 'active';

    main.dataset.page='home';
    main.innerHTML='<div class="w-full h-full flex min-h-0">'
        +'<aside id="chatSidebar" class="'+sidebarClass+' transition-all duration-200 overflow-hidden">'
            +'<div class="'+sidebarInnerClass+' h-full flex flex-col">'
                +'<div class="px-0 py-3 text-xs uppercase tracking-wide text-gray-400 border-b border-gray-800">Previous chats</div>'
                +'<div id="chatThreadList" class="flex-1 overflow-y-auto p-0 space-y-0"></div>'
            +'</div>'
        +'</aside>'
        +'<div class="flex-1 min-w-0 flex flex-col h-full relative">'
            +'<div class="px-3 py-2 border-b border-gray-800 flex items-center justify-between gap-2">'
                +'<button data-action="chat-sidebar-toggle" class="px-2.5 py-1.5 rounded-lg text-xs bg-gray-800 hover:bg-gray-700 transition-colors">'+sidebarToggleText+'</button>'
                +'<button data-action="chat-new" class="px-3 py-1.5 rounded-lg text-xs bg-blue-700 hover:bg-blue-600 transition-colors">New</button>'
            +'</div>'
            +'<div id="chatArea" class="flex-1 overflow-y-auto px-4 py-4 pb-24 space-y-3 min-h-0"></div>'
            +'<div class="absolute bottom-0 left-0 right-0 border-t border-gray-800 bg-gray-900 px-3 py-2 z-10">'
                +'<form id="chatComposer" class="flex items-center gap-2">'
                    +'<input id="chatInput" type="text" placeholder="Type a message" class="flex-1 min-w-0 rounded-lg bg-gray-800 border border-gray-700 px-3 py-2 text-sm text-gray-100 placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-600" />'
                    +'<button id="chatSendBtn" type="submit" class="px-3 py-2 rounded-lg text-sm bg-blue-700 hover:bg-blue-600 transition-colors">Send</button>'
                +'</form>'
            +'</div>'
        +'</div>'
    +'</div>';

    renderThreadList(selected);
    renderChatMessages(selected);
    updateChatComposerState();
}}
function getSelectedMessages(selectedId){{
    if(!selectedId || selectedId==='active') return S.chat;
    const t=(S.chatThreads||[]).find(x=>x.id===selectedId);
    return (t&&Array.isArray(t.messages)) ? t.messages : [];
}}
function renderThreadList(selectedId){{
    const list=document.getElementById('chatThreadList');
    if(!list) return;
    const currentActive = selectedId==='active' ? 'bg-blue-800 text-white' : 'bg-gray-800 text-gray-200 hover:bg-gray-700';
    const items = [];
    items.push('<button data-action="chat-select" data-thread-id="active" class="w-full text-left px-3 py-2 rounded-none text-sm transition-colors border-b border-gray-800 '+currentActive+'">Current chat</button>');
    (S.chatThreads||[]).forEach(t=>{{
        const title = esc((t.title||'Untitled').trim()||'Untitled');
        const activeCls = (t.id===selectedId) ? 'bg-blue-800 text-white' : 'bg-gray-800 text-gray-200 hover:bg-gray-700';
        items.push('<button data-action="chat-select" data-thread-id="'+esc(t.id)+'" class="w-full text-left px-3 py-2 rounded-none transition-colors border-b border-gray-800 '+activeCls+'"><div class="text-sm truncate">'+title+'</div></button>');
    }});
    list.innerHTML = items.join('');
}}
function renderChatMessages(selectedId){{
    const area=document.getElementById('chatArea');
    if(!area) return;
    area.innerHTML='';
    const msgs = getSelectedMessages(selectedId);
        const collated = collateChatMessages(msgs);
        collated.forEach(m=>area.appendChild(mkBubble(m)));
    scrollChat();
}}
function scrollChat(){{ const a=document.getElementById('chatArea'); if(a) a.scrollTop=a.scrollHeight; }}
function collateChatMessages(msgs){{
    const out=[];
    let assistantGroup=null;

    const flushAssistantGroup=()=>{{
        if(!assistantGroup) return;
        const segments=(assistantGroup.segments||[]).filter(s=>s&&String(s.text||'').trim().length>0);
        if(!segments.length){{ assistantGroup=null; return; }}
        out.push({{
            role:'assistant_group',
            request_id: assistantGroup.request_id,
            source: assistantGroup.source||'assistant',
            latest: segments[segments.length-1],
            previous: segments.slice(0, -1),
            segments,
        }});
        assistantGroup=null;
    }};

    (msgs||[]).forEach(m=>{{
        const role=(m&&m.role)||'';
        if(role!=='assistant'){{
            flushAssistantGroup();
            out.push(m);
            return;
        }}

        const requestId=(m.request_id===undefined||m.request_id===null) ? null : String(m.request_id);
        const canMerge=assistantGroup
            && (
                (assistantGroup.request_id!==null && assistantGroup.request_id===requestId)
                || (assistantGroup.request_id===null && requestId===null)
            );

        if(canMerge){{
            assistantGroup.segments.push(m);
            if(!assistantGroup.source && m.source) assistantGroup.source=m.source;
            return;
        }}

        flushAssistantGroup();
        assistantGroup={{
            request_id: requestId,
            source: m.source||'assistant',
            segments:[m],
        }};
    }});

    flushAssistantGroup();
    return out;
}}
function mkBubble(m){{
  const d=document.createElement('div');
    const role = (m&&m.role)||'';
    d.className='chat-msg flex '+(role==='user'?'justify-end':'justify-start');

    if(role==='assistant_group'){{
        const wrap=document.createElement('div');
        wrap.className='max-w-xs sm:max-w-sm lg:max-w-md';

        const isLive = !!(m.latest && m.latest.segment_kind === 'stream');
        if(isLive){{
            const live=document.createElement('div');
            live.className='mb-1 inline-flex items-center gap-1.5 px-2 py-0.5 rounded-full text-[10px] font-medium bg-emerald-900 text-emerald-200';
            live.innerHTML='<span class="inline-block w-1.5 h-1.5 rounded-full bg-emerald-300 animate-pulse"></span><span>Live</span>';
            wrap.appendChild(live);
        }}

        const b=document.createElement('div');
        b.className='px-4 py-2 rounded-2xl rounded-bl-md text-sm leading-relaxed bg-gray-700 text-gray-100';
        b.textContent=((m.latest&&m.latest.text)||'');
        wrap.appendChild(b);

        const prev=(m.previous||[]).map(x=>String((x&&x.text)||'').trim()).filter(Boolean);
        if(prev.length){{
            const details=document.createElement('details');
            details.className='mt-1 bg-gray-800/80 border border-gray-700 rounded-xl';
            const summary=document.createElement('summary');
            summary.className='cursor-pointer select-none px-3 py-1.5 text-[11px] text-gray-300';
            summary.textContent='Previous segments ('+prev.length+')';
            details.appendChild(summary);

            const body=document.createElement('div');
            body.className='px-3 pb-2 space-y-1';
            prev.forEach((txt,idx)=>{{
                const p=document.createElement('div');
                p.className='text-[11px] leading-relaxed text-gray-300 bg-gray-900/70 rounded-lg px-2 py-1';
                p.textContent=(idx+1)+'. '+txt;
                body.appendChild(p);
            }});
            details.appendChild(body);
            wrap.appendChild(details);
        }}

        d.appendChild(wrap);
        return d;
    }}

  const b=document.createElement('div');
    b.className='max-w-xs sm:max-w-sm lg:max-w-md px-4 py-2 rounded-2xl text-sm leading-relaxed '+
        (role==='user'?'bg-blue-700 text-white rounded-br-md':
         role==='system'?'bg-gray-700 text-gray-300 italic text-xs':
     'bg-gray-700 text-gray-100 rounded-bl-md');
  b.textContent=m.text||''; d.appendChild(b); return d;
}}

function renderMusicPage(main){{
  main.dataset.page='music';
  const m=S.music, q=S.musicQueue||[];
  const rows=q.map(item=>{{
    const active=item.pos===m.position;
    return '<tr class="hover:bg-gray-800 cursor-pointer '+(active?'bg-gray-800 font-semibold text-green-400':'')+'" data-action="music-play-track" data-position="'+item.pos+'"><td class="px-4 py-2 w-8 text-gray-500 text-xs">'+(item.pos+1)+'</td><td class="px-2 py-2 text-sm truncate max-w-xs">'+esc(item.title||item.file||'\u2014')+'</td><td class="px-2 py-2 text-xs text-gray-400 truncate">'+esc(item.artist||'')+'</td><td class="px-2 py-2 text-xs text-gray-500 text-right pr-4">'+fmtDur(item.duration)+'</td></tr>';
  }}).join('');
    main.innerHTML='<div class="max-w-2xl mx-auto px-2 py-4"><div class="flex items-center justify-between mb-4 px-2"><h2 class="font-semibold text-lg">Queue <span class="text-gray-400 font-normal text-sm ml-1">'+m.queue_length+' tracks</span></h2><button data-action="music-toggle" class="px-3 py-1 rounded-lg text-sm bg-gray-700 hover:bg-gray-600 transition-colors">'+(m.state==='play'?'\u23f9 Stop':'\u25b6 Play')+'</button></div>'+(q.length?'<div class="overflow-x-auto rounded-xl border border-gray-800"><table class="w-full text-left"><tbody>'+rows+'</tbody></table></div>':'<p class="text-gray-500 text-center py-8 text-sm">No tracks in queue</p>')+'</div>';
}}

function fmtDur(s){{ if(!s) return '\u2014'; const t=Math.round(Number(s)); return Math.floor(t/60)+':'+String(t%60).padStart(2,'0'); }}
function esc(s){{ return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }}

function wsUrl(){{ return (location.protocol==='https:'?'wss':'ws')+'://'+location.hostname+':'+WS_PORT+'/ws'; }}
function sendAction(payload){{ if(S.ws&&S.ws.readyState===WebSocket.OPEN) S.ws.send(JSON.stringify(payload)); }}

function connectWs(){{
  if(S.ws&&(S.ws.readyState===WebSocket.OPEN||S.ws.readyState===WebSocket.CONNECTING)) return;
  S.ws=new WebSocket(wsUrl()); S.ws.binaryType='arraybuffer';
  const dot=document.getElementById('wsDot');
  S.ws.onopen=()=>{{ S.wsConnected=true; dot.className=dot.className.replace('bg-red-500','bg-green-500'); S.ws.send(JSON.stringify({{type:'ui_ready'}})); }};
    S.ws.onclose=(evt)=>{{
        S.wsConnected=false;
        dot.className=dot.className.replace('bg-green-500','bg-red-500');
        // 4001 = replaced by newer client; do not auto-reconnect so newest client remains active.
        if (evt && evt.code === 4001) return;
        setTimeout(connectWs,1500);
    }};
  S.ws.onerror=()=>{{ dot.className=dot.className.replace('bg-green-500','bg-yellow-500'); }};
  S.ws.onmessage=evt=>{{ if(!(evt.data instanceof ArrayBuffer)){{ try{{ handleMsg(JSON.parse(evt.data)); }}catch(_){{}} }} }};
}}

function handleMsg(msg){{
  switch(msg.type){{
    case 'hello': break;
    case 'state_snapshot':
      if(msg.orchestrator) applyOrch(msg.orchestrator);
      if(msg.ui_control&&msg.ui_control.mic_enabled!==undefined){{ S.micEnabled=!!msg.ui_control.mic_enabled; applyMicState(); }}
      if(msg.music){{ applyMusic(msg.music); if(msg.music.queue) S.musicQueue=msg.music.queue; }}
      if(msg.timers) applyTimers(msg.timers);
            if(msg.chat) S.chat=msg.chat;
            if(Array.isArray(msg.chat_threads)) S.chatThreads=msg.chat_threads;
            if(msg.active_chat_id) S.activeChatId=msg.active_chat_id;
            if(!S.selectedChatId) S.selectedChatId='active';
            renderPage();
      break;
    case 'orchestrator_status': applyOrch(msg); break;
    case 'status': if(msg.orchestrator) applyOrch(msg.orchestrator); break;
        case 'chat_append': if(msg.message){{ S.chat.push(msg.message); if(S.page==='home'&&(!S.selectedChatId||S.selectedChatId==='active')) renderChatMessages('active'); }} break;
        case 'chat_threads_update':
            if(Array.isArray(msg.chat_threads)) S.chatThreads=msg.chat_threads;
            if(msg.active_chat_id) S.activeChatId=msg.active_chat_id;
            if(S.page==='home') renderPage();
            break;
        case 'chat_reset':
            S.chat=[];
            if(Array.isArray(msg.chat_threads)) S.chatThreads=msg.chat_threads;
            if(msg.active_chat_id) S.activeChatId=msg.active_chat_id;
            S.selectedChatId='active';
            if(S.page==='home') renderPage();
            break;
        case 'chat_text_ack':
            if(msg.client_msg_id) S.pendingChatSends.delete(String(msg.client_msg_id));
            if(S.page==='home') updateChatComposerState();
            break;
    case 'music_state': applyMusic(msg); if(msg.queue!==undefined) S.musicQueue=msg.queue; if(S.page==='music') renderPage(); else applyMusicHeader(); break;
    case 'timers_state': applyTimers(msg.timers||[]); break;
    case 'ui_control': if(msg.mic_enabled!==undefined){{ S.micEnabled=!!msg.mic_enabled; applyMicState(); }} break;
  }}
}}

function applyOrch(o){{
  if(o.voice_state!==undefined) S.voice_state=o.voice_state;
  if(o.wake_state!==undefined)  S.wake_state=o.wake_state;
  if(o.tts_playing!==undefined) S.tts_playing=!!o.tts_playing;
  if(o.mic_rms!==undefined)     S.mic_rms=Number(o.mic_rms)||0;
  if(o.mic_enabled!==undefined) S.micEnabled=!!o.mic_enabled;
  applyMicState();
}}
function applyMusic(m){{ Object.assign(S.music,m); applyMusicHeader(); }}
function applyTimers(t){{ S.timers=t; renderTimerBar(); }}

async function startBrowserCapture(){{
  S.mediaStream=await navigator.mediaDevices.getUserMedia({{audio:true,video:false}});
  S.audioCtx=new(window.AudioContext||window.webkitAudioContext)();
  const src=S.audioCtx.createMediaStreamSource(S.mediaStream);
  const proc=S.audioCtx.createScriptProcessor(2048,1,1);
  const mute=S.audioCtx.createGain(); mute.gain.value=0;
  src.connect(proc); proc.connect(mute); mute.connect(S.audioCtx.destination);
  proc.onaudioprocess=evt=>{{
    const inp=evt.inputBuffer.getChannelData(0);
    let ss=0; for(let i=0;i<inp.length;i++) ss+=inp[i]*inp[i];
    const rms=Math.sqrt(ss/Math.max(1,inp.length));
    if(!S.ws||S.ws.readyState!==WebSocket.OPEN) return;
    const now=performance.now();
    if(now-S.lastLevel>=120){{ S.lastLevel=now; S.ws.send(JSON.stringify({{type:'browser_audio_level',rms,peak:rms}})); }}
        // Always stream PCM when browser capture is active; backend decides authority.
        const out=new Int16Array(inp.length);
        for(let i=0;i<inp.length;i++){{const s=Math.max(-1,Math.min(1,inp[i]));out[i]=s<0?s*0x8000:s*0x7fff;}}
        S.ws.send(out.buffer);
  }};
  S.processor=proc;
}}

S.page=getPage(); renderPage(); applyMicState(); connectWs();
startBrowserCapture().catch(()=>{{}});
</script>
</body>
</html>
"""


class EmbeddedVoiceWebService:
    """Small embedded HTTP/WebSocket service for realtime UI and audio streaming."""

    def __init__(
        self,
        host: str = "0.0.0.0",
        ui_port: int = 18910,
        ws_port: int = 18911,
        status_hz: int = 12,
        hotword_active_ms: int = 2000,
        mic_starts_disabled: bool = True,
        audio_authority: str = "native",
        chat_history_limit: int = 200,
    ):
        self.host = host
        self.ui_port = ui_port
        self.ws_port = ws_port
        self.status_interval_s = 1.0 / max(1, status_hz)
        self.hotword_active_s = max(0.1, hotword_active_ms / 1000.0)
        self.mic_starts_disabled = mic_starts_disabled
        self.audio_authority = audio_authority
        self.chat_history_limit = max(20, chat_history_limit)

        self._http_server: HTTPServer | None = None
        self._http_thread: threading.Thread | None = None
        self._ws_server: Any = None
        self._status_task: asyncio.Task | None = None

        self._clients: set[Any] = set()
        self._active_client: Any | None = None
        self._latest_browser_audio: dict[str, float] = {"rms": 0.0, "peak": 0.0}
        self._browser_pcm_frames: deque[bytes] = deque(maxlen=400)
        self._last_hotword_ts: float | None = None

        self._orchestrator_status: dict[str, Any] = {
            "voice_state": "idle",
            "wake_state": "asleep",
            "speech_active": False,
            "tts_playing": False,
            "mic_rms": 0.0,
            "queue_depth": 0,
        }

        self._chat_messages: list[dict[str, Any]] = []
        self._chat_seq: int = 0
        self._chat_threads: list[dict[str, Any]] = []
        self._active_chat_id: str = "active"
        self._chat_thread_limit = 100
        self._music_state: dict[str, Any] = {
            "state": "stop", "title": "", "artist": "", "album": "",
            "queue_length": 0, "elapsed": 0.0, "duration": 0.0, "position": -1,
        }
        self._timers_state: list[dict[str, Any]] = []
        self._ui_control_state: dict[str, Any] = {"mic_enabled": not mic_starts_disabled}

        self._on_mic_toggle: Callable[[str], Awaitable[None]] | None = None
        self._on_music_toggle: Callable[[str], Awaitable[None]] | None = None
        self._on_music_stop: Callable[[str], Awaitable[None]] | None = None
        self._on_music_play_track: Callable[[int, str], Awaitable[None]] | None = None
        self._on_timer_cancel: Callable[[str, str], Awaitable[None]] | None = None
        self._on_chat_new: Callable[[str], Awaitable[None]] | None = None
        self._on_chat_text: Callable[[str, str], Awaitable[None]] | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        self._start_http_server()
        self._ws_server = await websockets.serve(self._ws_handler, self.host, self.ws_port)
        self._status_task = asyncio.create_task(self._status_loop())
        logger.info(
            "Embedded web UI started: http://%s:%d (ws://%s:%d)",
            self.host, self.ui_port, self.host, self.ws_port,
        )

    async def stop(self) -> None:
        if self._status_task:
            self._status_task.cancel()
            try:
                await self._status_task
            except asyncio.CancelledError:
                pass
            self._status_task = None

        if self._ws_server is not None:
            self._ws_server.close()
            await self._ws_server.wait_closed()
            self._ws_server = None

        if self._http_server is not None:
            self._http_server.shutdown()
            self._http_server.server_close()
            self._http_server = None

        if self._http_thread and self._http_thread.is_alive():
            self._http_thread.join(timeout=1.0)
        self._http_thread = None
        self._clients.clear()

    # ------------------------------------------------------------------
    # State update helpers (called from main.py)
    # ------------------------------------------------------------------

    def update_orchestrator_status(self, **status: Any) -> None:
        self._orchestrator_status.update(status)

    def note_hotword_detected(self) -> None:
        self._last_hotword_ts = time.monotonic()

    def update_chat_history(self, messages: list[dict[str, Any]]) -> None:
        self._chat_messages = list(messages[-self.chat_history_limit:])

    def _derive_chat_title(self, messages: list[dict[str, Any]]) -> str:
        for m in messages:
            if str(m.get("role", "")).lower() == "user":
                raw = str(m.get("text", "")).strip()
                if raw:
                    return raw[:72]
        return f"Chat {len(self._chat_threads) + 1}"

    def _archive_active_chat_if_needed(self) -> None:
        if not self._chat_messages:
            return
        now = time.time()
        archived = {
            "id": uuid.uuid4().hex[:12],
            "title": self._derive_chat_title(self._chat_messages),
            "messages": list(self._chat_messages),
            "created_ts": now,
            "updated_ts": now,
        }
        self._chat_threads.insert(0, archived)
        if len(self._chat_threads) > self._chat_thread_limit:
            self._chat_threads = self._chat_threads[: self._chat_thread_limit]

    def start_new_chat(self) -> None:
        self._archive_active_chat_if_needed()
        self._chat_messages = []
        self._active_chat_id = "active"
        asyncio.create_task(
            self.broadcast(
                {
                    "type": "chat_reset",
                    "active_chat_id": self._active_chat_id,
                    "chat": [],
                    "chat_threads": list(self._chat_threads),
                }
            )
        )

    def append_chat_message(self, message: dict[str, Any]) -> None:
        self._chat_seq += 1
        msg = dict(message)
        msg.setdefault("id", self._chat_seq)
        msg.setdefault("ts", time.time())
        self._chat_messages.append(msg)
        if len(self._chat_messages) > self.chat_history_limit:
            self._chat_messages = self._chat_messages[-self.chat_history_limit:]
        asyncio.create_task(self.broadcast({"type": "chat_append", "message": msg}))

    def update_music_state(self, queue: list[dict[str, Any]] | None = None, **state: Any) -> None:
        self._music_state.update(state)
        payload: dict[str, Any] = {"type": "music_state"}
        payload.update(self._music_state)
        if queue is not None:
            payload["queue"] = queue
        asyncio.create_task(self.broadcast(payload))

    def update_timers_state(self, timers: list[dict[str, Any]]) -> None:
        self._timers_state = list(timers)
        asyncio.create_task(self.broadcast({"type": "timers_state", "timers": self._timers_state}))

    def update_ui_control_state(self, **state: Any) -> None:
        self._ui_control_state.update(state)
        asyncio.create_task(self.broadcast({"type": "ui_control", **self._ui_control_state}))

    def has_active_client(self) -> bool:
        return self._active_client is not None and self._active_client in self._clients

    async def read_browser_frame(self, timeout: float = 0.0) -> bytes | None:
        if self._browser_pcm_frames:
            return self._browser_pcm_frames.popleft()
        if timeout <= 0:
            return None
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._browser_pcm_frames:
                return self._browser_pcm_frames.popleft()
            await asyncio.sleep(0.005)
        return None

    def latest_browser_audio(self) -> dict[str, float]:
        return dict(self._latest_browser_audio)

    # ------------------------------------------------------------------
    # Action handler registration
    # ------------------------------------------------------------------

    def set_action_handlers(
        self,
        on_mic_toggle: Callable[[str], Awaitable[None]] | None = None,
        on_music_toggle: Callable[[str], Awaitable[None]] | None = None,
        on_music_stop: Callable[[str], Awaitable[None]] | None = None,
        on_music_play_track: Callable[[int, str], Awaitable[None]] | None = None,
        on_timer_cancel: Callable[[str, str], Awaitable[None]] | None = None,
        on_chat_new: Callable[[str], Awaitable[None]] | None = None,
        on_chat_text: Callable[[str, str], Awaitable[None]] | None = None,
    ) -> None:
        if on_mic_toggle is not None:
            self._on_mic_toggle = on_mic_toggle
        if on_music_toggle is not None:
            self._on_music_toggle = on_music_toggle
        if on_music_stop is not None:
            self._on_music_stop = on_music_stop
        if on_music_play_track is not None:
            self._on_music_play_track = on_music_play_track
        if on_timer_cancel is not None:
            self._on_timer_cancel = on_timer_cancel
        if on_chat_new is not None:
            self._on_chat_new = on_chat_new
        if on_chat_text is not None:
            self._on_chat_text = on_chat_text

    # ------------------------------------------------------------------
    # Broadcast helper
    # ------------------------------------------------------------------

    async def broadcast(self, payload: dict[str, Any]) -> None:
        if not self._clients:
            return
        message = json.dumps(payload)
        stale: list[Any] = []
        for client in list(self._clients):
            try:
                await client.send(message)
            except Exception:
                stale.append(client)
        for c in stale:
            self._clients.discard(c)

    # ------------------------------------------------------------------
    # WebSocket handler
    # ------------------------------------------------------------------

    async def _ws_handler(self, websocket: Any) -> None:
        client_id = uuid.uuid4().hex[:8]

        # Single-client mode: newest connection replaces existing one.
        for existing in list(self._clients):
            if existing is not websocket:
                try:
                    await existing.close(code=4001, reason="Replaced by newer Web UI client")
                except Exception:
                    pass
                self._clients.discard(existing)

        self._clients.add(websocket)
        self._active_client = websocket
        logger.info("Web UI client connected (%s); clients=%d", client_id, len(self._clients))
        try:
            await websocket.send(json.dumps({
                "type": "hello",
                "client_id": client_id,
                "ws_port": self.ws_port,
                "ui_port": self.ui_port,
            }))
            await websocket.send(json.dumps(self._build_state_snapshot()))
            async for message in websocket:
                if isinstance(message, str):
                    asyncio.create_task(self._handle_text_action(message, client_id, websocket))
                elif isinstance(message, (bytes, bytearray)):
                    self._handle_pcm_chunk(bytes(message))
        except Exception as exc:
            logger.debug("Web UI client %s disconnected: %s", client_id, exc)
        finally:
            self._clients.discard(websocket)
            if self._active_client is websocket:
                self._active_client = None
            self._browser_pcm_frames.clear()
            logger.info("Web UI client disconnected (%s); clients=%d", client_id, len(self._clients))

    # ------------------------------------------------------------------
    # Incoming action dispatch
    # ------------------------------------------------------------------

    async def _handle_text_action(self, message: str, client_id: str, websocket: Any | None = None) -> None:
        try:
            payload = json.loads(message)
        except json.JSONDecodeError:
            return
        msg_type = payload.get("type", "")

        if msg_type == "browser_audio_level":
            try:
                self._latest_browser_audio["rms"] = float(payload.get("rms", 0.0))
                self._latest_browser_audio["peak"] = float(payload.get("peak", 0.0))
            except Exception:
                pass
            return

        if msg_type in ("ui_ready", "navigate"):
            return

        if msg_type == "mic_toggle" and self._on_mic_toggle:
            try:
                await self._on_mic_toggle(client_id)
            except Exception as exc:
                logger.warning("mic_toggle handler error: %s", exc)
            return

        if msg_type == "music_toggle" and self._on_music_toggle:
            try:
                await self._on_music_toggle(client_id)
            except Exception as exc:
                logger.warning("music_toggle handler error: %s", exc)
            return

        if msg_type == "music_stop" and self._on_music_stop:
            try:
                await self._on_music_stop(client_id)
            except Exception as exc:
                logger.warning("music_stop handler error: %s", exc)
            return

        if msg_type == "music_play_track" and self._on_music_play_track:
            pos = payload.get("position")
            if pos is not None:
                try:
                    await self._on_music_play_track(int(pos), client_id)
                except Exception as exc:
                    logger.warning("music_play_track handler error: %s", exc)
            return

        if msg_type == "timer_cancel" and self._on_timer_cancel:
            timer_id = payload.get("timer_id", "")
            if timer_id:
                try:
                    await self._on_timer_cancel(str(timer_id), client_id)
                except Exception as exc:
                    logger.warning("timer_cancel handler error: %s", exc)
            return

        if msg_type == "chat_new" and self._on_chat_new:
            try:
                await self._on_chat_new(client_id)
            except Exception as exc:
                logger.warning("chat_new handler error: %s", exc)
            return

        if msg_type == "chat_text" and self._on_chat_text:
            text = str(payload.get("text", "")).strip()
            client_msg_id = payload.get("client_msg_id")
            if text:
                try:
                    await self._on_chat_text(text, client_id)
                    if websocket is not None:
                        await websocket.send(
                            json.dumps(
                                {
                                    "type": "chat_text_ack",
                                    "client_msg_id": client_msg_id,
                                    "ok": True,
                                }
                            )
                        )
                except Exception as exc:
                    logger.warning("chat_text handler error: %s", exc)
                    if websocket is not None:
                        try:
                            await websocket.send(
                                json.dumps(
                                    {
                                        "type": "chat_text_ack",
                                        "client_msg_id": client_msg_id,
                                        "ok": False,
                                        "error": str(exc),
                                    }
                                )
                            )
                        except Exception:
                            pass
            return

        logger.debug("Web UI: unhandled action '%s' from %s", msg_type, client_id)

    def _handle_pcm_chunk(self, pcm_bytes: bytes) -> None:
        if len(pcm_bytes) < 2:
            return
        sample_count = len(pcm_bytes) // 2
        if sample_count <= 0:
            return
        pcm_view = memoryview(pcm_bytes)[:sample_count * 2].cast("h")
        sum_sq = 0.0
        peak = 0
        for sample in pcm_view:
            s = int(sample)
            abs_s = -s if s < 0 else s
            if abs_s > peak:
                peak = abs_s
            sum_sq += float(s * s)
        rms = math.sqrt(sum_sq / float(sample_count)) / 32768.0
        self._latest_browser_audio["rms"] = max(0.0, min(1.0, rms))
        self._latest_browser_audio["peak"] = max(0.0, min(1.0, float(peak) / 32768.0))
        self._browser_pcm_frames.append(pcm_bytes)

    # ------------------------------------------------------------------
    # Status broadcast loop
    # ------------------------------------------------------------------

    async def _status_loop(self) -> None:
        while True:
            await asyncio.sleep(self.status_interval_s)
            if not self._clients:
                continue
            payload = self._build_status_payload()
            message = json.dumps(payload)
            stale: list[Any] = []
            for client in list(self._clients):
                try:
                    await client.send(message)
                except Exception:
                    stale.append(client)
            for c in stale:
                self._clients.discard(c)

    def _build_status_payload(self) -> dict[str, Any]:
        now = time.monotonic()
        hotword_active = (
            self._last_hotword_ts is not None
            and (now - self._last_hotword_ts) <= self.hotword_active_s
        )
        orch = dict(self._orchestrator_status)
        orch["hotword_active"] = hotword_active
        orch["mic_enabled"] = self._ui_control_state.get("mic_enabled", False)
        return {
            "type": "orchestrator_status",
            "ts": time.time(),
            **orch,
            "browser_audio": dict(self._latest_browser_audio),
        }

    def _build_state_snapshot(self) -> dict[str, Any]:
        now = time.monotonic()
        hotword_active = (
            self._last_hotword_ts is not None
            and (now - self._last_hotword_ts) <= self.hotword_active_s
        )
        orch = dict(self._orchestrator_status)
        orch["hotword_active"] = hotword_active
        return {
            "type": "state_snapshot",
            "orchestrator": orch,
            "ui_control": dict(self._ui_control_state),
            "music": dict(self._music_state),
            "timers": list(self._timers_state),
            "chat": list(self._chat_messages[-50:]),
            "chat_threads": list(self._chat_threads),
            "active_chat_id": self._active_chat_id,
        }

    # ------------------------------------------------------------------
    # HTTP server
    # ------------------------------------------------------------------

    def _start_http_server(self) -> None:
        html = _build_ui_html(self.ws_port, self.mic_starts_disabled, self.audio_authority)

        class UIHandler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
                return

            def _send(self, body: bytes, status: int = 200, content_type: str = "text/html; charset=utf-8") -> None:
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Access-Control-Allow-Headers", "*")
                self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
                self.end_headers()
                self.wfile.write(body)

            def do_OPTIONS(self) -> None:  # noqa: N802
                self._send(b"", status=204, content_type="text/plain")

            def do_GET(self) -> None:  # noqa: N802
                path = self.path.split("?")[0]
                if path in ("/", "/index.html"):
                    self._send(html.encode("utf-8"))
                elif path == "/health":
                    self._send(
                        json.dumps({"status": "ok", "service": "embedded-voice-ui"}).encode(),
                        content_type="application/json",
                    )
                else:
                    self._send(b"Not found", status=404, content_type="text/plain")

        self._http_server = HTTPServer((self.host, self.ui_port), UIHandler)
        self._http_thread = threading.Thread(target=self._http_server.serve_forever, daemon=True)
        self._http_thread.start()
