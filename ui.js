/* 数字人 UI 模块：只负责生成界面结构；WebGL 渲染保留在 face_expression_viewer.html。 */
(() => {
  'use strict';
  const panel = document.querySelector('#ui-root');
  const chat = document.querySelector('#chat-ui-root');
  if (!panel || !chat) throw new Error('未找到 UI 挂载点');

  const style = document.createElement('style');
  style.textContent = `
    aside#ui-root{width:92px;padding:12px 9px;background:#fff;border-left:1px solid #e5e7eb;box-shadow:-8px 0 24px #0f172a12;overflow-y:auto}
    #ui-root section{margin:0 0 12px;padding:0;border:0;background:transparent} #ui-root h1,#ui-root h2{display:none}
    #ui-root .load{gap:7px} #ui-root label.file{height:42px;padding:9px;text-align:center;color:#334155;border:1px solid #dbe3ed;background:#f8fafc;font-size:17px}
    #ui-root label.file.loaded{color:#15803d;border-color:#bbf7d0;background:#f0fdf4} #ui-root label.file span{font-size:0} #ui-root label.file span::after{font-size:17px}#model-label span::after{content:'◫'}#blend-label span::after{content:'◌'}#map-label span::after{content:'⌘'}
    #ui-root .expression-grid{grid-template-columns:1fr;gap:6px} #ui-root button{min-height:39px;padding:6px;color:#334155;border-color:#dbe3ed;background:#fff;font-size:17px;box-shadow:0 1px 2px #0f172a0b}#ui-root button:hover,#ui-root button.active{color:#2563eb;border-color:#93c5fd;background:#eff6ff}
    #ui-root .range-row{justify-content:center;gap:0}#ui-root .range-row>span{display:none}#ui-root input[type=range]{width:70px;flex:none}#ui-root #strength-value,#ui-root #head-yaw-value,#ui-root #head-pitch-value{display:none}
    #ui-root .animation-actions,#ui-root .action-row{grid-template-columns:1fr;display:grid;gap:6px;margin-top:0}#ui-root .action-row select{height:39px;padding:4px;color:#334155;border-color:#dbe3ed;background:#fff;font-size:15px}#action-status{display:none}
    #chat-panel{background:#fffffff2;border-color:#e2e8f0;box-shadow:0 12px 36px #0f172a20;color:#1e293b}#chat-panel h2{display:none}.chat-log{max-height:110px}.chat-message.ai{background:#f1f5f9;color:#1e293b}.chat-message.user{background:#2563eb}
    .input-dock{display:grid;grid-template-columns:minmax(0,1fr) 130px;gap:8px}.voice-input-box{display:flex;gap:6px;padding:4px;border:1px solid #cbd5e1;border-radius:8px;background:#f8fafc}.voice-input-box button{flex:1;min-height:40px;color:#334155;background:#fff;border-color:#cbd5e1}.voice-input-box #record-button{color:#2563eb}.voice-input-box #record-button.recording{color:#dc2626;border-color:#fecaca;background:#fff1f2}.voice-row{display:flex;gap:6px}.voice-row input{min-width:0;flex:1;height:48px;color:#1e293b;background:#fff;border-color:#cbd5e1}.voice-row #chat-send{min-height:48px;color:#fff;background:#2563eb;border-color:#2563eb}.chat-actions{display:none}@media(max-width:560px){.input-dock{grid-template-columns:1fr}.voice-input-box{height:48px}}
  `;
  document.head.append(style);

  panel.innerHTML = `
    <section><div class="load">
      <label class="file" id="model-label" title="加载模型"><span>◫</span><input id="model-input" type="file" accept=".bin"></label>
      <label class="file" id="blend-label" title="加载表情数据"><span>◌</span><input id="blend-input" type="file" accept=".npy"></label>
      <label class="file" id="map-label" title="加载映射数据"><span>⌘</span><input id="map-input" type="file" accept=".npy"></label>
    </div></section>
    <section><div class="expression-grid" id="expression-grid"></div>
      <div class="range-row"><span></span><input id="strength" type="range" min="0" max="1.4" step="0.01" value="1.10"><span id="strength-value"></span></div>
    </section>
    <section><div class="animation-actions"><button id="render-button" title="播放表情">▶</button><button id="stop-button" title="停止表情" disabled>■</button></div></section>
    <section>
      <div class="action-row"><select id="action-select" title="选择动作"><option value="waveHello">👋</option><option value="thinking">💭</option><option value="jump">↟</option><option value="leftArmRaise">↖</option><option value="aPose">A</option><option value="sit">▱</option></select><button id="action-play" title="播放动作">▶</button></div>
      <div class="action-row" style="margin-top:8px"><button id="action-pause" title="停止动作">■</button><button id="action-rewind" title="首帧">↺</button></div><div id="action-status"></div>
    </section>
    <section>
      <div class="range-row"><span></span><input id="head-yaw" title="头部左右" type="range" min="-80" max="80" step="1" value="0"><span id="head-yaw-value"></span></div>
      <div class="range-row" style="margin-top:9px"><span></span><input id="head-pitch" title="头部上下" type="range" min="-55" max="55" step="1" value="0"><span id="head-pitch-value"></span></div>
    </section>`;

  chat.innerHTML = `<section id="chat-panel"><h2>语音对话</h2>
    <div class="chat-log" id="chat-log" aria-live="polite"></div>
    <div class="input-dock">
      <div class="voice-row"><input id="chat-input" type="text" placeholder="输入文字消息"><button id="chat-send" title="发送文字">发送</button></div>
      <div class="voice-input-box"><button id="record-button" title="点击开始或结束录音">🎙 语音</button><button id="chat-stop" title="停止当前回复" disabled>■</button></div>
    </div>
    <div id="voice-status">可输入文字或点击语音输入</div>
  </section>`;
})();
