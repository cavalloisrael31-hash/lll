
(() => {
'use strict';
const embedded = new URLSearchParams(location.search).get('embed') === '1'; if(embedded) document.body.classList.add('embed');
const { mat4, vec3 } = glMatrix;
const NUM_VERTICES = 10475, NUM_FACES = 20908, NUM_JOINTS = 55;
const PRESETS = {
  neutral:   { label:'自然 / 重置', channels:{}, note:'恢复未叠加任何面部表情的基础脸。' },
  blink:     { label:'双眼眨眼', channels:{8:1.00, 9:1.00,18:0.12,19:0.12}, note:'闭眼并轻微挤压眼周，避免眼皮显得僵硬。' },
  smile:     { label:'微笑', channels:{43:1.00,44:1.00,6:0.34,7:0.34,27:0.18,28:0.18}, note:'嘴角上扬，同时带动脸颊与酒窝区域。' },
  surprise:  { label:'惊讶', channels:{2:0.62,3:0.30,4:0.30,20:0.78,21:0.78,24:0.92,42:0.18,43:0.12,44:0.12}, note:'抬眉、睁眼、张嘴与轻微下唇变化。' },
  frown:     { label:'皱眉 / 不高兴', channels:{0:0.78,1:0.78,29:0.92,30:0.92,35:0.24,36:0.24}, note:'眉毛压低、嘴角下压，并轻微压紧嘴唇。' },
  pucker:    { label:'噘嘴', channels:{37:1.00,22:0.22,26:0.16}, note:'嘴唇前噘，配合轻微闭嘴与下巴前伸。' },
  thinking:  { label:'思考', channels:{2:1.05,3:1.00,4:0.12,18:0.55,19:0.20,35:0.33,36:0.10,29:0.10,30:0.10,41:0.16}, note:'单侧外眉明显抬起，配合轻眯眼、轻压嘴唇和下唇微收。' },
  sneer:     { label:'鼻翼皱起', channels:{49:1.00,50:1.00,35:0.16,36:0.16,29:0.16,30:0.16}, note:'鼻翼收紧，并带动上唇和嘴角的细微变化。' }
};
let gl, program, canvas, vao, positionBuffer, normalBuffer, boneIndexBuffer, boneWeightBuffer, faceIndexBuffer, indexBuffer, faceDeltaTexture;
let model = null, blendshapes = null, faceIds = null, activePreset = 'neutral', preferredPreset = 'neutral';
let faceWeights = new Float32Array(52), faceWeightsDirty = true, bonesDirty = true;
let boneMatrices = new Float32Array(NUM_JOINTS * 16), rotations = Array.from({length:NUM_JOINTS},()=>[0,0,0]);
// 静态资源和高频参数分离：换模型时释放并按新尺寸重建；交互帧只更新 uniforms。
let gpuRevision=0, uploadedRevision=-1, faceTexWidth=0, faceTexHeight=0;
const PARENTS = [-1,0,0,0,1,2,3,4,5,6,7,8,9,12,12,9,13,14,16,17,18,19,15,22,23,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,21,21,21,21,21,21,21,21,21,21,21,21,21,21,21];
const ANIMATION_SECONDS = 2, ANIMATION_FPS = 30, ANIMATION_FRAMES = ANIMATION_SECONDS * ANIMATION_FPS;
let animation = { playing:false, startedAt:0, lastFrame:-1 };
const ACTION_FRAMES=90, ACTION_FPS=30; let selectedAction='waveHello', actionIntensity=1, rootOffset=[0,0,0], actionPlayer={playing:false,frame:0,lastTime:0,accumulator:0};
let talkingMouth = 0, expressionIntensity = 1;
let speechExpressionAnimating = false, speechExpressionReleasing = false, speechExpressionWeight = 1, speechExpressionLastTime = 0;
let audioContext = null, speakingQueue = [], playingSpeech = false, currentSpeechAudio = null;
let chatController = null, isChatting = false, isRecording = false, mediaRecorder = null, recordStream = null, recordChunks = [], currentAssistantBubble = null;
let camera = { azimuth:Math.PI/2, elevation:0.02, distance:0.72, target:[0,0.20,0.02], drag:false, pan:false, x:0, y:0 };

const VS = `#version 300 es
precision highp float;
in vec3 a_position; in vec3 a_normal; in vec4 a_boneIndices; in vec4 a_boneWeights; in float a_faceIndex;
uniform mat4 u_view,u_projection,u_bones[55]; uniform sampler2D u_faceDeltaTex; uniform float u_faceWeights[52]; uniform int u_faceTexWidth;
out vec3 v_normal; out vec3 v_pos;
ivec2 texel(int i){ return ivec2(i % u_faceTexWidth, i / u_faceTexWidth); }
mat4 skin(){ ivec4 i=ivec4(a_boneIndices+.5); return u_bones[i.x]*a_boneWeights.x+u_bones[i.y]*a_boneWeights.y+u_bones[i.z]*a_boneWeights.z+u_bones[i.w]*a_boneWeights.w; }
void main(){
  vec3 p=a_position; int fv=int(a_faceIndex+.5);
  if(fv>=0) for(int c=0;c<52;c++) if(abs(u_faceWeights[c])>.00001) p+=texelFetch(u_faceDeltaTex,texel(c*5023+fv),0).xyz*u_faceWeights[c];
  mat4 m=skin(); vec4 world=m*vec4(p,1.); v_pos=world.xyz; v_normal=normalize(mat3(m)*a_normal); gl_Position=u_projection*u_view*world;
}`;
const FS = `#version 300 es
precision highp float; in vec3 v_normal; in vec3 v_pos; uniform vec3 u_eye; out vec4 outColor;
void main(){ vec3 n=normalize(v_normal), v=normalize(u_eye-v_pos); vec3 l1=normalize(vec3(-0.28,0.66,0.72)), l2=normalize(vec3(0.58,0.28,0.44));
float d1=max(dot(n,l1),0.0), d2=max(dot(n,l2),0.0); float rim=pow(1.0-max(dot(n,v),0.0),2.5); float spec=pow(max(dot(n,normalize(l1+v)),0.0),48.0);
vec3 c=vec3(.80,.67,.58)*(vec3(.17,.18,.23)+d1*vec3(.82,.69,.58)+d2*vec3(.18,.22,.31))+spec*.20+vec3(.18,.34,.60)*rim*.16;
outColor=vec4(pow(c,vec3(1.0/2.2)),1.0); }`;

function status(text, kind='') { console[kind==='error'?'error':'log'](text); }
function setLoaded(id, name) { const e=document.querySelector(id); e.classList.add('loaded'); e.title='已加载：'+name; }
function shader(type, src) { const s=gl.createShader(type); gl.shaderSource(s,src); gl.compileShader(s); if(!gl.getShaderParameter(s,gl.COMPILE_STATUS)) throw Error(gl.getShaderInfoLog(s)); return s; }
function setupGL() { canvas=document.querySelector('#glcanvas'); gl=canvas.getContext('webgl2',{antialias:true}); if(!gl) throw Error('浏览器不支持 WebGL2'); program=gl.createProgram(); gl.attachShader(program,shader(gl.VERTEX_SHADER,VS)); gl.attachShader(program,shader(gl.FRAGMENT_SHADER,FS)); gl.linkProgram(program); if(!gl.getProgramParameter(program,gl.LINK_STATUS)) throw Error(gl.getProgramInfoLog(program)); vao=gl.createVertexArray(); }
function parseModel(buf) {
  const expected=NUM_JOINTS*3*4+NUM_VERTICES*11*4+NUM_FACES*3*4;
  if(buf.byteLength!==expected) throw Error(`模型大小不匹配：${buf.byteLength}，预期 ${expected}`);
  const joints=new Float32Array(buf,0,NUM_JOINTS*3), raw=new Float32Array(buf,NUM_JOINTS*3*4,NUM_VERTICES*11);
  const positions=new Float32Array(NUM_VERTICES*3), boneIndices=new Float32Array(NUM_VERTICES*4), boneWeights=new Float32Array(NUM_VERTICES*4);
  for(let i=0;i<NUM_VERTICES;i++){ const s=i*11;positions.set(raw.subarray(s,s+3),i*3);boneIndices.set(raw.subarray(s+3,s+7),i*4);boneWeights.set(raw.subarray(s+7,s+11),i*4); }
  const faces=new Uint32Array(buf,NUM_JOINTS*3*4+NUM_VERTICES*11*4,NUM_FACES*3);
  return {joints,positions,boneIndices,boneWeights,faces};
}
function parseNpy(buf) { const b=new Uint8Array(buf); if(String.fromCharCode(...b.slice(0,6))!=='\x93NUMPY') throw Error('不是 NumPy .npy 文件'); const major=b[6], headerLen=major===1?new DataView(buf).getUint16(8,true):new DataView(buf).getUint32(8,true); const start=major===1?10:12; const header=new TextDecoder('latin1').decode(b.slice(start,start+headerLen)); const descr=(header.match(/'descr':\s*'([^']+)'/)||[])[1]; const shapeText=(header.match(/'shape':\s*\(([^)]*)\)/)||[])[1]; if(!descr||!shapeText) throw Error('无法解析 .npy 文件头'); const shape=shapeText.split(',').map(x=>x.trim()).filter(Boolean).map(Number); return {descr,shape,dataOffset:start+headerLen,buffer:buf}; }
function loadBlend(buf) { const n=parseNpy(buf); if(n.descr!=='<f8'||n.shape.join(',')!=='52,5023,3') throw Error(`表情文件格式不符：${n.descr} / (${n.shape})`); if(n.dataOffset%8) throw Error('表情数据未按 8 字节对齐'); blendshapes=new Float64Array(n.buffer,n.dataOffset,52*5023*3); }
function loadMap(buf) { const n=parseNpy(buf); if(n.descr!=='<i8'||n.shape.join(',')!=='5023') throw Error(`映射文件格式不符：${n.descr} / (${n.shape})`); const view=new DataView(n.buffer,n.dataOffset); faceIds=new Uint32Array(5023); for(let i=0;i<5023;i++){ const id=Number(view.getBigInt64(i*8,true)); if(id<0||id>=NUM_VERTICES) throw Error(`映射顶点越界：${id}`); faceIds[i]=id; } }
function normals(pos, faces) { const out=new Float32Array(pos.length); for(let i=0;i<faces.length;i+=3){ const a=faces[i]*3,b=faces[i+1]*3,c=faces[i+2]*3; const ux=pos[b]-pos[a],uy=pos[b+1]-pos[a+1],uz=pos[b+2]-pos[a+2],vx=pos[c]-pos[a],vy=pos[c+1]-pos[a+1],vz=pos[c+2]-pos[a+2]; const x=uy*vz-uz*vy,y=uz*vx-ux*vz,z=ux*vy-uy*vx; for(const q of [a,b,c]){out[q]+=x;out[q+1]+=y;out[q+2]+=z;} } for(let i=0;i<out.length;i+=3){const l=Math.hypot(out[i],out[i+1],out[i+2])||1;out[i]/=l;out[i+1]/=l;out[i+2]/=l;} return out; }
function bindAttribute(name,data,size){ const b=gl.createBuffer(),loc=gl.getAttribLocation(program,name);gl.bindBuffer(gl.ARRAY_BUFFER,b);gl.bufferData(gl.ARRAY_BUFFER,data,gl.STATIC_DRAW);gl.enableVertexAttribArray(loc);gl.vertexAttribPointer(loc,size,gl.FLOAT,false,0,0);return b; }
function disposeGpuAssets(){ for(const b of [positionBuffer,normalBuffer,boneIndexBuffer,boneWeightBuffer,faceIndexBuffer,indexBuffer])if(b)gl.deleteBuffer(b);if(faceDeltaTexture)gl.deleteTexture(faceDeltaTexture);positionBuffer=normalBuffer=boneIndexBuffer=boneWeightBuffer=faceIndexBuffer=indexBuffer=faceDeltaTexture=null;uploadedRevision=-1; }
function invalidateGpu(){ gpuRevision++; if(gl)disposeGpuAssets(); }
function buffers() { gl.bindVertexArray(vao); positionBuffer=bindAttribute('a_position',model.positions,3);normalBuffer=bindAttribute('a_normal',normals(model.positions,model.faces),3);boneIndexBuffer=bindAttribute('a_boneIndices',model.boneIndices,4);boneWeightBuffer=bindAttribute('a_boneWeights',model.boneWeights,4);const ids=new Float32Array(model.positions.length/3);ids.fill(-1);for(let i=0;i<faceIds.length;i++)ids[faceIds[i]]=i;faceIndexBuffer=bindAttribute('a_faceIndex',ids,1);indexBuffer=gl.createBuffer();gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER,indexBuffer);gl.bufferData(gl.ELEMENT_ARRAY_BUFFER,model.faces,gl.STATIC_DRAW);gl.bindVertexArray(null); }
function uploadFaceTexture(){ const texels=52*5023,limit=gl.getParameter(gl.MAX_TEXTURE_SIZE);faceTexWidth=Math.min(limit,Math.max(1,Math.min(4096,texels)));faceTexHeight=Math.ceil(texels/faceTexWidth);if(faceTexHeight>limit)throw Error(`表情形状基需要 ${faceTexWidth}×${faceTexHeight} 浮点纹理，超过本设备上限 ${limit}`);const packed=new Float32Array(faceTexWidth*faceTexHeight*4);for(let i=0;i<texels;i++){packed[i*4]=blendshapes[i*3];packed[i*4+1]=blendshapes[i*3+1];packed[i*4+2]=blendshapes[i*3+2];}faceDeltaTexture=gl.createTexture();gl.bindTexture(gl.TEXTURE_2D,faceDeltaTexture);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MIN_FILTER,gl.NEAREST);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MAG_FILTER,gl.NEAREST);gl.texImage2D(gl.TEXTURE_2D,0,gl.RGBA32F,faceTexWidth,faceTexHeight,0,gl.RGBA,gl.FLOAT,packed); }
function ready() { if(!model||!blendshapes||!faceIds) return; if(uploadedRevision!==gpuRevision){disposeGpuAssets();buffers();uploadFaceTexture();uploadedRevision=gpuRevision;}document.querySelector('#loading').classList.add('hidden');status(`静态资源已上传 GPU（纹理 ${faceTexWidth}×${faceTexHeight}）；运行时只提交脏的表情/骨骼参数。`,'ok');applyExpression(); }
function applyExpression(multiplier=1) {
  faceWeights.fill(0); const preset=PRESETS[activePreset],speechBlend=speechExpressionAnimating?speechExpressionWeight:1,s=Number(document.querySelector('#strength').value)*expressionIntensity*speechBlend;
  for(const [c,w] of Object.entries(preset.channels)) faceWeights[Number(c)]=w*s*multiplier;
  // 口型与表情共用下颌通道：降低预设张口、再叠加小幅语音运动，避免说话时“夸张张嘴”。
  faceWeights[24]=Math.min(.58,faceWeights[24]*.45+talkingMouth); faceWeightsDirty=true;
}
function updateBoneMatrices(){
  const world=Array(NUM_JOINTS);for(let i=0;i<NUM_JOINTS;i++){const p=PARENTS[i],local=mat4.create(),r=mat4.create(),j=model.joints,root=p<0?rootOffset:[0,0,0];const x=j[i*3]-(p<0?0:j[p*3])+root[0],y=j[i*3+1]-(p<0?0:j[p*3+1])+root[1],z=j[i*3+2]-(p<0?0:j[p*3+2])+root[2];mat4.fromTranslation(local,[x,y,z]);mat4.rotateX(r,r,rotations[i][0]);mat4.rotateY(r,r,rotations[i][1]);mat4.rotateZ(r,r,rotations[i][2]);mat4.multiply(local,local,r);world[i]=mat4.create();if(p<0)mat4.copy(world[i],local);else mat4.multiply(world[i],world[p],local);const inv=mat4.create();mat4.fromTranslation(inv,[-j[i*3],-j[i*3+1],-j[i*3+2]]);mat4.multiply(boneMatrices.subarray(i*16,i*16+16),world[i],inv);}bonesDirty=true;
}
function smooth(x){ x=Math.max(0,Math.min(1,x)); return x*x*(3-2*x); }
function blinkPulse(t, center){ const x=Math.abs(t-center)/.14; return x>=1 ? 0 : Math.sin(Math.PI*x); }
function animationWeight(t){
  if(activePreset==='neutral') return 0;
  if(activePreset==='blink') return Math.max(blinkPulse(t,.58),blinkPulse(t,1.30));
  const attack=activePreset==='surprise'?.20:.38, release=.44, holdEnd=ANIMATION_SECONDS-release;
  let w=t<attack?smooth(t/attack):t>holdEnd?1-smooth((t-holdEnd)/release):1;
  // 停留阶段极轻的肌肉松弛，避免像定格模型一样僵硬。
  if(t>=attack&&t<=holdEnd) w*=.985+.015*Math.sin((t-attack)*Math.PI*2.4);
  return w;
}
function updateAnimation(now){
  if(!animation.playing) return;
  const elapsed=Math.max(0,(now-animation.startedAt)/1000), frame=Math.min(ANIMATION_FRAMES-1,Math.floor(elapsed*ANIMATION_FPS));
  if(frame!==animation.lastFrame){ animation.lastFrame=frame; const t=frame/ANIMATION_FPS; applyExpression(animationWeight(t)); }
  if(elapsed>=ANIMATION_SECONDS){ animation.playing=false; applyExpression(0); document.querySelector('#render-button').disabled=false; document.querySelector('#stop-button').disabled=true; }
}
function startAnimation(){ if(!positionBuffer) return; animation={playing:true,startedAt:performance.now(),lastFrame:-1}; document.querySelector('#render-button').disabled=true;document.querySelector('#stop-button').disabled=false; }
function stopAnimation(reset=true){ if(!animation.playing) return; animation.playing=false;document.querySelector('#render-button').disabled=false;document.querySelector('#stop-button').disabled=true;if(reset) applyExpression(); }
function actionStatus(){const el=document.querySelector('#action-status');if(el)el.textContent=`${document.querySelector('#action-select').selectedOptions[0].text} · 帧 ${actionPlayer.frame+1}/${ACTION_FRAMES} · ${actionPlayer.playing?'渲染中':'已停止'}`;}
function stopAction(){actionPlayer.playing=false;actionPlayer.accumulator=0;actionStatus();}
function applyActionFrame(frame){const t=frame/(ACTION_FRAMES-1),ease=x=>x*x*(3-2*x),raised=ease(Math.min(1,t/.28));for(let i=0;i<NUM_JOINTS;i++)rotations[i]=[0,0,0];rootOffset=[0,0,0];const d=Math.PI/180;
 if(selectedAction==='waveHello'){const w=Math.sin(Math.max(0,(t-.28)/.72)*Math.PI*6)*ease(Math.max(0,(t-.28)/.72));rotations[13]=[0,0,12*d*raised];rotations[16]=[0,9*d*w,72*d*raised];rotations[18]=[-22*d*raised,0,0];rotations[20]=[-90*d*raised,0,12*d*w];}
 else if(selectedAction==='thinking'){const q=ease(Math.min(1,t/.78));rotations[17]=[0,85*d*q,25*d*q];rotations[19]=[0,42*d*q,-52*d*q];rotations[21]=[-45*d*q,12*d*q,8*d*q];rotations[15]=[0,0,4*d*q];rotations[43]=[0,0,24*d*q];rotations[44]=[0,0,30*d*q];}
 else if(selectedAction==='jump'){const p=Math.max(0,Math.min(1,(t-.25)/.55)),c=t<.25?.12*Math.sin(Math.PI*.5*Math.min(1,t/.25)):0,h=.18*Math.sin(Math.PI*p),bend=Math.min(1,c/.12);rootOffset=[0,h-c,0];for(const i of [1,2])rotations[i]=[-22*d*bend,0,0];for(const i of [4,5])rotations[i]=[42*d*bend,0,0];}
 else if(selectedAction==='leftArmRaise'){rotations[13]=[0,0,10*d*raised];rotations[16]=[0,0,55*d*raised];}
 else if(selectedAction==='aPose'){rotations[13]=[0,0,-.3*raised];rotations[14]=[0,0,.3*raised];rotations[16]=[0,0,-.5*raised];rotations[17]=[0,0,.5*raised];}
 else if(selectedAction==='sit'){const q=ease(Math.min(1,t/.7));rootOffset=[0,-.4*q,-.2*q];for(const i of [1,2])rotations[i]=[-88*d*q,0,0];for(const i of [4,5])rotations[i]=[92*d*q,0,0];}
 for(let i=0;i<NUM_JOINTS;i++)rotations[i]=rotations[i].map(v=>v*actionIntensity);rootOffset=rootOffset.map(v=>v*actionIntensity);bonesDirty=true;actionStatus();}
function playAction(intensity=1){if(!model)return;actionIntensity=Math.max(0,Math.min(1,Number(intensity)||1));actionPlayer={playing:true,frame:0,lastTime:performance.now(),accumulator:0};applyActionFrame(0);}
function seekAction(frame){stopAction();actionPlayer.frame=Math.max(0,Math.min(ACTION_FRAMES-1,Math.round(frame)));applyActionFrame(actionPlayer.frame);}
function updateAction(now){if(!actionPlayer.playing)return;actionPlayer.accumulator+=Math.min(100,now-actionPlayer.lastTime);actionPlayer.lastTime=now;while(actionPlayer.accumulator>=1000/ACTION_FPS){actionPlayer.accumulator-=1000/ACTION_FPS;if(actionPlayer.frame>=ACTION_FRAMES-1){stopAction();break;}applyActionFrame(++actionPlayer.frame);}}
function markPreset(key) { document.querySelectorAll('[data-preset]').forEach(b=>b.classList.toggle('active',b.dataset.preset===key)); }
function selectPreset(key) { stopAnimation(false); preferredPreset=key; activePreset=key; expressionIntensity=1; markPreset(key); applyExpression(); }
function setSpeechExpression(key, intensity=1) { if(!PRESETS[key]||key==='neutral') return; activePreset=key; expressionIntensity=Math.max(.35,Math.min(1.0,Number(intensity)||1)); markPreset(key); if(!speechExpressionAnimating||speechExpressionReleasing){speechExpressionAnimating=true;speechExpressionReleasing=false;speechExpressionWeight=0;speechExpressionLastTime=performance.now();} applyExpression(); }
function restorePreferredExpression() { activePreset=preferredPreset; expressionIntensity=1; markPreset(preferredPreset); applyExpression(); }
function releaseSpeechExpression(){ speechExpressionReleasing=true; }
function updateSpeechExpression(now){
  if(!speechExpressionAnimating) return;
  const dt=Math.min(.05,Math.max(0,(now-speechExpressionLastTime)/1000)); speechExpressionLastTime=now;
  const target=speechExpressionReleasing?0:(.90+.05*Math.sin(now*.006));
  speechExpressionWeight+=(target-speechExpressionWeight)*Math.min(1,dt*(speechExpressionReleasing?5:7));
  if(speechExpressionReleasing&&speechExpressionWeight<.012){speechExpressionAnimating=false;speechExpressionReleasing=false;speechExpressionWeight=1;restorePreferredExpression();return;}
  applyExpression();
}
function setVoiceStatus(text){ document.querySelector('#voice-status').textContent=text; }
function base64ToBlob(value){ const raw=atob(value), bytes=new Uint8Array(raw.length); for(let i=0;i<raw.length;i++) bytes[i]=raw.charCodeAt(i); return new Blob([bytes],{type:'audio/wav'}); }
function setChatting(value){ isChatting=value; document.querySelector('#chat-send').disabled=value; document.querySelector('#chat-stop').disabled=!value; }
function addChatMessage(text, role, append=false){ const log=document.querySelector('#chat-log'); if(append&&currentAssistantBubble){currentAssistantBubble.textContent+=text;}else{const node=document.createElement('div');node.className='chat-message '+role;node.textContent=text;log.append(node);if(role==='ai')currentAssistantBubble=node;}log.scrollTop=log.scrollHeight; }
function playNextSpeech(){
  if(playingSpeech||!speakingQueue.length) return;
  const part=speakingQueue.shift(); playingSpeech=true; if(part.action&&part.action!=='idle'){selectedAction=part.action;playAction(part.action_intensity);} setSpeechExpression(part.expression,part.expression_intensity); setVoiceStatus('正在语音回复：'+part.text);
  const audio=new Audio(URL.createObjectURL(base64ToBlob(part.audio))); currentSpeechAudio=audio;
  audioContext ||= new (window.AudioContext||window.webkitAudioContext)();
  const source=audioContext.createMediaElementSource(audio), analyser=audioContext.createAnalyser(); analyser.fftSize=256; source.connect(analyser); analyser.connect(audioContext.destination); const samples=new Uint8Array(analyser.fftSize);
  const animateMouth=()=>{ if(audio.paused||audio.ended) return; analyser.getByteTimeDomainData(samples); let sum=0; for(const x of samples){const d=(x-128)/128;sum+=d*d;} const rms=Math.sqrt(sum/samples.length); const target=Math.min(.26,Math.max(.018,rms*2.4)); talkingMouth+=(target-talkingMouth)*.16; applyExpression();requestAnimationFrame(animateMouth); };
  audio.onplay=()=>{audioContext.resume();animateMouth();};
  audio.onended=()=>{URL.revokeObjectURL(audio.src);talkingMouth=0;applyExpression();playingSpeech=false;currentSpeechAudio=null;if(part.endpoint&&!speakingQueue.length){releaseSpeechExpression();setVoiceStatus('回复完成');setChatting(false);}playNextSpeech();};
  audio.play().catch(e=>{playingSpeech=false;setVoiceStatus('音频播放失败：'+e.message);playNextSpeech();});
}
function stopChat(){ if(chatController)chatController.abort(); chatController=null; speakingQueue=[]; if(currentSpeechAudio){currentSpeechAudio.pause();currentSpeechAudio=null;} playingSpeech=false;talkingMouth=0;speechExpressionAnimating=false;speechExpressionReleasing=false;speechExpressionWeight=1;restorePreferredExpression();setChatting(false);setVoiceStatus('已停止回复'); }
async function startChat(payload){
  if(isChatting)stopChat(); setChatting(true);setVoiceStatus('正在生成语音和表情…');currentAssistantBubble=null;chatController=new AbortController();
  try { const response=await fetch('/eb_stream',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload),signal:chatController.signal}); if(!response.ok)throw Error(await response.text());
    const reader=response.body.getReader(),decoder=new TextDecoder();let pending='';while(true){const {value,done}=await reader.read();pending+=decoder.decode(value||new Uint8Array(),{stream:!done});let pos;while((pos=pending.indexOf('\n'))>=0){const line=pending.slice(0,pos).trim();pending=pending.slice(pos+1);if(!line)continue;const part=JSON.parse(line);if(part.prompt){addChatMessage(part.prompt,'user');continue;}if(part.type==='control'){if(part.action&&part.action!=='idle'){selectedAction=part.action;playAction(part.action_intensity);}if(part.expression&&part.expression!=='neutral')setSpeechExpression(part.expression,part.expression_intensity);continue;}if(part.text){addChatMessage(part.text,'ai',Boolean(currentAssistantBubble));}if(part.audio){speakingQueue.push(part);playNextSpeech();}}if(done)break;}
  }catch(e){if(e.name!=='AbortError'){setVoiceStatus('请求失败：'+e.message);setChatting(false);}}
}
function sendChat(){ const input=document.querySelector('#chat-input'),prompt=input.value.trim();if(!prompt)return;input.value='';addChatMessage(prompt,'user');startChat({input_mode:'text',prompt,voice_speed:1,voice_id:'demo'}); }
function blobToBase64(blob){return new Promise((resolve,reject)=>{const r=new FileReader();r.onload=()=>resolve(String(r.result).split(',')[1]);r.onerror=reject;r.readAsDataURL(blob);});}
async function toggleRecording(){
  const button=document.querySelector('#record-button');if(isRecording){mediaRecorder.stop();return;}if(isChatting)stopChat();
  try{recordStream=await navigator.mediaDevices.getUserMedia({audio:true});recordChunks=[];mediaRecorder=new MediaRecorder(recordStream);mediaRecorder.ondataavailable=e=>{if(e.data.size)recordChunks.push(e.data);};mediaRecorder.onstop=async()=>{isRecording=false;button.classList.remove('recording');button.textContent='● 点击说话';recordStream.getTracks().forEach(t=>t.stop());const blob=new Blob(recordChunks,{type:mediaRecorder.mimeType||'audio/webm'});if(blob.size<1000){setVoiceStatus('录音过短，请重新录制');return;}setVoiceStatus('正在识别语音…');startChat({input_mode:'audio',audio:await blobToBase64(blob),voice_speed:1,voice_id:'demo'});};mediaRecorder.start();isRecording=true;button.classList.add('recording');button.textContent='■ 点击结束';setVoiceStatus('正在录音，点击结束后发送');}catch(e){setVoiceStatus('无法访问麦克风：'+e.message);}
}
function buildPresetUI(){ const grid=document.querySelector('#expression-grid'),icons={neutral:'○',blink:'◉',smile:'⌣',surprise:'!',frown:'⌢',pucker:'◦',thinking:'?',sneer:'≈'};for(const [key,p] of Object.entries(PRESETS)){const b=document.createElement('button');b.textContent=icons[key]||'•';b.title=p.label;b.setAttribute('aria-label',p.label);b.dataset.preset=key;b.addEventListener('click',()=>selectPreset(key));grid.append(b);} selectPreset('neutral'); }
function eye(){ const a=camera.azimuth,e=camera.elevation,d=camera.distance; return [camera.target[0]+d*Math.cos(e)*Math.cos(a),camera.target[1]+d*Math.sin(e),camera.target[2]+d*Math.cos(e)*Math.sin(a)]; }
function render(now){ updateAnimation(now);updateSpeechExpression(now);updateAction(now); const dpr=Math.min(devicePixelRatio||1,2),w=Math.floor(canvas.clientWidth*dpr),h=Math.floor(canvas.clientHeight*dpr);if(canvas.width!==w||canvas.height!==h){canvas.width=w;canvas.height=h;}gl.viewport(0,0,w,h);gl.clearColor(.035,.045,.075,1);gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);gl.enable(gl.DEPTH_TEST);if(model&&positionBuffer){const view=mat4.create(),proj=mat4.create(),e=eye();mat4.lookAt(view,e,camera.target,[0,1,0]);mat4.perspective(proj,Math.PI/4,w/Math.max(h,1),.01,100);gl.useProgram(program);gl.uniformMatrix4fv(gl.getUniformLocation(program,'u_view'),false,view);gl.uniformMatrix4fv(gl.getUniformLocation(program,'u_projection'),false,proj);gl.uniform3f(gl.getUniformLocation(program,'u_eye'),...e);if(bonesDirty){updateBoneMatrices();gl.uniformMatrix4fv(gl.getUniformLocation(program,'u_bones[0]'),false,boneMatrices);bonesDirty=false;}if(faceWeightsDirty){gl.uniform1fv(gl.getUniformLocation(program,'u_faceWeights[0]'),faceWeights);faceWeightsDirty=false;}gl.activeTexture(gl.TEXTURE0);gl.bindTexture(gl.TEXTURE_2D,faceDeltaTexture);gl.uniform1i(gl.getUniformLocation(program,'u_faceDeltaTex'),0);gl.uniform1i(gl.getUniformLocation(program,'u_faceTexWidth'),faceTexWidth);gl.bindVertexArray(vao);gl.drawElements(gl.TRIANGLES,model.faces.length,gl.UNSIGNED_INT,0);gl.bindVertexArray(null);}requestAnimationFrame(render); }
function cameraControls(){ canvas.addEventListener('mousedown',e=>{camera.drag=e.button===0;camera.pan=e.button===2;camera.x=e.clientX;camera.y=e.clientY;});window.addEventListener('mouseup',()=>{camera.drag=camera.pan=false;});window.addEventListener('mousemove',e=>{const dx=e.clientX-camera.x,dy=e.clientY-camera.y;if(camera.drag){camera.azimuth-=dx*.008;camera.elevation=Math.max(-1.45,Math.min(1.45,camera.elevation+dy*.008));}if(camera.pan){const sc=camera.distance*.0015;camera.target[0]+=Math.sin(camera.azimuth)*dx*sc;camera.target[1]+=-dy*sc;camera.target[2]+=-Math.cos(camera.azimuth)*dx*sc;}camera.x=e.clientX;camera.y=e.clientY;});canvas.addEventListener('wheel',e=>{e.preventDefault();camera.distance=Math.max(.25,Math.min(3,camera.distance*(1+e.deltaY*.001)));},{passive:false});canvas.addEventListener('contextmenu',e=>e.preventDefault()); }
async function fromInput(input, fn, label){try{await fn((await input.files[0].arrayBuffer()));invalidateGpu();setLoaded(label,input.files[0].name);ready();}catch(e){status('加载失败：'+e.message,'error');console.error(e);}}
async function autoLoad(){ const list=[['obj_model.bin',b=>{model=parseModel(b);},'#model-label'],['flame_arkit_bs.npy',loadBlend,'#blend-label'],['SMPL-X__FLAME_vertex_ids.npy',loadMap,'#map-label']]; for(const [file,fn,label] of list){try{const r=await fetch('./'+file);if(!r.ok)continue;await fn(await r.arrayBuffer());setLoaded(label,file);}catch(_){}} ready();}
async function init(){ try{setupGL();buildPresetUI();const requested=new URLSearchParams(location.search).get('preset');if(PRESETS[requested])selectPreset(requested);cameraControls();document.querySelector('#model-input').addEventListener('change',e=>fromInput(e.target,b=>{model=parseModel(b);},'#model-label'));document.querySelector('#blend-input').addEventListener('change',e=>fromInput(e.target,loadBlend,'#blend-label'));document.querySelector('#map-input').addEventListener('change',e=>fromInput(e.target,loadMap,'#map-label'));document.querySelector('#strength').addEventListener('input',e=>{document.querySelector('#strength-value').textContent=Number(e.target.value).toFixed(2);if(!animation.playing)applyExpression();});for(const [id,axis,label] of [['head-yaw',1,'head-yaw-value'],['head-pitch',0,'head-pitch-value']])document.querySelector('#'+id).addEventListener('input',e=>{rotations[15][axis]=Number(e.target.value)*Math.PI/180;bonesDirty=true;document.querySelector('#'+label).textContent=e.target.value+'°';});document.querySelector('#render-button').addEventListener('click',startAnimation);document.querySelector('#stop-button').addEventListener('click',()=>stopAnimation(true));document.querySelector('#chat-send').addEventListener('click',sendChat);document.querySelector('#chat-input').addEventListener('keydown',e=>{if(e.key==='Enter')sendChat();});document.querySelector('#record-button').addEventListener('click',toggleRecording);document.querySelector('#chat-stop').addEventListener('click',stopChat);await autoLoad();if(!model||!blendshapes||!faceIds){document.querySelector('#loading').classList.add('hidden');status('请依次加载模型、表情形状基和顶点映射文件。');}render();}catch(e){document.querySelector('#loading-text').textContent='初始化失败：'+e.message;console.error(e);}}
document.querySelector('#action-select').addEventListener('change',e=>{selectedAction=e.target.value;seekAction(0);});
document.querySelector('#action-play').addEventListener('click',()=>playAction());
document.querySelector('#action-pause').addEventListener('click',stopAction);
document.querySelector('#action-rewind').addEventListener('click',()=>seekAction(0));
window.addEventListener('message',event=>{const msg=event.data||{};if(msg.type!=='avatar-control')return;if(msg.command==='expression'&&PRESETS[msg.value])selectPreset(msg.value);if(msg.command==='action'){selectedAction=msg.value;playAction(msg.intensity);}if(msg.command==='chat'){const input=document.querySelector('#chat-input');input.value=msg.value||'';sendChat();}if(msg.command==='stop')stopChat();});
init();
})();
