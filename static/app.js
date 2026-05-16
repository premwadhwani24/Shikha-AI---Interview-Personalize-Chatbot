/*
  app.js — Starter logic for Shiksha AI animated assistant
  Features implemented:
  - Webcam access + face-api.js dynamic loader for simple eye-contact detection
  - Simple animated girl avatar drawn on #avatarCanvas with eye-tracking, blinking, and mouth movement
  - Speech input (SpeechRecognition) and speech output (SpeechSynthesis)
  - Basic conversation flow: clarifying questions for company/role -> role-specific questions -> grading by keyword-match
  - UI updates (messages area, score, fluency, feedback list)
  - Export report (simple JSON download)

  Note: This is a starter implementation to demo behavior locally. For production you'll want to
  - Move heavy ML to a secure backend or use hosted models
  - Add authentication, HTTPS, encrypted storage
  - Use robust NLP (LLMs or fine-tuned models) for grading and feedback
*/

// ------------------------------
// Helpers & simple state
// ------------------------------
const state = {
  sessionId: null,
  company: null,
  jobType: 'auto',
  role: null,
  questions: [],
  currentQ: -1,
  score: 0,
  total: 0,
  feedback: [],
  speaking: false,
  faceApiLoaded: false
};

// Simple question bank (expandable / replaceable with server-driven content)
const QUESTION_BANK = {
  technical: {
    'Data Analyst': [
      {q: 'Explain the difference between inner join and outer join in SQL.', keywords: ['inner join','outer join','left','right','full']},
      {q: 'How would you clean a dataset with many missing values?', keywords: ['drop','impute','mean','median','interpolate','remove']},
    ],
    'Software Engineer': [
      {q: 'Explain the difference between processes and threads.', keywords: ['process','thread','memory','context switch','concurrency']},
      {q: 'What is a race condition and how do you prevent it?', keywords: ['race','mutex','lock','atomic','synchronization']},
    ]
  },
  nontechnical: {
    'HR': [
      {q: 'Describe a time you handled a conflict at work.', keywords: ['conflict','listened','compromise','mediate','resolution']},
      {q: 'How do you handle giving negative feedback to an employee?', keywords: ['constructive','specific','private','examples','improvement']}
    ]
  }
}

// ------------------------------
// DOM refs
// ------------------------------
const webcamEl = document.getElementById('webcam');
const avatarCanvas = document.getElementById('avatarCanvas');
const ctx = avatarCanvas.getContext('2d');
const messagesEl = document.getElementById('messages');
const userInput = document.getElementById('userInput');
const sendBtn = document.getElementById('sendBtn');
const micBtn = document.getElementById('micBtn');
const startInterviewBtn = document.getElementById('startInterview');
const companyInput = document.getElementById('companyInput');
const roleInput = document.getElementById('roleInput');
const jobTypeSelect = document.getElementById('jobType');
const scoreEl = document.getElementById('score');
const fluencyEl = document.getElementById('fluency');
const feedbackList = document.getElementById('feedbackList');
const eyeStatus = document.getElementById('eyeStatus');
const postureStatus = document.getElementById('postureStatus');
const calibrateBtn = document.getElementById('calibrateBtn');
const downloadReportBtn = document.getElementById('downloadReport');

// ------------------------------
// Avatar: simple animated girl
// ------------------------------
let avatar = {
  eyeX: avatarCanvas.width/2,
  eyeY: avatarCanvas.height/2 - 20,
  blinkTimer: 0,
  mouthOpen: 0 // 0..1
};

function drawAvatar() {
  const w = avatarCanvas.width;
  const h = avatarCanvas.height;
  ctx.clearRect(0,0,w,h);

  // background circle
  ctx.fillStyle = '#0f172a00';

  // face
  ctx.fillStyle = '#fce7f3';
  roundRect(ctx, 20, 20, w-40, h-40, 28);
  ctx.fill();

  // hair
  ctx.fillStyle = '#1f2937';
  roundRect(ctx, 12, 6, w-24, 80, 28);
  ctx.fill();

  // eyes
  const leftEye = {x: w*0.35, y: h*0.38};
  const rightEye = {x: w*0.65, y: h*0.38};
  // eyeballs track a smoothed target
  ctx.fillStyle = '#fff';
  circle(ctx, leftEye.x, leftEye.y, 16);
  circle(ctx, rightEye.x, rightEye.y, 16);

  // pupils
  const pupilOffsetX = (avatar.eyeX - w/2)/12;
  const pupilOffsetY = (avatar.eyeY - h/2)/18;
  ctx.fillStyle = '#111827';
  circle(ctx, leftEye.x + pupilOffsetX, leftEye.y + pupilOffsetY, 6);
  circle(ctx, rightEye.x + pupilOffsetX, rightEye.y + pupilOffsetY, 6);

  // mouth
  const mouthX = w/2;
  const mouthY = h*0.62;
  const mouthW = 70;
  const mouthH = 10 + avatar.mouthOpen*18;
  ctx.fillStyle = '#9b111e';
  roundRect(ctx, mouthX - mouthW/2, mouthY - mouthH/2, mouthW, mouthH, 10);
  ctx.fill();

  // simple blush
  ctx.fillStyle = 'rgba(255,182,193,0.35)';
  circle(ctx, w*0.27, h*0.55, 10);
  circle(ctx, w*0.73, h*0.55, 10);

  // hair highlight
  ctx.fillStyle = 'rgba(255,255,255,0.03)';
  roundRect(ctx, 14, 12, 40, 60, 14);
  ctx.fill();
}

function circle(c, x, y, r){ c.beginPath(); c.arc(x,y,r,0,Math.PI*2); c.fill(); }
function roundRect(c, x, y, w, h, r){ c.beginPath(); c.moveTo(x+r,y); c.arcTo(x+w,y,x+w,y+h,r); c.arcTo(x+w,y+h,x,y+h,r); c.arcTo(x,y+h,x,y,r); c.arcTo(x,y,x+w,y,r); c.closePath(); }

// animate avatar: blinking and slight breathing
function avatarTick(){
  avatar.blinkTimer += 1;
  if(avatar.blinkTimer > 220 + Math.random()*200) { avatar.blinkTimer = 0; }
  // mouth smoothing
  if(state.speaking) avatar.mouthOpen = Math.min(1, avatar.mouthOpen + 0.2);
  else avatar.mouthOpen = Math.max(0, avatar.mouthOpen - 0.15);

  // simple blink effect reduces pupil size (not fully implemented); can expand later
  drawAvatar();
  requestAnimationFrame(avatarTick);
}

// let avatar eyes follow mouse over the canvas region
avatarCanvas.addEventListener('mousemove', (e)=>{
  const rect = avatarCanvas.getBoundingClientRect();
  avatar.eyeX = e.clientX - rect.left;
  avatar.eyeY = e.clientY - rect.top;
});

// start avatar loop
avatarTick();

// ------------------------------
// Webcam + simple face detection
// We dynamically load face-api from CDN (lite usage)
// ------------------------------
async function loadFaceApi(){
  if(window.faceapi) { state.faceApiLoaded = true; return; }
  await new Promise((res,rej)=>{
    const s = document.createElement('script');
    s.src = 'https://cdn.jsdelivr.net/npm/@vladmandic/face-api@1.0.2/dist/face-api.min.js';
    s.onload = res; s.onerror = rej; document.head.appendChild(s);
  });
  // load small models from CDN — production: host your own models
  const MODEL_URL = 'https://models.s3.amazonaws.com/face-api/';
  try{
    await faceapi.nets.tinyFaceDetector.loadFromUri(MODEL_URL);
    await faceapi.nets.faceLandmark68TinyNet.loadFromUri(MODEL_URL);
    state.faceApiLoaded = true;
  }catch(e){
    console.warn('face-api model load failed', e);
    state.faceApiLoaded = false;
  }
}

async function startWebcam(){
  try{
    const stream = await navigator.mediaDevices.getUserMedia({video: {width: 640, height: 480}, audio: false});
    webcamEl.srcObject = stream;
    webcamEl.onloadedmetadata = ()=> webcamEl.play();
    // start simple detector loop
    monitorFaceLoop();
  }catch(e){
    console.error('webcam error', e);
    postureStatus.textContent = 'Camera access denied';
  }
}

async function monitorFaceLoop(){
  if(!state.faceApiLoaded){ eyeStatus.textContent = 'Eye contact: face-api not ready'; setTimeout(monitorFaceLoop,1000); return; }
  if(!webcamEl || webcamEl.readyState < 2){ setTimeout(monitorFaceLoop,500); return; }

  const options = new faceapi.TinyFaceDetectorOptions({inputSize: 224, scoreThreshold: 0.5});
  try{
    const res = await faceapi.detectSingleFace(webcamEl, options).withFaceLandmarks(true);
    if(res && res.landmarks){
      // simple heuristic: eyes midpoint relative to face center
      const lm = res.landmarks;
      const leftEye = lm.getLeftEye();
      const rightEye = lm.getRightEye();
      const eyeCenterX = (leftEye[0].x + rightEye[3].x)/2;
      const faceBox = res.detection.box;
      const faceCenterX = faceBox.x + faceBox.width/2;
      const dx = Math.abs(eyeCenterX - faceCenterX);
      const ratio = dx/faceBox.width;
      if(ratio < 0.12) { eyeStatus.innerHTML = 'Eye contact: <span class="font-medium text-green-300">Good</span>'; }
      else { eyeStatus.innerHTML = 'Eye contact: <span class="font-medium text-yellow-300">Avoiding</span>'; }

      // posture: check head tilt by comparing eye y-levels
      const eyeDy = Math.abs(leftEye[0].y - rightEye[3].y);
      postureStatus.innerHTML = 'Posture: <span class="font-medium">' + (eyeDy < 8 ? 'Straight' : 'Tilted') + '</span>';

      // move avatar eyes to face center for fun
      const rect = webcamEl.getBoundingClientRect();
      avatar.eyeX = avatarCanvas.width/2 + (eyeCenterX - faceCenterX)/2;
      avatar.eyeY = avatarCanvas.height/2 - (res.detection.box.y - 50)/10;
    } else {
      eyeStatus.innerHTML = 'Eye contact: <span class="font-medium text-red-300">No face</span>';
      postureStatus.innerHTML = 'Posture: <span class="font-medium">Unknown</span>';
    }
  }catch(e){ console.warn('face detect err', e); }
  setTimeout(monitorFaceLoop, 400);
}

// ------------------------------
// Speech: recognition + synthesis
// ------------------------------
let recognizer = null;
function startRecognition(){
  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  if(!SpeechRecognition) { alert('SpeechRecognition not supported in this browser.'); return; }
  recognizer = new SpeechRecognition();
  recognizer.lang = 'en-US';
  recognizer.interimResults = false;
  recognizer.maxAlternatives = 1;
  recognizer.onresult = (e)=>{
    const t = e.results[0][0].transcript;
    appendUserMessage(t);
    processUserInput(t);
  };
  recognizer.onerror = (e)=>{ console.warn('recog err', e); };
  recognizer.start();
}

function speak(text){
  if(!window.speechSynthesis) return;
  state.speaking = true;
  const utt = new SpeechSynthesisUtterance(text);
  utt.lang = 'en-US';
  utt.rate = 1.0;
  utt.onend = ()=>{ state.speaking = false; };
  window.speechSynthesis.speak(utt);
  appendBotMessage(text);
}

// ------------------------------
// Simple conversation flow
// ------------------------------
function appendBotMessage(text){
  const el = document.createElement('div');
  el.className = 'bot-msg p-3 bg-gray-800/50 rounded-lg text-sm';
  el.innerText = text;
  messagesEl.appendChild(el);
  messagesEl.scrollTop = messagesEl.scrollHeight;
}
function appendUserMessage(text){
  const el = document.createElement('div');
  el.className = 'user-msg self-end p-3 bg-green-600/20 rounded-lg text-sm';
  el.innerText = text;
  messagesEl.appendChild(el);
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function askNextQuestion(){
  state.currentQ += 1;
  if(state.currentQ >= state.questions.length){
    finalizeSession();
    return;
  }
  const q = state.questions[state.currentQ].q;
  state.total += 1;
  speak('Question ' + (state.currentQ+1) + '. ' + q);
}

function processUserInput(text){
  // if we're at the start, interpret company/role picks
  if(!state.sessionId){
    state.company = companyInput.value || 'Unknown Company';
    state.jobType = jobTypeSelect.value || 'auto';
    state.role = roleInput.value || null;
    // choose role if auto
    if(state.jobType === 'auto' && !state.role){
      // ask clarifying question
      state.sessionId = 's_' + Date.now();
      speak('Hi, I will help you prepare. Are you aiming for a technical or non-technical role?');
      return;
    }
    startSession();
    return;
  }

  // If session active and there are questions, grade the answer
  if(state.questions.length && state.currentQ >=0 && state.currentQ < state.questions.length){
    const current = state.questions[state.currentQ];
    const score = gradeAnswer(current, text);
    state.score += score;
    // add feedback
    const fb = createFeedback(current, text, score);
    state.feedback.push(fb);

    // update UI
    scoreEl.innerText = Math.round((state.score/state.total)*100) || 0;
    fluencyEl.innerText = Math.round(Math.max(60, 80 - (text.split(' ').length < 4 ? 10 : 0))) + '%';
    const li = document.createElement('li'); li.innerText = fb; feedbackList.appendChild(li);

    // ask next question
    askNextQuestion();
  } else {
    // no session -> small talk
    speak('I can run a mock interview. Click Start Interview and provide company and role.');
  }
}

function gradeAnswer(qObj, text){
  const lowered = text.toLowerCase();
  let hits = 0;
  for(const k of qObj.keywords){ if(lowered.includes(k)) hits++; }
  // normalize
  const score = Math.min(1, hits / Math.max(1, qObj.keywords.length))*1;
  return score; // 0..1
}

function createFeedback(qObj, text, score){
  if(score > 0.6) return `Good — you covered key points for: "${qObj.q}"`;
  if(score > 0.2) return `Partial — try to mention more specifics for: "${qObj.q}"`;
  return `Missed — include basics like ${qObj.keywords.slice(0,2).join(', ')} when answering: "${qObj.q}"`;
}

function startSession(){
  state.sessionId = 's_' + Date.now();
  // choose questions
  if(state.role){
    // pick bank
    const bucket = (state.jobType === 'technical') ? QUESTION_BANK.technical : QUESTION_BANK.nontechnical;
    const list = (bucket[state.role] || bucket[Object.keys(bucket)[0]] || []).slice(0);
    state.questions = list;
    state.currentQ = -1;
    speak(`Starting mock interview for ${state.role || 'General role'} at ${state.company}. I will ask ${state.questions.length} questions.`);
    setTimeout(()=> askNextQuestion(), 800);
  } else {
    // ask clarifying question
    speak('Which role? e.g. Data Analyst, Software Engineer, HR. Type the role and press Send.');
  }
}

function finalizeSession(){
  speak('This concludes the mock interview. I will prepare a short report now.');
  const finalScore = Math.round((state.score/state.total)*100) || 0;
  scoreEl.innerText = finalScore;
  appendBotMessage('Final Score: ' + finalScore + '\nThank you for practicing!');
}

// ------------------------------
// Buttons and bindings
// ------------------------------
sendBtn.addEventListener('click', ()=>{
  const t = userInput.value.trim(); if(!t) return; userInput.value=''; appendUserMessage(t); processUserInput(t);
});
micBtn.addEventListener('click', ()=>{ startRecognition(); });
startInterviewBtn.addEventListener('click', ()=>{
  // reset
  state.sessionId = null; state.questions = []; state.currentQ = -1; state.score = 0; state.total = 0; state.feedback = [];
  // prefill from controls
  state.company = companyInput.value || 'Unknown';
  state.jobType = jobTypeSelect.value || 'auto';
  state.role = roleInput.value || null;
  appendBotMessage('Preparing interview session...');
  // if role provided, start
  if(state.role) startSession(); else { speak('Please type the role you want to practice for, e.g. Data Analyst, then press Send.'); }
});

calibrateBtn.addEventListener('click', ()=>{ speak('Calibrating camera. Please look into the camera for two seconds.'); });

downloadReportBtn.addEventListener('click', ()=>{
  const report = {sessionId: state.sessionId, company: state.company, role: state.role, score: scoreEl.innerText, feedback: state.feedback, date: new Date().toISOString()};
  const blob = new Blob([JSON.stringify(report, null, 2)], {type:'application/json'});
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a'); a.href = url; a.download = `shiksha-report-${Date.now()}.json`; document.body.appendChild(a); a.click(); a.remove();
});

// ------------------------------
// Init
// ------------------------------
(async function init(){
  appendBotMessage('Welcome to Shiksha AI — the virtual interview coach. Configure the company & role on the left, then click Start Interview.');
  // start webcam and load face-api in background
  startWebcam();
  loadFaceApi();
})();
