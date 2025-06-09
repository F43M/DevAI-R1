function showHelpOverlay(){
  document.getElementById('help-overlay').classList.remove('hidden');
}
function hideHelpOverlay(){
  document.getElementById('help-overlay').classList.add('hidden');
}

function requireFile(){
  if(!window.currentFile){
    alert('⚠️ Para usar esta função, abra um arquivo primeiro no editor.');
    return false;
  }
  return true;
}

function showLoading(){
  const indicator=document.getElementById('loading-indicator');
  if(indicator) indicator.style.display='block';
  try{
    document.querySelectorAll('button').forEach(btn=>btn.disabled=true);
  }catch(e){
    if(indicator) indicator.textContent='⏳ Processando…';
  }
}

function hideLoading(){
  const indicator=document.getElementById('loading-indicator');
  if(indicator) indicator.style.display='none';
  try{
    document.querySelectorAll('button').forEach(btn=>btn.disabled=false);
  }catch(e){}
}

function showStatus(msg){
  const el=document.getElementById('statusMessage');
  if(!el) return;
  el.textContent=msg;
  setTimeout(()=>{ if(el.textContent===msg) el.textContent=''; },4000);
}

function showApprovalDialog(msg){
  return new Promise(resolve=>{
    const overlay=document.createElement('div');
    overlay.className='modal';
    overlay.innerHTML='<div class="modal-content"><p>'+msg+'</p><button id="apYes">Sim</button> <button id="apNo">Não</button></div>';
    document.body.appendChild(overlay);
    overlay.querySelector('#apYes').onclick=()=>{ cleanup(); resolve(true); };
    overlay.querySelector('#apNo').onclick=()=>{ cleanup(); resolve(false); };
    function cleanup(){ document.body.removeChild(overlay); }
  });
}

async function listenApprovalRequests(){
  while(true){
    try{
      const r=await fetch('/approval_request');
      const data=await r.json();
      if(data.message){
        const ok=await showApprovalDialog(data.message);
        await fetch('/approval_request',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({approved:ok})});
      }
    }catch(e){await new Promise(res=>setTimeout(res,1000));}
  }
}

function toggleReasoning(){
  const el=document.getElementById('reasoningOutput');
  if(!el) return;
  el.style.display=el.style.display==='none'?'block':'none';
}

async function toggleContext(){
  const panel=document.getElementById('contextPanel');
  if(!panel) return;
  if(panel.style.display==='none'){
    panel.textContent='⌛ carregando...';
    panel.style.display='block';
    try{
      const r=await fetch('/session/context');
      const data=await r.json();
      if(data.error){panel.textContent=data.error;return;}
      const list=p=>p.map(i=>'<li>'+i+'</li>').join('');
      panel.innerHTML='<h4>🗣️ Últimas mensagens</h4><ul>'+list(data.history_preview)+'</ul>'+
        '<h4>📌 Memórias Simbólicas Ativadas</h4><ul>'+list(data.symbolic_memories)+'</ul>'+
        '<h4>🔍 Blocos técnicos usados</h4><ul>'+list(data.logs_or_code)+'</ul>'+
        (data.warnings.length?'<h4>⚠️ Avisos</h4><p>'+data.warnings.join('<br>')+'</p>':'');
    }catch(e){panel.textContent='Erro ao carregar contexto';}
  }else{
    panel.style.display='none';
  }
}

function renderStructuredResponse(data){
  const container=document.createElement('div');
  const addSection=(title,list,cls)=>{
    if(!list.length) return;
    const sec=document.createElement('div');
    sec.classList.add('result-section');
    const h=document.createElement('h4');
    if(cls) h.classList.add(cls);
    h.textContent=title;
    sec.appendChild(h);
    const ul=document.createElement('ul');
    list.forEach(i=>{const li=document.createElement('li');li.textContent=i;ul.appendChild(li);});
    sec.appendChild(ul);
    container.appendChild(sec);
  };
  if(data.summary){
    const s=document.createElement('div');
    s.classList.add('result-section','result-success');
    s.textContent='✅ '+data.summary;
    container.appendChild(s);
  }
  const risks=[];
  if(Array.isArray(data.risks)) risks.push(...data.risks);
  if(data.findings&&Array.isArray(data.findings.risks)) risks.push(...data.findings.risks);
  if(Array.isArray(data.details)) data.details.forEach(d=>{if(d.type==='risk'&&d.msg) risks.push(d.msg);});
  addSection('Riscos:',risks,'result-danger');
  const suggestions=[];
  if(Array.isArray(data.suggestions)) suggestions.push(...data.suggestions);
  if(data.findings&&Array.isArray(data.findings.refactor)) suggestions.push(...data.findings.refactor);
  if(Array.isArray(data.details)) data.details.forEach(d=>{if(d.type==='suggestion'&&d.msg) suggestions.push(d.msg);});
  addSection('Sugestões:',suggestions,'result-warning');
  const todos=[];
  if(Array.isArray(data.todo)) todos.push(...data.todo); else if(data.todo) todos.push(data.todo);
  if(data.findings&&Array.isArray(data.findings.todo)) todos.push(...data.findings.todo);
  addSection('Tarefas pendentes:',todos,'');
  return container.children.length?container:null;
}

function renderFallbackJSON(data){
  const pre=document.createElement('pre');
  pre.textContent=typeof data==='string'?data:JSON.stringify(data,null,2);
  return pre;
}

function renderReport(text){
  const pre=document.createElement('pre');
  text.split('\n').forEach(line=>{
    const span=document.createElement('span');
    if(line.includes('🚩')||line.includes('❌')) span.classList.add('report-danger');
    else if(line.includes('⚠️')) span.classList.add('report-warning');
    else if(line.includes('✅')) span.classList.add('report-success');
    else if(line.includes('💡')) span.classList.add('report-info');
    span.textContent=line;
    pre.appendChild(span);
    pre.appendChild(document.createElement('br'));
  });
  return pre;
}

function displayAIResponseFormatted(data){
  const panel=document.getElementById('aiOutput');
  panel.innerHTML='';
  try{
    const parsed=typeof data==='string'?JSON.parse(data):data;
    if(parsed.report){
      panel.appendChild(renderReport(parsed.report));
    }
    const structured=renderStructuredResponse(parsed);
    if(structured) panel.appendChild(structured);
    else if(!parsed.report) panel.appendChild(renderFallbackJSON(parsed));
  }catch(e){
    panel.appendChild(renderFallbackJSON(data));
  }
}

const SESSION_KEY='devai_history';
const CHAT_KEY='chatHistory';
let showReasoningByDefault=false;
let showContextButton=false;

window.chatHistory=[];
let storageOK=true;

function saveChat(){
  try{localStorage.setItem(CHAT_KEY,JSON.stringify(window.chatHistory));}
  catch(e){storageOK=false;showHistoryWarning();}
}

function loadChat(){
  try{const d=JSON.parse(localStorage.getItem(CHAT_KEY));if(Array.isArray(d)) window.chatHistory=d;}
  catch(e){storageOK=false;}
}

function renderChatHistory(){
  const panel=document.getElementById('ia-panel');
  if(!panel) return;
  panel.innerHTML='';
  window.chatHistory.forEach(msg=>{
    const bubble=document.createElement('div');
    bubble.className=msg.role==='user'?'user-msg':'ai-msg';
    bubble.textContent=msg.content;
    panel.appendChild(bubble);
  });
}

function addChat(role,content){
  window.chatHistory.push({role,content});
  saveChat();
  renderChatHistory();
}

function clearChat(){
  window.chatHistory=[];
  try{localStorage.removeItem(CHAT_KEY);}catch(e){}
  renderChatHistory();
}

function showHistoryWarning(){
  const el=document.getElementById('history-warning');
  if(el){
    el.textContent='⚠️ Histórico de conversa será perdido ao recarregar.';
    el.classList.remove('hidden');
  }
}

async function syncChatFromBackend(){
  try{
    const r=await fetch('/history');
    const hist=await r.json();
    if(Array.isArray(hist)){
      window.chatHistory=hist;
      saveChat();
      renderChatHistory();
    }
  }catch(e){}
}

function saveSession(obj){
  try{localStorage.setItem(SESSION_KEY,JSON.stringify(obj));}catch(e){}
}

function loadSession(){
  try{return JSON.parse(localStorage.getItem(SESSION_KEY))||null;}catch(e){return null;}
}

function persistUI(){
  const consoleLines=document.getElementById('console').textContent.split('\n').slice(-20).join('\n');
  saveSession({
    console:consoleLines,
    planOutput:document.getElementById('planOutput')?document.getElementById('planOutput').innerHTML:'',
    aiOutput:document.getElementById('aiOutput').innerHTML,
    reasoning:document.getElementById('reasoningOutput').innerHTML,
    diffOutput:document.getElementById('diffOutput').innerHTML,
    ts:Date.now()
  });
}

function clearUIConversation(){
  localStorage.removeItem(SESSION_KEY);
  clearChat();
  document.getElementById('console').textContent='';
  const plan=document.getElementById('planOutput');
  if(plan) plan.innerHTML='';
  document.getElementById('aiOutput').innerHTML='';
  document.getElementById('reasoningOutput').innerHTML='';
  document.getElementById('diffOutput').innerHTML='';
}

function addSystemMessage(msg){
  addChat('system', msg);
}

async function resetSession(){
  await fetch('/session/reset?session_id=default', {method:'POST'});
  clearUIConversation();
  try{localStorage.removeItem(CHAT_KEY);}catch(e){}
  addSystemMessage('✅ Sessão reiniciada com sucesso.');
}

window.addEventListener('load',async()=>{
  loadChat();
  renderChatHistory();
  listenApprovalRequests();
  const data=loadSession();
  if(data){
    document.getElementById('console').textContent=data.console||'';
    if('planOutput' in data) document.getElementById('planOutput').innerHTML=data.planOutput||'';
    document.getElementById('aiOutput').innerHTML=data.aiOutput||'';
    document.getElementById('reasoningOutput').innerHTML=data.reasoning||'';
    document.getElementById('diffOutput').innerHTML=data.diffOutput||'';
    appendConsole('🔄 Sessão recuperada – continue de onde parou.');
  }
  try{
    const r=await fetch('/status');
    const info=await r.json();
    if(info.api_key_missing){
      document.getElementById('aiOutput').textContent='🚫 Nenhuma chave de API foi detectada. Configure OPENROUTER_API_KEY para habilitar a IA.';
    }
    if('show_reasoning_default' in info) showReasoningByDefault=info.show_reasoning_default;
    if('show_context_button' in info) showContextButton=info.show_context_button;
  }catch(e){}
  syncChatFromBackend();
  const btn=document.getElementById('clearHistoryBtn');
  if(btn) btn.onclick=clearChat;
  const reset=document.getElementById('reset-session-btn');
  if(reset) reset.onclick=resetSession;
  const ctx=document.getElementById('showContextBtn');
  if(ctx) ctx.style.display=showContextButton?'block':'none';
  if(!storageOK) showHistoryWarning();
});
