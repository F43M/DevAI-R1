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

function displayAIResponseFormatted(data){
  const panel=document.getElementById('aiOutput');
  panel.innerHTML='';
  try{
    const parsed=typeof data==='string'?JSON.parse(data):data;
    const structured=renderStructuredResponse(parsed);
    if(structured) panel.appendChild(structured); else panel.appendChild(renderFallbackJSON(parsed));
  }catch(e){
    panel.appendChild(renderFallbackJSON(data));
  }
}

const SESSION_KEY='devai_history';

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
    aiOutput:document.getElementById('aiOutput').innerHTML,
    diffOutput:document.getElementById('diffOutput').innerHTML,
    ts:Date.now()
  });
}

function clearSession(){
  localStorage.removeItem(SESSION_KEY);
  document.getElementById('console').textContent='';
  document.getElementById('aiOutput').innerHTML='';
  document.getElementById('diffOutput').innerHTML='';
  appendConsole('🧹 Sessão apagada com sucesso.');
}

window.addEventListener('load',()=>{
  const data=loadSession();
  if(data){
    document.getElementById('console').textContent=data.console||'';
    document.getElementById('aiOutput').innerHTML=data.aiOutput||'';
    document.getElementById('diffOutput').innerHTML=data.diffOutput||'';
    appendConsole('🔄 Sessão recuperada – continue de onde parou.');
  }
});
