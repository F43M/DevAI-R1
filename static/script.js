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
