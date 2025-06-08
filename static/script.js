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
