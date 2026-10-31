const P = window.PROJECT_CONFIG;
const $ = (selector) => document.querySelector(selector);
const fmt = (s) => String(s).replaceAll('_', ' ').replace(/\b\w/g, c => c.toUpperCase());
const escapeHtml = (s) => String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const HISTORY_KEY = `model-field-history-${P.id}`;
let latestPrediction = null;
let toastTimer;

function getFeatures(){
  return Object.fromEntries(P.features.map(name => [name, Number(document.querySelector(`[data-feature="${name}"]`).value)]));
}
function notify(message, error=false){
  const el=$('#toast'); el.textContent=message; el.classList.toggle('error',error); el.classList.add('show'); clearTimeout(toastTimer); toastTimer=setTimeout(()=>el.classList.remove('show'),2400);
}
function number(value){return Number(value).toLocaleString(undefined,{maximumFractionDigits:4});}
function setFeature(name,value){
  const numeric=Math.min(4,Math.max(-4,Number(value)||0));
  const input=document.querySelector(`[data-feature="${name}"]`); const range=document.querySelector(`[data-range="${name}"]`);
  if(input)input.value=numeric.toFixed(2); if(range)range.value=numeric;
}
function renderInputs(){
  $('#featureList').innerHTML=P.features.map((name,index)=>`<div class="feature-control"><label for="f-${index}">${escapeHtml(fmt(name))}<span>F${String(index+1).padStart(2,'0')}</span></label><input id="f-${index}" data-feature="${escapeHtml(name)}" type="number" min="-4" max="4" step="0.05" value="0.00" /><input aria-label="Range for ${escapeHtml(fmt(name))}" data-range="${escapeHtml(name)}" type="range" min="-4" max="4" step="0.05" value="0" /></div>`).join('');
  P.features.forEach(name=>{
    document.querySelector(`[data-feature="${name}"]`).addEventListener('input',e=>{document.querySelector(`[data-range="${name}"]`).value=e.target.value||0;});
    document.querySelector(`[data-range="${name}"]`).addEventListener('input',e=>{document.querySelector(`[data-feature="${name}"]`).value=Number(e.target.value).toFixed(2);});
  });
}
function renderResult(result){
  latestPrediction=result; $('#resultEmpty').classList.add('hidden'); const el=$('#resultContent'); el.classList.remove('hidden');
  const classification=result.task_type==='classification';
  const probabilityValues=Object.entries(result.probabilities||{});
  const confidence=typeof result.confidence==='number'?result.confidence:null;
  el.innerHTML=`<div class="result-target">${classification?'PREDICTED LABEL':'ESTIMATED TARGET'} · ${escapeHtml(result.target)}</div><div class="result-value">${escapeHtml(String(result.prediction))}</div>${confidence===null?'':`<div class="confidence-row"><span>Top class confidence</span><b>${(confidence*100).toFixed(1)}%</b></div><div class="bar-track"><div class="bar-fill" style="width:${Math.max(2,confidence*100)}%"></div></div>`}${probabilityValues.length?`<div class="class-bars">${probabilityValues.map(([label,value])=>`<div class="class-line"><span>Class ${escapeHtml(label)}</span><div class="bar-track"><div class="bar-fill" style="width:${Math.max(1,value*100)}%"></div></div><b>${(value*100).toFixed(1)}%</b></div>`).join('')}</div>`:''}<div class="confidence-row"><span>Algorithm</span><b>${escapeHtml(result.model)}</b></div>`;
  saveHistory(result);
}
async function api(path,payload){
  const response=await fetch(path,{method:payload?'POST':'GET',headers:payload?{'Content-Type':'application/json'}:{},body:payload?JSON.stringify(payload):undefined});
  const body=await response.json().catch(()=>({detail:'Unexpected response'}));
  if(!response.ok)throw new Error(body.detail||`Request failed (${response.status})`); return body;
}
async function predict(){
  const button=document.querySelector('#predictionForm button[type="submit"]');button.disabled=true;button.querySelector('span').textContent='Running…';
  try{const result=await api('/api/predict',{features:getFeatures()});renderResult(result);notify('Prediction completed by the saved model.');return result;}
  catch(error){notify(error.message,true);throw error;}
  finally{button.disabled=false;button.querySelector('span').textContent='Run model';}
}
function saveHistory(result){
  const items=JSON.parse(localStorage.getItem(HISTORY_KEY)||'[]');items.unshift({time:new Date().toISOString(),prediction:result.prediction,features:getFeatures()});localStorage.setItem(HISTORY_KEY,JSON.stringify(items.slice(0,20)));
}
function renderMetricList(){
  const training=P.model_training||{};const items=Object.entries(training.metrics||{});
  $('#metrics').innerHTML=items.length?items.map(([key,value])=>`<div class="metric-pill"><span>${escapeHtml(key.replaceAll('_',' '))}</span><b>${number(value)}</b></div>`).join(''):`<div class="metric-pill"><span>FIT ROWS</span><b>${number(training.fit_rows||0)}</b></div>`;
  $('#modelSummary').textContent=`${P.model_class} · ${number(training.fit_rows||0)} fit rows · ${P.gui_feature.mode.replaceAll('-',' ')} workflow`;
}
function renderWorkflow(){
  const mode=P.gui_feature.mode;const area=$('#workflowTool');
  if(mode==='scenario'||mode==='range'){
    const focus=P.gui_feature.focus_feature;
    area.innerHTML=`<div class="tool-row"><button class="small-button" id="scenarioButton">${mode==='range'?'Shift the target signal':'Compare a nearby scenario'}</button><label class="tool-control">Change ${escapeHtml(fmt(focus))}<input id="scenarioStep" type="range" min="-2" max="2" step="0.1" value="0.8"></label><span class="tool-control" id="scenarioValue">+0.8</span></div><div class="tool-output" id="scenarioOutput">Select a scenario shift to compare predictions from this saved model.</div>`;
    $('#scenarioStep').addEventListener('input',e=>$('#scenarioValue').textContent=Number(e.target.value)>=0?`+${e.target.value}`:e.target.value);
    $('#scenarioButton').addEventListener('click',async()=>{try{const base=getFeatures(),variant={...base};variant[focus]+=Number($('#scenarioStep').value);const data=await api('/api/compare',{base,scenarios:[{label:`${fmt(focus)} shift`,features:variant}]});$('#scenarioOutput').textContent=`Baseline ${number(data.results[0].prediction)} → scenario ${number(data.results[1].prediction)} (same trained model)`;}catch(e){notify(e.message,true);}});
  } else if(mode==='threshold'){
    area.innerHTML=`<div class="tool-row"><label class="tool-control">Alert threshold <input id="threshold" type="range" min="0.1" max="0.9" step="0.01" value="0.5"></label><b class="tool-control" id="thresholdValue">50%</b><button class="small-button" id="thresholdButton">Apply to last prediction</button></div><div class="tool-output" id="thresholdOutput">Run a prediction first. The threshold changes the displayed decision only; the model probabilities stay unchanged.</div>`;
    $('#threshold').addEventListener('input',e=>$('#thresholdValue').textContent=`${Math.round(Number(e.target.value)*100)}%`);
    $('#thresholdButton').addEventListener('click',()=>{if(!latestPrediction||!Object.keys(latestPrediction.probabilities||{}).length){notify('Run a classification prediction first.',true);return;}const entries=Object.entries(latestPrediction.probabilities);const positive=entries[entries.length-1];const threshold=Number($('#threshold').value);$('#thresholdOutput').textContent=`Decision at ${Math.round(threshold*100)}%: ${Number(positive[1])>=threshold?'flagged':'not flagged'} · class ${positive[0]} probability ${(Number(positive[1])*100).toFixed(1)}%.`;});
  } else if(mode==='batch'){
    const example=[P.features.join(','),P.features.map(()=>0).join(','),P.features.map((_,i)=>i%2?0.6:-0.35).join(',')].join('\n');
    area.innerHTML=`<div class="tool-row"><textarea class="tool-input" id="batchCsv" rows="3" aria-label="CSV rows">${escapeHtml(example)}</textarea><button class="small-button" id="batchButton">Score CSV</button></div><div class="tool-output" id="batchOutput">Header row should use these ${P.features.length} feature names. Limit: 100 rows per request.</div>`;
    $('#batchButton').addEventListener('click',async()=>{try{const lines=$('#batchCsv').value.trim().split(/\r?\n/).filter(Boolean).map(line=>line.split(',').map(v=>v.trim()));const headers=lines.shift().map(h=>h.replace(/^"|"$/g,''));const rows=lines.map(line=>Object.fromEntries(headers.map((h,i)=>[h,Number(line[i])])));const res=await api('/api/batch',{rows});$('#batchOutput').textContent=`${res.count} rows scored. First outputs: ${res.results.slice(0,5).map(r=>number(r.prediction)).join(' · ')}`;}catch(e){notify(e.message,true);}});
  } else if(mode==='explain'||mode==='sensitivity'){
    area.innerHTML=`<div class="tool-row"><button class="small-button" id="explainButton">${mode==='explain'?'Explain this input':'Build sensitivity map'}</button><span class="tool-control">Local ±0.5 feature perturbation</span></div><div class="tool-output" id="explainOutput">Run a local sensitivity analysis for the current input. This is a perturbation view, not a causal explanation.</div>`;
    $('#explainButton').addEventListener('click',async()=>{try{const res=await api('/api/explain',{features:getFeatures(),step:0.5});$('#explainOutput').innerHTML=res.drivers.slice(0,5).map(d=>`<div>${escapeHtml(fmt(d.feature))} · ${escapeHtml(d.direction)} · relative effect ${(d.relative_effect*100).toFixed(0)}%</div>`).join('');}catch(e){notify(e.message,true);}});
  } else if(mode==='history'){
    area.innerHTML=`<div class="tool-row"><button class="small-button" id="historyButton">Export prediction history</button><button class="small-button" id="clearHistoryButton">Clear this project’s history</button></div><div class="tool-output" id="historyOutput">History is kept in this browser only and is not uploaded to a database.</div>`;
    $('#historyButton').addEventListener('click',()=>{const data=localStorage.getItem(HISTORY_KEY)||'[]';const blob=new Blob([data],{type:'application/json'});const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download=`project-${P.id}-history.json`;a.click();URL.revokeObjectURL(a.href);});
    $('#clearHistoryButton').addEventListener('click',()=>{localStorage.removeItem(HISTORY_KEY);$('#historyOutput').textContent='Local history cleared.';});
  } else if(mode==='quality'){
    area.innerHTML=`<div class="tool-row"><button class="small-button" id="qualityButton">Check current signals</button><span class="tool-control">Expected demo range: −4 to +4</span></div><div class="tool-output" id="qualityOutput">Check the current feature vector before running inference.</div>`;
    $('#qualityButton').addEventListener('click',()=>{const values=getFeatures();const bad=Object.entries(values).filter(([,v])=>!Number.isFinite(v)||v < -4||v > 4);$('#qualityOutput').textContent=bad.length?`${bad.length} field(s) outside the demo range: ${bad.map(([k])=>fmt(k)).join(', ')}`:`All ${P.features.length} signals are numeric and inside the synthetic demo range.`;});
  } else if(mode==='presets'){
    area.innerHTML=`<div class="tool-row"><button class="small-button" data-preset="-1.2">Load low-signal profile</button><button class="small-button" data-preset="0">Load neutral profile</button><button class="small-button" data-preset="1.2">Load elevated profile</button><span class="tool-control">Presets are synthetic test vectors.</span></div>`;
    area.querySelectorAll('[data-preset]').forEach(btn=>btn.addEventListener('click',()=>{const value=Number(btn.dataset.preset);P.features.forEach((f,i)=>setFeature(f,value*(i%2?-0.6:1)));notify('Synthetic profile loaded.');}));
  } else {
    area.innerHTML=`<div class="tool-row"><button class="small-button" id="scorecardButton">Refresh model scorecard</button><span class="tool-control">Training and holdout metrics for this project’s fitted artifact</span></div><div class="tool-output" id="scorecardOutput">${escapeHtml(P.model_class)} · ${number((P.model_training||{}).fit_rows||0)} fit rows · trained on synthetic data.</div>`;
    $('#scorecardButton').addEventListener('click',async()=>{try{const m=await api('/api/model');$('#scorecardOutput').textContent=`${m.algorithm} · ${number(m.training.fit_rows)} fit rows · ${Object.entries(m.training.metrics||{}).map(([k,v])=>`${k} ${number(v)}`).join(' · ')}`;}catch(e){notify(e.message,true);}});
  }
}
function init(){
  renderInputs();renderMetricList();renderWorkflow();
  $('#predictionForm').addEventListener('submit',e=>{e.preventDefault();predict().catch(()=>{});});
  $('#resetButton').addEventListener('click',()=>{P.features.forEach(f=>setFeature(f,0));$('#qualityHint').textContent='Inputs reset to synthetic baseline.';});
  $('#sampleButton').addEventListener('click',()=>{P.features.forEach((f,i)=>setFeature(f,((i*7+Number(P.id))%9-4)*0.22));$('#qualityHint').textContent='Project-specific synthetic sample loaded.';});
  $('#qualityHint').textContent=`${P.features.length} project-specific signals · synthetic demo scale`;
}
init();
