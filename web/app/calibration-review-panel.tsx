"use client";
import { useEffect, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import ExtractionInspector from './extraction-inspector';

type XY = [number, number];
type Result = Record<string, any>;
type Props = {
  group: string; scope: string; version: string; matrix: number[] | null; disabled: boolean; activeId?: string | null;
  getRequest: () => Result;
  reviewBasis?: string;
  annotationStamps?: Record<string,string>;
  current: Result | null;
  onPreview: (id: string) => void;
  onHighlight: (value: {id: string; indices: number[]; color: number} | null) => void;
  onStatuses: (statuses: Record<string,string>) => void;
  onApply: (matrix: number[]) => void;
  onStructurePreview?: (item:any) => void;
  onStructureLine?: (line:number[][]|null) => void;
  extractionSceneKey?: string;
  getSceneFrames?: ()=>Result[];
  onExtractionView?: (item:Result|null)=>void;
};

export async function run(action: string, payload: Result, signal: AbortSignal, progress: (s:string)=>void) {
  const response = await fetch(`/api/calibration/review/${action}`, {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(payload), signal});
  if (!response.ok || !response.body) throw new Error(`请求失败：${response.status}`);
  const reader=response.body.getReader(), decoder=new TextDecoder(); let buffer=''; let result: Result | undefined;
  const consume=(line:string)=>{
    if (!line.trim()) return;
    const event=JSON.parse(line);
    if (event.type==='error') throw new Error(event.message);
    if (event.type==='progress') progress(`${event.percent}% · ${event.message}`);
    if (event.type==='result') result=event.result;
  };
  while(true) {const {done,value}=await reader.read(); if(done) break; buffer+=decoder.decode(value,{stream:true}); const lines=buffer.split('\n');buffer=lines.pop()!;lines.forEach(consume);}
  consume(buffer+decoder.decode()); if(!result) throw new Error('服务未返回结果'); return result;
}

export default function CalibrationReviewPanel(props: Props) {
  const host=useRef<HTMLElement|null>(null);
  const [toolbar,setToolbar]=useState<HTMLElement|null>(null);
  const [mainImage,setMainImage]=useState<HTMLElement|null>(null),[pairToolbar,setPairToolbar]=useState<HTMLElement|null>(null);
  const [editingId,setEditingId]=useState<string|null>(null),[editingImage,setEditingImage]=useState(false);
  const [matchStats,setMatchStats]=useState<Result|null>(null);
  const [inspectionActive,setInspectionActive]=useState(false);
  const pendingStructure=useRef<Result|null>(null);
  useEffect(()=>{setToolbar(host.current?.closest('.calibration-stage')?.querySelector<HTMLElement>('.calibration-workflow-actions')??null);},[props.scope]);
  const [open,setOpen]=useState(false), [busy,setBusy]=useState(false), [message,setMessage]=useState('按需运行；不会修改标注或当前外参。');
  const [report,setReport]=useState<Result|null>(null), [trial,setTrial]=useState<Result|null>(null), [selected,setSelected]=useState<string>('');
  const [reportStamps,setReportStamps]=useState<Record<string,string>>({});
  const stampsKey=JSON.stringify(props.annotationStamps??{});
  const [tab,setTab]=useState<'review'|'structure'>('review');
  const [structures,setStructures]=useState<Result[]>([]), [storeRevision,setStoreRevision]=useState(0), [storeReady,setStoreReady]=useState(false);
  const [mode,setMode]=useState('planes'),[a,setA]=useState<number[]>([]),[b,setB]=useState<number[]>([]);
  const [roi,setRoi]=useState<XY[]>([]),[imageLine,setImageLine]=useState<XY[]>([]),[lines,setLines]=useState<Result[]>([]),[fit,setFit]=useState<Result|null>(null);
  const [imageTool,setImageTool]=useState<'roi'|'line'>('roi'),[confirmed,setConfirmed]=useState(false);
  const [lineCurve,setLineCurve]=useState<XY[]>([]);
  const [highlightKind,setHighlightKind]=useState('outsideIndices');
  const controller=useRef<AbortController|null>(null), generation=useRef(0), currentProps=useRef(props);currentProps.current=props;
  const capturedRequest=useRef<Result|null>(null), capturedVersion=useRef('');
  const context=props.current ? `${props.current.datasetId}:${props.current.frame}:${props.current.cloudUrl}:${props.current.distortionModel}` : '';
  useEffect(()=>{const stage=host.current?.closest('.calibration-stage');setMainImage(stage?.querySelector<HTMLElement>('.unified-image-panel .image-canvas-shell')??null);setPairToolbar(stage?.querySelector<HTMLElement>('.embedded-pair-items')??null);},[context,open,tab]);
  useEffect(()=>{generation.current++;controller.current?.abort();setBusy(false);setReport(null);setTrial(null);setSelected('');props.onHighlight(null);props.onStatuses({});setMessage('诊断未运行或已过期，请重新复查。');},[props.reviewBasis,props.scope]);
  useEffect(()=>{generation.current++;controller.current?.abort();setBusy(false);setTrial(null);},[props.version]);
  useEffect(()=>{
    if(!report)return;
    const statuses:Record<string,string>={};
    for(const p of report.pairs){
      if(props.annotationStamps && !(p.annotationId in props.annotationStamps))continue;
      statuses[p.annotationId]=reportStamps[p.annotationId]!==props.annotationStamps?.[p.annotationId]?'已修改，待复查':p.status;
    }
    for(const id of Object.keys(props.annotationStamps??{}))if(!(id in statuses))statuses[id]='新增，待复查';
    props.onStatuses(statuses);
  },[report,reportStamps,stampsKey]);
  useEffect(()=>{generation.current++;controller.current?.abort();setBusy(false);setA([]);setB([]);setRoi([]);setImageLine([]);setLines([]);setFit(null);setConfirmed(false);},[context]);
  useEffect(()=>{const s=pendingStructure.current;if(!s||s.datasetId!==props.current?.datasetId||s.frame!==props.current?.frame)return;setMode(s.mode);setA(s.indicesA);setB(s.indicesB??[]);setImageLine(s.imageLine);setFit(s.fit);setEditingId(s.id);setConfirmed(false);pendingStructure.current=null;},[context,props.current]);
  useEffect(()=>{setConfirmed(false);setTrial(null);},[imageLine,fit]);
  useEffect(()=>{props.onStructureLine?.(open&&tab==='structure'?fit?.endpoints??null:null);},[fit,open,tab]);
  useEffect(()=>()=>props.onStructureLine?.(null),[]);
  useEffect(()=>{setTrial(null);},[structures]);
  useEffect(()=>{
    const item=report?.pairs.find((p:Result)=>p.annotationId===props.activeId);
    if(item&&selected!==item.annotationId){setSelected(item.annotationId);setHighlightKind('outsideIndices');props.onHighlight({id:item.annotationId,indices:item.outsideIndices,color:0xe45d56});}
  },[props.activeId,report]);
  useEffect(()=>{
    setLineCurve([]);if(imageLine.length!==2||!props.current)return;
    const abort=new AbortController();const timer=setTimeout(()=>{
      run('curve',{...props.current,imageLine},abort.signal,()=>{}).then(r=>setLineCurve(r.curve)).catch(()=>{});
    },300);
    return()=>{clearTimeout(timer);abort.abort();};
  },[imageLine,context]);
  useEffect(()=>{let alive=true;setStoreReady(false);setStructures([]);fetch(`/api/calibration/review/structures/${encodeURIComponent(props.group)}`).then(r=>{if(!r.ok)throw new Error('结构标注读取失败');return r.json();}).then(v=>{if(alive){setStructures(v.structures);setStoreRevision(v.revision);setStoreReady(true);}}).catch(e=>{if(alive)setMessage(e.message);});return()=>{alive=false;};},[props.group]);
  useEffect(()=>()=>controller.current?.abort(),[]);
  async function job(action:string, payload:Result, accept:(r:Result)=>void) {
    if(busy)return;const id=++generation.current;const abort=new AbortController();controller.current=abort;setBusy(true);
    try {const value=await run(action,payload,abort.signal,s=>{if(generation.current===id)setMessage(s);});if(generation.current===id){accept(value);setMessage('已完成；结果仅供复查，未自动修改标注或外参。');}}
    catch(e){if(generation.current===id)setMessage(e instanceof Error ? e.message : '操作失败');}
    finally{if(generation.current===id)setBusy(false);}
  }
  function requestSnapshot(){const request=props.getRequest();capturedRequest.current=JSON.parse(JSON.stringify(request));capturedVersion.current=props.version;return request;}
  function inspect(){if(!props.matrix)return;try{const request=requestSnapshot();const stamps={...props.annotationStamps};job('review',{request,matrix:props.matrix},r=>{setReport(r);setReportStamps(stamps);props.onStatuses(Object.fromEntries(r.pairs.map((p:Result)=>[p.annotationId,p.status])));});}catch(e){setMessage(String(e));}}
  function choose(item:Result,kind='outsideIndices') {setSelected(item.annotationId);setHighlightKind(kind);props.onPreview(item.annotationId);props.onHighlight({id:item.annotationId,indices:item[kind]??[],color:kind==='outsideIndices'?0xe45d56:kind==='isolatedIndices'?0xb85ab5:0xd6a13b});}
  function editStructure(s:Result){pendingStructure.current=s;setOpen(true);setTab('structure');setEditingImage(false);props.onStructurePreview?.(s);}
  function exclude(id:string){if(!props.matrix)return;try{const request=requestSnapshot();setTrial(null);job('influence',{request,matrix:props.matrix,excludeId:id},r=>setTrial({...r,kind:'influence',version:props.version}));}catch(e){setMessage(String(e));}}
  async function save(items:Result[]){
    if(!storeReady)return;const group=props.group;
    try {const r=await fetch(`/api/calibration/review/structures/${encodeURIComponent(group)}`,{method:'PUT',headers:{'Content-Type':'application/json'},body:JSON.stringify({revision:storeRevision,structures:items})});if(!r.ok)throw new Error(r.status===409?'其他页面更新了结构标注，请刷新后重试':'结构标注保存失败');const v=await r.json();if(currentProps.current.group===group){setStructures(v.structures);setStoreRevision(v.revision);setMessage('结构标注已单独保存；原物体标注不变。');}}
    catch(e){setMessage(String(e));}
  }
  function clickImage(event:React.MouseEvent<SVGSVGElement>){if(!props.current)return;const transform=event.currentTarget.getScreenCTM();if(!transform)return;const pixel=new DOMPoint(event.clientX,event.clientY).matrixTransform(transform.inverse());const p:XY=[pixel.x,pixel.y];if(p[0]<0||p[1]<0||p[0]>=props.current.width||p[1]>=props.current.height)return;if(imageTool==='roi')setRoi(v=>v.length===2?[p]:[...v,p]);else setImageLine(v=>v.length===2?[p]:[...v,p]);}
  const active=report?.pairs.find((p:Result)=>p.annotationId===selected);
  const activeVisiblePoints=props.activeId===selected && props.current
    ? new Set<number>(props.current.selectedIndices??[]) : null;
  const markedIndices=new Set<number>(active?.[highlightKind]??[]);
  const source=capturedRequest.current?.pairs.find((p:Result)=>p.annotation_id===selected);
  const matching=structures.filter(s=>s.scope===props.scope && s.confirmed && s.enabled!==false);
  const structuralPayload=props.current ? {...props.current,indicesA:a,indicesB:b,mode,imageLine,scope:props.scope,confirmed,extractionVersion:structures.find(s=>s.id===editingId)?.extractionVersion} : null;
  const trialCurrent=trial?.version===props.version && capturedVersion.current===props.version && (trial?.kind!=='joint'||trial.structureRevision===storeRevision);
  function jointSolve(){try{const request=requestSnapshot();job('joint',{matrix:props.matrix,request,structures:matching},r=>setTrial({...r,before:{...r.before,meanCost:r.before.objects.meanCost},after:{...r.after,meanCost:r.after.objects.meanCost},lineBeforePx:r.before.lineMeanPx,lineAfterPx:r.after.lineMeanPx,kind:'joint',version:props.version,structureRevision:storeRevision}));}catch(e){setMessage(String(e));}}
  const trigger=<button className="review-toggle" onClick={()=>setOpen(!open)} aria-expanded={open} aria-controls="calibration-review-drawer">配对复查{open?' ▴':' ▾'}</button>;
  return <section ref={host} className={`calibration-review${open?' is-open':''}`}>
    {toolbar&&matching.length>0&&createPortal(<button disabled={busy||props.disabled||!props.matrix||Boolean(editingId&&!confirmed)} onClick={()=>{setOpen(true);setTab('structure');jointSolve();}}>物体＋棱线联合优化</button>,toolbar)}
    {pairToolbar && createPortal(<>{structures.filter(s=>s.scope===props.scope).map((s,i)=><div className="embedded-pair-item" key={s.id}><button className="embedded-pair-preview" title={`${s.origin==='automatic'?'自动候选':'人工棱线'} · ${s.confirmed?'已确认':'待确认'} · 帧 ${s.frame}`} onClick={()=>editStructure(s)}><b>S{i+1}</b><small>{s.confirmed?'棱线':'待确认'}</small></button><input type="checkbox" aria-label={`棱线${i+1}参与优化`} checked={s.confirmed&&s.enabled!==false} disabled={!s.confirmed||busy} onChange={e=>save(structures.map(x=>x.id===s.id?{...x,enabled:e.target.checked}:x))}/><button className="embedded-pair-delete" aria-label={`删除棱线${i+1}`} onClick={()=>{if(window.confirm('删除这条棱线配对？原始点云和图像不受影响。'))save(structures.filter(x=>x.id!==s.id));}}>×</button></div>)}</>,pairToolbar)}
    {toolbar ? createPortal(trigger,toolbar) : <span className="review-trigger-placeholder">{trigger}</span>}
    {open && <aside className="review-drawer" id="calibration-review-drawer" aria-label="配对与结构检查" onKeyDown={e=>{if(e.key==='Escape')setOpen(false);}}>
      <header className="review-drawer-head"><div><strong>标定检查</strong><small>复查配对 · 辅助结构约束</small></div><button aria-label="关闭标定检查" onClick={()=>setOpen(false)}>×</button></header>
      <div className="review-tabs"><button onClick={()=>setTab('review')} aria-pressed={tab==='review'}>配对复查</button><button onClick={()=>setTab('structure')} aria-pressed={tab==='structure'}>结构棱线 <small>实验</small></button></div>
      <div className="review-drawer-content">
      {busy&&<button onClick={()=>{generation.current++;controller.current?.abort();setBusy(false);setMessage('任务已取消');}}>取消任务</button>}
      <p role="status">{message}</p>
      {tab==='review' ? <>
        <button disabled={busy||props.disabled||!props.matrix} onClick={inspect}>{report?'统一重新复查全部配对':'复查当前模型的全部配对'}</button>
        {report && <p className="review-warning">显示上次复查记录；修改后请先保存配对，再统一复查。距离与排序尚未重算。</p>}
        {report?.warning && <p className="review-warning">{report.warning}</p>}
        <div className="review-pair-list">{report?.pairs.filter((p:Result)=>!props.annotationStamps || p.annotationId in props.annotationStamps).map((p:Result,i:number)=><article key={p.annotationId} className={p.status==='建议复查'?'review-warning':''}>
          <button onClick={()=>choose(p)}>#{i+1} · {reportStamps[p.annotationId]!==props.annotationStamps?.[p.annotationId]?'已修改，待复查':p.status}</button><span>上次：越界 {(p.outsideRatio*100).toFixed(1)}% / {p.outsideMeanPx.toFixed(2)} px · 归一化 {p.score.toFixed(3)}</span>
          <button onClick={()=>choose(p,'outsideIndices')}>越界点 {p.outsideIndices.length}</button><button onClick={()=>choose(p,'isolatedIndices')}>孤立点 {p.isolatedIndices.length}</button><button onClick={()=>choose(p,'lowSupportIndices')}>低支持边界 {p.lowSupportIndices.length}</button>
          <button disabled={busy||props.disabled} onClick={()=>exclude(p.annotationId)}>暂时排除并试算</button>
          {selected===p.annotationId&&p.reasons.map((r:string)=><p key={r}>{r}</p>)}
        </article>)}</div>
        {active&&source&&<><p>图像与原始点云同步高亮：{highlightKind==='outsideIndices'?'越界点（红）':highlightKind==='isolatedIndices'?'孤立点（紫）':'低支持边界（黄）'}；白线为原 Mask。请对照真实边缘，再使用原有增补/擦除工具。</p><svg className="review-image" viewBox={`0 0 ${source.mask_width} ${source.mask_height}`}>
          <image href={source.image_url} width={source.mask_width} height={source.mask_height}/>
          {active.maskContours?.map((c:XY[],i:number)=><polygon key={i} points={c.map(p=>p.join(',')).join(' ')} fill="none" stroke="#eee" strokeWidth="1"/>)}
          {active.projection.filter((p:Result)=>!activeVisiblePoints||activeVisiblePoints.has(p.index)).map((p:Result)=>{const marked=markedIndices.has(p.index);return <circle key={p.index} cx={p.xy[0]} cy={p.xy[1]} r={marked?2:1} fill={marked?(highlightKind==='outsideIndices'?'#e45d56':highlightKind==='isolatedIndices'?'#b85ab5':'#d6a13b'):'#dce6ed'}/>;})}
          {trialCurrent&&trial?.kind==='influence'&&trial.excludeId===selected&&trial.excludedProjection.map((xy:XY,i:number)=>xy?.every(v=>v!==null)&&<circle key={`trial-${i}`} cx={xy[0]} cy={xy[1]} r={1.4} fill="#63c9e3"/>)}
        </svg></>}
      </> : <>
        <p>适用于任何刚性物体的真实棱线。图像纹理、阴影不等于几何边缘。先在主点云窗口选点，再捕获支撑。</p>
        {props.getSceneFrames&&props.onExtractionView&&<ExtractionInspector group={props.group} sceneKey={props.extractionSceneKey??''} getFrames={props.getSceneFrames} current={props.current} disabled={props.disabled} run={run} onView={view=>{setInspectionActive(Boolean(view));props.onExtractionView?.(view);}}/>}
        <details><summary>配对与联合优化（完成提取检查后再使用）</summary>
        <button disabled={busy||props.disabled||!storeReady||!props.current||!props.matrix} onClick={()=>job('match',{...props.current,matrix:props.matrix,scope:props.scope},r=>{
          setMatchStats(r.extraction??null);
          const fresh=r.candidates.filter((s:Result)=>!structures.some(old=>old.extractionVersion===s.extractionVersion&&old.datasetId===s.datasetId&&old.frame===s.frame&&JSON.stringify(old.imageLine)===JSON.stringify(s.imageLine)&&JSON.stringify(old.indicesA)===JSON.stringify(s.indicesA)&&JSON.stringify(old.indicesB)===JSON.stringify(s.indicesB)));
          if(fresh.length)void save([...structures,...fresh.map((s:Result)=>({...s,id:crypto.randomUUID(),confirmed:false,enabled:false}))]);
          else window.alert('当前帧没有新增可靠候选。可以手动指定；没有候选不代表没有棱线。');
        })}>根据当前外参自动匹配棱线</button>
        <p>图像独立提取边缘线；点云独立提取相邻平面及有限交线。外参只用于最后匹配。</p>
        {matchStats&&<p>上次提取：图像边缘 {matchStats.imageEdgeCount} 条 · 点云平面 {matchStats.planeCount} 个 · 有支撑交线 {matchStats.intersectionCount} 条 · 匹配候选 {matchStats.matchCount} 组</p>}
        <p>候选加入顶部配对列表（S 编号），默认不参与优化。点击候选，在主视图检查并调整。</p>
        <button onClick={()=>{setEditingId(null);setA([]);setB([]);setFit(null);setImageLine([]);setConfirmed(false);}}>新建人工棱线配对</button>
        <div className="review-actions"><select aria-label="结构拟合方式" value={mode} onChange={e=>{setMode(e.target.value);setFit(null);}}><option value="planes">两个表面求交</option><option value="line">窄点带拟合</option></select>
          <button disabled={!props.current?.selectedIndices?.length} onClick={()=>{setA([...props.current!.selectedIndices]);setFit(null);}}>捕获{mode==='planes'?'表面 A':'点带'}（{a.length}点）</button>
          {mode==='planes'&&<button disabled={!props.current?.selectedIndices?.length} onClick={()=>{setB([...props.current!.selectedIndices]);setFit(null);}}>捕获表面 B（{b.length}点）</button>}
          <button disabled={busy||!a.length||(mode==='planes'&&!b.length)} onClick={()=>job('fit',structuralPayload!,setFit)}>拟合三维棱线</button>
        </div>
        {fit&&<div><p>{fit.label}；{fit.support.map((s:Result,i:number)=>`支撑 ${i+1} 中位拟合误差 ${(s.residualM*100).toFixed(2)} cm，低支持 ${s.unsupportedIndices.length} 点`).join('；')}</p></div>}
        <div className="review-actions"><button onClick={()=>{setImageTool('roi');setEditingImage(true);}}>主图框定区域</button><button onClick={()=>{setImageTool('line');setEditingImage(true);}}>主图重选两端点</button><button onClick={()=>setEditingImage(false)}>结束图像编辑</button><button disabled={busy||roi.length!==2||!props.current} onClick={()=>job('detect',{...props.current,roi},r=>setLines(r.lines))}>检测区域内线段</button></div>
        {!inspectionActive&&props.current&&mainImage&&createPortal(<svg className="structure-main-overlay" style={{pointerEvents:editingImage?'auto':'none'}} viewBox={`0 0 ${props.current.width} ${props.current.height}`} onClick={clickImage}>
          {roi.length===2&&<rect x={Math.min(roi[0][0],roi[1][0])} y={Math.min(roi[0][1],roi[1][1])} width={Math.abs(roi[1][0]-roi[0][0])} height={Math.abs(roi[1][1]-roi[0][1])} fill="none" stroke="#eee"/>}
          {lines.map((line,i)=><polyline key={i} points={line.curve.map((p:XY)=>p.join(',')).join(' ')} fill="none" stroke="#e0be57" strokeWidth="2" style={{cursor:'pointer'}} onClick={e=>{e.stopPropagation();setImageLine(line.imageLine);}}/>)}
          {lineCurve.length>1&&<polyline points={lineCurve.map(p=>p.join(',')).join(' ')} fill="none" stroke="#ce68ae" strokeWidth="2"/>}
          {imageLine.map((p,i)=><circle key={i} cx={p[0]} cy={p[1]} r="3" fill="#ce68ae"/>)}
        </svg>,mainImage)}
        <div className="review-actions">{imageLine.map((p,i)=><label key={i}>端点 {i+1}{p.map((v,j)=><input key={j} type="number" step="0.5" aria-label={`端点${i+1}${j?'Y':'X'}`} value={v.toFixed(1)} onChange={e=>setImageLine(old=>old.map((q,n)=>n===i?q.map((z,k)=>k===j?Number(e.target.value):z) as XY:q))}/>)}</label>)}</div>
        <label><input type="checkbox" checked={confirmed} disabled={!fit||imageLine.length!==2} onChange={e=>setConfirmed(e.target.checked)}/>确认两种模态对应同一条真实几何棱线</label>
        <button disabled={!confirmed||!storeReady||busy} onClick={()=>{
          const item={...structures.find(s=>s.id===editingId),...structuralPayload,id:editingId??crypto.randomUUID(),fit,confirmed:true,enabled:true};
          void save([...structures.filter(s=>s.id!==item.id),item]);setEditingImage(false);
        }}>{editingId?'更新并确认棱线配对':'保存并确认棱线配对'}</button>
        <ul>{structures.filter(s=>s.scope===props.scope).map(s=><li key={s.id}><button onClick={()=>editStructure(s)}>帧 {s.frame} · {s.confirmed?'已确认':'自动候选，待确认'}</button><button disabled={!s.confirmed} onClick={()=>save(structures.map(x=>x.id===s.id?{...x,enabled:x.enabled===false}:x))}>{s.enabled===false?'参与优化':'暂停参与'}</button><button onClick={()=>{if(window.confirm('删除这条棱线配对？'))save(structures.filter(x=>x.id!==s.id));}}>删除</button></li>)}</ul>
        <button disabled={busy||props.disabled||!props.matrix||matching.length<1||Boolean(editingId&&!confirmed)} onClick={jointSolve}>联合优化全部物体配对＋{matching.length} 条确认棱线</button>
        <p>同一外参下共同优化两类约束；原三模型按钮仍为物体配对流程。联合结果需确认应用，不自动覆盖。</p>
        </details>
      </>}
      {trial&&<div className="review-trial"><strong>{trialCurrent?'候选对比':'候选已过期，不可应用'}</strong>
        <p>原物体目标：{trial.before.meanCost?.toFixed(5)} → {trial.after.meanCost?.toFixed(5)}（固定同一评价尺度）</p>
        {trial.kind==='influence'?<><p>{trial.conclusion}；外参变化 {trial.rotationChangeDeg.toFixed(3)}° / {(trial.translationChangeM*100).toFixed(2)} cm。{trial.note}</p><p>其余配对均值 {trial.remainingBefore.toFixed(5)} → {trial.remainingAfter.toFixed(5)}；排除项 {trial.excludedBefore.toFixed(5)} → {trial.excludedAfter.toFixed(5)}。青色投影为排除后的该配对。</p></>:<><p>棱线残差 {trial.lineBeforePx.toFixed(2)} → {trial.lineAfterPx.toFixed(2)} px · {trial.recommended?'联合目标下降，待人工审阅':'未收敛或联合目标未下降，保留原外参'}</p><p>联合目标 {trial.before.jointCost?.toFixed(5)} → {trial.after.jointCost?.toFixed(5)}；请同时检查上方原物体指标是否退化。</p><button disabled={!trialCurrent||!trial.recommended||busy} onClick={()=>{props.onApply(trial.matrix);setTrial(null);}}>应用候选为新版本（保留原版）</button></>}
      </div>}
      </div>
    </aside>}
  </section>;
}

function StructurePreview({fit}:{fit:Result}) {
  return <div className="review-geometry">{[[0,1,'XY'],[0,2,'XZ']].map(([a,b,label])=>{
    const ax=Number(a),by=Number(b),points=fit.points as number[][], ends=fit.endpoints as number[][];
    const values=[...points,...ends];let xmin=Infinity,xmax=-Infinity,ymin=Infinity,ymax=-Infinity;
    values.forEach(p=>{xmin=Math.min(xmin,p[ax]);xmax=Math.max(xmax,p[ax]);ymin=Math.min(ymin,p[by]);ymax=Math.max(ymax,p[by]);});
    const scale=200/Math.max(xmax-xmin,ymax-ymin,1e-6);const xy=(p:number[])=>[20+(p[ax]-xmin)*scale,230-(p[by]-ymin)*scale];
    const first=xy(ends[0]),last=xy(ends[1]);return <svg key={label} viewBox="0 0 260 260"><text x="10" y="15" fill="#ddd">{label} · 拟合线与真实支撑点</text>{points.map((p,i)=>{const pos=xy(p);return <circle key={i} cx={pos[0]} cy={pos[1]} r="1.5" fill="#a8bcc7"/>;})}<line x1={first[0]} y1={first[1]} x2={last[0]} y2={last[1]} stroke="#d779b1" strokeWidth="2"/></svg>;
  })}</div>;
}
