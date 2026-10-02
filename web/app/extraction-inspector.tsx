"use client";
import {useEffect,useRef,useState} from 'react';
import {createPortal} from 'react-dom';
type Item=Record<string,any>;
type Props={group:string;sceneKey:string;getFrames:()=>Item[];current:Item|null;disabled:boolean;
  run:(action:string,payload:Item,signal:AbortSignal,progress:(s:string)=>void)=>Promise<Item>;
  onView:(item:Item|null)=>void;};
const COLORS=[0x38c4d4,0xf3ab47,0xb78ad8,0x68bf75,0xea7777,0x71a0f4];
export default function ExtractionInspector(props:Props){
  const [report,setReport]=useState<Item|null>(null),[busy,setBusy]=useState(false),[message,setMessage]=useState('独立提取全部帧，不匹配、不创建配对、不优化外参。');
  const [index,setIndex]=useState(0),[line,setLine]=useState(-1),[plane,setPlane]=useState(-1),[edge,setEdge]=useState(-1);
  const [viewing,setViewing]=useState(false),[isolated,setIsolated]=useState(false),[host,setHost]=useState<HTMLElement|null>(null);
  const root=useRef<HTMLDivElement>(null),abort=useRef<AbortController|null>(null),generation=useRef(0);
  const currentProps=useRef(props);currentProps.current=props;
  const frame=report?.frames[index];
  const context=props.current?`${props.current.datasetId}:${props.current.frame}`:'';
  useEffect(()=>{
    const id=++generation.current;abort.current?.abort();setBusy(false);setReport(null);setViewing(false);props.onView(null);
    const controller=new AbortController();abort.current=controller;
    fetch(`/api/calibration/review/extraction/${encodeURIComponent(props.group)}?scene_key=${encodeURIComponent(props.sceneKey)}`,{signal:controller.signal})
      .then(async r=>{if(r.status===404)return null;if(!r.ok)throw new Error('检查报告读取失败');return r.json();})
      .then(r=>{if(generation.current===id&&r){setReport(r);setIndex(0);setMessage('已载入上次独立提取报告，可逐帧检查或重新提取。');}})
      .catch(e=>{if(generation.current===id&&e.name!=='AbortError')setMessage(e.message);});
    return()=>{generation.current++;controller.abort();abort.current?.abort();currentProps.current.onView(null);};
  },[props.group,props.sceneKey]);
  useEffect(()=>{setHost(root.current?.closest('.calibration-stage')?.querySelector<HTMLElement>('.unified-image-panel .image-canvas-shell')??null);},[context,viewing,frame]);
  useEffect(()=>{if(!viewing||!report)return;const i=report.frames.findIndex((f:Item)=>`${f.datasetId}:${f.frame}`===context);if(i>=0&&i!==index){setIndex(i);setLine(-1);setPlane(-1);setEdge(-1);show(report.frames[i]);}},[context]);
  async function extract(){
    const id=++generation.current;abort.current?.abort();const controller=new AbortController();abort.current=controller;
    setBusy(true);setViewing(false);props.onView(null);setMessage('开始逐帧独立提取…');
    try{const result=await props.run('extract-scene',{group:props.group,sceneKey:props.sceneKey,frames:props.getFrames()},controller.signal,s=>{if(generation.current===id)setMessage(s);});
      if(generation.current===id){setReport(result);setIndex(0);setLine(-1);setPlane(-1);setEdge(-1);setMessage(`完成 ${result.successfulFrames}/${result.frameCount} 帧。报告已保存，没有生成配对。`);}}
    catch(e){if(generation.current===id)setMessage(String(e));}
    finally{if(generation.current===id)setBusy(false);}
  }
  function show(target:Item,chosenLine=-1,chosenPlane=-1,isolate=isolated,focus=false){
    if(!target||target.status!=='ok'){props.onView(null);return;}
    const selectedLine=target.intersections.find((v:Item)=>v.id===chosenLine);
    const planes=selectedLine?target.planes.filter((p:Item)=>selectedLine.planeIds.includes(p.id)):
      chosenPlane>=0?target.planes.filter((p:Item)=>p.id===chosenPlane):target.planes;
    props.onView({datasetId:target.datasetId,frame:target.frame,
      planes:planes.map((p:Item,i:number)=>({indices:p.indices,color:COLORS[i%COLORS.length]})),
      lines:selectedLine?[selectedLine.endpoints]:chosenPlane>=0?[]:target.intersections.map((v:Item)=>v.endpoints),isolate,focus});
    setViewing(true);
  }
  function selectFrame(i:number){setIndex(i);setLine(-1);setPlane(-1);setEdge(-1);show(report!.frames[i]);}
  const sameFrame=viewing&&frame&&`${frame.datasetId}:${frame.frame}`===context;
  return <div ref={root} className="extraction-inspector">
    <strong>整场景独立提取检查</strong>
    <p>图像边缘与点云平面/交线分别提取，不用外参筛选。</p>
    <button disabled={busy||props.disabled||!props.sceneKey} onClick={extract}>提取当前场景全部 {props.getFrames().length} 帧</button>
    {busy&&<button onClick={()=>{generation.current++;abort.current?.abort();setBusy(false);setMessage('已取消；上次完整报告不变。');}}>取消整场景提取</button>}
    <p role="status">{message}</p>
    {report&&<>
      <small>报告时间：{new Date(report.generatedAt).toLocaleString()} · 成功 {report.successfulFrames}/{report.frameCount} 帧</small>
      <select aria-label="提取检查来源帧" value={index} onChange={e=>selectFrame(Number(e.target.value))}>
        {report.frames.map((f:Item,i:number)=><option key={i} value={i}>帧 {f.frame} · {f.status==='ok'?`图像 ${f.stats.imageEdgeCount} / 平面 ${f.stats.planeCount} / 交线 ${f.stats.intersectionCount}`:'失败'}</option>)}
      </select>
      <div className="review-actions"><button disabled={index===0} onClick={()=>selectFrame(index-1)}>上一帧</button><button disabled={index+1>=report.frames.length} onClick={()=>selectFrame(index+1)}>下一帧</button><button disabled={frame?.status!=='ok'} onClick={()=>show(frame,line,plane)}>在主视图检查</button><button onClick={()=>{setViewing(false);props.onView(null);}}>退出检查</button></div>
      {frame?.status==='error'?<p>{frame.error}</p>:frame&&<>
        <label>图像边缘<select value={edge} onChange={e=>{setEdge(Number(e.target.value));if(!viewing)show(frame,line,plane);}}><option value={-1}>全部 {frame.imageEdges.length} 条（不按匹配过滤）</option>{frame.imageEdges.map((l:Item,i:number)=><option key={i} value={i}>E{i+1} · {l.length.toFixed(1)} px</option>)}</select></label>
        <label>三维交线<select value={line} onChange={e=>{const value=Number(e.target.value);setLine(value);setPlane(-1);show(frame,value,-1,isolated,value>=0);}}><option value={-1}>全部 {frame.intersections.length} 条</option>{frame.intersections.map((l:Item)=> <option key={l.id} value={l.id}>L{l.id+1} · 平面 {l.planeIds.join(' / ')}</option>)}</select></label>
        <label>单独检查平面<select value={plane} onChange={e=>{const value=Number(e.target.value);setPlane(value);setLine(-1);show(frame,-1,value);}}><option value={-1}>全部 {frame.planes.length} 个</option>{frame.planes.map((p:Item)=><option key={p.id} value={p.id}>P{p.id} · {p.indices.length} 点 · {(p.medianResidualM*100).toFixed(2)} cm</option>)}</select></label>
        <label><input type="checkbox" checked={isolated} onChange={e=>{setIsolated(e.target.checked);show(frame,line,plane,e.target.checked);}}/>仅显示检查中的平面支撑点（三维）</label>
        <p>图像黄色线和三维紫色线是独立特征，不表示对应。选一条三维线时，两侧平面分别着色；未归入平面的原始点不会被删除。</p>
      </>}
    </>}
    {sameFrame&&host&&createPortal(<svg className="structure-main-overlay" style={{pointerEvents:'none'}} viewBox={`0 0 ${frame.width} ${frame.height}`}>
      {frame.imageEdges.map((l:Item,i:number)=>(edge<0||edge===i)&&<polyline key={i} points={l.curve.map((p:number[])=>p.join(',')).join(' ')} fill="none" stroke="#ffe06a" strokeWidth={edge===i?2.5:1} opacity={edge<0?.75:1} vectorEffect="non-scaling-stroke" style={{pointerEvents:'stroke',cursor:'pointer'}} onClick={e=>{e.stopPropagation();setEdge(edge===i?-1:i);}}/>)}
    </svg>,host)}
  </div>;
}
