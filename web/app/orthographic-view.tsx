"use client";

import { useEffect, useRef } from "react";
import { type CloudData } from './workspace-core';

export function OrthographicCloudView({
  source,
  target,
  plane,
  label,
  span,
  center,
  rotation,
  zoom,
  onRotate,
  onZoom,
  onReset,
}: {
  source: CloudData;
  target: CloudData;
  plane: "xy" | "xz" | "yz";
  label: string;
  span: number;
  center: [number, number, number];
  rotation: { yaw: number; pitch: number };
  zoom: number;
  onRotate: (deltaX: number, deltaY: number) => void;
  onZoom: (delta: number) => void;
  onReset: () => void;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const dragRef = useRef<{ x: number; y: number } | null>(null);
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const draw = () => {
      const rect = canvas.getBoundingClientRect();
      const ratio = Math.min(window.devicePixelRatio || 1, 2);
      canvas.width = Math.max(1, Math.round(rect.width * ratio));
      canvas.height = Math.max(1, Math.round(rect.height * ratio));
      const context = canvas.getContext("2d");
      if (!context) return;
      context.setTransform(ratio, 0, 0, ratio, 0, 0);
      context.fillStyle = "#07100f";
      context.fillRect(0, 0, rect.width, rect.height);
      const padding = 12;
      const scale = Math.max(1, Math.min(rect.width - padding * 2, rect.height - padding * 2)) / Math.max(span, .2) * zoom;
      const axes = plane === "xy" ? [0, 1] : plane === "xz" ? [0, 2] : [1, 2];
      const yawCos = Math.cos(rotation.yaw), yawSin = Math.sin(rotation.yaw);
      const pitchCos = Math.cos(rotation.pitch), pitchSin = Math.sin(rotation.pitch);
      const plot = (cloud: CloudData, color: string) => {
        const count = cloud.positions.length / 3;
        const stride = Math.max(1, Math.ceil(count / 45000));
        context.fillStyle = color;
        for (let index = 0; index < count; index += stride) {
          const offset = index * 3;
          const localX = cloud.positions[offset] - center[0];
          const localY = cloud.positions[offset + 1] - center[1];
          const localZ = cloud.positions[offset + 2] - center[2];
          const yawX = yawCos * localX - yawSin * localY;
          const yawY = yawSin * localX + yawCos * localY;
          const rotated = [yawX, pitchCos * yawY - pitchSin * localZ, pitchSin * yawY + pitchCos * localZ];
          const x = rect.width / 2 + rotated[axes[0]] * scale;
          const y = rect.height / 2 - rotated[axes[1]] * scale;
          context.fillRect(x, y, 1.35, 1.35);
        }
      };
      context.globalCompositeOperation = "lighter";
      plot(target, "rgba(100,143,255,.82)");
      plot(source, "rgba(255,176,0,.82)");
      context.globalCompositeOperation = "source-over";
    };
    draw();
    const observer = new ResizeObserver(draw);
    observer.observe(canvas);
    return () => observer.disconnect();
  }, [center, plane, rotation.pitch, rotation.yaw, source, span, target, zoom]);
  return <div className="orthographic-cloud-view"><canvas
    ref={canvasRef}
    onPointerDown={(event) => { event.currentTarget.setPointerCapture(event.pointerId); dragRef.current = { x: event.clientX, y: event.clientY }; }}
    onPointerMove={(event) => { const previous = dragRef.current; if (!previous) return; onRotate(event.clientX - previous.x, event.clientY - previous.y); dragRef.current = { x: event.clientX, y: event.clientY }; }}
    onPointerUp={(event) => { event.currentTarget.releasePointerCapture(event.pointerId); dragRef.current = null; }}
    onPointerCancel={() => { dragRef.current = null; }}
    onWheel={(event) => { event.preventDefault(); onZoom(event.deltaY); }}
    onDoubleClick={onReset}
  /><b>{label}</b></div>;
}
