type Mask = { width: number; height: number; rle: number[] };
type Layer = Mask & { id: string; granularity: string; brushModified: boolean };
type Content = {
  pointIndices: number[];
  imageMask: Mask | null;
  imageMaskLayers?: Layer[];
  manualAddedMask?: Mask | null;
  manualRemovedMask?: Mask | null;
};

function sameMask(a?: Mask | null, b?: Mask | null): boolean {
  if (!a || !b) return !a && !b;
  return a.width === b.width && a.height === b.height &&
    a.rle.length === b.rle.length && a.rle.every((value, index) => value === b.rle[index]);
}

// Compare saved content, not edit counters: undoing an edit can be clean again.
export function annotationContentChanged(current: Content, saved: Content): boolean {
  if (JSON.stringify(current.pointIndices) !== JSON.stringify(saved.pointIndices) ||
      !sameMask(current.imageMask, saved.imageMask)) return true;
  // Legacy annotations acquire an in-memory layer when loaded; that alone is not an edit.
  if (saved.imageMaskLayers?.length) {
    const layers = current.imageMaskLayers ?? [];
    if (layers.length !== saved.imageMaskLayers.length || saved.imageMaskLayers.some((layer, index) => {
      const other = layers[index];
      return !other || layer.id !== other.id || layer.granularity !== other.granularity ||
        layer.brushModified !== other.brushModified || !sameMask(layer, other);
    })) return true;
  }
  return !sameMask(current.manualAddedMask, saved.manualAddedMask) ||
    !sameMask(current.manualRemovedMask, saved.manualRemovedMask);
}
