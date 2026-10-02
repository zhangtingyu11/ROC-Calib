export type NormalizedRect = { x1: number; y1: number; x2: number; y2: number };

export type ImageMask = {
  width: number;
  height: number;
  data: Uint8Array;
};

export type MaskDirtyRect = {
  left: number;
  top: number;
  right: number;
  bottom: number;
  changedPixels: number;
};

export type ProjectionDepthSample = { x: number; y: number; depth: number };

export function filterOccludedProjectionPoints<T extends ProjectionDepthSample>(
  points: T[],
  cellSize = 8,
  depthTolerance = 0.35,
  neighborRadius = 1,
): T[] {
  if (points.length < 2) return points.slice();
  const safeCellSize = Math.max(1, cellSize);
  const nearestDepth = new Map<number, number>();
  for (const point of points) {
    const cellX = Math.floor(point.x / safeCellSize);
    const cellY = Math.floor(point.y / safeCellSize);
    const key = cellY * 1048576 + cellX;
    const previous = nearestDepth.get(key);
    if (previous === undefined || point.depth < previous) nearestDepth.set(key, point.depth);
  }
  return points.filter((point) => {
    const cellX = Math.floor(point.x / safeCellSize);
    const cellY = Math.floor(point.y / safeCellSize);
    let nearest = point.depth;
    for (let offsetY = -neighborRadius; offsetY <= neighborRadius; offsetY += 1) {
      for (let offsetX = -neighborRadius; offsetX <= neighborRadius; offsetX += 1) {
        const candidate = nearestDepth.get((cellY + offsetY) * 1048576 + cellX + offsetX);
        if (candidate !== undefined && candidate < nearest) nearest = candidate;
      }
    }
    return point.depth <= nearest + Math.max(depthTolerance, nearest * 0.025);
  });
}

function colorStatistics(
  pixels: Uint8ClampedArray,
  width: number,
  sample: (x: number, y: number) => boolean,
  bounds: { left: number; top: number; right: number; bottom: number },
) {
  const sum = [0, 0, 0];
  const squared = [0, 0, 0];
  let count = 0;
  for (let y = bounds.top; y <= bounds.bottom; y += 2) {
    for (let x = bounds.left; x <= bounds.right; x += 2) {
      if (!sample(x, y)) continue;
      const offset = (y * width + x) * 4;
      for (let channel = 0; channel < 3; channel += 1) {
        const value = pixels[offset + channel];
        sum[channel] += value;
        squared[channel] += value * value;
      }
      count += 1;
    }
  }
  const safeCount = Math.max(count, 1);
  const mean = sum.map((value) => value / safeCount);
  const variance = squared.map((value, channel) =>
    Math.max(value / safeCount - mean[channel] * mean[channel], 180),
  );
  return { mean, variance };
}

function distanceToModel(
  pixels: Uint8ClampedArray,
  offset: number,
  model: { mean: number[]; variance: number[] },
) {
  let distance = 0;
  for (let channel = 0; channel < 3; channel += 1) {
    const delta = pixels[offset + channel] - model.mean[channel];
    distance += delta * delta / model.variance[channel];
  }
  return distance;
}

function smoothMask(mask: Uint8Array, width: number, bounds: { left: number; top: number; right: number; bottom: number }) {
  let current = mask;
  for (let pass = 0; pass < 2; pass += 1) {
    const next = current.slice();
    for (let y = bounds.top + 1; y < bounds.bottom; y += 1) {
      for (let x = bounds.left + 1; x < bounds.right; x += 1) {
        let neighbors = 0;
        for (let dy = -1; dy <= 1; dy += 1) {
          for (let dx = -1; dx <= 1; dx += 1) {
            if (dx || dy) neighbors += current[(y + dy) * width + x + dx] ? 1 : 0;
          }
        }
        const index = y * width + x;
        if (!current[index] && neighbors >= 5) next[index] = 1;
        if (current[index] && neighbors <= 1) next[index] = 0;
      }
    }
    current = next;
  }
  return current;
}

export function segmentImageBox(
  pixels: Uint8ClampedArray,
  width: number,
  height: number,
  rect: NormalizedRect,
  sensitivity: number,
): ImageMask {
  const left = Math.max(0, Math.floor(rect.x1 * width));
  const top = Math.max(0, Math.floor(rect.y1 * height));
  const right = Math.min(width - 1, Math.ceil(rect.x2 * width));
  const bottom = Math.min(height - 1, Math.ceil(rect.y2 * height));
  const roiWidth = Math.max(right - left, 1);
  const roiHeight = Math.max(bottom - top, 1);
  const centerLeft = left + Math.floor(roiWidth * 0.3);
  const centerRight = right - Math.floor(roiWidth * 0.3);
  const centerTop = top + Math.floor(roiHeight * 0.3);
  const centerBottom = bottom - Math.floor(roiHeight * 0.3);
  const borderX = Math.max(2, Math.floor(roiWidth * 0.1));
  const borderY = Math.max(2, Math.floor(roiHeight * 0.1));
  const bounds = { left, top, right, bottom };
  const foreground = colorStatistics(
    pixels, width,
    (x, y) => x >= centerLeft && x <= centerRight && y >= centerTop && y <= centerBottom,
    bounds,
  );
  const background = colorStatistics(
    pixels, width,
    (x, y) => x <= left + borderX || x >= right - borderX || y <= top + borderY || y >= bottom - borderY,
    bounds,
  );
  const candidate = new Uint8Array(width * height);
  const threshold = 0.72 + Math.max(0, Math.min(1, sensitivity)) * 0.72;
  for (let y = top; y <= bottom; y += 1) {
    for (let x = left; x <= right; x += 1) {
      const offset = (y * width + x) * 4;
      const foregroundDistance = distanceToModel(pixels, offset, foreground);
      const backgroundDistance = distanceToModel(pixels, offset, background);
      const normalizedX = (x - (left + right) * 0.5) / roiWidth;
      const normalizedY = (y - (top + bottom) * 0.5) / roiHeight;
      const spatialPenalty = (normalizedX * normalizedX + normalizedY * normalizedY) * 0.6;
      if (foregroundDistance + spatialPenalty < backgroundDistance * threshold) {
        candidate[y * width + x] = 1;
      }
    }
  }

  const labels = new Int32Array(width * height);
  const queue = new Int32Array(Math.max(roiWidth * roiHeight + roiWidth + roiHeight + 1, 1));
  let label = 0;
  let bestLabel = 0;
  let bestScore = 0;
  for (let y = top; y <= bottom; y += 1) {
    for (let x = left; x <= right; x += 1) {
      const start = y * width + x;
      if (!candidate[start] || labels[start]) continue;
      label += 1;
      let head = 0;
      let tail = 0;
      let count = 0;
      let centerHits = 0;
      queue[tail++] = start;
      labels[start] = label;
      while (head < tail) {
        const index = queue[head++];
        const pointY = Math.floor(index / width);
        const pointX = index - pointY * width;
        count += 1;
        if (pointX >= centerLeft && pointX <= centerRight && pointY >= centerTop && pointY <= centerBottom) {
          centerHits += 1;
        }
        const neighbors = [index - 1, index + 1, index - width, index + width];
        for (const neighbor of neighbors) {
          const neighborY = Math.floor(neighbor / width);
          const neighborX = neighbor - neighborY * width;
          if (neighborX < left || neighborX > right || neighborY < top || neighborY > bottom ||
              !candidate[neighbor] || labels[neighbor]) continue;
          labels[neighbor] = label;
          queue[tail++] = neighbor;
        }
      }
      const score = count * (centerHits > 0 ? 1.6 : 0.35);
      if (score > bestScore) {
        bestScore = score;
        bestLabel = label;
      }
    }
  }

  let mask: Uint8Array = new Uint8Array(width * height);
  if (bestLabel) {
    for (let y = top; y <= bottom; y += 1) {
      for (let x = left; x <= right; x += 1) {
        const index = y * width + x;
        if (labels[index] === bestLabel) mask[index] = 1;
      }
    }
  } else {
    for (let y = centerTop; y <= centerBottom; y += 1) {
      mask.fill(1, y * width + centerLeft, y * width + centerRight + 1);
    }
  }
  mask = smoothMask(mask, width, bounds);
  return { width, height, data: mask };
}

export function mergeImageMasks(current: ImageMask | null, incoming: ImageMask, mode: "replace" | "add") {
  if (!current || mode === "replace" || current.width !== incoming.width || current.height !== incoming.height) {
    return incoming;
  }
  const data = current.data.slice();
  for (let index = 0; index < data.length; index += 1) {
    if (incoming.data[index]) data[index] = 1;
  }
  return { width: current.width, height: current.height, data };
}

/**
 * Convert a raw SAM candidate into one calibration object: retain one
 * 4-connected foreground component, then fill every enclosed background
 * region. Prefer the component under the prompt-box center; if SAM leaves the
 * exact center empty, retain its largest component.
 */
export function normalizeSamObjectMask(mask: ImageMask, prompt: NormalizedRect): ImageMask {
  const { width, height } = mask;
  const size = width * height;
  const labels = new Int32Array(size);
  const queue = new Int32Array(size);
  const centerX = Math.max(0, Math.min(width - 1, Math.floor((prompt.x1 + prompt.x2) * width * 0.5)));
  const centerY = Math.max(0, Math.min(height - 1, Math.floor((prompt.y1 + prompt.y2) * height * 0.5)));
  const centerIndex = centerY * width + centerX;
  let nextLabel = 0;
  let largestLabel = 0;
  let largestSize = 0;

  for (let start = 0; start < size; start += 1) {
    if (!mask.data[start] || labels[start]) continue;
    nextLabel += 1;
    let head = 0, tail = 0, componentSize = 0;
    queue[tail++] = start;
    labels[start] = nextLabel;
    while (head < tail) {
      const index = queue[head++];
      componentSize += 1;
      const x = index % width;
      if (x > 0 && mask.data[index - 1] && !labels[index - 1]) {
        labels[index - 1] = nextLabel; queue[tail++] = index - 1;
      }
      if (x + 1 < width && mask.data[index + 1] && !labels[index + 1]) {
        labels[index + 1] = nextLabel; queue[tail++] = index + 1;
      }
      if (index >= width && mask.data[index - width] && !labels[index - width]) {
        labels[index - width] = nextLabel; queue[tail++] = index - width;
      }
      if (index + width < size && mask.data[index + width] && !labels[index + width]) {
        labels[index + width] = nextLabel; queue[tail++] = index + width;
      }
    }
    if (componentSize > largestSize) {
      largestSize = componentSize;
      largestLabel = nextLabel;
    }
  }

  const selectedLabel = labels[centerIndex] || largestLabel;
  const data = new Uint8Array(size);
  if (!selectedLabel) return { width, height, data };
  for (let index = 0; index < size; index += 1) {
    if (labels[index] === selectedLabel) data[index] = 1;
  }

  // Flood-fill background reachable from the image boundary. Any remaining
  // zero is an enclosed hole in the retained object and must become foreground.
  const exterior = new Uint8Array(size);
  let head = 0, tail = 0;
  const enqueueExterior = (index: number) => {
    if (data[index] || exterior[index]) return;
    exterior[index] = 1;
    queue[tail++] = index;
  };
  for (let x = 0; x < width; x += 1) {
    enqueueExterior(x);
    enqueueExterior((height - 1) * width + x);
  }
  for (let y = 1; y + 1 < height; y += 1) {
    enqueueExterior(y * width);
    enqueueExterior(y * width + width - 1);
  }
  while (head < tail) {
    const index = queue[head++];
    const x = index % width;
    if (x > 0) enqueueExterior(index - 1);
    if (x + 1 < width) enqueueExterior(index + 1);
    if (index >= width) enqueueExterior(index - width);
    if (index + width < size) enqueueExterior(index + width);
  }
  for (let index = 0; index < size; index += 1) {
    if (!data[index] && !exterior[index]) data[index] = 1;
  }
  return { width, height, data };
}

// One native-pixel pass on a NEW SAM layer only. Never run on saved masks or
// brush edits: repeated filtering would progressively alter thin structures.
export function smoothSamMask(mask: ImageMask): ImageMask {
  const { width, height, data } = mask;
  const output = data.slice();
  for (let y = 1; y < height - 1; y += 1) {
    for (let x = 1; x < width - 1; x += 1) {
      const i = y * width + x;
      const orthogonal = data[i - 1] + data[i + 1] + data[i - width] + data[i + width];
      const neighbors = orthogonal + data[i - width - 1] + data[i - width + 1]
        + data[i + width - 1] + data[i + width + 1];
      // Only a supported, one-pixel spur: keep straight corners, diagonal
      // filaments and endpoints without a nearby two-dimensional interior.
      if (data[i] && orthogonal === 1 && neighbors >= 3 && neighbors <= 4) {
        // Check that the supporting side spans three pixels, so a thin
        // isolated endpoint is not mistaken for a spur.
        const hasInterior = (data[i - 1] && data[i - width - 1] && data[i + width - 1]) === 1
          || (data[i + 1] && data[i - width + 1] && data[i + width + 1]) === 1
          || (data[i - width] && data[i - width - 1] && data[i - width + 1]) === 1
          || (data[i + width] && data[i + width - 1] && data[i + width + 1]) === 1;
        if (hasInterior) output[i] = 0;
      } else if (!data[i] && orthogonal >= 3 && neighbors >= 5) {
        output[i] = 1;
      }
    }
  }
  return { width, height, data: output };
}

export function paintMaskCircle(mask: ImageMask, x: number, y: number, radius: number, value: 0 | 1) {
  const left = Math.max(0, Math.floor(x - radius));
  const right = Math.min(mask.width - 1, Math.ceil(x + radius));
  const top = Math.max(0, Math.floor(y - radius));
  const bottom = Math.min(mask.height - 1, Math.ceil(y + radius));
  const squaredRadius = radius * radius;
  for (let py = top; py <= bottom; py += 1) {
    for (let px = left; px <= right; px += 1) {
      const dx = px - x;
      const dy = py - y;
      if (dx * dx + dy * dy <= squaredRadius) mask.data[py * mask.width + px] = value;
    }
  }
}

/**
 * Paint one continuous brush stroke and return only the pixels that need to be
 * redrawn. A one-pixel brush is handled separately so it always edits exactly
 * one source pixel at a stationary pointer position.
 */
export function paintMaskStroke(
  mask: ImageMask,
  from: { x: number; y: number },
  to: { x: number; y: number },
  diameter: number,
  value: 0 | 1,
): MaskDirtyRect | null {
  const safeDiameter = Math.max(1, Math.round(diameter));
  const distance = Math.hypot(to.x - from.x, to.y - from.y);
  const spacing = safeDiameter === 1 ? 0.45 : Math.max(0.75, safeDiameter * 0.24);
  const steps = Math.max(1, Math.ceil(distance / spacing));
  let left = mask.width;
  let top = mask.height;
  let right = -1;
  let bottom = -1;
  let changedPixels = 0;

  const setPixel = (px: number, py: number) => {
    if (px < 0 || px >= mask.width || py < 0 || py >= mask.height) return;
    const index = py * mask.width + px;
    if (mask.data[index] === value) return;
    mask.data[index] = value;
    left = Math.min(left, px);
    top = Math.min(top, py);
    right = Math.max(right, px);
    bottom = Math.max(bottom, py);
    changedPixels += 1;
  };

  for (let step = 0; step <= steps; step += 1) {
    const amount = step / steps;
    const x = from.x + (to.x - from.x) * amount;
    const y = from.y + (to.y - from.y) * amount;
    if (safeDiameter === 1) {
      setPixel(Math.floor(x), Math.floor(y));
      continue;
    }

    const radius = safeDiameter * 0.5;
    const circleLeft = Math.floor(x - radius);
    const circleRight = Math.ceil(x + radius);
    const circleTop = Math.floor(y - radius);
    const circleBottom = Math.ceil(y + radius);
    const squaredRadius = radius * radius;
    for (let py = circleTop; py <= circleBottom; py += 1) {
      for (let px = circleLeft; px <= circleRight; px += 1) {
        const dx = px + 0.5 - x;
        const dy = py + 0.5 - y;
        if (dx * dx + dy * dy <= squaredRadius) setPixel(px, py);
      }
    }
  }

  if (!changedPixels) return null;
  return { left, top, right, bottom, changedPixels };
}

export function countMask(mask: ImageMask | null) {
  if (!mask) return 0;
  return mask.data.reduce((total, value) => total + (value ? 1 : 0), 0);
}

export function encodeMaskRle(mask: ImageMask) {
  const runs: number[] = [];
  let current = 0;
  let length = 0;
  for (let index = 0; index < mask.data.length; index += 1) {
    const value = mask.data[index] ? 1 : 0;
    if (value === current) {
      length += 1;
    } else {
      runs.push(length);
      current = value;
      length = 1;
    }
  }
  runs.push(length);
  return runs;
}

export function decodeMaskRle(width: number, height: number, runs: number[]): ImageMask {
  const data = new Uint8Array(width * height);
  let offset = 0;
  let value = 0;
  for (const length of runs) {
    if (value && length > 0) data.fill(1, offset, Math.min(offset + length, data.length));
    offset += length;
    value = value ? 0 : 1;
    if (offset >= data.length) break;
  }
  return { width, height, data };
}

function sortedUniquePointIndices(values: Uint32Array) {
  if (values.length < 2) return values;
  let sorted = true;
  for (let index = 1; index < values.length; index += 1) {
    if (values[index] <= values[index - 1]) { sorted = false; break; }
  }
  const source = sorted ? values : Uint32Array.from(values).sort();
  if (sorted) return source;
  let write = 1;
  for (let read = 1; read < source.length; read += 1) {
    if (source[read] !== source[write - 1]) source[write++] = source[read];
  }
  return write === source.length ? source : source.slice(0, write);
}

export function combinePointIndices(
  current: Uint32Array,
  incoming: Uint32Array,
  mode: "replace" | "add" | "remove",
) {
  const left = sortedUniquePointIndices(current);
  const right = sortedUniquePointIndices(incoming);
  if (mode === "replace") return right;
  const output = new Uint32Array(mode === "add" ? left.length + right.length : left.length);
  let leftIndex = 0, rightIndex = 0, write = 0;
  while (leftIndex < left.length && rightIndex < right.length) {
    const leftValue = left[leftIndex], rightValue = right[rightIndex];
    if (leftValue < rightValue) output[write++] = left[leftIndex++];
    else if (leftValue > rightValue) {
      if (mode === "add") output[write++] = rightValue;
      rightIndex += 1;
    } else {
      if (mode === "add") output[write++] = leftValue;
      leftIndex += 1;
      rightIndex += 1;
    }
  }
  while (leftIndex < left.length) output[write++] = left[leftIndex++];
  if (mode === "add") while (rightIndex < right.length) output[write++] = right[rightIndex++];
  return write === output.length ? output : output.slice(0, write);
}

type PointCloudVoxelKey = number | string;

type PointCloudVoxel = {
  x: number;
  y: number;
  z: number;
  points: number[];
  count: number;
  sumX: number;
  sumY: number;
  sumZ: number;
  sumXX: number;
  sumXY: number;
  sumXZ: number;
  sumYY: number;
  sumYZ: number;
  sumZZ: number;
  normal: [number, number, number] | null;
};

export type PointCloudConnectivityIndex = {
  positions: Float32Array;
  stable: Uint8Array | null;
  ground: Uint8Array | null;
  stableOnly: boolean;
  removeGround: boolean;
  voxelSize: number;
  groundLimit: number;
  eligible: Uint8Array;
  cells: Map<PointCloudVoxelKey, PointCloudVoxel>;
  growthCache: Map<string, Uint32Array>;
  minX: number;
  minY: number;
  minZ: number;
  maxX: number;
  maxY: number;
  maxZ: number;
  spanY: number;
  spanZ: number;
  numericKeys: boolean;
};

export type PointCloudGrowResult = {
  indices: Uint32Array;
  reason: "ok" | "invalid-seed" | "not-static" | "ground" | "isolated" | "surface-unavailable";
  cacheHit: boolean;
  surfaceConstrained: boolean;
};

export type PointCloudGrowOptions = {
  surfaceAware?: boolean;
  maxNormalAngleDegrees?: number;
  maxPlaneDistance?: number;
};

function connectivityKey(
  index: Pick<PointCloudConnectivityIndex, "minX" | "minY" | "minZ" | "spanY" | "spanZ" | "numericKeys">,
  x: number,
  y: number,
  z: number,
): PointCloudVoxelKey {
  if (!index.numericKeys) return `${x},${y},${z}`;
  return ((x - index.minX) * index.spanY + (y - index.minY)) * index.spanZ + (z - index.minZ);
}

function smallestCovarianceEigenvector(
  xx: number,
  xy: number,
  xz: number,
  yy: number,
  yz: number,
  zz: number,
): { vector: [number, number, number]; eigenvalues: [number, number, number] } {
  const matrix = [xx, xy, xz, xy, yy, yz, xz, yz, zz];
  const vectors = [1, 0, 0, 0, 1, 0, 0, 0, 1];
  for (let iteration = 0; iteration < 10; iteration += 1) {
    let p = 0;
    let q = 1;
    let largest = Math.abs(matrix[1]);
    if (Math.abs(matrix[2]) > largest) { p = 0; q = 2; largest = Math.abs(matrix[2]); }
    if (Math.abs(matrix[5]) > largest) { p = 1; q = 2; largest = Math.abs(matrix[5]); }
    if (largest < 1e-10) break;
    const pp = matrix[p * 3 + p];
    const qq = matrix[q * 3 + q];
    const pq = matrix[p * 3 + q];
    const angle = 0.5 * Math.atan2(2 * pq, qq - pp);
    const cosine = Math.cos(angle);
    const sine = Math.sin(angle);
    for (let column = 0; column < 3; column += 1) {
      const pc = matrix[p * 3 + column];
      const qc = matrix[q * 3 + column];
      matrix[p * 3 + column] = cosine * pc - sine * qc;
      matrix[q * 3 + column] = sine * pc + cosine * qc;
    }
    for (let row = 0; row < 3; row += 1) {
      const rp = matrix[row * 3 + p];
      const rq = matrix[row * 3 + q];
      matrix[row * 3 + p] = cosine * rp - sine * rq;
      matrix[row * 3 + q] = sine * rp + cosine * rq;
    }
    for (let row = 0; row < 3; row += 1) {
      const rp = vectors[row * 3 + p];
      const rq = vectors[row * 3 + q];
      vectors[row * 3 + p] = cosine * rp - sine * rq;
      vectors[row * 3 + q] = sine * rp + cosine * rq;
    }
  }
  const order = [0, 1, 2].sort((left, right) => matrix[left * 3 + left] - matrix[right * 3 + right]);
  const smallest = order[0];
  const vector: [number, number, number] = [
    vectors[smallest],
    vectors[3 + smallest],
    vectors[6 + smallest],
  ];
  const length = Math.hypot(vector[0], vector[1], vector[2]) || 1;
  vector[0] /= length;
  vector[1] /= length;
  vector[2] /= length;
  return {
    vector,
    eigenvalues: order.map((axis) => Math.max(0, matrix[axis * 3 + axis])) as [number, number, number],
  };
}

function estimateVoxelNormals(index: PointCloudConnectivityIndex) {
  for (const cell of index.cells.values()) {
    let count = 0;
    let sumX = 0, sumY = 0, sumZ = 0;
    let sumXX = 0, sumXY = 0, sumXZ = 0, sumYY = 0, sumYZ = 0, sumZZ = 0;
    for (let dx = -2; dx <= 2; dx += 1) {
      for (let dy = -2; dy <= 2; dy += 1) {
        for (let dz = -2; dz <= 2; dz += 1) {
          const x = cell.x + dx;
          const y = cell.y + dy;
          const z = cell.z + dz;
          if (x < index.minX || x > index.maxX || y < index.minY || y > index.maxY ||
              z < index.minZ || z > index.maxZ) continue;
          const neighbor = index.cells.get(connectivityKey(index, x, y, z));
          if (!neighbor) continue;
          count += neighbor.count;
          sumX += neighbor.sumX;
          sumY += neighbor.sumY;
          sumZ += neighbor.sumZ;
          sumXX += neighbor.sumXX;
          sumXY += neighbor.sumXY;
          sumXZ += neighbor.sumXZ;
          sumYY += neighbor.sumYY;
          sumYZ += neighbor.sumYZ;
          sumZZ += neighbor.sumZZ;
        }
      }
    }
    if (count < 5) continue;
    const meanX = sumX / count;
    const meanY = sumY / count;
    const meanZ = sumZ / count;
    const eigen = smallestCovarianceEigenvector(
      sumXX / count - meanX * meanX,
      sumXY / count - meanX * meanY,
      sumXZ / count - meanX * meanZ,
      sumYY / count - meanY * meanY,
      sumYZ / count - meanY * meanZ,
      sumZZ / count - meanZ * meanZ,
    );
    const [, middle] = eigen.eigenvalues;
    const minimumSpread = index.voxelSize * index.voxelSize * 0.0001;
    if (middle <= minimumSpread || eigen.eigenvalues[0] / middle > 0.7) continue;
    cell.normal = eigen.vector;
  }
}

/**
 * Build the reusable spatial index used by projection-point picking. The
 * index is intentionally independent from the camera: a projected pixel keeps
 * its original point index, while connectivity is always evaluated in 3-D.
 */
export function buildPointCloudConnectivityIndex(
  positions: Float32Array,
  stable: Uint8Array | null,
  voxelSize: number,
  stableOnly: boolean,
  removeGround: boolean,
  ground: Uint8Array | null = null,
): PointCloudConnectivityIndex {
  const pointCount = Math.floor(positions.length / 3);
  const safeVoxelSize = Math.max(0.01, voxelSize);
  let groundLimit = Number.NEGATIVE_INFINITY;

  if (removeGround && !ground && pointCount) {
    let heightCount = 0;
    for (let pointIndex = 0; pointIndex < pointCount; pointIndex += 1) {
      if (stable?.[pointIndex] === 2 || (stableOnly && stable?.[pointIndex] !== 1)) continue;
      const z = positions[pointIndex * 3 + 2];
      if (Number.isFinite(z)) heightCount += 1;
    }
    if (heightCount) {
      const heights = new Float32Array(heightCount);
      let output = 0;
      for (let pointIndex = 0; pointIndex < pointCount; pointIndex += 1) {
        if (stable?.[pointIndex] === 2 || (stableOnly && stable?.[pointIndex] !== 1)) continue;
        const z = positions[pointIndex * 3 + 2];
        if (Number.isFinite(z)) heights[output++] = z;
      }
      heights.sort();
      groundLimit = heights[Math.floor((heights.length - 1) * 0.12)] + 0.24;
    }
  }

  const eligible = new Uint8Array(pointCount);
  let minX = Number.POSITIVE_INFINITY;
  let minY = Number.POSITIVE_INFINITY;
  let minZ = Number.POSITIVE_INFINITY;
  let maxX = Number.NEGATIVE_INFINITY;
  let maxY = Number.NEGATIVE_INFINITY;
  let maxZ = Number.NEGATIVE_INFINITY;
  for (let pointIndex = 0; pointIndex < pointCount; pointIndex += 1) {
    if (stable?.[pointIndex] === 2 || (stableOnly && stable?.[pointIndex] !== 1)) continue;
    const offset = pointIndex * 3;
    const px = positions[offset];
    const py = positions[offset + 1];
    const pz = positions[offset + 2];
    if (!Number.isFinite(px) || !Number.isFinite(py) || !Number.isFinite(pz) ||
        (removeGround && ground?.[pointIndex] === 1) || pz <= groundLimit) continue;
    eligible[pointIndex] = 1;
    const x = Math.floor(px / safeVoxelSize);
    const y = Math.floor(py / safeVoxelSize);
    const z = Math.floor(pz / safeVoxelSize);
    minX = Math.min(minX, x);
    minY = Math.min(minY, y);
    minZ = Math.min(minZ, z);
    maxX = Math.max(maxX, x);
    maxY = Math.max(maxY, y);
    maxZ = Math.max(maxZ, z);
  }

  if (!Number.isFinite(minX)) {
    minX = minY = minZ = maxX = maxY = maxZ = 0;
  }
  const spanX = maxX - minX + 1;
  const spanY = maxY - minY + 1;
  const spanZ = maxZ - minZ + 1;
  const numericKeys = spanX <= Number.MAX_SAFE_INTEGER / Math.max(spanY, 1) &&
    spanX * spanY <= Number.MAX_SAFE_INTEGER / Math.max(spanZ, 1);
  const index: PointCloudConnectivityIndex = {
    positions,
    stable,
    ground,
    stableOnly,
    removeGround,
    voxelSize: safeVoxelSize,
    groundLimit,
    eligible,
    cells: new Map(),
    growthCache: new Map(),
    minX,
    minY,
    minZ,
    maxX,
    maxY,
    maxZ,
    spanY,
    spanZ,
    numericKeys,
  };

  for (let pointIndex = 0; pointIndex < pointCount; pointIndex += 1) {
    if (!eligible[pointIndex]) continue;
    const offset = pointIndex * 3;
    const x = Math.floor(positions[offset] / safeVoxelSize);
    const y = Math.floor(positions[offset + 1] / safeVoxelSize);
    const z = Math.floor(positions[offset + 2] / safeVoxelSize);
    const key = connectivityKey(index, x, y, z);
    const cell = index.cells.get(key);
    const px = positions[offset];
    const py = positions[offset + 1];
    const pz = positions[offset + 2];
    if (cell) {
      cell.points.push(pointIndex);
      cell.count += 1;
      cell.sumX += px; cell.sumY += py; cell.sumZ += pz;
      cell.sumXX += px * px; cell.sumXY += px * py; cell.sumXZ += px * pz;
      cell.sumYY += py * py; cell.sumYZ += py * pz; cell.sumZZ += pz * pz;
    } else index.cells.set(key, {
      x, y, z,
      points: [pointIndex],
      count: 1,
      sumX: px, sumY: py, sumZ: pz,
      sumXX: px * px, sumXY: px * py, sumXZ: px * pz,
      sumYY: py * py, sumYZ: py * pz, sumZZ: pz * pz,
      normal: null,
    });
  }
  estimateVoxelNormals(index);
  return index;
}

/** Grow the seed's spatial component while preserving its local surface. */
export function growConnectedPointCloud(
  index: PointCloudConnectivityIndex,
  seedIndex: number,
  options: PointCloudGrowOptions = {},
): PointCloudGrowResult {
  const surfaceAware = options.surfaceAware ?? true;
  const maxNormalAngleDegrees = Math.max(1, Math.min(90, options.maxNormalAngleDegrees ?? 35));
  const maxPlaneDistance = Math.max(0.01, options.maxPlaneDistance ?? index.voxelSize * 0.45);
  const empty = (reason: PointCloudGrowResult["reason"]): PointCloudGrowResult => ({
    indices: new Uint32Array(), reason, cacheHit: false, surfaceConstrained: surfaceAware,
  });
  const pointCount = index.eligible.length;
  if (!Number.isInteger(seedIndex) || seedIndex < 0 || seedIndex >= pointCount) {
    return empty("invalid-seed");
  }
  if (index.stable?.[seedIndex] === 2 || (index.stableOnly && index.stable?.[seedIndex] !== 1)) {
    return empty("not-static");
  }
  if (index.removeGround && (index.ground?.[seedIndex] === 1 ||
      index.positions[seedIndex * 3 + 2] <= index.groundLimit)) {
    return empty("ground");
  }
  if (!index.eligible[seedIndex]) {
    return empty("isolated");
  }

  const offset = seedIndex * 3;
  const startX = Math.floor(index.positions[offset] / index.voxelSize);
  const startY = Math.floor(index.positions[offset + 1] / index.voxelSize);
  const startZ = Math.floor(index.positions[offset + 2] / index.voxelSize);
  const startKey = connectivityKey(index, startX, startY, startZ);
  const start = index.cells.get(startKey);
  if (!start) return empty("isolated");
  let seedNormal = start.normal;
  if (surfaceAware && !seedNormal) {
    let best: { normal: [number, number, number]; distanceSquared: number } | null = null;
    for (let dx = -1; dx <= 1; dx += 1) {
      for (let dy = -1; dy <= 1; dy += 1) {
        for (let dz = -1; dz <= 1; dz += 1) {
          const neighbor = index.cells.get(connectivityKey(index, startX + dx, startY + dy, startZ + dz));
          if (!neighbor?.normal) continue;
          const centerX = neighbor.sumX / neighbor.count;
          const centerY = neighbor.sumY / neighbor.count;
          const centerZ = neighbor.sumZ / neighbor.count;
          const distanceSquared = (centerX - index.positions[offset]) ** 2 +
            (centerY - index.positions[offset + 1]) ** 2 + (centerZ - index.positions[offset + 2]) ** 2;
          if (!best || distanceSquared < best.distanceSquared) best = { normal: neighbor.normal, distanceSquared };
        }
      }
    }
    seedNormal = best?.normal ?? null;
  }
  if (surfaceAware && !seedNormal) return empty("surface-unavailable");
  const cacheKey = `${String(startKey)}|${surfaceAware ? 1 : 0}|${maxNormalAngleDegrees}|${maxPlaneDistance}`;
  const cached = index.growthCache.get(cacheKey);
  if (cached) return { indices: cached, reason: "ok", cacheHit: true, surfaceConstrained: surfaceAware };
  const normalCosineLimit = Math.cos(maxNormalAngleDegrees * Math.PI / 180);
  const centerOf = (cell: PointCloudVoxel): [number, number, number] => [
    cell.sumX / cell.count, cell.sumY / cell.count, cell.sumZ / cell.count,
  ];

  const queue: PointCloudVoxel[] = [start];
  const visited = new Set<PointCloudVoxelKey>([startKey]);
  const points: number[] = [];
  for (let head = 0; head < queue.length; head += 1) {
    const cell = queue[head];
    const cellNormal = cell.normal ?? seedNormal;
    const cellCenter = centerOf(cell);
    if (surfaceAware && seedNormal) {
      for (const pointIndex of cell.points) {
        const pointOffset = pointIndex * 3;
        const pointPlaneDistance = cellNormal
          ? Math.abs(
            (index.positions[pointOffset] - cellCenter[0]) * cellNormal[0] +
            (index.positions[pointOffset + 1] - cellCenter[1]) * cellNormal[1] +
            (index.positions[pointOffset + 2] - cellCenter[2]) * cellNormal[2]
          )
          : 0;
        if (pointPlaneDistance <= maxPlaneDistance) points.push(pointIndex);
      }
    } else points.push(...cell.points);
    for (let dx = -1; dx <= 1; dx += 1) {
      for (let dy = -1; dy <= 1; dy += 1) {
        for (let dz = -1; dz <= 1; dz += 1) {
          if (dx === 0 && dy === 0 && dz === 0) continue;
          const x = cell.x + dx;
          const y = cell.y + dy;
          const z = cell.z + dz;
          if (x < index.minX || x > index.maxX || y < index.minY || y > index.maxY ||
              z < index.minZ || z > index.maxZ) continue;
          const key = connectivityKey(index, x, y, z);
          if (visited.has(key)) continue;
          const neighbor = index.cells.get(key);
          if (!neighbor) continue;
          if (surfaceAware && seedNormal) {
            if (!neighbor.normal) continue;
            const seedSimilarity = Math.abs(
              seedNormal[0] * neighbor.normal[0] + seedNormal[1] * neighbor.normal[1] + seedNormal[2] * neighbor.normal[2],
            );
            const localSimilarity = cellNormal ? Math.abs(
              cellNormal[0] * neighbor.normal[0] + cellNormal[1] * neighbor.normal[1] + cellNormal[2] * neighbor.normal[2],
            ) : seedSimilarity;
            if (seedSimilarity < normalCosineLimit || localSimilarity < normalCosineLimit) continue;
            const neighborCenter = centerOf(neighbor);
            const deltaX = neighborCenter[0] - cellCenter[0];
            const deltaY = neighborCenter[1] - cellCenter[1];
            const deltaZ = neighborCenter[2] - cellCenter[2];
            const currentPlaneGap = cellNormal ? Math.abs(
              deltaX * cellNormal[0] + deltaY * cellNormal[1] + deltaZ * cellNormal[2],
            ) : 0;
            const neighborPlaneGap = Math.abs(
              deltaX * neighbor.normal[0] + deltaY * neighbor.normal[1] + deltaZ * neighbor.normal[2],
            );
            if (currentPlaneGap > maxPlaneDistance || neighborPlaneGap > maxPlaneDistance) continue;
          }
          visited.add(key);
          queue.push(neighbor);
        }
      }
    }
  }
  points.sort((left, right) => left - right);
  const component = Uint32Array.from(points);
  index.growthCache.set(cacheKey, component);
  return { indices: component, reason: "ok", cacheHit: false, surfaceConstrained: surfaceAware };
}

export function clusterPointSelection(
  positions: Float32Array,
  selected: Uint32Array,
  voxelSize: number,
  removeGround: boolean,
) {
  if (!selected.length) return selected;
  let groundLimit = Number.NEGATIVE_INFINITY;
  if (removeGround) {
    const heights = Array.from(selected, (index) => positions[index * 3 + 2]).sort((a, b) => a - b);
    groundLimit = heights[Math.floor((heights.length - 1) * 0.12)] + 0.24;
  }
  const candidates: number[] = [];
  selected.forEach((index) => {
    const z = positions[index * 3 + 2];
    if (z <= groundLimit) return;
    candidates.push(index);
  });
  if (!candidates.length) return selected;

  // Estimate local sampling density from the actual boxed points. A fixed
  // voxel works for dense scans but fragments distant/sparse objects. Search
  // neighboring base cells for a deterministic sample, then enlarge the
  // connectivity voxel only as much as the observed spacing requires.
  const baseSize = Math.max(voxelSize, 0.01);
  const densityCells = new Map<string, number[]>();
  for (const index of candidates) {
    const x = Math.floor(positions[index * 3] / baseSize);
    const y = Math.floor(positions[index * 3 + 1] / baseSize);
    const z = Math.floor(positions[index * 3 + 2] / baseSize);
    const key = `${x},${y},${z}`;
    const cell = densityCells.get(key);
    if (cell) cell.push(index);
    else densityCells.set(key, [index]);
  }
  const nearestDistances: number[] = [];
  const sampleStep = Math.max(1, Math.floor(candidates.length / 900));
  for (let sampleOffset = 0; sampleOffset < candidates.length; sampleOffset += sampleStep) {
    const index = candidates[sampleOffset];
    const px = positions[index * 3], py = positions[index * 3 + 1], pz = positions[index * 3 + 2];
    const cellX = Math.floor(px / baseSize), cellY = Math.floor(py / baseSize), cellZ = Math.floor(pz / baseSize);
    let nearestSquared = Number.POSITIVE_INFINITY;
    for (let radius = 0; radius <= 6 && !Number.isFinite(nearestSquared); radius += 1) {
      for (let dx = -radius; dx <= radius; dx += 1) for (let dy = -radius; dy <= radius; dy += 1) for (let dz = -radius; dz <= radius; dz += 1) {
        if (radius > 0 && Math.max(Math.abs(dx), Math.abs(dy), Math.abs(dz)) !== radius) continue;
        for (const other of densityCells.get(`${cellX + dx},${cellY + dy},${cellZ + dz}`) ?? []) {
          if (other === index) continue;
          const distanceSquared = (positions[other * 3] - px) ** 2 +
            (positions[other * 3 + 1] - py) ** 2 + (positions[other * 3 + 2] - pz) ** 2;
          nearestSquared = Math.min(nearestSquared, distanceSquared);
        }
      }
    }
    if (Number.isFinite(nearestSquared)) nearestDistances.push(Math.sqrt(nearestSquared));
    if (nearestDistances.length >= 900) break;
  }
  nearestDistances.sort((left, right) => left - right);
  const observedSpacing = nearestDistances[Math.floor(Math.max(0, nearestDistances.length - 1) * 0.75)] ?? baseSize;
  const adaptiveVoxelSize = Math.min(baseSize * 6, Math.max(baseSize, observedSpacing * 0.95));

  const voxels = new Map<string, number[]>();
  candidates.forEach((index) => {
    const z = positions[index * 3 + 2];
    const x = Math.floor(positions[index * 3] / adaptiveVoxelSize);
    const y = Math.floor(positions[index * 3 + 1] / adaptiveVoxelSize);
    const vz = Math.floor(z / adaptiveVoxelSize);
    const key = `${x},${y},${vz}`;
    const values = voxels.get(key);
    if (values) values.push(index);
    else voxels.set(key, [index]);
  });
  if (!voxels.size) return selected;
  const visited = new Set<string>();
  let best: number[] = [];
  for (const start of voxels.keys()) {
    if (visited.has(start)) continue;
    const queue = [start];
    visited.add(start);
    const cluster: number[] = [];
    for (let head = 0; head < queue.length; head += 1) {
      const key = queue[head];
      cluster.push(...(voxels.get(key) ?? []));
      const [x, y, z] = key.split(",").map(Number);
      for (let dx = -1; dx <= 1; dx += 1) {
        for (let dy = -1; dy <= 1; dy += 1) {
          for (let dz = -1; dz <= 1; dz += 1) {
            const neighbor = `${x + dx},${y + dy},${z + dz}`;
            if (!visited.has(neighbor) && voxels.has(neighbor)) {
              visited.add(neighbor);
              queue.push(neighbor);
            }
          }
        }
      }
    }
    if (cluster.length > best.length) best = cluster;
  }
  return Uint32Array.from(best.sort((left, right) => left - right));
}

export type ScreenDepthPoint = { index: number; x: number; y: number; depth: number };

export function filterForegroundScreenPoints(
  points: ScreenDepthPoint[],
  cellSize = 10,
  depthTolerance = 0.8,
) {
  if (points.length < 2) return Uint32Array.from(points.map((point) => point.index));
  const depths = points.map((point) => point.depth).sort((left, right) => left - right);
  const foregroundDepth = depths[Math.floor((depths.length - 1) * .10)];
  const foregroundThickness = Math.max(depthTolerance * 1.5, foregroundDepth * .08);
  const localMinimum = new Map<string, number>();
  for (const point of points) {
    const key = `${Math.floor(point.x / cellSize)},${Math.floor(point.y / cellSize)}`;
    localMinimum.set(key, Math.min(localMinimum.get(key) ?? Number.POSITIVE_INFINITY, point.depth));
  }
  const selected = points.filter((point) => {
    if (point.depth > foregroundDepth + foregroundThickness) return false;
    const cellX = Math.floor(point.x / cellSize), cellY = Math.floor(point.y / cellSize);
    let nearbyMinimum = Number.POSITIVE_INFINITY;
    for (let dx = -1; dx <= 1; dx += 1) for (let dy = -1; dy <= 1; dy += 1) {
      nearbyMinimum = Math.min(nearbyMinimum, localMinimum.get(`${cellX + dx},${cellY + dy}`) ?? Number.POSITIVE_INFINITY);
    }
    return point.depth <= nearbyMinimum + depthTolerance;
  });
  return Uint32Array.from(selected.map((point) => point.index));
}
