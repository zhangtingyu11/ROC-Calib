export type WaymoTemporalProjection = {
  model: "waymo-camera-model-v1";
  vehicleFromLidar: number[];
  referenceWorldFromVehicle: number[];
  cameraWorldFromVehicle: number[];
  linearVelocityWorld: number[];
  angularVelocityVehicle: number[];
  poseTimestamp: number;
  shutter: number;
  triggerTime: number;
  readoutDoneTime: number;
  rollingShutterDirection: number;
  imageWidth: number;
  imageHeight: number;
};

export type WaymoProjectedPoint = { x: number; y: number; depth: number };

const WAYMO_CAMERA_FROM_OPENCV = [
  0, 0, 1, 0,
  -1, 0, 0, 0,
  0, -1, 0, 0,
  0, 0, 0, 1,
];
const MIN_TRUSTED_RADIAL_DISTORTION = 0.8;
const MAX_TRUSTED_RADIAL_DISTORTION = 1.2;

function multiply4(left: number[], right: number[]) {
  const result = new Array<number>(16).fill(0);
  for (let row = 0; row < 4; row += 1) for (let column = 0; column < 4; column += 1) {
    for (let inner = 0; inner < 4; inner += 1) result[row * 4 + column] += left[row * 4 + inner] * right[inner * 4 + column];
  }
  return result;
}

function invertRigid(matrix: number[]) {
  return [
    matrix[0], matrix[4], matrix[8], -(matrix[0] * matrix[3] + matrix[4] * matrix[7] + matrix[8] * matrix[11]),
    matrix[1], matrix[5], matrix[9], -(matrix[1] * matrix[3] + matrix[5] * matrix[7] + matrix[9] * matrix[11]),
    matrix[2], matrix[6], matrix[10], -(matrix[2] * matrix[3] + matrix[6] * matrix[7] + matrix[10] * matrix[11]),
    0, 0, 0, 1,
  ];
}

function distort(x: number, y: number, distortion: number[]) {
  const [k1 = 0, k2 = 0, p1 = 0, p2 = 0, k3 = 0] = distortion;
  const radius2 = x * x + y * y;
  const radial = 1 + k1 * radius2 + k2 * radius2 ** 2 + k3 * radius2 ** 3;
  return [
    x * radial + 2 * p1 * x * y + p2 * (radius2 + 2 * x * x),
    y * radial + p1 * (radius2 + 2 * y * y) + 2 * p2 * x * y,
  ];
}

function undistort(pixelX: number, pixelY: number, intrinsic: number[], distortion: number[]) {
  const targetX = (pixelX - intrinsic[2]) / intrinsic[0];
  const targetY = (pixelY - intrinsic[5]) / intrinsic[4];
  const [k1 = 0, k2 = 0, p1 = 0, p2 = 0, k3 = 0] = distortion;
  let x = targetX, y = targetY;
  for (let iteration = 0; iteration < 20; iteration += 1) {
    const radius2 = x * x + y * y;
    const radial = 1 + k1 * radius2 + k2 * radius2 ** 2 + k3 * radius2 ** 3;
    const tangentialX = 2 * p1 * x * y + p2 * (radius2 + 2 * x * x);
    const tangentialY = 2 * p2 * x * y + p1 * (radius2 + 2 * y * y);
    x = (targetX - tangentialX) / radial;
    y = (targetY - tangentialY) / radial;
  }
  return [x, y];
}

function pixelTimestamp(metadata: WaymoTemporalProjection, x: number, y: number) {
  const readout = metadata.readoutDoneTime - metadata.triggerTime - metadata.shutter;
  const base = metadata.triggerTime + 0.5 * metadata.shutter;
  if (metadata.rollingShutterDirection === 1) return base + readout / metadata.imageHeight * y;
  if (metadata.rollingShutterDirection === 2) return base + readout / metadata.imageWidth * x;
  if (metadata.rollingShutterDirection === 3) return base + readout / metadata.imageHeight * (metadata.imageHeight - y);
  if (metadata.rollingShutterDirection === 4) return base + readout / metadata.imageWidth * (metadata.imageWidth - x);
  if (metadata.rollingShutterDirection === 5) return base;
  throw new Error(`unsupported Waymo rolling-shutter direction: ${metadata.rollingShutterDirection}`);
}

function multiplyRotationVector(matrix: number[], vector: number[]) {
  return [
    matrix[0] * vector[0] + matrix[1] * vector[1] + matrix[2] * vector[2],
    matrix[4] * vector[0] + matrix[5] * vector[1] + matrix[6] * vector[2],
    matrix[8] * vector[0] + matrix[9] * vector[1] + matrix[10] * vector[2],
  ];
}

function cross(left: number[], right: number[]) {
  return [
    left[1] * right[2] - left[2] * right[1],
    left[2] * right[0] - left[0] * right[2],
    left[0] * right[1] - left[1] * right[0],
  ];
}

export function createWaymoCameraProjector(
  metadata: WaymoTemporalProjection,
  opencvCameraFromLidar: number[],
  intrinsic: number[],
  distortion: number[],
) {
  if (metadata.model !== "waymo-camera-model-v1") throw new Error("unsupported temporal projection model");
  const nativeCameraFromLidar = multiply4(WAYMO_CAMERA_FROM_OPENCV, opencvCameraFromLidar);
  const vehicleFromCamera = multiply4(metadata.vehicleFromLidar, invertRigid(nativeCameraFromLidar));
  const referenceWorldFromLidar = multiply4(metadata.referenceWorldFromVehicle, metadata.vehicleFromLidar);
  const worldFromCamera = multiply4(metadata.cameraWorldFromVehicle, vehicleFromCamera);
  const cameraFromWorld = [
    worldFromCamera[0], worldFromCamera[4], worldFromCamera[8],
    worldFromCamera[1], worldFromCamera[5], worldFromCamera[9],
    worldFromCamera[2], worldFromCamera[6], worldFromCamera[10],
  ];
  const omegaCamera = [
    vehicleFromCamera[0] * metadata.angularVelocityVehicle[0] + vehicleFromCamera[4] * metadata.angularVelocityVehicle[1] + vehicleFromCamera[8] * metadata.angularVelocityVehicle[2],
    vehicleFromCamera[1] * metadata.angularVelocityVehicle[0] + vehicleFromCamera[5] * metadata.angularVelocityVehicle[1] + vehicleFromCamera[9] * metadata.angularVelocityVehicle[2],
    vehicleFromCamera[2] * metadata.angularVelocityVehicle[0] + vehicleFromCamera[6] * metadata.angularVelocityVehicle[1] + vehicleFromCamera[10] * metadata.angularVelocityVehicle[2],
  ];
  const skewOmega = [
    0, -omegaCamera[2], omegaCamera[1],
    omegaCamera[2], 0, -omegaCamera[0],
    -omegaCamera[1], omegaCamera[0], 0,
  ];
  const rotationRate = new Array<number>(9).fill(0);
  for (let row = 0; row < 3; row += 1) for (let column = 0; column < 3; column += 1) {
    for (let inner = 0; inner < 3; inner += 1) rotationRate[row * 3 + column] -= skewOmega[row * 3 + inner] * cameraFromWorld[inner * 3 + column];
  }
  const omegaWorld = multiplyRotationVector(metadata.cameraWorldFromVehicle, metadata.angularVelocityVehicle);
  const leverWorld = multiplyRotationVector(metadata.cameraWorldFromVehicle, [vehicleFromCamera[3], vehicleFromCamera[7], vehicleFromCamera[11]]);
  const leverVelocity = cross(omegaWorld, leverWorld);
  const cameraVelocity = metadata.linearVelocityWorld.map((value, index) => value + leverVelocity[index]);
  const cameraPosition = [worldFromCamera[3], worldFromCamera[7], worldFromCamera[11]];
  const horizontal = metadata.rollingShutterDirection === 2 || metadata.rollingShutterDirection === 4;
  let readoutFactor = 0;
  if (metadata.rollingShutterDirection !== 5) {
    let first: number, last: number;
    if (horizontal) {
      first = undistort(0, metadata.imageHeight * 0.5, intrinsic, distortion)[0];
      last = undistort(metadata.imageWidth, metadata.imageHeight * 0.5, intrinsic, distortion)[0];
    } else {
      first = undistort(metadata.imageWidth * 0.5, 0, intrinsic, distortion)[1];
      last = undistort(metadata.imageWidth * 0.5, metadata.imageHeight, intrinsic, distortion)[1];
    }
    const readout = metadata.readoutDoneTime - metadata.triggerTime - metadata.shutter;
    readoutFactor = (metadata.rollingShutterDirection === 3 || metadata.rollingShutterDirection === 4 ? -1 : 1) * readout / (last - first);
  }
  const principalTime = pixelTimestamp(metadata, intrinsic[2], intrinsic[5]);
  const poseOffset = metadata.poseTimestamp - principalTime;

  const cameraAt = (time: number, world: number[]) => {
    const q = world.map((value, index) => value - cameraPosition[index] - time * cameraVelocity[index]);
    return [0, 1, 2].map((row) =>
      (cameraFromWorld[row * 3] + time * rotationRate[row * 3]) * q[0]
      + (cameraFromWorld[row * 3 + 1] + time * rotationRate[row * 3 + 1]) * q[1]
      + (cameraFromWorld[row * 3 + 2] + time * rotationRate[row * 3 + 2]) * q[2],
    );
  };

  return (lidarX: number, lidarY: number, lidarZ: number): WaymoProjectedPoint | null => {
    const world = [
      referenceWorldFromLidar[0] * lidarX + referenceWorldFromLidar[1] * lidarY + referenceWorldFromLidar[2] * lidarZ + referenceWorldFromLidar[3],
      referenceWorldFromLidar[4] * lidarX + referenceWorldFromLidar[5] * lidarY + referenceWorldFromLidar[6] * lidarZ + referenceWorldFromLidar[7],
      referenceWorldFromLidar[8] * lidarX + referenceWorldFromLidar[9] * lidarY + referenceWorldFromLidar[10] * lidarZ + referenceWorldFromLidar[11],
    ];
    let time = 0;
    if (metadata.rollingShutterDirection !== 5) for (let iteration = 0; iteration < 4; iteration += 1) {
      const camera = cameraAt(time, world);
      const normalizedX = -camera[1] / camera[0];
      const normalizedY = -camera[2] / camera[0];
      const spacing = horizontal ? normalizedX : normalizedY;
      const residual = time - spacing * readoutFactor + poseOffset;
      const rotation = cameraFromWorld.map((value, index) => value + time * rotationRate[index]);
      const velocityIndex = [0, 1, 2].map((row) =>
        -(rotation[row * 3] * cameraVelocity[0] + rotation[row * 3 + 1] * cameraVelocity[1] + rotation[row * 3 + 2] * cameraVelocity[2])
        - (skewOmega[row * 3] * camera[0] + skewOmega[row * 3 + 1] * camera[1] + skewOmega[row * 3 + 2] * camera[2]),
      );
      const combined = horizontal
        ? normalizedX * velocityIndex[0] - velocityIndex[1]
        : normalizedY * velocityIndex[0] - velocityIndex[2];
      const jacobian = 1 - readoutFactor / camera[0] * combined;
      if (!Number.isFinite(jacobian) || Math.abs(jacobian) <= 1e-9) return null;
      time -= residual / jacobian;
    }
    const camera = cameraAt(time, world);
    if (!(camera[0] > 0.1) || !Number.isFinite(time)) return null;
    const normalizedX = -camera[1] / camera[0];
    const normalizedY = -camera[2] / camera[0];
    const [k1 = 0, k2 = 0, , , k3 = 0] = distortion;
    const radius2 = normalizedX * normalizedX + normalizedY * normalizedY;
    const radial = 1 + k1 * radius2 + k2 * radius2 ** 2 + k3 * radius2 ** 3;
    // This validity range is part of Waymo CameraModel::DirectionToImage.
    // It prevents an untrustworthy RadTan polynomial from folding side/rear
    // LiDAR directions back into the FRONT image as wave-shaped aliases.
    if (radial < MIN_TRUSTED_RADIAL_DISTORTION || radial > MAX_TRUSTED_RADIAL_DISTORTION) return null;
    const [distortedX, distortedY] = distort(normalizedX, normalizedY, distortion);
    const x = intrinsic[0] * distortedX + intrinsic[2];
    const y = intrinsic[4] * distortedY + intrinsic[5];
    return Number.isFinite(x) && Number.isFinite(y) ? { x, y, depth: camera[0] } : null;
  };
}
