import * as THREE from './vendor/three/build/three.module.js';
import { OrbitControls } from './vendor/three/examples/jsm/controls/OrbitControls.js';
import { PLYLoader } from './vendor/three/examples/jsm/loaders/PLYLoader.js';
import { OBJLoader } from './vendor/three/examples/jsm/loaders/OBJLoader.js';

const viewer = document.querySelector('[data-viewer]');
const canvas = document.querySelector('[data-viewer-canvas]');
const statusEl = document.querySelector('[data-viewer-status]');
const buttons = Array.from(document.querySelectorAll('[data-viewer-asset]'));

const assets = {
  sugar: {
    label: 'SuGaR textured OBJ',
    type: 'obj',
    path: '../outputs/sugar/truck/sugarfine_3Dgs7000_densityestim02_sdfnorm02_level03_decim200000_normalconsistency01_gaussperface6.obj',
    texture: '../outputs/sugar/truck/sugarfine_3Dgs7000_densityestim02_sdfnorm02_level03_decim200000_normalconsistency01_gaussperface6.png',
    rotateX: Math.PI,
    groundNormal: [0.03886, 0.99595, 0.08104],
  },
  gof: {
    label: 'GOF TSDF mesh',
    type: 'ply-mesh',
    path: '../outputs/gof/truck/gof_truck_tsdf_7000.ply',
    rotateX: Math.PI,
  },
  opensplat: {
    label: 'OpenSplat Gaussian centers',
    type: 'sample-points',
    sampleKey: 'opensplat',
    rotateX: Math.PI,
    groundNormal: [0.09907, 0.99371, 0.0523],
  },
};

const scene = new THREE.Scene();
scene.background = new THREE.Color(0xf4f6f8);

const camera = new THREE.PerspectiveCamera(45, 16 / 9, 0.01, 1000);
camera.position.set(0, 0.35, 3.2);

let renderer = null;
let controls = null;

try {
  renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: false });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.outputColorSpace = THREE.SRGBColorSpace;

  controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.autoRotate = true;
  controls.autoRotateSpeed = 0.45;
} catch (error) {
  console.error(error);
  setStatus('WebGL unavailable: interactive viewer needs browser WebGL support');
  buttons.forEach((button) => {
    button.disabled = true;
    button.classList.remove('active');
  });
  canvas.style.background = '#e8edf2';
}

const groundY = -0.92;

scene.add(new THREE.HemisphereLight(0xffffff, 0x94a3b8, 2.1));
const keyLight = new THREE.DirectionalLight(0xffffff, 1.6);
keyLight.position.set(3, 5, 4);
scene.add(keyLight);

const ground = new THREE.GridHelper(3.2, 16, 0xcbd5df, 0xe1e7ee);
ground.position.y = groundY;
scene.add(ground);

let activeObject = null;
let activeKey = null;

function setStatus(text) {
  statusEl.textContent = text;
}

function resize() {
  const rect = viewer.getBoundingClientRect();
  const width = Math.max(320, Math.floor(rect.width));
  const height = Math.max(320, Math.floor(width * 0.56));
  renderer.setSize(width, height, false);
  camera.aspect = width / height;
  camera.updateProjectionMatrix();
}

function normalizeObject(object) {
  const box = new THREE.Box3().setFromObject(object);
  const size = new THREE.Vector3();
  const center = new THREE.Vector3();
  box.getSize(size);
  box.getCenter(center);
  const maxDim = Math.max(size.x, size.y, size.z, 1e-6);
  const scale = 1.85 / maxDim;
  object.scale.multiplyScalar(scale);
  object.position.addScaledVector(center, -scale);

  const alignedBox = new THREE.Box3().setFromObject(object);
  object.position.y += groundY - alignedBox.min.y;

  const finalBox = new THREE.Box3().setFromObject(object);
  const finalCenter = new THREE.Vector3();
  finalBox.getCenter(finalCenter);
  controls.target.copy(finalCenter);
  camera.position.set(0, 0.35, 3.2);
  camera.near = 0.01;
  camera.far = 100;
  camera.updateProjectionMatrix();
  controls.update();
}

function normalizeSampleObject(object) {
  const box = new THREE.Box3().setFromObject(object);
  const alignedBox = new THREE.Box3().setFromObject(object);
  object.position.y += groundY - alignedBox.min.y;

  const finalBox = new THREE.Box3().setFromObject(object);
  const finalCenter = new THREE.Vector3();
  finalBox.getCenter(finalCenter);
  controls.target.copy(finalCenter);
  camera.position.set(0, 0.35, 3.2);
  camera.near = 0.01;
  camera.far = 100;
  camera.updateProjectionMatrix();
  controls.update();
}

function applyAssetTransform(object, asset) {
  if (asset.rotateX) object.rotation.x += asset.rotateX;
  if (asset.rotateY) object.rotation.y += asset.rotateY;
  if (asset.rotateZ) object.rotation.z += asset.rotateZ;
  if (asset.groundNormal) {
    const normal = new THREE.Vector3(...asset.groundNormal).normalize();
    const up = new THREE.Vector3(0, 1, 0);
    const leveling = new THREE.Quaternion().setFromUnitVectors(normal, up);
    object.quaternion.premultiply(leveling);
  }
  object.updateMatrixWorld(true);
}

function makePointObject(geometry, stride = 1) {
  convertFloat64Attributes(geometry);
  let source = geometry;
  if (stride > 1 && geometry.attributes.position) {
    const pos = geometry.attributes.position.array;
    const colorAttr = geometry.attributes.color;
    const count = geometry.attributes.position.count;
    const keep = Math.ceil(count / stride);
    const nextPos = new Float32Array(keep * 3);
    const nextColor = new Float32Array(keep * 3);
    let out = 0;
    for (let i = 0; i < count; i += stride) {
      nextPos[out * 3] = pos[i * 3];
      nextPos[out * 3 + 1] = pos[i * 3 + 1];
      nextPos[out * 3 + 2] = pos[i * 3 + 2];
      if (colorAttr) {
        nextColor[out * 3] = colorAttr.array[i * 3];
        nextColor[out * 3 + 1] = colorAttr.array[i * 3 + 1];
        nextColor[out * 3 + 2] = colorAttr.array[i * 3 + 2];
      } else {
        nextColor[out * 3] = 0.72;
        nextColor[out * 3 + 1] = 0.72;
        nextColor[out * 3 + 2] = 0.72;
      }
      out += 1;
    }
    source = new THREE.BufferGeometry();
    source.setAttribute('position', new THREE.BufferAttribute(nextPos, 3));
    source.setAttribute('color', new THREE.BufferAttribute(nextColor, 3));
  } else if (!geometry.attributes.color) {
    const count = geometry.attributes.position.count;
    const colors = new Float32Array(count * 3);
    colors.fill(0.72);
    source.setAttribute('color', new THREE.BufferAttribute(colors, 3));
  }
  source.computeBoundingSphere();
  return new THREE.Points(
    source,
    new THREE.PointsMaterial({ size: 0.008, vertexColors: true, sizeAttenuation: true })
  );
}

function makeSamplePointObject(asset) {
  const sample = window.INTERACTIVE_POINTS?.scenes?.[asset.sampleKey];
  if (!sample) throw new Error(`Missing sample point asset: ${asset.sampleKey}`);

  const range = window.INTERACTIVE_POINTS.range || 1.35;
  const pointBytes = Uint8Array.from(atob(sample.points), (char) => char.charCodeAt(0)).buffer;
  const colorBytes = Uint8Array.from(atob(sample.colors), (char) => char.charCodeAt(0));
  const qPoints = new Uint16Array(pointBytes);
  const positions = new Float32Array(sample.count * 3);
  const colors = new Float32Array(sample.count * 3);

  for (let i = 0; i < sample.count * 3; i += 1) {
    positions[i] = (qPoints[i] / 65535) * range * 2 - range;
    colors[i] = colorBytes[i] / 255;
  }

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
  geometry.computeBoundingSphere();

  return new THREE.Points(
    geometry,
    new THREE.PointsMaterial({ size: 0.012, vertexColors: true, sizeAttenuation: true })
  );
}

function applyShDcColors(geometry, attrName = 'shDc') {
  const shDc = geometry.attributes[attrName];
  const position = geometry.attributes.position;
  if (!shDc || !position || shDc.itemSize < 3) return;

  const sh0 = 0.28209479177387814;
  const colors = new Float32Array(position.count * 3);
  for (let i = 0; i < position.count; i += 1) {
    colors[i * 3] = THREE.MathUtils.clamp(shDc.array[i * shDc.itemSize] * sh0 + 0.5, 0, 1);
    colors[i * 3 + 1] = THREE.MathUtils.clamp(shDc.array[i * shDc.itemSize + 1] * sh0 + 0.5, 0, 1);
    colors[i * 3 + 2] = THREE.MathUtils.clamp(shDc.array[i * shDc.itemSize + 2] * sh0 + 0.5, 0, 1);
  }
  geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
}

function makeMeshObject(geometry) {
  convertFloat64Attributes(geometry);
  geometry.computeVertexNormals();
  const hasColor = Boolean(geometry.attributes.color);
  const material = new THREE.MeshStandardMaterial({
    color: hasColor ? 0xffffff : 0xb7c2cf,
    vertexColors: hasColor,
    roughness: 0.82,
    metalness: 0.0,
    side: THREE.DoubleSide,
  });
  return new THREE.Mesh(geometry, material);
}

function convertFloat64Attributes(geometry) {
  Object.entries(geometry.attributes).forEach(([name, attribute]) => {
    if (attribute.array instanceof Float64Array) {
      geometry.setAttribute(
        name,
        new THREE.BufferAttribute(new Float32Array(attribute.array), attribute.itemSize, attribute.normalized)
      );
    }
  });
}

function disposeObject(object) {
  if (!object) return;
  object.traverse((child) => {
    if (child.geometry) child.geometry.dispose();
    if (child.material) {
      const materials = Array.isArray(child.material) ? child.material : [child.material];
      materials.forEach((material) => {
        if (material.map) material.map.dispose();
        material.dispose();
      });
    }
  });
}

async function loadAsset(key) {
  const asset = assets[key];
  if (!asset || key === activeKey) return;
  activeKey = key;
  buttons.forEach((button) => button.classList.toggle('active', button.dataset.viewerAsset === key));
  setStatus(`Loading ${asset.label}...`);

  const previous = activeObject;
  activeObject = null;
  if (previous) {
    scene.remove(previous);
    disposeObject(previous);
  }

  try {
    let object;
    if (asset.type === 'obj') {
      const texture = await new THREE.TextureLoader().loadAsync(asset.texture);
      texture.colorSpace = THREE.SRGBColorSpace;
      object = await new OBJLoader().loadAsync(asset.path);
      object.traverse((child) => {
        if (child.isMesh) {
          child.material = new THREE.MeshStandardMaterial({
            map: texture,
            roughness: 0.86,
            metalness: 0,
            side: THREE.DoubleSide,
          });
        }
      });
    } else if (asset.type === 'sample-points') {
      object = makeSamplePointObject(asset);
    } else {
      const loader = new PLYLoader();
      if (asset.shDc) loader.setCustomPropertyNameMapping({ shDc: asset.shDc });
      const geometry = await loader.loadAsync(asset.path);
      if (asset.shDc) applyShDcColors(geometry);
      object = asset.type === 'ply-points' ? makePointObject(geometry, asset.stride || 1) : makeMeshObject(geometry);
    }
    applyAssetTransform(object, asset);
    activeObject = object;
    scene.add(object);
    if (asset.type === 'sample-points') normalizeSampleObject(object);
    else normalizeObject(object);
    setStatus(`${asset.label} loaded`);
  } catch (error) {
    console.error(error);
    setStatus(`Failed to load ${asset.label}`);
  }
}

function animate() {
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
}

buttons.forEach((button) => button.addEventListener('click', () => loadAsset(button.dataset.viewerAsset)));
if (renderer && controls) {
  window.addEventListener('resize', resize);
  resize();
  animate();
  loadAsset('sugar');
}
