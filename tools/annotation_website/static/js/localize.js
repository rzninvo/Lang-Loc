// Three.js first-person localizer.
//
// Loads the scene mesh, drops the user in at the centroid at fixed eye
// height (1.6 m), and lets them walk around with WASD + mouse-look in
// pointer lock. On submit, POSTs (x, y, z, yaw) IN MESH COORDS to
// /api/localize/submit. The server computes the error against the
// keyframe's stored GT pose; the user then sees their own error.
//
// We deliberately do not send the GT pose to the client — otherwise
// it would be trivial to cheat by reading the JSON in dev-tools.
//
// Coordinate-frame fix: ScanNet / 3RScan meshes are z-up, but
// Three.js's standard controls (PointerLockControls, default camera)
// assume y-up. To use those battle-tested controls without re-deriving
// the conventions ourselves, we rotate the mesh -π/2 around X on load,
// so mesh-z becomes Three.js-y. At submit time we convert the camera
// pose in Three.js coords back to mesh coords:
//
//   mesh_x = three_x
//   mesh_y = -three_z
//   mesh_z = three_y   (locked at 1.6 m)
//   mesh_yaw = atan2(-threeFwd_z, threeFwd_x)

import * as THREE from "three";
import { PLYLoader } from "three/addons/loaders/PLYLoader.js";
import { PointerLockControls } from "three/addons/controls/PointerLockControls.js";

const root = document.getElementById("localize-root");
const viewerEl = document.getElementById("localize-viewer");
const statusEl = document.getElementById("loc-status");
const submitBtn = document.getElementById("localize-submit-button");
const skipBtn = document.getElementById("localize-skip-button");
const resultEl = document.getElementById("localize-result");
const respDist = document.getElementById("resp-dist");
const respAngle = document.getElementById("resp-angle");
const respIou = document.getElementById("resp-iou");
const rdx = document.getElementById("rdx");
const rdy = document.getElementById("rdy");
const rdyaw = document.getElementById("rdyaw");

const sceneId = root.dataset.scene;
const frameId = root.dataset.frame;
const meshUrl = root.dataset.meshUrl;
const eyeHeight = parseFloat(root.dataset.eyeHeight) || 1.6;
const editMode = root.dataset.editMode === "1";
const promptAnnotatorId = root.dataset.promptAnnotator || null;
// Per-dataset camera FoV (paper supp Tab. 7 / configs/localization/*.yaml).
// ScanNet 58.30°×45.33°  ·  3RScan 39.31°×64.76° (portrait).
const hFovDeg = parseFloat(root.dataset.hFovDeg) || 60.0;
const vFovDeg = parseFloat(root.dataset.vFovDeg) || 45.0;
const paperAspect = hFovDeg / vFovDeg;
const existingPose = root.dataset.existingX !== undefined
  ? {
      x: parseFloat(root.dataset.existingX),
      y: parseFloat(root.dataset.existingY),
      z: parseFloat(root.dataset.existingZ),
      yaw: parseFloat(root.dataset.existingYaw),
    }
  : null;

function setStatus(msg, kind) {
  statusEl.textContent = msg;
  statusEl.className = "save-status" + (kind ? " " + kind : "");
}

// ---------------------------------------------------------------------------
// Coord conversions: mesh (z-up) ↔ three (y-up)
// ---------------------------------------------------------------------------
// Mesh is parented under `meshGroup` which is rotated -π/2 around X.
// That makes mesh-z map to three-y (up) and mesh-y map to three-(-z).
function meshToThree(mx, my, mz) {
  return new THREE.Vector3(mx, mz, -my);
}
function threeToMesh(tx, ty, tz) {
  return { x: tx, y: -tz, z: ty };
}
// Convert a yaw around mesh-z into the equivalent yaw around three-y.
// In mesh: forward = (cos yaw, sin yaw, 0). After our rotation:
//   threeFwd = (cos yaw, 0, -sin yaw).
// Three.js PointerLockControls' getObject().rotation.y is the yaw
// around three's y-axis. Three's yaw=0 means looking along three-(-z).
// For mesh forward = (cos θ, sin θ, 0) we need three forward = (cos θ, 0, -sin θ),
// which is three.yaw = π/2 + θ around three-y (where three.yaw=0 looks -z).
// Actually atan2(threeFwd.x, -threeFwd.z) gives three.yaw.
// Easier: at submit time, read camera.getWorldDirection(threeFwd) and
// compute mesh yaw = atan2(-threeFwd.z, threeFwd.x).
function meshYawFromThree(threeFwd) {
  return Math.atan2(-threeFwd.z, threeFwd.x);
}

// ---------------------------------------------------------------------------
// Three.js scene
// ---------------------------------------------------------------------------
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0a0a14);

const camera = new THREE.PerspectiveCamera(vFovDeg, paperAspect, 0.05, 200);
// We use Three.js's default y-up convention now (camera.up = (0,1,0)).

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(window.devicePixelRatio);
viewerEl.appendChild(renderer.domElement);

function fitCanvasToCard() {
  const cw = viewerEl.clientWidth;
  const ch = viewerEl.clientHeight;
  let w, h;
  if (cw / ch > paperAspect) {
    h = ch; w = Math.round(ch * paperAspect);
  } else {
    w = cw; h = Math.round(cw / paperAspect);
  }
  renderer.setSize(w, h, false);
  renderer.domElement.style.width = w + "px";
  renderer.domElement.style.height = h + "px";
  renderer.domElement.style.display = "block";
  renderer.domElement.style.margin = "0 auto";
  camera.aspect = paperAspect;
  camera.fov = vFovDeg;
  camera.updateProjectionMatrix();
}
fitCanvasToCard();
window.addEventListener("resize", fitCanvasToCard);
document.addEventListener("fullscreenchange", () => {
  // resize once the browser has settled into / out of fullscreen
  setTimeout(fitCanvasToCard, 50);
});

// Lights — meshes are vertex-coloured (.ply has rgb per vertex). Add a
// soft ambient so non-coloured meshes are still legible.
scene.add(new THREE.AmbientLight(0xffffff, 0.7));
const sun = new THREE.DirectionalLight(0xffffff, 0.5);
sun.position.set(5, 10, 5);
scene.add(sun);

// Group that holds the mesh, rotated so mesh-z (up) → three-y (up).
const meshGroup = new THREE.Group();
meshGroup.rotation.x = -Math.PI / 2;
scene.add(meshGroup);

// Reference floor grid in three-space (mesh z=0 floor → three y=0).
const grid = new THREE.GridHelper(20, 40, 0x444466, 0x222233);
grid.position.y = 0;
scene.add(grid);

// ---------------------------------------------------------------------------
// PointerLockControls (battle-tested) + WASD
// ---------------------------------------------------------------------------
const controls = new PointerLockControls(camera, renderer.domElement);
scene.add(controls.getObject());

renderer.domElement.addEventListener("click", () => {
  if (!controls.isLocked) controls.lock();
});
controls.addEventListener("lock", () => setStatus("Move with WASD · esc to release"));
controls.addEventListener("unlock", () => setStatus("Click viewer to walk"));

const keys = { w: false, a: false, s: false, d: false, shift: false };
window.addEventListener("keydown", (e) => {
  const k = e.key.toLowerCase();
  if (k in keys) keys[k] = true;
  if (e.key === "Shift") keys.shift = true;
  if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
    if (!submitBtn.disabled) submitBtn.click();
  }
  // F → fullscreen toggle (also exposed as a button)
  if (k === "f" && !e.target.matches("input, textarea")) {
    toggleFullscreen();
  }
});
window.addEventListener("keyup", (e) => {
  const k = e.key.toLowerCase();
  if (k in keys) keys[k] = false;
  if (e.key === "Shift") keys.shift = false;
});

const moveSpeed = 1.5;
const sprintMult = 2.5;
const clock = new THREE.Clock();
const fwdTmp = new THREE.Vector3();

function updateMovement(dt) {
  if (!controls.isLocked) return;
  const speed = (keys.shift ? sprintMult : 1.0) * moveSpeed * dt;
  if (keys.w) controls.moveForward(speed);
  if (keys.s) controls.moveForward(-speed);
  if (keys.d) controls.moveRight(speed);
  if (keys.a) controls.moveRight(-speed);
  // lock eye height: three-y maps to mesh-z, and we want mesh-z = 1.6 m
  controls.getObject().position.y = eyeHeight;
}

function currentMeshPose() {
  const p = controls.getObject().position;
  const meshPos = threeToMesh(p.x, p.y, p.z);
  camera.getWorldDirection(fwdTmp);
  const meshYaw = meshYawFromThree(fwdTmp);
  return { ...meshPos, yaw: meshYaw };
}

function updateReadout() {
  const mp = currentMeshPose();
  rdx.textContent = mp.x.toFixed(2);
  rdy.textContent = mp.y.toFixed(2);
  rdyaw.textContent = ((mp.yaw * 180) / Math.PI).toFixed(0);
}

// ---------------------------------------------------------------------------
// Fullscreen toggle
// ---------------------------------------------------------------------------
function toggleFullscreen() {
  if (!document.fullscreenElement) {
    viewerEl.requestFullscreen?.().catch(() => {});
  } else {
    document.exitFullscreen?.().catch(() => {});
  }
}
// expose a global so an in-template button can call this
window.__langlocToggleFullscreen = toggleFullscreen;

// ---------------------------------------------------------------------------
// PLY load → place camera at scene centroid (or restore previous pose)
// ---------------------------------------------------------------------------
setStatus("Loading mesh…");
const loader = new PLYLoader();
loader.load(
  meshUrl,
  (geometry) => {
    geometry.computeBoundingBox();
    geometry.computeVertexNormals();
    const matOpts = geometry.hasAttribute("color")
      ? { vertexColors: true, side: THREE.DoubleSide }
      : { color: 0x9999aa, side: THREE.DoubleSide };
    const mesh = new THREE.Mesh(geometry, new THREE.MeshLambertMaterial(matOpts));
    meshGroup.add(mesh);

    // bbox is in mesh frame (z-up). Centre of the floor plane:
    const bb = geometry.boundingBox;
    const cx = 0.5 * (bb.min.x + bb.max.x);
    const cy = 0.5 * (bb.min.y + bb.max.y);

    if (editMode && existingPose) {
      const t = meshToThree(existingPose.x, existingPose.y, eyeHeight);
      controls.getObject().position.copy(t);
      // restore yaw: we want mesh-yaw = existingPose.yaw, which means
      // three.yaw such that meshYawFromThree(getWorldDirection()) = existingPose.yaw.
      // Three.js PLC stores yaw on object.rotation.y; three.yaw=0 looks -z.
      // mesh-yaw=θ means three forward = (cos θ, 0, -sin θ); three.yaw = atan2(threeFwd.x, -threeFwd.z) = atan2(cos θ, sin θ) = π/2 − θ.
      const threeYaw = Math.PI / 2 - existingPose.yaw;
      const obj = controls.getObject();
      obj.rotation.set(0, threeYaw, 0);
      // pitch is on the inner camera in PLC; default to level
      camera.rotation.set(0, 0, 0);
    } else {
      const t = meshToThree(cx, cy, eyeHeight);
      controls.getObject().position.copy(t);
      controls.getObject().rotation.set(0, 0, 0);
      camera.rotation.set(0, 0, 0);
    }

    submitBtn.disabled = false;
    setStatus("Click viewer to walk");
    updateReadout();
  },
  (xhr) => {
    if (xhr.total) {
      const pct = ((xhr.loaded / xhr.total) * 100).toFixed(0);
      setStatus(`Loading mesh… ${pct}%`);
    }
  },
  (err) => {
    setStatus("Could not load mesh — try refreshing", "error");
    console.error(err);
  },
);

function tick() {
  const dt = Math.min(clock.getDelta(), 0.1);
  updateMovement(dt);
  updateReadout();
  renderer.render(scene, camera);
  requestAnimationFrame(tick);
}
tick();

// ---------------------------------------------------------------------------
// Submit
// ---------------------------------------------------------------------------
const t0 = Date.now();
if (skipBtn) {
  skipBtn.addEventListener("click", async () => {
    skipBtn.disabled = true;
    setStatus("Skipping…", "saving");
    try {
      const res = await fetch("/api/localize/skip", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ scene_id: sceneId, frame_id: frameId }),
      });
      if (!res.ok) {
        setStatus("Could not skip — try again", "error");
        skipBtn.disabled = false;
        return;
      }
    } catch (e) {
      setStatus("Network error", "error");
      skipBtn.disabled = false;
      return;
    }
    // unlock pointer + navigate to next assignment
    controls.unlock();
    window.location.href = "/localize";
  });
}

submitBtn.addEventListener("click", async () => {
  submitBtn.disabled = true;
  const mp = currentMeshPose();
  const payload = {
    scene_id: sceneId,
    frame_id: frameId,
    pred_x: mp.x,
    pred_y: mp.y,
    pred_z: eyeHeight,         // locked
    pred_yaw: mp.yaw,
    duration_ms: Date.now() - t0,
    prompt_annotator_id: promptAnnotatorId,
  };
  setStatus("Saving…", "saving");
  try {
    const res = await fetch("/api/localize/submit", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!res.ok) {
      const body = await res.json().catch(() => ({ detail: "submit failed" }));
      setStatus("Error: " + (body.detail || res.statusText), "error");
      submitBtn.disabled = false;
      return;
    }
    const data = await res.json();
    setStatus("Saved", "ok");
    if (typeof data.distance_error === "number") {
      respDist.textContent = data.distance_error.toFixed(2);
    }
    if (typeof data.angular_error_deg === "number") {
      respAngle.textContent = data.angular_error_deg.toFixed(1);
    }
    if (typeof data.iou === "number") {
      respIou.textContent = data.iou.toFixed(3);
    } else {
      respIou.textContent = "—";
    }
    resultEl.hidden = false;
    controls.unlock();
  } catch (e) {
    setStatus("Network error", "error");
    submitBtn.disabled = false;
  }
});
