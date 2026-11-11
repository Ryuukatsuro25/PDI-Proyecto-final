/* =====================================================================
   Coincidencia emocional entre obra y visitante — GUI
   Backend FastAPI (mismo origen):
     GET  /api/estado     · GET  /api/metricas
     POST /api/frame      · POST /api/analizar
     GET  /api/dataset/estado | /api/dataset/aleatorias | /api/dataset/imagen
     POST /api/dataset/analizar | /api/dataset/recargar   (carpeta precarga-sin-analizar)
   ===================================================================== */
"use strict";

const API = location.protocol.startsWith("http") ? "" : "http://localhost:8000";
const CFG = {
  MAX_MB: 10,
  LIVE_ANCHO: 480,
  LIVE_PAUSA_MS: 120,
  OBS_ANCHO: 640,
  OBS_INTERVALO_MS: 400,
  PREP_SEG: 3,
  TIMEOUT_MS: 60000,
  FETCH_TIMEOUT_MS: 15000,
  FETCH_RETRIES: 2,
};

const EMO = {
  felicidad: { es: "Felicidad", color: "#eda100" },
  sorpresa:  { es: "Sorpresa",  color: "#e87ba4" },
  neutral:   { es: "Neutral",   color: "#9aa0b4" },
  tristeza:  { es: "Tristeza",  color: "#2a78d6" },
  miedo:     { es: "Miedo",     color: "#6250d6" },
  enojo:     { es: "Enojo",     color: "#e34948" },
  asco:      { es: "Asco",      color: "#008300" },
  desprecio: { es: "Desprecio", color: "#eb6834" },
};
const MODULOS = [
  ["image_loader","ImageLoader"],["preprocesamiento","PreprocesamientoImagen"],["cnn","ClasificadorDeObras"],
  ["gradcam","Grad-CAM"],["paleta","ExtractorDePaleta"],["gestos","AnalizarGestosFaciales"],
  ["resultado","ResultadoProceso"],["integracion","Integración"],
];
const ICONOS = { servidor:"🛠️", rostro_no_detectado:"🙈", formato:"🖼️", archivo_corrupto:"🧩", archivo_ausente:"📭",
  calidad_insuficiente:"🌫️", fuera_de_rango:"📏", no_reconocido:"❓", entrada_incompleta:"🧷", multiples_rostros:"👥",
  dataset_no_encontrado:"📁", dataset_no_disponible:"📭",
  imagen_no_encontrada:"🔍", imagen_conocida:"🎓", camara:"📷" };
const TITULOS = { servidor:"Error del servidor", rostro_no_detectado:"Rostro no detectado", formato:"Formato no soportado",
  archivo_corrupto:"Archivo dañado", archivo_ausente:"Archivo ausente", calidad_insuficiente:"Calidad de imagen insuficiente",
  fuera_de_rango:"Valor fuera de rango", no_reconocido:"Elemento no reconocido", entrada_incompleta:"Entrada incompleta",
  multiples_rostros:"Varias personas en cuadro", dataset_no_encontrado:"Carpeta de precarga no encontrada",
  dataset_no_disponible:"Carpeta de precarga vacía", imagen_no_encontrada:"Imagen no encontrada",
  imagen_conocida:"Imagen conocida por la CNN" };
const gravedadCls = g => g === "error" ? "error" : g === "info" ? "info" : "aviso";

const $ = id => document.getElementById(id);
const state = {
  obra: null, obraURL: null, stream: null, frames: [], resultado: null, servidorOk: false,
  live: false, liveTimer: null, observando: false, cancelarObs: null, ultimoLive: null,
  analizando: false, enfocarAlCerrarObs: null,
};

/* ============================================================ utilidades */
function el(tag, cls, text) { const e = document.createElement(tag); if (cls) e.className = cls; if (text != null) e.textContent = text; return e; }
function pct(v, d = 0) { return (Number(v || 0) * 100).toFixed(d) + "%"; }
function mostrarError(id, txt) { const n = $(id); n.textContent = txt || ""; n.hidden = !txt; }
function esc(s) { return String(s).replace(/[&<>"']/g, c => ({ "&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;" }[c])); }
function pill(id, txt, cls) { const p = $(id); p.textContent = txt; p.className = "pill " + (cls || ""); }
const sleep = ms => new Promise(r => setTimeout(r, ms));

/* sleep cancelable: responde rápido a la cancelación */
function sleepCancel(ms, cancelCheck) {
  return new Promise(resolve => {
    const start = performance.now();
    const id = setInterval(() => {
      if (cancelCheck() || performance.now() - start >= ms) { clearInterval(id); resolve(); }
    }, 50);
  });
}

/* Toast flotante */
function toast(msg, tipo = "", duracion = 3500) {
  let cont = $("toasts");
  if (!cont) {
    cont = el("div", "toast-container"); cont.id = "toasts";
    cont.setAttribute("role", "status"); cont.setAttribute("aria-live", "polite");
    document.body.append(cont);
  }
  const t = el("div", "toast" + (tipo ? " " + tipo : ""), msg);
  cont.append(t);
  setTimeout(() => {
    t.classList.add("out");
    t.addEventListener("animationend", () => t.remove(), { once: true });
    setTimeout(() => t.remove(), 500);
  }, duracion);
}

/* Anuncio a lectores de pantalla */
let _announceNode;
function announce(msg) {
  if (!_announceNode) {
    _announceNode = el("div");
    _announceNode.setAttribute("aria-live", "polite");
    _announceNode.setAttribute("aria-atomic", "true");
    _announceNode.style.cssText = "position:absolute;left:-9999px;width:1px;height:1px;overflow:hidden";
    document.body.append(_announceNode);
  }
  _announceNode.textContent = "";
  setTimeout(() => { _announceNode.textContent = msg; }, 60);
}

/* fetch con timeout + reintentos con backoff exponencial */
async function fetchRetry(url, opts = {}, { retries = CFG.FETCH_RETRIES, timeout = CFG.FETCH_TIMEOUT_MS } = {}) {
  let lastErr;
  for (let i = 0; i <= retries; i++) {
    const ctrl = new AbortController();
    const to = setTimeout(() => ctrl.abort(), timeout);
    try {
      const r = await fetch(url, { ...opts, signal: ctrl.signal });
      clearTimeout(to);
      if (r.status >= 500 && i < retries) {
        lastErr = new Error("HTTP " + r.status);
        await sleep(300 * Math.pow(2, i));
        continue;
      }
      return r;
    } catch (e) {
      clearTimeout(to);
      lastErr = e;
      if (i < retries) await sleep(300 * Math.pow(2, i));
    }
  }
  throw lastErr;
}

function irAPaso(n) {
  [1,2,3,4].forEach(i => $("paso" + i).hidden = i !== n);
  [...$("stepper").children].forEach((li, i) => {
    li.classList.toggle("active", i + 1 === n);
    li.classList.toggle("done", i + 1 < n);
    li.classList.remove("skip");
  });
  if (n === 4 && state.resultado && !state.resultado.excepciones.some(x => x.gravedad !== "info"))
    $("stepper").children[2].classList.add("skip");
  const h = $("t" + n);
  if (h) { h.setAttribute("tabindex", "-1"); h.focus({ preventScroll: true }); }
  window.scrollTo({ top: 0, behavior: "smooth" });
  announce(`Paso ${n} de 4`);
}

/* tooltip compartido */
const tip = $("tooltip");
document.addEventListener("mousemove", e => {
  const t = e.target.closest("[data-tip]");
  if (!t) { tip.hidden = true; return; }
  tip.innerHTML = t.dataset.tip; tip.hidden = false;
  const x = Math.min(e.clientX + 14, innerWidth - tip.offsetWidth - 8);
  tip.style.left = x + "px"; tip.style.top = (e.clientY + 14) + "px";
});

/* ============================================================ estado del servidor */
async function cargarEstado() {
  try {
    const r = await fetchRetry(API + "/api/estado");
    const s = await r.json();
    state.servidorOk = true;
    const set = (k, ok, t) => { const li = document.querySelector(`#status [data-k="${k}"]`); li.className = ok ? "ok" : "bad"; if (t) li.title = t; };
    set("cnn", s.cnn.disponible, s.cnn.error || "CNN de obras entrenada");
    set("det", !!s.rostro.detector, s.rostro.detector);
    set("emo", s.rostro.emociones, s.rostro.errores.join(" "));
    const avisos = [];
    if (!s.cnn.disponible) avisos.push(s.cnn.error);
    if (!s.rostro.emociones) avisos.push(...s.rostro.errores);
    if (avisos.length) { $("bannerServidor").textContent = "Atención: " + avisos.join(" "); $("bannerServidor").hidden = false; }
  } catch {
    state.servidorOk = false;
    $("bannerServidor").innerHTML = "No se pudo contactar al servidor. Ejecuta el notebook completo (Ejecutar todo) y abre el enlace de la última celda.";
    $("bannerServidor").hidden = false;
    document.querySelectorAll("#status li").forEach(li => li.className = "bad");
  }
}

/* ============================================================ PASO 1a — ImageLoader */
function validarArchivo(f) {
  if (!["image/jpeg","image/png"].includes(f.type)) return "Formato no soportado. Ingresa una imagen JPG o PNG.";
  if (f.size > CFG.MAX_MB * 1024 * 1024) return `La imagen pesa más de ${CFG.MAX_MB} MB. Elige una más ligera.`;
  return null;
}
function setObra(file) {
  mostrarError("errObra");
  const err = validarArchivo(file);
  if (err) { mostrarError("errObra", err); toast(err, "err"); return; }
  if (state.obraURL) URL.revokeObjectURL(state.obraURL);
  const url = URL.createObjectURL(file);
  const img = $("imgObra");
  img.onerror = () => { quitarObra(); mostrarError("errObra", "No se pudo leer la imagen. Prueba con otro archivo JPG o PNG."); toast("No se pudo leer la imagen.", "err"); };
  img.onload = () => {
    state.obra = file; state.obraURL = url;
    $("drop").hidden = true; $("prevObra").hidden = false; $("btnQuitarObra").hidden = false;
    $("metaObra").textContent = `${file.name} · ${img.naturalWidth}×${img.naturalHeight} px · ${(file.size / 1024 / 1024).toFixed(2)} MB`;
    $("metaObra").hidden = false;
    pill("pillObra", "Lista", "ok");
    actualizarBoton();
    toast("Obra cargada.", "ok", 1800);
  };
  img.src = url;
}
function quitarObra() {
  if (state.obraURL) { URL.revokeObjectURL(state.obraURL); state.obraURL = null; }
  state.obra = null; $("imgObra").removeAttribute("src"); $("fileObra").value = "";
  $("prevObra").hidden = true; $("drop").hidden = false; $("btnQuitarObra").hidden = true; $("metaObra").hidden = true;
  pill("pillObra", "Pendiente"); actualizarBoton();
}
$("drop").addEventListener("click", () => $("fileObra").click());
$("drop").addEventListener("keydown", e => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); $("fileObra").click(); } });
$("fileObra").addEventListener("change", e => e.target.files[0] && setObra(e.target.files[0]));
["dragover","dragenter"].forEach(ev => $("drop").addEventListener(ev, e => { e.preventDefault(); $("drop").classList.add("over"); }));
["dragleave","drop"].forEach(ev => $("drop").addEventListener(ev, e => { e.preventDefault(); $("drop").classList.remove("over"); }));
$("drop").addEventListener("drop", e => e.dataTransfer.files[0] && setObra(e.dataTransfer.files[0]));
$("btnQuitarObra").addEventListener("click", quitarObra);

function actualizarBoton() {
  const listo = !!(state.obra || state.frames.length);
  $("btnAnalizar").disabled = !listo || state.analizando;
  $("hintAnalizar").textContent = state.obra && state.frames.length
    ? "Obra y rostro listos: se calculará la coincidencia emocional."
    : state.obra ? "Obra lista. Puedes analizarla sola o capturar también tu rostro."
    : state.frames.length ? "Rostro capturado. Carga una obra para calcular la coincidencia."
    : "Carga una obra o captura tu rostro para continuar.";
}

/* ============================================================ PASO 1b — Cámara */
$("chkConsent").addEventListener("change", e => { $("btnCam").disabled = !e.target.checked; });

async function iniciarCamara() {
  mostrarError("errCam");
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    const m = window.isSecureContext === false
      ? "La cámara solo funciona en http://localhost o https://. Abre la página desde http://localhost:8000."
      : "Tu navegador no permite usar la cámara. Usa Chrome, Edge o Firefox actualizados.";
    mostrarError("errCam", m); toast(m, "err"); return;
  }
  try {
    const dispositivos = await navigator.mediaDevices.enumerateDevices();
    if (!dispositivos.some(d => d.kind === "videoinput")) {
      const m = "No se encontró ninguna cámara en este equipo. Conecta una webcam y vuelve a intentarlo.";
      mostrarError("errCam", m); toast(m, "err"); return;
    }
  } catch { /* algunos navegadores no permiten enumerar antes del permiso */ }
  $("btnCam").disabled = true; $("btnCam").textContent = "Solicitando permiso…";
  try {
    state.stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false });
  } catch (e) {
    const msgs = {
      NotAllowedError: "Permiso de cámara denegado. Habilítalo en la configuración del navegador y vuelve a intentarlo.",
      NotFoundError: "No se encontró ninguna cámara en este dispositivo.",
      NotReadableError: "La cámara está siendo usada por otra aplicación. Ciérrala e inténtalo de nuevo.",
      OverconstrainedError: "La cámara no cumple con los requisitos solicitados.",
      SecurityError: "El navegador bloqueó la cámara por seguridad. Abre la página desde http://localhost:8000.",
      AbortError: "La cámara no respondió (puede estar ocupada). Ciérrala en otras aplicaciones e inténtalo de nuevo.",
      TrackStartError: "La cámara está siendo usada por otra aplicación. Ciérrala e inténtalo de nuevo.",
      DevicesNotFoundError: "No se encontró ninguna cámara en este dispositivo.",
      PermissionDeniedError: "Permiso de cámara denegado. Habilítalo en la configuración del navegador y vuelve a intentarlo.",
    };
    const m = msgs[e.name] || `No se pudo activar la cámara (${e.name || "error desconocido"}). Inténtalo de nuevo.`;
    mostrarError("errCam", m); toast(m, "err");
    $("btnCam").disabled = !$("chkConsent").checked; $("btnCam").textContent = "Activar cámara";
    return;
  }
  $("btnCam").textContent = "Activar cámara";
  state.stream.getVideoTracks().forEach(t => t.addEventListener("ended", () => {
    if (!state.stream) return;
    if (state.cancelarObs) state.cancelarObs();
    detenerCamara();
    mostrarError("errCam", "La cámara se desconectó o fue tomada por otra aplicación. Vuelve a activarla.");
  }));
  const v = $("video"); v.srcObject = state.stream; v.hidden = false;
  await new Promise(r => v.readyState >= 2 ? r() : v.addEventListener("loadeddata", r, { once: true }));
  $("camOff").hidden = true; $("liveTag").hidden = false; $("livePanel").hidden = false; $("opts").hidden = false;
  $("btnCam").hidden = true; $("btnObservar").hidden = false; $("btnFoto").hidden = false; $("btnApagar").hidden = false;
  $("consent").hidden = true;
  $("btnObservar").textContent = `Iniciar observación (${$("selDur").value} s)`;
  pill("pillCam", "En vivo", "live");
  iniciarLive();
  toast("Cámara activada.", "ok", 1800);
  announce("Cámara activada");
}
function detenerCamara() {
  detenerLive();
  if (state.stream) { state.stream.getTracks().forEach(t => t.stop()); state.stream = null; }
  const v = $("video"); v.srcObject = null; v.hidden = true;
  $("camOff").hidden = false; $("liveTag").hidden = true; $("livePanel").hidden = true; $("opts").hidden = true;
  $("btnCam").hidden = false; $("btnObservar").hidden = true; $("btnFoto").hidden = true; $("btnApagar").hidden = true;
  $("btnCam").disabled = !$("chkConsent").checked;
  $("consent").hidden = false;
  limpiarOverlay();
  pill("pillCam", state.frames.length ? "Capturado" : "Apagada", state.frames.length ? "ok" : "");
}
$("btnCam").addEventListener("click", iniciarCamara);
$("btnApagar").addEventListener("click", detenerCamara);
$("selDur").addEventListener("change", () => $("btnObservar").textContent = `Iniciar observación (${$("selDur").value} s)`);

function capturarBlob(ancho, calidad = 0.85) {
  const v = $("video");
  if (!v.videoWidth) return Promise.resolve(null);
  const e = Math.min(1, ancho / v.videoWidth);
  const c = document.createElement("canvas");
  c.width = Math.round(v.videoWidth * e); c.height = Math.round(v.videoHeight * e);
  c.getContext("2d").drawImage(v, 0, 0, c.width, c.height);
  return new Promise(r => c.toBlob(b => r(b), "image/jpeg", calidad));
}

/* ---------- Vista en vivo ---------- */
function iniciarLive() { state.live = true; cicloLive(); }
function detenerLive() { state.live = false; clearTimeout(state.liveTimer); }
async function cicloLive() {
  if (!state.live) return;
  if (state.observando || document.hidden || !state.servidorOk) { state.liveTimer = setTimeout(cicloLive, 400); return; }
  const t0 = performance.now();
  try {
    const blob = await capturarBlob(CFG.LIVE_ANCHO, 0.75);
    if (blob) {
      const fd = new FormData(); fd.append("frame", blob, "frame.jpg");
      const r = await fetch(API + "/api/frame", { method: "POST", body: fd });
      const d = await r.json();
      if (state.live && !state.observando) pintarLive(d, performance.now() - t0);
    }
  } catch { $("liveTxt").textContent = "Sin conexión"; }
  state.liveTimer = setTimeout(cicloLive, CFG.LIVE_PAUSA_MS);
}

function geometriaVideo() {
  const v = $("video"), cv = $("overlay");
  const cw = cv.clientWidth, ch = cv.clientHeight, vw = v.videoWidth || 4, vh = v.videoHeight || 3;
  const s = Math.max(cw / vw, ch / vh);
  return { cw, ch, dw: vw * s, dh: vh * s, ox: (cw - vw * s) / 2, oy: (ch - vh * s) / 2 };
}
function limpiarOverlay() { const cv = $("overlay"); cv.getContext("2d").clearRect(0, 0, cv.width, cv.height); }
function pintarLive(d, ms) {
  state.ultimoLive = d;
  const cv = $("overlay"), dpr = devicePixelRatio || 1;
  cv.width = cv.clientWidth * dpr; cv.height = cv.clientHeight * dpr;
  const ctx = cv.getContext("2d"); ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.clearRect(0, 0, cv.width, cv.height);
  const g = geometriaVideo();
  const mapX = x => g.cw - (g.ox + x * g.dw);
  const mapY = y => g.oy + y * g.dh;
  (d.detecciones || []).forEach(det => {
    const esPrincipal = d.principal && det.caja.x === d.principal.caja.x && det.caja.y === d.principal.caja.y;
    const x1 = mapX(det.caja.x + det.caja.w), y1 = mapY(det.caja.y), w = det.caja.w * g.dw, h = det.caja.h * g.dh;
    ctx.lineWidth = esPrincipal ? 3 : 2;
    ctx.strokeStyle = esPrincipal ? "#2F9E6A" : "rgba(255,255,255,.6)";
    ctx.beginPath(); ctx.roundRect(x1, y1, w, h, 6); ctx.stroke();
    (det.puntos || []).forEach(([px, py]) => { ctx.fillStyle = "#fff"; ctx.beginPath(); ctx.arc(mapX(px), mapY(py), 2.5, 0, 7); ctx.fill(); });
    if (esPrincipal && det.emocion) {
      const txt = `${EMO[det.emocion].es} ${pct(det.confianza)}`;
      ctx.font = "600 13px Instrument Sans, Segoe UI, sans-serif";
      const tw = ctx.measureText(txt).width + 14;
      ctx.fillStyle = "#2F9E6A"; ctx.beginPath(); ctx.roundRect(x1 - 1.5, Math.max(0, y1 - 24), tw, 22, 5); ctx.fill();
      ctx.fillStyle = "#fff"; ctx.fillText(txt, x1 + 6, Math.max(15, y1 - 8));
    }
  });
  const p = d.principal;
  $("liveEmo").textContent = p && p.emocion ? EMO[p.emocion].es : "—";
  $("liveCalif").textContent = p && p.calificacion ? `${Math.round(p.calificacion.puntaje)} · ${p.calificacion.nivel}` : "—";
  $("liveFace").textContent = d.rostros > 1 ? `${d.rostros} rostros` : p ? `Sí · ${pct(p.score_deteccion)}` : "No detectado";
  const b = d.calidad ? d.calidad.brillo : null;
  $("liveLuz").textContent = b == null ? "—" : b < 45 ? "Muy oscura" : b > 225 ? "Excesiva" : "Adecuada";
  $("liveTxt").textContent = `En vivo · ${Math.round(1000 / Math.max(ms + CFG.LIVE_PAUSA_MS, 1))} fps`;
  renderBarras($("liveBars"), p && p.probabilidades, true);
}

function renderBarras(ul, probs, mini) {
  ul.replaceChildren();
  const orden = Object.keys(EMO);
  const top = probs ? orden.reduce((a, b) => (probs[a] >= probs[b] ? a : b)) : null;
  orden.forEach(k => {
    const v = probs ? probs[k] || 0 : 0;
    const li = el("li", k === top ? "top" : "");
    li.dataset.tip = `<b>${EMO[k].es}</b>: ${pct(v, 1)}`;
    li.append(el("span", "lab", EMO[k].es));
    const tr = el("span", "track"), fi = el("span", "fill");
    fi.style.width = (v * 100).toFixed(1) + "%"; fi.style.background = EMO[k].color;
    tr.append(fi); li.append(tr); li.append(el("span", "num", probs ? pct(v) : "—"));
    ul.append(li);
  });
}

/* ---------- Observación guiada ---------- */
async function observar() {
  if (state.observando) return;
  const dur = Number($("selDur").value) * 1000;
  const mostrar = $("chkMostrarObra").checked && state.obraURL;
  state.observando = true; limpiarOverlay();
  const capa = $("observe"), cuenta = $("obsCount"), prog = $("obsProg");
  $("obsImg").hidden = !mostrar; $("obsFallback").hidden = !!mostrar;
  if (mostrar) $("obsImg").src = state.obraURL;
  capa.hidden = false; prog.style.width = "0";
  state.enfocarAlCerrarObs = document.activeElement;
  setTimeout(() => $("btnCancelarObs").focus(), 80);
  let cancelado = false;
  state.cancelarObs = () => { cancelado = true; };
  announce("Observación iniciada");

  $("obsTitulo").textContent = "Prepárate";
  $("obsSub").textContent = mostrar ? "En un momento verás la obra en grande." : "Mira hacia la cámara.";
  cuenta.hidden = false;
  for (let n = CFG.PREP_SEG; n > 0 && !cancelado; n--) {
    cuenta.innerHTML = `${n}<small>${mostrar ? "Observa la obra con naturalidad" : "Mantén el rostro dentro del cuadro"}</small>`;
    await sleepCancel(1000, () => cancelado);
  }
  cuenta.hidden = true;

  const capturados = [];
  $("obsTitulo").textContent = mostrar ? "Observa la obra" : "Observación en curso";
  $("obsSub").textContent = "La cámara registra tu expresión. No hace falta posar.";
  const inicio = performance.now();
  while (!cancelado && performance.now() - inicio < dur) {
    const t = performance.now() - inicio;
    const blob = await capturarBlob(CFG.OBS_ANCHO, 0.85);
    if (blob) capturados.push({ blob, t: Math.round(t) });
    prog.style.width = Math.min(100, (t / dur) * 100) + "%";
    await sleepCancel(Math.max(0, CFG.OBS_INTERVALO_MS - (performance.now() - inicio - t)), () => cancelado);
  }
  prog.style.width = "100%";
  capa.hidden = true; state.observando = false; state.cancelarObs = null;
  if (state.enfocarAlCerrarObs && state.enfocarAlCerrarObs.focus) {
    try { state.enfocarAlCerrarObs.focus(); } catch {}
    state.enfocarAlCerrarObs = null;
  }
  if (cancelado) { toast("Observación cancelada.", "warn"); announce("Observación cancelada"); return; }
  if (!capturados.length) { mostrarError("errCam", "No se pudo capturar ningún fotograma. Inténtalo de nuevo."); toast("Sin fotogramas capturados.", "err"); return; }
  guardarFrames(capturados, `${capturados.length} fotogramas en ${(dur / 1000).toFixed(0)} s`);
  announce(`Observación completada: ${capturados.length} fotogramas`);
}
async function fotoUnica() {
  const blob = await capturarBlob(CFG.OBS_ANCHO, 0.9);
  if (!blob) { mostrarError("errCam", "La cámara aún no está lista. Espera un segundo e inténtalo de nuevo."); toast("Cámara no lista.", "warn"); return; }
  guardarFrames([{ blob, t: 0 }], "1 foto capturada");
  toast("Foto capturada.", "ok", 1800);
}
function guardarFrames(lista, texto) {
  state.frames.forEach(f => f.url && URL.revokeObjectURL(f.url));
  state.frames = lista.map(f => ({ ...f, url: URL.createObjectURL(f.blob) }));
  const strip = $("strip"); strip.replaceChildren();
  state.frames.forEach((f, i) => { const im = el("img"); im.src = f.url; im.alt = `Fotograma ${i + 1}`; strip.append(im); });
  $("framesTxt").textContent = "✓ " + texto;
  $("framesBox").hidden = false;
  $("btnObservar").textContent = `Repetir observación (${$("selDur").value} s)`;
  pill("pillCam", state.stream ? "Capturado · en vivo" : "Capturado", "ok");
  actualizarBoton();
}
function descartarFrames() {
  state.frames.forEach(f => f.url && URL.revokeObjectURL(f.url)); state.frames = [];
  $("framesBox").hidden = true; actualizarBoton();
  $("btnObservar").textContent = `Iniciar observación (${$("selDur").value} s)`;
  pill("pillCam", state.stream ? "En vivo" : "Apagada", state.stream ? "live" : "");
}
$("btnObservar").addEventListener("click", observar);
$("btnFoto").addEventListener("click", fotoUnica);
$("btnDescartar").addEventListener("click", descartarFrames);
$("btnCancelarObs").addEventListener("click", () => state.cancelarObs && state.cancelarObs());
document.addEventListener("keydown", e => { if (e.key === "Escape" && state.cancelarObs) state.cancelarObs(); });

/* ============================================================ PASO 2 — envío */
function animarPipeline() {
  const ol = $("pipeline"); ol.replaceChildren();
  const lis = MODULOS.map(([, n]) => { const li = el("li", "", n); ol.append(li); return li; });
  let i = 0;
  const id = setInterval(() => { if (i < lis.length) { lis[i].classList.add("run"); if (i) lis[i - 1].classList.replace("run", "done"); i++; } }, 260);
  return () => { clearInterval(id); lis.forEach(li => { li.classList.remove("run"); li.classList.add("done"); }); };
}

async function analizar() {
  if (state.analizando) return;
  state.analizando = true;
  $("btnAnalizar").disabled = true;
  try {
    detenerCamara();
    irAPaso(2);
    const fin = animarPipeline();
    const fd = new FormData();
    if (state.obra) fd.append("obra", state.obra, state.obra.name);
    state.frames.forEach((f, i) => fd.append("frames", f.blob, `frame_${i}.jpg`));
    if (state.frames.length) fd.append("tiempos", JSON.stringify(state.frames.map(f => f.t)));
    const ctrl = new AbortController(); const to = setTimeout(() => ctrl.abort(), CFG.TIMEOUT_MS);
    const t0 = performance.now();
    try {
      const r = await fetch(API + "/api/analizar", { method: "POST", body: fd, signal: ctrl.signal });
      if (!r.ok) throw new Error("HTTP " + r.status);
      state.resultado = await r.json();
    } catch (e) {
      state.resultado = { estado: "error", obra: null, rostro: null, coincidencia: null, tiempos_ms: {}, interpretacion: "",
        excepciones: [{ tipo: "servidor", gravedad: "error", modulo: "GUI",
          mensaje: e.name === "AbortError" ? "El servidor tardó demasiado en responder. Inténtalo de nuevo."
            : `No se pudo contactar al modelo (${e.message || "error de red"}). Verifica que el servidor esté activo.` }] };
    } finally { clearTimeout(to); }
    await sleep(Math.max(0, 900 - (performance.now() - t0)));
    fin(); await sleep(250);
    const r = state.resultado;
    if (r.excepciones.some(x => x.gravedad !== "info")) { renderExcepciones(); irAPaso(3); announce("Análisis con excepciones"); }
    else { renderResultado(); irAPaso(4); announce("Análisis completado"); }
  } finally {
    state.analizando = false;
    actualizarBoton();
  }
}
$("btnAnalizar").addEventListener("click", analizar);

/* ============================================================ PASO 3 — excepciones */
function renderExcepciones() {
  const r = state.resultado, ul = $("listaExc"); ul.replaceChildren();
  const errores = r.excepciones.filter(x => x.gravedad === "error").length;
  $("leadExc").textContent = errores
    ? "El análisis encontró problemas que impiden parte del resultado. Revisa y ajusta la entrada."
    : "El análisis terminó con avisos que conviene revisar antes de interpretar el resultado.";
  r.excepciones.forEach(x => {
    const li = el("li", "exc " + gravedadCls(x.gravedad));
    li.append(el("span", "ic", ICONOS[x.tipo] || "⚠️"));
    li.append(el("b", null, TITULOS[x.tipo] || (x.gravedad === "error" ? "Error" : x.gravedad === "info" ? "Información" : "Aviso")));
    li.append(el("span", "mod", x.modulo || ""));
    li.append(el("span", null, x.mensaje || "Situación no especificada."));
    ul.append(li);
  });
  const hayAlgo = (r.obra && r.obra.categoria) || (r.rostro && !r.rostro.sin_rostro) || (r.obra && r.obra.paleta);
  $("btnVerResultado").disabled = !hayAlgo;
}
$("btnReintentar").addEventListener("click", () => irAPaso(1));
$("btnVerResultado").addEventListener("click", () => { renderResultado(); irAPaso(4); });
$("btnVolverExc").addEventListener("click", () => { renderExcepciones(); irAPaso(3); });

/* ============================================================ PASO 4 — Resultado */
function stage(src, cajas, opts = {}) {
  const wrap = el("div", "stage" + (opts.espejo ? " mirror-img" : ""));
  const img = el("img"); img.alt = opts.alt || "Imagen analizada"; img.src = src; img.loading = "lazy"; wrap.append(img);
  const mx = b => opts.espejo ? 1 - b.x - b.w : b.x;
  const lista = cajas || [];
  lista.forEach((b, i) => {
    const d = el("div", "box " + (opts.cls || "") + (i > 0 ? " bottom" : b.y < 0.08 ? " top" : ""));
    d.style.cssText = `left:${mx(b) * 100}%;top:${b.y * 100}%;width:${b.w * 100}%;height:${b.h * 100}%`;
    if (b.etiqueta) d.append(el("span", null, b.etiqueta));
    wrap.append(d);
  });
  (opts.puntos || []).forEach(([x, y]) => {
    const p = el("i", "lm"); p.style.left = (opts.espejo ? 1 - x : x) * 100 + "%"; p.style.top = y * 100 + "%"; wrap.append(p);
  });
  return wrap;
}

function gaugeSVG(valor, nivel) {
  const p = valor == null ? 0 : Math.round(valor * 100);
  return `<svg viewBox="0 0 200 112" role="img" aria-label="Coincidencia emocional ${valor == null ? "no disponible" : p + " por ciento"}">
    <path d="M14 100 A86 86 0 0 1 186 100" pathLength="100" fill="none" stroke="var(--line-2)" stroke-width="16" stroke-linecap="round"/>
    <path d="M14 100 A86 86 0 0 1 186 100" pathLength="100" fill="none" stroke="var(--ink)" stroke-width="16" stroke-linecap="round" stroke-dasharray="${p} 100"/>
  </svg><div class="val">${valor == null ? "—" : p + "%"}</div><div class="cap">Coincidencia emocional</div>
  ${nivel ? `<span class="lvl">Coincidencia ${nivel}</span>` : ""}`;
}

function escalaValencia(vo, vr) {
  if (vo == null && vr == null) return "";
  const X = v => 10 + (v + 1) / 2 * 180;
  const marca = (v, color, etiqueta, arriba) => v == null ? "" : `
    <g data-tip="<b>${etiqueta}</b>: valencia ${v.toFixed(2)}">
      <circle cx="${X(v)}" cy="30" r="7" fill="${color}" stroke="var(--card)" stroke-width="2"/>
      <text x="${X(v)}" y="${arriba ? 16 : 52}" text-anchor="middle" style="fill:var(--ink);font-weight:600">${etiqueta}</text>
    </g>`;
  return `<div class="chart" style="margin-top:14px"><svg viewBox="0 0 200 76" role="img" aria-label="Valencia de obra y visitante">
    <defs><linearGradient id="gv"><stop offset="0" stop-color="#eb6834"/><stop offset=".5" stop-color="#c9ccd8"/><stop offset="1" stop-color="#2a78d6"/></linearGradient></defs>
    <rect x="10" y="27" width="180" height="6" rx="3" fill="url(#gv)"/>
    <text x="10" y="74" style="font-size:9px">−1 desagradable</text><text x="190" y="74" text-anchor="end" style="font-size:9px">agradable +1</text>
    ${marca(vo, "#232B5D", "Obra", true)}${marca(vr, "#2F9E6A", "Visitante", false)}
  </svg></div>`;
}

function renderResultado() {
  const r = state.resultado;
  const obra = r.obra && r.obra.categoria ? r.obra : null;
  const pal = r.obra && r.obra.paleta ? r.obra.paleta : null;
  const ro = r.rostro && !r.rostro.sin_rostro ? r.rostro : null;
  $("btnVolverExc").hidden = !r.excepciones.length;

  const rObra = $("rObra"); rObra.replaceChildren();
  if (state.obraURL && (obra || pal)) {
    const cont = el("div"); let vista = "original";
    const pintar = () => {
      cont.replaceChildren(vista === "gradcam" && obra && obra.gradcam
        ? stage(obra.gradcam, [], { alt: "Mapa Grad-CAM de la obra" })
        : vista === "rostros" ? stageRostros(state.obraURL, r.obra.rostros, true)
        : stage(state.obraURL, obra ? obra.regiones : [], { alt: "Obra analizada" }));
    };
    if (obra && obra.gradcam) {
      const tg = el("div", "toggle"); tg.setAttribute("role", "group"); tg.setAttribute("aria-label", "Vista de la obra");
      const opciones = [["original", "Regiones"], ["gradcam", "Grad-CAM"]];
      if (r.obra.rostros && r.obra.rostros.n) opciones.push(["rostros", `Rostros (${r.obra.rostros.n})`]);
      opciones.forEach(([k, t]) => {
        const b = el("button", null, t); b.setAttribute("aria-pressed", String(k === vista));
        b.onclick = () => { vista = k; [...tg.children].forEach(c => c.setAttribute("aria-pressed", String(c === b))); pintar(); };
        tg.append(b);
      });
      rObra.append(tg);
    }
    pintar(); rObra.append(cont);
    if (obra) {
      const t = el("div", "tag"); t.innerHTML = `${esc(obra.etiqueta)}: ${pct(obra.confianza)}<small>${esc(obra.descripcion)}</small>`;
      rObra.append(t);
    } else rObra.append(el("div", "tag", "Solo paleta (CNN no disponible)"));
  } else rObra.append(el("div", "vacio", "Sin imagen de obra analizada."));

  const rRo = $("rRostro"); rRo.replaceChildren();
  if (ro) {
    rRo.append(stage(ro.frame, [{ ...ro.caja, etiqueta: ro.etiqueta }], { cls: "face", espejo: true, puntos: ro.puntos, alt: "Fotograma representativo del visitante" }));
    const t = el("div", "tag face"); t.innerHTML = `${esc(ro.etiqueta)} (${pct(ro.confianza)})<small>${ro.frames_con_rostro} de ${ro.frames_analizados} fotogramas con rostro</small>`;
    rRo.append(t);
  } else if (r.rostro && r.rostro.sin_rostro) {
    rRo.append(stage(r.rostro.frame, [], { espejo: true, alt: "Fotograma sin rostro detectado" }));
    rRo.append(el("div", "tag", "Rostro no detectado"));
  } else rRo.append(el("div", "vacio", "Sin rostro analizado."));

  $("gauge").innerHTML = gaugeSVG(r.coincidencia, r.nivel_coincidencia) + escalaValencia(obra ? obra.valencia : null, ro ? ro.valencia : null);
  $("resumen").textContent = r.interpretacion || "";
  const chips = $("excChips"); chips.replaceChildren();
  r.excepciones.forEach(x => { const c = el("span", gravedadCls(x.gravedad), `${ICONOS[x.tipo] || "⚠️"} ${TITULOS[x.tipo] || "Aviso"}`); c.dataset.tip = esc(x.mensaje); chips.append(c); });

  renderClasificacion(obra); renderPaleta(pal); renderEmociones(ro); renderGestos(ro); renderTiempo(ro); renderTiempos(r.tiempos_ms);
  renderRostrosObra(r.obra && r.obra.rostros, $("cardRostrosObra"), "Rostros dentro de la obra");
}

function renderClasificacion(o, c = $("cardClasif")) {
  c.replaceChildren();
  if (!o) return;
  c.innerHTML = `<h3>Clasificación de la obra <span class="badge ${o.revision_requerida ? "warn" : "ok"}">${o.revision_requerida ? "Revisión requerida" : "Confiable"}</span></h3>
    <p class="note">CNN entrenada en el Parcial 2 · Softmax sobre 2 logits · umbral de aceptación ${pct(o.umbral)}</p>`;
  const ul = el("ul", "bars");
  [["positivo", "Positiva", "var(--pos)"], ["negativo", "Negativa", "var(--neg)"]].forEach(([k, n, col]) => {
    const v = o.puntuaciones[k];
    const li = el("li", o.categoria === k ? "top" : ""); li.dataset.tip = `<b>${n}</b>: ${pct(v, 1)} · logit ${o.logits[k === "negativo" ? 0 : 1]}`;
    li.append(el("span", "lab", n));
    const tr = el("span", "track"), fi = el("span", "fill"), th = el("span", "thr");
    fi.style.width = v * 100 + "%"; fi.style.background = col; th.style.left = o.umbral * 100 + "%"; th.title = "Umbral";
    tr.append(fi, th); li.append(tr, el("span", "num", pct(v)));
    ul.append(li);
  });
  c.append(ul);
  const kv = el("div", "kv");
  kv.innerHTML = `<div><span>Logits [neg, pos]</span><b>${o.logits.map(v => v.toFixed(2)).join(", ")}</b></div>
    <div><span>Tensor de entrada</span><b>[${o.tensor.forma.join(", ")}]</b></div>
    <div><span>Dimensiones</span><b>${o.dimensiones.ancho}×${o.dimensiones.alto}</b></div>
    <div><span>Brillo · nitidez</span><b>${o.calidad.brillo} · ${Math.round(o.calidad.nitidez)}</b></div>`;
  c.append(kv);
}

function renderPaleta(p, c = $("cardPaleta")) {
  c.replaceChildren();
  if (!p) return;
  c.innerHTML = `<h3>Paleta cromática dominante</h3><p class="note">K-Means (k=${p.colores.length}) en espacio CIE Lab · clic en un color para copiar su HEX</p>`;
  const bar = el("div", "palette-bar"); bar.setAttribute("role", "img");
  bar.setAttribute("aria-label", "Distribución de colores: " + p.colores.map(x => `${x.nombre} ${pct(x.proporcion)}`).join(", "));
  p.colores.forEach(x => { const i = el("i"); i.style.cssText = `flex:${x.proporcion};background:${x.hex}`; i.dataset.tip = `<b>${x.hex}</b> · ${x.nombre} · ${pct(x.proporcion, 1)}`; bar.append(i); });
  c.append(bar);
  const sw = el("div", "swatches");
  p.colores.forEach(x => {
    const b = el("button", "sw"); b.type = "button"; b.title = "Copiar " + x.hex;
    b.innerHTML = `<i style="background:${x.hex}"></i><span><b>${x.hex}</b>${esc(x.nombre)} · ${pct(x.proporcion)}</span>`;
    b.onclick = () => {
      const done = () => {
        b.querySelector("b").textContent = "¡Copiado!";
        setTimeout(() => b.querySelector("b").textContent = x.hex, 900);
        toast(`Copiado ${x.hex}`, "ok", 1400);
      };
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(x.hex).then(done).catch(() => toast("No se pudo copiar al portapapeles.", "warn"));
      } else done();
    };
    sw.append(b);
  });
  c.append(sw);
  const kv = el("div", "kv");
  kv.innerHTML = `<div><span>Brillo medio</span><b>${pct(p.brillo_medio)}</b></div><div><span>Saturación media</span><b>${pct(p.saturacion_media)}</b></div>
    <div><span>Colores cálidos</span><b>${pct(p.proporcion_calida)}</b></div><div><span>Colores fríos/neutros</span><b>${pct(1 - p.proporcion_calida)}</b></div>`;
  c.append(kv);
}

function renderEmociones(ro, c = $("cardEmociones")) {
  c.replaceChildren();
  if (!ro) return;
  c.innerHTML = `<h3>Emociones del visitante</h3><p class="note">FER+ · probabilidad media en los ${ro.frames_con_rostro} fotograma(s) con rostro válido</p>`;
  const ul = el("ul", "bars"); renderBarras(ul, ro.probabilidades); c.append(ul);
}

function renderGestos(ro, c = $("cardGestos")) {
  c.replaceChildren();
  if (!ro) return;
  const g = ro.gestos || {};
  const incl = g.inclinacion_cabeza_grados, giro = g.giro_relativo;
  c.innerHTML = `<h3>Gestos faciales observables</h3><p class="note">Calculados con los 5 puntos faciales de YuNet (ojos, nariz y comisuras), promediados en la secuencia</p>
  <div class="kv">
    <div><span>Inclinación de cabeza</span><b>${incl == null ? "—" : incl.toFixed(1) + "°"}</b></div>
    <div><span>Giro (yaw relativo)</span><b>${giro == null ? "—" : (Math.abs(giro) < 0.1 ? "De frente" : giro > 0 ? "Hacia un lado" : "Hacia el otro")}</b></div>
    <div><span>Ancho de boca / ojos</span><b>${g.ancho_boca_relativo == null ? "—" : g.ancho_boca_relativo.toFixed(2)}</b></div>
    <div><span>Mirando al frente</span><b>${g.porcentaje_mirando_al_frente == null ? "—" : pct(g.porcentaje_mirando_al_frente)}</b></div>
    <div><span>Cambios de expresión</span><b>${ro.cambios_de_expresion}</b></div>
    <div><span>Cámara: brillo · nitidez</span><b>${ro.calidad.brillo} · ${Math.round(ro.calidad.nitidez)}</b></div>
    <div><span>Calificación del rostro</span><b>${ro.calificacion ? `${Math.round(ro.calificacion.puntaje)}/100 · ${ro.calificacion.nivel}` : "—"}</b></div>
  </div>${ro.calificacion && ro.calificacion.componentes ? componentesCalif(ro.calificacion.componentes) : ""}`;
}

function renderTiempo(ro) {
  const c = $("cardTiempo"); c.replaceChildren();
  if (!ro) return;
  const pts = ro.linea_tiempo;
  c.innerHTML = `<h3>Reacción a lo largo de la observación</h3><p class="note">Valencia estimada de la expresión en cada fotograma (−1 desagradable … +1 agradable). Cada punto toma el color de su emoción dominante.</p>`;
  if (pts.length < 2) { c.append(el("p", "note", "Se capturó una sola foto: usa «Iniciar observación» para obtener la secuencia temporal.")); return; }
  const W = 720, H = 200, L = 40, R = 12, T = 12, B = 30;
  const tmax = Math.max(...pts.map(p => p.t), 1);
  const X = t => L + (t / tmax) * (W - L - R), Y = v => T + (1 - (v + 1) / 2) * (H - T - B);
  let path = "", abierto = false;
  pts.forEach(p => { if (p.rostro && p.valencia != null) { path += (abierto ? "L" : "M") + X(p.t).toFixed(1) + " " + Y(p.valencia).toFixed(1); abierto = true; } else abierto = false; });
  const grid = [-1, -0.5, 0, 0.5, 1].map(v => `<line x1="${L}" x2="${W - R}" y1="${Y(v)}" y2="${Y(v)}" ${v === 0 ? 'style="stroke:var(--muted)"' : ""}/><text x="${L - 8}" y="${Y(v) + 4}" text-anchor="end">${v > 0 ? "+" : ""}${v}</text>`).join("");
  const ticks = Array.from({ length: 6 }, (_, i) => tmax * i / 5).map(t => `<text x="${X(t)}" y="${H - 8}" text-anchor="middle">${(t / 1000).toFixed(1)} s</text>`).join("");
  const dots = pts.map(p => p.rostro && p.valencia != null
    ? `<g data-tip="<b>${(p.t / 1000).toFixed(1)} s</b> · ${EMO[p.emocion].es} ${pct(p.confianza)}<br>valencia ${p.valencia.toFixed(2)}"><circle cx="${X(p.t)}" cy="${Y(p.valencia)}" r="12" fill="transparent"/><circle cx="${X(p.t)}" cy="${Y(p.valencia)}" r="5" fill="${EMO[p.emocion].color}" stroke="var(--card)" stroke-width="2"/></g>`
    : `<g data-tip="<b>${(p.t / 1000).toFixed(1)} s</b> · sin rostro"><line x1="${X(p.t)}" x2="${X(p.t)}" y1="${H - B - 6}" y2="${H - B + 2}" stroke="#C2503A" stroke-width="2"/></g>`).join("");
  const wrap = el("div", "chart");
  wrap.innerHTML = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Línea de tiempo de la valencia facial"><g class="grid">${grid}</g>${ticks}
    <path d="${path}" fill="none" stroke="var(--ink)" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>${dots}</svg>`;
  c.append(wrap);
  const usadas = [...new Set(pts.filter(p => p.emocion).map(p => p.emocion))];
  const lg = el("div", "legend");
  lg.innerHTML = usadas.map(k => `<span><i style="background:${EMO[k].color}"></i>${EMO[k].es}</span>`).join("") +
    (pts.some(p => !p.rostro) ? `<span><i style="background:#C2503A;border-radius:1px;width:3px"></i>Sin rostro</span>` : "");
  c.append(lg);
  const det = el("details", "tabla");
  det.innerHTML = `<summary>Ver tabla de fotogramas</summary><table class="mini"><thead><tr><th>Tiempo</th><th>Rostro</th><th>Emoción</th><th>Confianza</th><th>Valencia</th></tr></thead><tbody>${
    pts.map(p => `<tr><td>${(p.t / 1000).toFixed(1)} s</td><td>${p.rostro ? "Sí" : "No"}</td><td>${p.emocion ? EMO[p.emocion].es : "—"}</td><td>${p.confianza != null ? pct(p.confianza) : "—"}</td><td>${p.valencia != null ? p.valencia.toFixed(2) : "—"}</td></tr>`).join("")}</tbody></table>`;
  c.append(det);
}

function renderTiempos(t, c = $("cardTiempos")) {
  c.replaceChildren();
  if (!t || !t.total) return;
  c.innerHTML = `<h3>Tiempo de procesamiento por módulo</h3><p class="note">Total en el servidor: ${Math.round(t.total)} ms</p>`;
  const nombres = { image_loader: "ImageLoader", rostros_obra: "Rostros en la imagen", preprocesamiento: "Preprocesamiento", cnn: "CNN",
    gradcam: "Grad-CAM", paleta: "Paleta", gestos: "Gestos (cámara)" };
  const filas = Object.entries(nombres).filter(([k]) => t[k] != null);
  const max = Math.max(...filas.map(([k]) => t[k]), 1);
  const ul = el("ul", "bars ancho");
  filas.forEach(([k, n]) => {
    const li = el("li"); li.dataset.tip = `<b>${n}</b>: ${t[k]} ms`;
    li.append(el("span", "lab", n));
    const tr = el("span", "track"), fi = el("span", "fill"); fi.style.width = (t[k] / max * 100) + "%"; fi.style.background = "var(--ink)";
    tr.append(fi); li.append(tr, el("span", "num", Math.round(t[k]) + " ms")); ul.append(li);
  });
  c.append(ul);
}

function stageRostros(src, rostros, etiquetas) {
  const cajas = (rostros && rostros.detecciones || []).map(d => ({ ...d.caja, etiqueta: etiquetas ? `#${d.id} ${d.etiqueta || ""}` : "" }));
  return stage(src, cajas, { cls: "face", alt: "Rostros detectados en la imagen" });
}
function componentesCalif(c) {
  const nombres = { deteccion: "Detección", tamano: "Tamaño", frontalidad: "Frontalidad", nitidez: "Nitidez", iluminacion: "Iluminación" };
  const ul = Object.entries(nombres).map(([k, n]) => `<li data-tip="<b>${n}</b>: ${pct(c[k])}"><span class="lab">${n}</span><span class="track"><span class="fill" style="width:${c[k] * 100}%;background:var(--ok)"></span></span><span class="num">${pct(c[k])}</span></li>`).join("");
  return `<p class="note" style="margin:12px 0 6px">Componentes de la calificación (30 % detección · 20 % tamaño · 20 % frontalidad · 15 % nitidez · 15 % iluminación)</p><ul class="bars">${ul}</ul>`;
}
function califHTML(cal) {
  return cal ? `<span class="calif" data-tip="Calificación del rostro: ${cal.puntaje}/100"><i style="--v:${cal.puntaje}%"></i>${Math.round(cal.puntaje)} · ${cal.nivel}</span>` : "—";
}
function renderRostrosObra(rostros, c, titulo) {
  c.replaceChildren();
  if (!rostros) return;
  if (!rostros.n) { c.innerHTML = `<h3>${titulo}</h3><p class="note">No se detectaron rostros en la imagen (umbral del detector ≥ 60 %).</p>`; return; }
  c.innerHTML = `<h3>${titulo} <span class="badge ok">${rostros.n}</span></h3>
    <p class="note">YuNet + FER+ sobre la propia imagen · emoción predominante: <b>${EMO[rostros.emocion_predominante] ? EMO[rostros.emocion_predominante].es : "—"}</b> · calificación media ${rostros.calificacion_media != null ? Math.round(rostros.calificacion_media) + "/100" : "—"}</p>
    <table class="mini"><thead><tr><th>#</th><th>Emoción</th><th>Confianza</th><th>Valencia</th><th>Detección</th><th>Calificación</th></tr></thead><tbody>${
    rostros.detecciones.map(d => `<tr><td>${d.id}</td><td class="l"><i style="display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:6px;background:${d.emocion ? EMO[d.emocion].color : "#ccc"}"></i>${d.etiqueta || "—"}</td><td>${d.confianza != null ? pct(d.confianza) : "—"}</td><td>${d.valencia != null ? d.valencia.toFixed(2) : "—"}</td><td>${pct(d.score_deteccion)}</td><td>${califHTML(d.calificacion)}</td></tr>`).join("")}</tbody></table>`;
  const p = rostros.principal;
  if (p && p.probabilidades) {
    c.append(el("p", "note", `Rostro principal (#${p.id}): probabilidades por emoción`));
    const ul = el("ul", "bars"); renderBarras(ul, p.probabilidades); c.append(ul);
  }
  if (p && p.calificacion) c.insertAdjacentHTML("beforeend", componentesCalif(p.calificacion.componentes));
}

/* ---------- reporte ---------- */
$("btnJSON").addEventListener("click", () => {
  const r = { generado: new Date().toISOString(), obra_archivo: state.obra ? state.obra.name : null, ...state.resultado };
  const blob = new Blob([JSON.stringify(r, null, 2)], { type: "application/json" });
  const a = el("a"); a.href = URL.createObjectURL(blob); a.download = `reporte_coincidencia_${Date.now()}.json`; a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 2000);
  toast("Reporte descargado.", "ok", 1800);
});
$("btnImprimir").addEventListener("click", () => window.print());

/* ---------- reiniciar ---------- */
function reiniciar() {
  detenerCamara(); quitarObra(); descartarFrames(); mostrarError("errCam"); mostrarError("errObra");
  state.resultado = null; pill("pillCam", "Apagada"); actualizarBoton(); irAPaso(1);
}
$("btnNueva").addEventListener("click", reiniciar);

/* ---------- cleanup global ---------- */
window.addEventListener("pagehide", () => {
  detenerCamara();
  if (state.obraURL) URL.revokeObjectURL(state.obraURL);
  state.frames.forEach(f => f.url && URL.revokeObjectURL(f.url));
  ds.items.forEach(item => { if (item.url && item.url.startsWith("blob:")) URL.revokeObjectURL(item.url); });
});

/* ============================================================ Acerca del modelo */
async function abrirAcerca() {
  const dlg = $("dlgAcerca"), body = $("dlgBody"); dlg.showModal();
  let m = null;
  try { const r = await fetch(API + "/api/metricas"); if (r.ok) m = await r.json(); } catch { /* sin servidor */ }
  const flujo = `<div class="flow"><span>ImageLoader</span><em>→</em><span>PreprocesamientoImagen</span><em>→</em><span>ClasificadorDeObras</span><em>+</em><span>ExtractorDePaleta</span></div>
    <div class="flow"><span>CámaraAutorizada</span><em>→</em><span>CapturaDeFrame</span><em>→</em><span>AnalizarGestosFaciales</span></div>
    <div class="flow"><em>→</em><span>ResultadoProceso</span><em>→</em><span>Integración</span><em>→</em><span>Presentación y uso</span></div>`;
  let html = `<p>La obra se clasifica en <b>positiva</b> o <b>negativa</b> con la CNN de obras (ResNet18 con transfer learning); el rostro se detecta con YuNet y su expresión con FER+ (8 emociones). Ambas lecturas se llevan a una misma escala de <b>valencia</b> (−1 a +1) y la coincidencia es <code>1 − |v_obra − v_visitante| / 2</code>.</p>
  <h3>Arquitectura</h3>${flujo}`;
  if (m) {
    const p = m.prueba, cm = p.matriz_confusion;
    html += `<h3>Desempeño de la CNN (${m.k_folds ? "validación cruzada de " + m.k_folds + " folds" : "conjunto de prueba"}, ${p.n} imágenes)</h3>
    <div class="kv"><div><span>Accuracy</span><b>${p.accuracy}</b></div><div><span>Precision</span><b>${p.precision}</b></div><div><span>Recall</span><b>${p.recall}</b></div><div><span>F1-score</span><b>${p.f1}</b></div></div>
    <table class="mini"><thead><tr><th>Real \\ Predicho</th><th>Negativa</th><th>Positiva</th></tr></thead>
    <tbody><tr><th>Negativa</th><td>${cm[0][0]}</td><td>${cm[0][1]}</td></tr><tr><th>Positiva</th><td>${cm[1][0]}</td><td>${cm[1][1]}</td></tr></tbody></table>
    <h3>Curva de entrenamiento</h3>${curvaPerdida(m.historial, m.mejor_epoca)}
    <p>${esc(m.modelo)}. Mejor época: ${m.mejor_epoca}. ${m.k_folds ? "Cada imagen se evaluó con un modelo que no la vio; accuracy por fold " + m.accuracy_media + " ± " + m.accuracy_std + (p.auc != null ? " · AUC " + p.auc : "") + ". Modelo final entrenado con las " + m.particion.entrenamiento + " imágenes" : "Partición: " + m.particion.entrenamiento + " / " + m.particion.validacion + " / " + m.particion.prueba}. Normalización: media ${m.media.map(v => v.toFixed(3)).join(", ")} · std ${m.std.map(v => v.toFixed(3)).join(", ")}.</p>`;
  } else html += `<p>No hay métricas disponibles (el servidor no respondió o el modelo no está entrenado).</p>`;
  html += `<h3>Valencia asignada a cada emoción</h3><p>Felicidad +1 · Sorpresa +0.3 · Neutral 0 · Desprecio −0.5 · Miedo −0.7 · Tristeza −0.8 · Asco −0.8 · Enojo −0.9</p>
  <p><small>Los resultados son exploratorios: la CNN se entrenó con ${m ? m.particion.entrenamiento : "pocas"} imágenes y la expresión facial no equivale a la emoción interna.</small></p>`;
  body.innerHTML = html;
}
function curvaPerdida(h, mejor) {
  const W = 640, H = 200, L = 40, R = 90, T = 10, B = 26, n = h.train_loss.length;
  const vals = [...h.train_loss, ...h.val_loss], lo = Math.min(...vals) * 0.95, hi = Math.max(...vals) * 1.02;
  const X = i => L + (i / Math.max(n - 1, 1)) * (W - L - R), Y = v => T + (1 - (v - lo) / (hi - lo)) * (H - T - B);
  const linea = arr => arr.map((v, i) => (i ? "L" : "M") + X(i).toFixed(1) + " " + Y(v).toFixed(1)).join("");
  const serie = (arr, col, nom) => `<path d="${linea(arr)}" fill="none" stroke="${col}" stroke-width="2" stroke-linejoin="round"/>` +
    arr.map((v, i) => `<g data-tip="<b>${nom}</b> · época ${i + 1}: ${v.toFixed(3)}"><circle cx="${X(i)}" cy="${Y(v)}" r="9" fill="transparent"/></g>`).join("") +
    `<text x="${X(n - 1) + 8}" y="${Y(arr[n - 1]) + 4}" style="fill:#232B5D">${nom}</text>`;
  const yt = [lo, (lo + hi) / 2, hi].map(v => `<line x1="${L}" x2="${W - R}" y1="${Y(v)}" y2="${Y(v)}"/><text x="${L - 6}" y="${Y(v) + 4}" text-anchor="end">${v.toFixed(2)}</text>`).join("");
  const xt = h.train_loss.map((_, i) => (i % 2 === 0 || i === n - 1) ? `<text x="${X(i)}" y="${H - 6}" text-anchor="middle">${i + 1}</text>` : "").join("");
  return `<div class="chart"><svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Pérdida de entrenamiento y validación por época">
    <g class="grid">${yt}</g>${xt}<line x1="${X(mejor - 1)}" x2="${X(mejor - 1)}" y1="${T}" y2="${H - B}" stroke="#9aa0b4" stroke-dasharray="4 4"/>
    <text x="${X(mejor - 1) + 4}" y="${T + 10}">mejor época</text>
    ${serie(h.train_loss, "#2a78d6", "Entrenamiento")}${serie(h.val_loss, "#eb6834", "Validación")}</svg></div>
    <div class="legend"><span><i style="background:#2a78d6"></i>Pérdida de entrenamiento</span><span><i style="background:#eb6834"></i>Pérdida de validación</span></div>`;
}
$("btnAcerca").addEventListener("click", abrirAcerca);
$("btnCerrarDlg").addEventListener("click", () => $("dlgAcerca").close());

/* =====================================================================
   PESTAÑAS: Obra y visitante  |  Imágenes precargadas
   ===================================================================== */
function cambiarModo(modo, enfocar = false) {
  const ds = modo === "dataset";
  const tabs = [[$("tabVisitante"), !ds], [$("tabDataset"), ds]];
  tabs.forEach(([t, sel]) => { t.setAttribute("aria-selected", String(sel)); t.tabIndex = sel ? 0 : -1; });
  $("vistaVisitante").hidden = ds; $("vistaDataset").hidden = !ds;
  if (ds) { if (state.stream) detenerCamara(); cargarEstadoDataset(); }   // libera la cámara al salir del flujo con visitante
  if (enfocar) (ds ? $("tabDataset") : $("tabVisitante")).focus();
  window.scrollTo({ top: 0, behavior: "smooth" });
}
$("tabVisitante").addEventListener("click", () => cambiarModo("visitante"));
$("tabDataset").addEventListener("click", () => cambiarModo("dataset"));
[$("tabVisitante"), $("tabDataset")].forEach(t => t.addEventListener("keydown", e => {   // flechas entre pestañas
  if (["ArrowLeft", "ArrowRight", "Home", "End"].includes(e.key)) {
    e.preventDefault();
    cambiarModo(t.id === "tabVisitante" ? "dataset" : "visitante", true);
  }
}));

/* =====================================================================
   IMÁGENES PRECARGADAS — carpeta precarga-sin-analizar (Google Drive)
   Mismo flujo de análisis que una imagen subida.
   ===================================================================== */
const ds = { items: new Map(), cola: [], activos: 0, propias: 0 };
const DS_CONCURRENCIA = 2;
const RUTA_PRECARGA = "Mi unidad / Proyecto-final-de-PDI / dataset usado / precarga-sin-analizar";

async function cargarEstadoDataset() {
  try {
    const e = await (await fetchRetry(API + "/api/dataset/estado")).json();
    pintarEstadoDataset(e);
  } catch { $("dsEstado").textContent = "No se pudo consultar la carpeta de imágenes precargadas (servidor no disponible)."; }
}
function pintarEstadoDataset(e) {
  if (!e) return;
  const carpeta = `<code>${esc(e.carpeta || "precarga-sin-analizar")}</code>`;
  $("dsEstado").innerHTML = e.listo
    ? `<b>${e.n_imagenes} imágenes</b> disponibles<br>${carpeta}`
    : `<b>Carpeta no disponible.</b><br>Se busca en ${carpeta}`;
}
function mostrarErrorDs(err) {
  const box = $("dsError");
  if (!err) { box.hidden = true; return; }
  const consejos = {
    dataset_no_encontrado: `<ol><li>Ejecuta la celda <b>«Localizar el dataset»</b> del notebook y autoriza el acceso a Google Drive.</li>
      <li>Verifica que exista la carpeta <code>${esc(RUTA_PRECARGA)}</code>.</li>
      <li>Si está en otro lugar, cambia <code>config.PRECARGA_DIR</code> en la celda de configuración y vuelve a ejecutar el notebook.</li></ol>`,
    dataset_no_disponible: `<ol><li>Copia imágenes JPG o PNG en <code>${esc(RUTA_PRECARGA)}</code>.</li>
      <li>Pulsa «Volver a leer la carpeta» (no hace falta reiniciar el servidor).</li></ol>`,
  };
  box.innerHTML = `<b>${ICONOS[err.tipo] || "⚠️"} ${esc(TITULOS[err.tipo] || "Error")}</b>${esc(err.mensaje)}${consejos[err.tipo] || ""}`;
  box.hidden = false;
}
async function fetchJSON(url, opts) {
  const r = await fetch(API + url, opts);
  let d; try { d = await r.json(); } catch { throw { tipo: "servidor", mensaje: `Respuesta inválida del servidor (HTTP ${r.status}).` }; }
  if (d && d.estado === "error" && d.error) { if (d.dataset) pintarEstadoDataset(d.dataset); throw d.error; }
  if (!r.ok) throw { tipo: "servidor", mensaje: (d && d.detail) || `HTTP ${r.status}` };
  return d;
}
function setCargando(btn, cargando, texto) {
  if (cargando) { btn.dataset.txt = btn.textContent; btn.innerHTML = `<span class="spin-sm"></span>${texto}`; btn.disabled = true; }
  else { btn.textContent = btn.dataset.txt || btn.textContent; btn.disabled = false; }
}

/* Paso 1: si el visitante no quiere subir una imagen propia, se toma una al azar de la carpeta */
async function obraPrecargada() {
  const btn = $("btnObraPrecargada");
  mostrarError("errObra");
  setCargando(btn, true, "Eligiendo…");
  try {
    const excluir = state.obra ? state.obra.name : "";
    const d = await fetchJSON(`/api/dataset/aleatorias?n=1&excluir=${encodeURIComponent(excluir)}`);
    const info = d.imagenes[0];
    const r = await fetch(API + info.url);
    if (!r.ok) throw { mensaje: `No se pudo cargar «${info.nombre}» (HTTP ${r.status}).` };
    const blob = await r.blob();
    setObra(new File([blob], info.nombre, { type: blob.type || "image/jpeg" }));
  } catch (e) {
    const msg = e.mensaje || "No se pudo contactar al servidor.";
    mostrarError("errObra", msg); toast(msg, "err");
  } finally { setCargando(btn, false); }
}
$("btnObraPrecargada").addEventListener("click", obraPrecargada);

async function pedirAleatorias() {
  const btn = $("btnDsAleatorias");
  mostrarErrorDs(null);
  setCargando(btn, true, "Eligiendo al azar…");
  try {
    const excluir = [...ds.items.keys()].filter(k => !k.startsWith("propia:")).join("|");
    const d = await fetchJSON(`/api/dataset/aleatorias?n=${$("dsN").value}&excluir=${encodeURIComponent(excluir)}`);
    ds.items.clear(); ds.cola = []; $("dsGrid").replaceChildren(); $("dsVacio").hidden = true;
    d.imagenes.forEach(info => agregarTarjeta(info));
    if (d.avisos && d.avisos.length) mostrarErrorDs({ tipo: "dataset_no_disponible", mensaje: `${d.avisos.length} imagen(es) no se pudieron leer y se omitieron.` });
    $("btnDsAnalizarTodas").disabled = false;
    cargarEstadoDataset();
    if ($("chkDsAuto").checked) analizarTodas();
  } catch (e) {
    mostrarErrorDs(e.tipo ? e : { tipo: "servidor", mensaje: "No se pudo contactar al servidor. ¿Sigue ejecutándose el notebook?" });
  } finally { setCargando(btn, false); }
}

function etiquetaParticion(cnn, propia) {
  if (propia) return ["Imagen propia", ""];
  if (!cnn) return ["Nunca vista por la CNN", "prueba"];
  return cnn.particion === "prueba" ? ["Conjunto de prueba de la CNN", "prueba"]
    : [cnn.particion === "entrenamiento" ? "Usada para entrenar la CNN" : "Usada en validación de la CNN", "vista"];
}
function agregarTarjeta(info, archivoPropio) {
  const item = { info, archivo: archivoPropio || null, resultado: null, estado: "pendiente", url: info.url.startsWith("/api") ? API + info.url : info.url };
  ds.items.set(info.nombre, item);
  const card = el("article", "ds-card"); card.dataset.nombre = info.nombre;
  item.card = card;
  $("dsGrid").append(card);
  pintarTarjeta(item);
}
function pintarTarjeta(item) {
  const { info, resultado: r, card } = item;
  const [txtTag, clsTag] = etiquetaParticion(info.cnn, !!item.archivo);
  const rostros = r && r.obra && r.obra.rostros;
  card.className = "ds-card" + (item.estado === "analizando" ? " analizando" : "");
  card.replaceChildren();

  const zona = el("div", "ds-img");
  zona.append(rostros && rostros.n ? stageRostros(item.url, rostros, false) : stage(item.url, [], { alt: info.nombre }));
  const tag = el("span", "ds-tag " + clsTag, txtTag); tag.dataset.tip = info.cnn ? `Etiqueta real en el entrenamiento: <b>${info.cnn.clase}</b>` : "La CNN no vio esta imagen al entrenar";
  zona.append(tag);
  card.append(zona);

  const body = el("div", "ds-body");
  const nm = el("div", "ds-name", info.nombre.replace(/^propia:\d+:/, "")); nm.title = info.nombre; body.append(nm);
  if (item.estado === "pendiente") body.append(el("p", "ds-msg", "Sin analizar."));
  else if (item.estado === "analizando") body.insertAdjacentHTML("beforeend", `<p class="ds-msg"><span class="spin-sm"></span>Analizando: CNN, rostros, emociones…</p>`);
  else if (item.estado === "error") body.append(el("p", "ds-msg err", item.error || "No se pudo analizar la imagen."));
  else if (r) {
    const o = r.obra || {};
    const p = rostros && rostros.principal;
    const g = el("div", "ds-res");
    g.innerHTML = `
      <div><span>CNN</span><b>${o.etiqueta ? `${esc(o.etiqueta)} ${pct(o.confianza)}` : "—"}</b></div>
      <div><span>Rostros</span><b>${rostros ? rostros.n : "—"}</b></div>
      <div><span>Emoción</span><b>${p && p.etiqueta ? `${p.etiqueta} ${pct(p.confianza)}` : "Sin rostro"}</b></div>
      <div><span>Calificación</span><b>${p && p.calificacion ? `${Math.round(p.calificacion.puntaje)} · ${p.calificacion.nivel}` : "—"}</b>
        ${p && p.calificacion ? `<div class="meter"><i style="width:${p.calificacion.puntaje}%"></i></div>` : ""}</div>
      ${r.coherencia != null ? `<div class="full" data-tip="1 − |valencia de la obra − valencia del rostro| / 2"><span>Coherencia obra ↔ rostro</span><b>${pct(r.coherencia)} · ${r.nivel_coherencia}</b></div>` : ""}`;
    body.append(g);
    const errores = r.excepciones.filter(x => x.gravedad === "error");
    if (errores.length) body.append(el("p", "ds-msg err", errores[0].mensaje));
  }
  card.append(body);

  const acc = el("div", "ds-actions");
  const bA = el("button", null, item.estado === "listo" ? "Reanalizar" : "Analizar");
  bA.disabled = item.estado === "analizando"; bA.onclick = () => encolar(item.info.nombre, true);
  const bD = el("button", null, "Ver detalle"); bD.disabled = !r; bD.onclick = () => abrirDetalle(item.info.nombre);
  const bU = el("button", "ghost", "Usar con mi cámara"); bU.onclick = () => usarConCamara(item.info.nombre);
  acc.append(bA, bD, bU);
  card.append(acc);
}

function encolar(nombre, prioridad) {
  const item = ds.items.get(nombre);
  if (!item || item.estado === "analizando" || ds.cola.includes(nombre)) return;
  prioridad ? ds.cola.unshift(nombre) : ds.cola.push(nombre);
  bombear();
}
function bombear() {
  while (ds.activos < DS_CONCURRENCIA && ds.cola.length) {
    const nombre = ds.cola.shift();
    ds.activos++;
    analizarUna(nombre).finally(() => { ds.activos--; bombear(); });
  }
}
function analizarTodas() { [...ds.items.keys()].forEach(n => { if (ds.items.get(n).estado !== "listo") encolar(n); }); }

async function analizarUna(nombre) {
  const item = ds.items.get(nombre);
  if (!item) return;
  item.estado = "analizando"; pintarTarjeta(item);
  try {
    const fd = new FormData();
    let url = "/api/dataset/analizar";
    if (item.archivo) { url = "/api/analizar"; fd.append("obra", item.archivo, item.archivo.name); fd.append("modo", "dataset"); }
    else fd.append("nombre", nombre);
    item.resultado = await fetchJSON(url, { method: "POST", body: fd });
    item.estado = item.resultado.obra ? "listo" : "error";
    if (!item.resultado.obra) item.error = (item.resultado.excepciones[0] || {}).mensaje;
  } catch (e) {
    item.estado = "error"; item.error = e.mensaje || "No se pudo contactar al servidor.";
    if (e.tipo && e.tipo !== "servidor") mostrarErrorDs(e);
  }
  if (ds.items.get(nombre) === item) pintarTarjeta(item);
}

function analizarPropia(file) {
  const err = validarArchivo(file);
  if (err) { mostrarErrorDs({ tipo: "formato", mensaje: err }); return; }
  mostrarErrorDs(null);
  const nombre = `propia:${++ds.propias}:${file.name}`;
  $("dsVacio").hidden = true;
  agregarTarjeta({ nombre, cnn: null, url: URL.createObjectURL(file) }, file);
  ds.items.get(nombre).card.querySelector(".ds-name").textContent = file.name;
  $("btnDsAnalizarTodas").disabled = false;
  encolar(nombre, true);
}

/* ---------- detalle ---------- */
let detalleActual = null;
function abrirDetalle(nombre) {
  const item = ds.items.get(nombre);
  if (!item || !item.resultado) return;
  detalleActual = nombre;
  const r = item.resultado, o = r.obra || {}, rostros = o.rostros;
  $("detT").textContent = item.archivo ? item.archivo.name : nombre;
  const body = $("detBody"); body.replaceChildren();
  const grid = el("div", "det-grid");

  const izq = el("div", "det-img");
  const cont = el("div");
  const vistas = [];
  if (rostros && rostros.n) vistas.push(["rostros", `Rostros (${rostros.n})`]);
  vistas.push(["regiones", "Regiones CNN"]);
  if (o.gradcam) vistas.push(["gradcam", "Grad-CAM"]);
  let vista = vistas[0][0];
  const pintar = () => cont.replaceChildren(
    vista === "rostros" ? stageRostros(item.url, rostros, true)
      : vista === "gradcam" ? stage(o.gradcam, [], { alt: "Grad-CAM" }) : stage(item.url, o.regiones || [], { alt: "Regiones de la CNN" }));
  const tg = el("div", "toggle");
  vistas.forEach(([k, t]) => { const b = el("button", null, t); b.setAttribute("aria-pressed", String(k === vista));
    b.onclick = () => { vista = k; [...tg.children].forEach(c => c.setAttribute("aria-pressed", String(c === b))); pintar(); }; tg.append(b); });
  izq.append(tg, cont); pintar();
  const sum = el("div", "summary", r.interpretacion); izq.append(sum);
  const chips = el("div", "exc-chips");
  r.excepciones.forEach(x => { const c = el("span", gravedadCls(x.gravedad), `${ICONOS[x.tipo] || "⚠️"} ${TITULOS[x.tipo] || "Aviso"}`); c.dataset.tip = esc(x.mensaje); chips.append(c); });
  izq.append(chips);
  if (r.excepciones.length) {
    const ul = el("ul", "exc-list"); ul.style.marginTop = "10px";
    r.excepciones.forEach(x => { const li = el("li", "exc " + gravedadCls(x.gravedad));
      li.append(el("span", "ic", ICONOS[x.tipo] || "⚠️"), el("b", null, TITULOS[x.tipo] || "Aviso"), el("span", "mod", x.modulo || ""), el("span", null, x.mensaje)); ul.append(li); });
    izq.append(ul);
  }

  const der = el("div", "det-cards");
  const cR = el("article", "card"), cC = el("article", "card");
  renderRostrosObra(rostros, cR, "Rostros, emociones y calificación");
  renderClasificacion(o.categoria ? o : null, cC);
  der.append(cR, cC);
  grid.append(izq, der);
  body.append(grid);

  const abajo = el("div", "details");
  const cP = el("article", "card"), cT = el("article", "card");
  renderPaleta(o.paleta, cP); renderTiempos(r.tiempos_ms, cT);
  abajo.append(cP, cT);
  body.append(abajo);
  $("dlgDetalle").showModal();
}
$("btnCerrarDet").addEventListener("click", () => $("dlgDetalle").close());
$("btnDetUsar").addEventListener("click", () => { $("dlgDetalle").close(); if (detalleActual) usarConCamara(detalleActual); });
$("btnDetJSON").addEventListener("click", () => {
  const item = ds.items.get(detalleActual); if (!item) return;
  const blob = new Blob([JSON.stringify({ generado: new Date().toISOString(), imagen: detalleActual, ...item.resultado }, null, 2)], { type: "application/json" });
  const a = el("a"); a.href = URL.createObjectURL(blob); a.download = `analisis_${detalleActual.replace(/[^\w.-]+/g, "_")}.json`; a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 2000);
});

/* ---------- llevar una imagen precargada al flujo con cámara ---------- */
async function usarConCamara(nombre) {
  const item = ds.items.get(nombre);
  if (!item) return;
  try {
    let file = item.archivo;
    if (!file) {
      const r = await fetch(item.url);
      if (!r.ok) throw new Error("HTTP " + r.status);
      const blob = await r.blob();
      file = new File([blob], nombre, { type: blob.type || "image/jpeg" });
    }
    cambiarModo("visitante"); irAPaso(1);
    setObra(file);
    $("hintAnalizar").textContent = "Imagen precargada lista como obra. Activa la cámara y observa la obra.";
  } catch (e) { mostrarErrorDs({ tipo: "imagen_no_encontrada", mensaje: `No se pudo cargar «${nombre}» (${e.message}).` }); }
}

$("btnDsAleatorias").addEventListener("click", pedirAleatorias);
$("btnDsAnalizarTodas").addEventListener("click", analizarTodas);
$("btnDsPropia").addEventListener("click", () => $("fileDsPropia").click());
$("fileDsPropia").addEventListener("change", e => { if (e.target.files[0]) analizarPropia(e.target.files[0]); e.target.value = ""; });
$("btnDsRecargar").addEventListener("click", async () => {
  const btn = $("btnDsRecargar"); mostrarErrorDs(null); setCargando(btn, true, "Leyendo carpeta…");
  try { const e = await fetchJSON("/api/dataset/recargar", { method: "POST" }); pintarEstadoDataset(e); toast(`${e.n_imagenes} imágenes en la carpeta.`, "ok", 2200); }
  catch (e) { mostrarErrorDs(e.tipo ? e : { tipo: "servidor", mensaje: "No se pudo contactar al servidor." }); }
  finally { setCargando(btn, false); }
});

/* ---------- inicio ---------- */
cargarEstado();
actualizarBoton();
irAPaso(1);
