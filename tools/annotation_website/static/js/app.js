// LangLoc annotation page client logic.
// Plain ES-modules-free vanilla JS so it runs without a build step.

(function () {
  "use strict";

  const root = document.querySelector(".annotate");
  if (!root) return;

  const sceneId = root.dataset.scene;
  const frameId = root.dataset.frame;
  const editMode = root.dataset.editMode === "1";
  const minWords = parseInt(root.dataset.minWords, 10) || 25;
  const minChars = parseInt(root.dataset.minChars, 10) || 80;
  const maxChars = parseInt(root.dataset.maxChars, 10) || 2000;

  const ta = document.getElementById("description");
  const wcEl = document.getElementById("word-count");
  const barFill = document.getElementById("length-bar-fill");
  const hint = document.getElementById("length-hint");
  const nextBtn = document.getElementById("next-button");
  const status = document.getElementById("save-status");

  let firstKeystrokeAt = null;
  let lastSavedText = ta.value || "";
  let hasSavedRoundTrip = !!lastSavedText; // restored draft counts as saved

  function wordCount(s) {
    const m = (s || "").match(/[A-Za-z']+/g);
    return m ? m.length : 0;
  }

  function setStatus(text, klass) {
    status.textContent = text;
    status.classList.remove("saving", "error", "saved");
    if (klass) status.classList.add(klass);
    else if (text && /^(Saved|Loaded|Restored)/.test(text)) status.classList.add("saved");
  }

  function updateLengthFeedback() {
    const wc = wordCount(ta.value);
    const cc = (ta.value || "").length;
    wcEl.textContent = wc;
    // Bar fills based on chars vs minChars when min_words is disabled (=0),
    // otherwise on the word-count progress like before.
    const ratio = minWords > 0
      ? Math.max(0, Math.min(1, wc / minWords))
      : Math.max(0, Math.min(1, cc / Math.max(1, minChars)));
    barFill.style.width = (ratio * 100).toFixed(0) + "%";
    const wordsOk = minWords <= 0 || wc >= minWords;
    if (wordsOk && cc >= minChars) {
      barFill.classList.add("ok");
      hint.classList.add("ok");
      hint.classList.remove("hidden");
      hint.textContent = "Looks good";
      nextBtn.disabled = false;
    } else {
      barFill.classList.remove("ok");
      hint.classList.remove("ok");
      hint.classList.remove("hidden");
      hint.textContent =
        cc < 12
          ? "Just getting started — keep going"
          : "A bit short — try one more sentence";
      nextBtn.disabled = true;
    }
    if (cc > maxChars) {
      hint.textContent = "Way too long — please trim";
      nextBtn.disabled = true;
    }
  }

  function durationMs() {
    if (!firstKeystrokeAt) return 0;
    return Date.now() - firstKeystrokeAt;
  }

  async function send({ submit = false } = {}) {
    const text = ta.value || "";
    if (!submit && text === lastSavedText) return { skipped: true };

    setStatus("Saving…", "saving");
    let url, method;
    if (editMode && submit) {
      url = "/api/edit";
      method = "POST";
    } else if (submit) {
      url = "/api/submit";
      method = "POST";
    } else {
      url = "/api/save";
      method = "PUT";
    }

    try {
      const res = await fetch(url, {
        method,
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          scene_id: sceneId,
          frame_id: frameId,
          text,
          duration_ms: durationMs(),
        }),
      });
      if (res.status === 409) {
        setStatus("Reassigning…", "error");
        window.location.href = "/annotate";
        return { reassigned: true };
      }
      if (!res.ok) {
        const body = await res.json().catch(() => ({ detail: "save failed" }));
        setStatus("Error: " + (body.detail || res.statusText), "error");
        return { error: body.detail };
      }
      lastSavedText = text;
      hasSavedRoundTrip = true;
      const ts = new Date();
      setStatus("Saved " + ts.toLocaleTimeString());
      return await res.json();
    } catch (err) {
      setStatus("Network issue — retrying", "error");
      return { error: String(err) };
    }
  }

  // -- typing handler with debounced auto-save -------------------------------
  let saveTimer = null;
  ta.addEventListener("input", () => {
    if (firstKeystrokeAt === null) firstKeystrokeAt = Date.now();
    updateLengthFeedback();
    setStatus("Typing…", "saving");
    if (saveTimer) clearTimeout(saveTimer);
    saveTimer = setTimeout(() => send(), 1200);
  });
  ta.addEventListener("blur", () => {
    if (saveTimer) clearTimeout(saveTimer);
    send();
  });

  // Cmd/Ctrl-Enter submits when the next button is enabled
  ta.addEventListener("keydown", (ev) => {
    if ((ev.metaKey || ev.ctrlKey) && ev.key === "Enter") {
      ev.preventDefault();
      if (!nextBtn.disabled) nextBtn.click();
    }
  });

  // -- chips: insert phrase at cursor ---------------------------------------
  document.querySelectorAll(".chip[data-insert]").forEach((chip) => {
    chip.addEventListener("click", () => {
      const phrase = chip.dataset.insert;
      const start = ta.selectionStart || 0;
      const end = ta.selectionEnd || 0;
      const before = ta.value.slice(0, start);
      const after = ta.value.slice(end);
      const sep = before.length === 0 || /[\s\n.!?]$/.test(before) ? "" : " ";
      ta.value = before + sep + phrase + after;
      const newPos = (before + sep + phrase).length;
      ta.focus();
      ta.setSelectionRange(newPos, newPos);
      ta.dispatchEvent(new Event("input"));
    });
  });

  // -- next button -----------------------------------------------------------
  nextBtn.addEventListener("click", async () => {
    nextBtn.disabled = true;
    const res = await send({ submit: true });
    if (res && (res.submitted || res.edited || res.reassigned)) {
      window.location.href = editMode ? "/history" : "/annotate";
    } else {
      updateLengthFeedback();
    }
  });

  // -- lightbox --------------------------------------------------------------
  const dialog = document.getElementById("lightbox");
  const lightboxImg = document.getElementById("lightbox-img");
  const closeBtn = document.getElementById("lightbox-close");
  const zoomBtn = document.getElementById("image-zoom");
  const frameImg = document.getElementById("frame-image");

  function openLightbox() {
    if (!dialog || !dialog.showModal) return;
    lightboxImg.src = frameImg.src;
    dialog.showModal();
  }
  function closeLightbox() {
    if (!dialog || !dialog.close) return;
    dialog.close();
  }
  if (zoomBtn) zoomBtn.addEventListener("click", openLightbox);
  if (closeBtn) closeBtn.addEventListener("click", closeLightbox);
  if (dialog) {
    dialog.addEventListener("click", (ev) => {
      if (ev.target === dialog) closeLightbox();
    });
  }

  // -- worked-example modal --------------------------------------------------
  const exampleModal = document.getElementById("example-modal");
  const exampleClose = document.getElementById("example-modal-close");
  const exampleX = document.getElementById("example-modal-x");
  const showExampleBtn = document.getElementById("show-example-button");
  const EXAMPLE_FLAG = "langloc:exampleSeen";

  function openExample() {
    if (!exampleModal || !exampleModal.showModal) return;
    if (!exampleModal.open) exampleModal.showModal();
  }
  function closeExample() {
    if (!exampleModal || !exampleModal.close) return;
    exampleModal.close();
    try { localStorage.setItem(EXAMPLE_FLAG, "1"); } catch (e) {}
    // return focus to the trigger if there is one (a11y)
    if (showExampleBtn && document.activeElement !== showExampleBtn) {
      try { showExampleBtn.focus({ preventScroll: true }); } catch (e) {}
    }
  }

  if (exampleClose) exampleClose.addEventListener("click", closeExample);
  if (exampleX) exampleX.addEventListener("click", closeExample);
  if (showExampleBtn) showExampleBtn.addEventListener("click", openExample);
  if (exampleModal) {
    exampleModal.addEventListener("click", (ev) => {
      if (ev.target === exampleModal) closeExample();
    });
  }

  // first /annotate visit per browser → auto-open. Skip in edit mode and
  // skip if the user prefers reduced motion (don't surprise them with a
  // popping modal). Open via DOMContentLoaded rather than a fixed timeout
  // so it doesn't interrupt a user who's already typing.
  const reduced = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  if (!editMode && !reduced) {
    let seen = "0";
    try { seen = localStorage.getItem(EXAMPLE_FLAG) || "0"; } catch (e) {}
    if (seen !== "1") {
      // wait for the page to be settled, then open without a startle delay
      if (document.readyState === "complete") openExample();
      else window.addEventListener("load", openExample, { once: true });
    }
  }

  // -- initial state ---------------------------------------------------------
  updateLengthFeedback();
  if (editMode && lastSavedText) setStatus("Loaded your earlier description");
  else if (lastSavedText) setStatus("Restored your earlier draft");
  else setStatus("Ready");
})();
