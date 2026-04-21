/* ForensicAI */
(function () {
    "use strict";

    const $ = s => document.querySelector(s);
    const $$ = s => document.querySelectorAll(s);

    const uploadArea   = $("#uploadArea");
    const fileInput    = $("#fileInput");
    const previewBox   = $("#previewContainer");
    const previewImg   = $("#previewImage");
    const previewName  = $("#previewName");
    const previewSize  = $("#previewSize");
    const analyzeBtn   = $("#analyzeBtn");
    const clearBtn     = $("#clearBtn");
    const uploadSec    = $("#uploadSection");
    const scanSec      = $("#scanningSection");
    const scanImg      = $("#scanImage");
    const scanStatus   = $("#scanStatus");
    const progressFill = $("#progressFill");
    const resultsSec   = $("#resultsSection");
    const newBtn       = $("#newAnalysisBtn");
    const modal        = $("#vizModal");
    const modalClose   = $("#modalClose");
    const modalTitle   = $("#modalTitle");

    let file = null;

    /* ── upload ──────────────────────────────────────────── */
    uploadArea.addEventListener("click", () => fileInput.click());
    uploadArea.addEventListener("dragover", e => { e.preventDefault(); uploadArea.classList.add("over"); });
    uploadArea.addEventListener("dragleave", () => uploadArea.classList.remove("over"));
    uploadArea.addEventListener("drop", e => {
        e.preventDefault(); uploadArea.classList.remove("over");
        if (e.dataTransfer.files.length && e.dataTransfer.files[0].type.startsWith("image/"))
            pick(e.dataTransfer.files[0]);
    });
    fileInput.addEventListener("change", () => { if (fileInput.files.length) pick(fileInput.files[0]); });

    function pick(f) {
        if (f.size > 25e6) { alert("Max 25 MB"); return; }
        file = f;
        const r = new FileReader();
        r.onload = e => {
            previewImg.src = e.target.result;
            previewName.textContent = f.name;
            previewSize.textContent = fmt(f.size);
            previewBox.style.display = "block";
            uploadArea.style.display = "none";
            // hide capabilities row when preview is shown
            const caps = $(".capabilities");
            if (caps) caps.style.display = "none";
        };
        r.readAsDataURL(f);
    }

    clearBtn.addEventListener("click", reset);
    newBtn.addEventListener("click", () => { reset(); window.scrollTo({ top: 0, behavior: "smooth" }); });

    function reset() {
        file = null; fileInput.value = "";
        previewBox.style.display = "none";
        uploadArea.style.display = "block";
        const caps = $(".capabilities");
        if (caps) caps.style.display = "flex";
        scanSec.style.display = "none";
        resultsSec.style.display = "none";
        uploadSec.style.display = "block";
    }

    /* ── analyze ─────────────────────────────────────────── */
    analyzeBtn.addEventListener("click", () => { if (file) run(); });

    async function run() {
        uploadSec.style.display = "none";
        scanSec.style.display = "block";
        resultsSec.style.display = "none";
        scanImg.src = previewImg.src;

        const items = $$(".module-item");
        items.forEach(el => { el.classList.remove("done","active"); el.querySelector(".module-check").textContent = "·"; });

        let prog = 0;
        const tick = setInterval(() => { if (prog < 82) { prog += Math.random() * 9; progressFill.style.width = Math.min(prog, 82) + "%"; } }, 350);
        const anim = setInterval(() => {
            const cur = document.querySelector(".module-item.active");
            const nxt = document.querySelector(".module-item:not(.done):not(.active)");
            if (cur) { cur.classList.replace("active","done"); cur.querySelector(".module-check").textContent = "✓"; }
            if (nxt) {
                nxt.classList.add("active");
                nxt.querySelector(".module-check").textContent = "→";
                scanStatus.textContent = nxt.querySelector("span:last-child").textContent;
            }
        }, 700);

        try {
            const fd = new FormData(); fd.append("file", file);
            const res = await fetch("/api/analyze", { method: "POST", body: fd });
            clearInterval(tick); clearInterval(anim);
            if (!res.ok) throw new Error((await res.json()).detail || "Failed");
            const data = await res.json();

            items.forEach(el => { el.classList.remove("active"); el.classList.add("done"); el.querySelector(".module-check").textContent = "✓"; });
            progressFill.style.width = "100%";
            scanStatus.textContent = "Complete";
            setTimeout(() => render(data), 600);
        } catch (e) {
            clearInterval(tick); clearInterval(anim);
            alert(e.message); reset();
        }
    }

    /* ── render ───────────────────────────────────────────── */
    function render(data) {
        scanSec.style.display = "none";
        resultsSec.style.display = "block";

        const score = data.overall_score;
        const v = data.overall_verdict;

        // gauge
        const circ = 2 * Math.PI * 52;
        const off = circ - (score / 100) * circ;
        const fill = $("#gaugeFill");
        let col = score < 30 ? "#3dcc7a" : score < 60 ? "#e0a535" : "#e05252";
        fill.setAttribute("stroke", col);
        setTimeout(() => { fill.style.strokeDashoffset = off; }, 60);

        const num = $("#scoreNumber");
        num.style.color = col;
        animNum(num, 0, Math.round(score), 1200);

        const pill = $("#verdictBadge");
        pill.textContent = v;
        pill.className = "verdict-pill " + v.toLowerCase();

        $("#analysisTime").textContent = data.elapsed_seconds + "s";
        $("#analysisFile").textContent = data.filename;

        // cards
        const grid = $("#moduleResultsGrid");
        grid.innerHTML = "";
        const order = ["ela","copymove","noise","ai_detection","heatmap","metadata"];

        for (const key of order) {
            const m = data.modules[key]; if (!m) continue;
            const vc = vclass(m.verdict);
            const pct = Math.round(m.score * 100);

            const el = document.createElement("div");
            el.className = "card";
            el.innerHTML = `
                <div class="card-head">
                    <span class="card-name">${esc(m.display_name)}</span>
                    <span class="card-pill ${vc}">${esc(m.verdict)}</span>
                </div>
                <div class="card-body">
                    <div class="card-bar"><div class="card-bar-fill ${vc}" style="width:${pct}%"></div></div>
                    <ul class="card-flags">${m.flags.map(f => `<li>${esc(f)}</li>`).join("")}</ul>
                </div>
                <div class="card-foot">
                    <button class="link-btn j-details">Details</button>
                    ${m.visualization ? `<button class="viz-btn j-viz">Visualization</button>` : ""}
                </div>`;
            el._d = m;
            grid.appendChild(el);
        }

        grid.addEventListener("click", e => {
            const card = e.target.closest(".card"); if (!card) return;
            if (e.target.closest(".j-details")) openDetails(card._d);
            if (e.target.closest(".j-viz")) openViz(card._d);
        });

        resultsSec.scrollIntoView({ behavior: "smooth" });
    }

    /* ── modals ───────────────────────────────────────────── */
    function openViz(m) {
        modalTitle.textContent = m.display_name;
        $(".modal-body").innerHTML = `<img id="modalImage" alt="" src="data:image/png;base64,${m.visualization}">`;
        $("#modalControls").style.display = "block";
        modal.style.display = "flex";
    }
    function openDetails(m) {
        modalTitle.textContent = m.display_name;
        let h = '<div class="details-grid">';
        for (const [k,v] of Object.entries(m.details || {})) {
            h += `<div class="detail-item"><span class="label">${fmtKey(k)}</span><span class="value">${esc(typeof v==="object"?JSON.stringify(v):String(v))}</span></div>`;
        }
        h += '</div>';
        $(".modal-body").innerHTML = h;
        $("#modalControls").style.display = "none";
        modal.style.display = "flex";
    }

    modalClose.addEventListener("click", closeModal);
    modal.addEventListener("click", e => { if (e.target === modal) closeModal(); });
    document.addEventListener("keydown", e => { if (e.key === "Escape") closeModal(); });

    function closeModal() {
        modal.style.display = "none";
        $(".modal-body").innerHTML = '<img id="modalImage" alt="">';
        $("#modalControls").style.display = "block";
    }

    /* ── helpers ──────────────────────────────────────────── */
    function fmt(b) { return b < 1024 ? b+" B" : b < 1048576 ? (b/1024).toFixed(1)+" KB" : (b/1048576).toFixed(1)+" MB"; }
    function fmtKey(k) { return k.replace(/_/g," ").replace(/\b\w/g,c=>c.toUpperCase()); }
    function vclass(v) { v=(v||"").toLowerCase(); return v.includes("authentic")||v.includes("real")?"authentic":v.includes("suspicious")?"suspicious":"manipulated"; }
    function esc(s) { const d=document.createElement("div"); d.textContent=s; return d.innerHTML; }
    function animNum(el,from,to,ms) {
        const t0=performance.now();
        (function f(t){ const p=Math.min((t-t0)/ms,1); el.textContent=Math.round(from+(to-from)*(1-Math.pow(1-p,3))); if(p<1) requestAnimationFrame(f); })(t0);
    }
})();
