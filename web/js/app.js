/* ForensicAI — Dashboard Logic */
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
    const progressPct  = $("#progressPct");
    const resultsSec   = $("#resultsSection");
    const newBtn       = $("#newAnalysisBtn");
    const modal        = $("#vizModal");
    const modalClose   = $("#modalClose");
    const modalTitle   = $("#modalTitle");
    const opacitySlider = $("#opacitySlider");
    const sliderValue  = $("#sliderValue");

    let file = null;

    /* ── Interactive Glow on Drop Zone ─────────────────────── */
    uploadArea.addEventListener("mousemove", e => {
        const rect = uploadArea.getBoundingClientRect();
        const x = ((e.clientX - rect.left) / rect.width) * 100;
        const y = ((e.clientY - rect.top) / rect.height) * 100;
        uploadArea.style.setProperty("--mx", `${x}%`);
        uploadArea.style.setProperty("--my", `${y}%`);
    });

    /* ── Topbar scroll effect ──────────────────────────────── */
    const topbar = $("#topbar");
    window.addEventListener("scroll", () => {
        topbar.style.borderBottomColor = window.scrollY > 10
            ? "rgba(30,30,42,0.8)" : "var(--border)";
    });

    /* ── Upload Logic ──────────────────────────────────────── */
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
        if (f.size > 25e6) { alert("File size exceeds 25MB limit"); return; }
        file = f;
        const r = new FileReader();
        r.onload = e => {
            previewImg.src = e.target.result;
            previewName.textContent = f.name;
            previewSize.textContent = fmt(f.size);
            previewBox.style.display = "block";
            uploadArea.style.display = "none";
            $("#stepsRow").style.display = "none";
            $(".features-section").style.display = "none";
            $(".trust-bar").style.display = "none";
        };
        r.readAsDataURL(f);
    }

    clearBtn.addEventListener("click", reset);
    newBtn.addEventListener("click", () => { reset(); window.scrollTo({ top: 0, behavior: "smooth" }); });

    function reset() {
        file = null; fileInput.value = "";
        previewBox.style.display = "none";
        uploadArea.style.display = "block";
        $("#stepsRow").style.display = "flex";
        $(".features-section").style.display = "block";
        $(".trust-bar").style.display = "flex";
        scanSec.style.display = "none";
        resultsSec.style.display = "none";
        uploadSec.style.display = "block";
    }

    /* ── Analysis Execution ────────────────────────────────── */
    analyzeBtn.addEventListener("click", () => { if (file) run(); });

    async function run() {
        uploadSec.style.display = "none";
        scanSec.style.display = "block";
        scanImg.src = previewImg.src;

        const items = $$(".module-item");
        items.forEach(el => {
            el.classList.remove("done", "active");
        });

        let progress = 0;
        const pInterval = setInterval(() => {
            if (progress < 85) {
                progress += Math.random() * 4;
                const p = Math.min(progress, 85);
                progressFill.style.width = p + "%";
                if (progressPct) progressPct.textContent = Math.round(p) + "%";
            }
        }, 300);

        let moduleIdx = 0;
        const mInterval = setInterval(() => {
            if (moduleIdx > 0 && moduleIdx <= items.length) {
                items[moduleIdx - 1].classList.remove("active");
                items[moduleIdx - 1].classList.add("done");
            }
            if (moduleIdx < items.length) {
                items[moduleIdx].classList.add("active");
                scanStatus.textContent = "Scanning: " + items[moduleIdx].querySelector("span:last-child").textContent;
                moduleIdx++;
            }
        }, 800);

        try {
            const fd = new FormData(); fd.append("file", file);

            // AbortController with 120s timeout for large images
            const controller = new AbortController();
            const timeout = setTimeout(() => controller.abort(), 120000);

            const res = await fetch("/api/analyze", {
                method: "POST",
                body: fd,
                signal: controller.signal,
                credentials: "same-origin",
            });

            clearTimeout(timeout);
            clearInterval(pInterval);
            clearInterval(mInterval);

            if (!res.ok) {
                let errMsg = "Analysis pipeline failed";
                try { errMsg = (await res.json()).detail || errMsg; } catch(_) {}
                throw new Error(errMsg);
            }
            const data = await res.json();

            items.forEach(el => {
                el.classList.remove("active");
                el.classList.add("done");
            });
            progressFill.style.width = "100%";
            if (progressPct) progressPct.textContent = "100%";
            scanStatus.textContent = "Analysis Complete";

            setTimeout(() => render(data), 800);
        } catch (err) {
            clearInterval(pInterval);
            clearInterval(mInterval);
            let msg = err.message;
            if (err.name === "AbortError") {
                msg = "Analysis timed out. The image may be too large or the server is busy. Please try again.";
            } else if (msg === "Failed to fetch" || msg.includes("NetworkError")) {
                msg = "Network error: Could not reach the server. Make sure the server is running at " + window.location.origin;
            }
            alert(msg);
            reset();
        }
    }

    /* ── Results Rendering ─────────────────────────────────── */
    function render(data) {
        scanSec.style.display = "none";
        resultsSec.style.display = "block";

        const score = data.overall_score;
        const verdict = data.overall_verdict;

        // Gauge animation
        const circ = 2 * Math.PI * 52;
        const offset = circ - (score / 100) * circ;
        const fill = $("#gaugeFill");
        const color = score < 30 ? "var(--green)" : score < 60 ? "var(--amber)" : "var(--red)";

        fill.setAttribute("stroke", color);
        setTimeout(() => { fill.style.strokeDashoffset = offset; }, 100);

        const num = $("#scoreNumber");
        num.style.color = color;
        animateNumber(num, 0, Math.round(score), 1500);

        const badge = $("#verdictBadge");
        badge.textContent = verdict;
        badge.className = "verdict-pill " + verdict.toLowerCase();

        $("#analysisTime").textContent = `${data.elapsed_seconds}s processing`;
        $("#analysisFile").textContent = data.filename;

        // Module Cards
        const grid = $("#moduleResultsGrid");
        grid.innerHTML = "";
        const keys = ["ela", "copymove", "noise", "ai_detection", "heatmap", "metadata"];

        keys.forEach(key => {
            const m = data.modules[key]; if (!m) return;
            const cls = getVerdictClass(m.verdict);
            const pct = Math.round(m.score * 100);

            const card = document.createElement("div");
            card.className = "card";
            card.innerHTML = `
                <div class="card-head">
                    <span class="card-name">${escape(m.display_name)}</span>
                    <span class="card-pill ${cls}">${escape(m.verdict)}</span>
                </div>
                <div class="card-body">
                    <div class="card-bar"><div style="width:${pct}%;height:100%;background:var(--${cls === 'green' ? 'green' : cls === 'amber' ? 'amber' : 'red'});border-radius:2px;transition:width .6s ease"></div></div>
                    <ul class="card-flags">${m.flags.map(f => `<li>${escape(f)}</li>`).join("")}</ul>
                </div>
                <div class="card-foot">
                    <button class="link-btn j-details">View Details</button>
                    ${m.visualization ? `<button class="viz-btn j-viz">Show Visualization</button>` : ""}
                </div>`;
            card._data = m;
            grid.appendChild(card);
        });

        grid.onclick = e => {
            const btn = e.target.closest("button"); if (!btn) return;
            const card = btn.closest(".card");
            if (btn.classList.contains("j-details")) openDetails(card._data);
            if (btn.classList.contains("j-viz")) openViz(card._data);
        };

        resultsSec.scrollIntoView({ behavior: "smooth" });
    }

    /* ── Modal Components ──────────────────────────────────── */
    function openViz(m) {
        modalTitle.textContent = "Visualization — " + m.display_name;
        $(".modal-body").innerHTML = `<img id="modalImage" alt="" src="data:image/png;base64,${m.visualization}" style="filter:contrast(1.1) brightness(1.1);border-radius:8px;">`;
        $("#modalControls").style.display = "block";
        modal.style.display = "flex";
        opacitySlider.value = 100;
        if (sliderValue) sliderValue.textContent = "100%";
        opacitySlider.oninput = () => {
            $("#modalImage").style.opacity = opacitySlider.value / 100;
            if (sliderValue) sliderValue.textContent = opacitySlider.value + "%";
        };
    }

    function openDetails(m) {
        modalTitle.textContent = "Details — " + m.display_name;
        let html = '<div class="details-grid">';
        for (const [key, val] of Object.entries(m.details || {})) {
            html += `
                <div class="detail-item">
                    <span class="label">${formatKey(key)}</span>
                    <span class="value">${escape(typeof val === "object" ? JSON.stringify(val) : String(val))}</span>
                </div>`;
        }
        html += '</div>';
        $(".modal-body").innerHTML = html;
        $("#modalControls").style.display = "none";
        modal.style.display = "flex";
    }

    modalClose.onclick = () => { modal.style.display = "none"; };
    window.addEventListener("click", e => { if (e.target === modal) modal.style.display = "none"; });
    document.addEventListener("keydown", e => { if (e.key === "Escape") modal.style.display = "none"; });

    /* ── Utilities ─────────────────────────────────────────── */
    function fmt(bytes) {
        if (bytes < 1024) return bytes + " B";
        if (bytes < 1048576) return (bytes / 1024).toFixed(1) + " KB";
        return (bytes / 1048576).toFixed(1) + " MB";
    }
    function formatKey(k) { return k.replace(/_/g, " ").replace(/\b\w/g, c => c.toUpperCase()); }
    function getVerdictClass(v) {
        v = (v || "").toLowerCase();
        if (v.includes("authentic") || v.includes("real")) return "green";
        if (v.includes("suspicious")) return "amber";
        return "red";
    }
    function escape(str) { const d = document.createElement("div"); d.textContent = str; return d.innerHTML; }
    function animateNumber(el, start, end, duration) {
        const startTime = performance.now();
        function update(currentTime) {
            const elapsed = currentTime - startTime;
            const progress = Math.min(elapsed / duration, 1);
            const current = Math.floor(start + (end - start) * (1 - Math.pow(1 - progress, 3)));
            el.textContent = current;
            if (progress < 1) requestAnimationFrame(update);
        }
        requestAnimationFrame(update);
    }

    /* ── Intersection Observer for feature cards ───────────── */
    const featureCards = $$(".feature-card");
    if (featureCards.length) {
        const observer = new IntersectionObserver(entries => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    entry.target.style.opacity = "1";
                    entry.target.style.transform = "translateY(0)";
                }
            });
        }, { threshold: 0.1 });

        featureCards.forEach((card, i) => {
            card.style.opacity = "0";
            card.style.transform = "translateY(24px)";
            card.style.transition = `all 0.5s cubic-bezier(.16,1,.3,1) ${i * 0.08}s`;
            observer.observe(card);
        });
    }
})();
