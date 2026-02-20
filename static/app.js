// viz-bio2token frontend

(function () {
    "use strict";

    // --- DOM elements ---
    const tokenEditor = document.getElementById("token-editor");
    const tokenCount = document.getElementById("token-count");
    const statusIndicator = document.getElementById("status-indicator");
    const fileInput = document.getElementById("file-input");
    const styleSelect = document.getElementById("style-select");
    const showOriginalCheckbox = document.getElementById("show-original");
    const viewerContainer = document.getElementById("viewer-container");
    const divider = document.getElementById("divider");
    const leftPane = document.getElementById("left-pane");

    // Metadata fields (bio2token)
    const metaAtomNames = document.getElementById("meta-atom-names");
    const metaResidueTypes = document.getElementById("meta-residue-types");
    const metaResidueIds = document.getElementById("meta-residue-ids");
    const metaClear = document.getElementById("meta-clear");
    const metaStatus = document.getElementById("meta-status");
    const metadataPanel = document.getElementById("metadata-panel");

    // Tokenizer selector
    const tokenizerSelect = document.getElementById("tokenizer-select");

    // APT options
    const aptOptions = document.getElementById("apt-options");
    const aptSteps = document.getElementById("apt-steps");
    const aptStepsValue = document.getElementById("apt-steps-value");
    const aptNumResidues = document.getElementById("apt-num-residues");
    const aptNumTokens = document.getElementById("apt-num-tokens");

    // Kanzi options
    const kanziOptions = document.getElementById("kanzi-options");
    const kanziSteps = document.getElementById("kanzi-steps");
    const kanziStepsValue = document.getElementById("kanzi-steps-value");
    const kanziCfg = document.getElementById("kanzi-cfg");
    const kanziCfgValue = document.getElementById("kanzi-cfg-value");
    const kanziNumTokens = document.getElementById("kanzi-num-tokens");

    // --- State ---
    let viewer = null;
    let currentDecodedPdb = null;
    let currentOriginalPdb = null;
    let storedMetadata = null;  // metadata from encode, used in decode
    let decodeTimer = null;
    let modelReady = { bio2token: false, apt: false, kanzi: false };
    let currentTokenizer = "bio2token";
    let aptState = { numResidues: null, numTokens: 0 };

    // --- 3Dmol viewer init ---
    function initViewer() {
        viewer = $3Dmol.createViewer(viewerContainer, {
            backgroundColor: "#1a1a2e",
        });
        viewer.resize();
    }

    // PDB code elements
    const pdbCodeInput = document.getElementById("pdb-code-input");
    const pdbCodeLoad = document.getElementById("pdb-code-load");

    // --- Token parsing ---
    // Returns {ids: number[], spans: [{start, end}, ...]}
    // spans[i] gives the character range in the original text for token i
    function parseTokensWithSpans(text) {
        const ids = [];
        const spans = [];
        const regex = /(\d+)/g;
        let match;
        while ((match = regex.exec(text)) !== null) {
            const num = parseInt(match[1], 10);
            if (!isNaN(num) && num >= 0 && num <= 4095) {
                ids.push(num);
                spans.push({ start: match.index, end: match.index + match[0].length });
            }
        }
        return { ids, spans };
    }

    function parseTokens(text) {
        return parseTokensWithSpans(text).ids;
    }

    // --- Metadata parsing ---
    function parseMetadataFields() {
        const atomNamesRaw = metaAtomNames.value.trim();
        const residueTypesRaw = metaResidueTypes.value.trim();
        const residueIdsRaw = metaResidueIds.value.trim();

        if (!atomNamesRaw && !residueTypesRaw && !residueIdsRaw) {
            return null;  // No metadata set
        }

        const result = {};
        if (atomNamesRaw) {
            result.atom_names = atomNamesRaw.split(",").map(s => s.trim()).filter(s => s);
        }
        if (residueTypesRaw) {
            result.residue_types = residueTypesRaw.split(",").map(s => s.trim()).filter(s => s);
        }
        if (residueIdsRaw) {
            result.residue_ids = residueIdsRaw.split(",").map(s => parseInt(s.trim(), 10)).filter(n => !isNaN(n));
        }
        return result;
    }

    function populateMetadataFields(metadata) {
        if (!metadata) return;
        if (metadata.atom_names) {
            metaAtomNames.value = metadata.atom_names.join(", ");
        }
        if (metadata.residue_types) {
            metaResidueTypes.value = metadata.residue_types.join(", ");
        }
        if (metadata.residue_ids) {
            metaResidueIds.value = metadata.residue_ids.join(", ");
        }
        metaStatus.textContent = `${metadata.atom_names ? metadata.atom_names.length : 0} atoms loaded from encode`;
    }

    function clearMetadataFields() {
        metaAtomNames.value = "";
        metaResidueTypes.value = "";
        metaResidueIds.value = "";
        storedMetadata = null;
        metaStatus.textContent = "Using default CA/ALA metadata";
    }

    // --- Tokenizer switching ---
    function clearState() {
        tokenEditor.value = "";
        tokenCount.textContent = "0 tokens";
        currentDecodedPdb = null;
        currentOriginalPdb = null;
        storedMetadata = null;
        aptState = { numResidues: null, numTokens: 0 };
        showOriginalCheckbox.checked = false;
        showOriginalCheckbox.disabled = true;
        clearMetadataFields();
        aptNumResidues.textContent = "-";
        aptNumTokens.textContent = "-";
        kanziNumTokens.textContent = "-";
        if (viewer) {
            viewer.removeAllModels();
            viewer.removeAllLabels();
            viewer.render();
        }
    }

    function switchTokenizer(tokenizer) {
        if (tokenizer === currentTokenizer) return;
        currentTokenizer = tokenizer;
        clearState();

        if (tokenizer === "apt") {
            metadataPanel.style.display = "none";
            aptOptions.style.display = "";
            kanziOptions.style.display = "none";
            tokenEditor.placeholder = "Paste APT tokens here (1\u2013128 global tokens), e.g.:\n3842 1027 2955 512 3701 88 1444 2231";
            styleSelect.value = "sphere";
        } else if (tokenizer === "kanzi") {
            metadataPanel.style.display = "none";
            aptOptions.style.display = "none";
            kanziOptions.style.display = "";
            tokenEditor.placeholder = "Paste Kanzi tokens here (1 per residue), e.g.:\n3842 1027 2955 512 3701 88 1444 2231";
            styleSelect.value = "sphere";
        } else {
            metadataPanel.style.display = "";
            aptOptions.style.display = "none";
            kanziOptions.style.display = "none";
            tokenEditor.placeholder = "Paste bio2token tokens here, e.g.:\n2239 2751 2619 1082 3131 3127 1591 2847";
            styleSelect.value = "cartoon";
        }
    }

    // --- API calls ---
    async function decodeTokens(tokenIds) {
        const body = { token_ids: tokenIds, tokenizer: currentTokenizer };

        if (currentTokenizer === "kanzi") {
            body.n_steps = parseInt(kanziSteps.value, 10);
            body.cfg_weight = parseFloat(kanziCfg.value);
        } else if (currentTokenizer === "apt") {
            body.n_steps = parseInt(aptSteps.value, 10);
            if (aptState.numResidues) {
                body.num_residues = aptState.numResidues;
            }
        } else {
            // bio2token: include metadata
            const uiMetadata = parseMetadataFields();
            const metadata = uiMetadata || storedMetadata;
            if (metadata) {
                if (metadata.atom_names) body.atom_names = metadata.atom_names;
                if (metadata.residue_types) body.residue_types = metadata.residue_types;
                if (metadata.residue_ids) body.residue_ids = metadata.residue_ids;
            }
            // chain_ids aren't editable in the UI, always use storedMetadata
            if (storedMetadata && storedMetadata.chain_ids) {
                body.chain_ids = storedMetadata.chain_ids;
            }
        }

        // Send ground-truth PDB for Kabsch alignment when available
        if (currentOriginalPdb) {
            body.gt_pdb_string = currentOriginalPdb;
        }

        const res = await fetch("/api/decode", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(body),
        });

        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.detail || "Decode failed");
        }
        return await res.json();
    }

    async function encodeFile(file) {
        const formData = new FormData();
        formData.append("file", file);

        const res = await fetch(`/api/encode?tokenizer=${currentTokenizer}`, {
            method: "POST",
            body: formData,
        });

        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.detail || "Encode failed");
        }
        return await res.json();
    }

    async function encodePdbCode(code) {
        const res = await fetch("/api/encode-pdb-code", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ pdb_code: code, tokenizer: currentTokenizer }),
        });

        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.detail || "PDB fetch failed");
        }
        return await res.json();
    }

    async function handleEncodeResult(result) {
        tokenEditor.value = result.token_string;
        tokenCount.textContent = `${result.num_tokens} tokens`;
        currentOriginalPdb = result.gt_pdb_string;

        if (result.tokenizer === "kanzi") {
            kanziNumTokens.textContent = result.num_tokens;
        } else if (result.tokenizer === "apt") {
            aptState.numResidues = result.num_residues;
            aptState.numTokens = result.num_tokens;
            aptNumResidues.textContent = result.num_residues || "-";
            aptNumTokens.textContent = result.num_tokens;
        } else {
            storedMetadata = {
                atom_names: result.atom_names,
                residue_types: result.residue_types,
                residue_ids: result.residue_ids,
                chain_ids: result.chain_ids,
            };
            populateMetadataFields(storedMetadata);
        }

        showOriginalCheckbox.disabled = false;
        showOriginalCheckbox.checked = true;

        const decoded = await decodeTokens(result.token_ids);
        currentDecodedPdb = decoded.pdb_string;
        renderStructures();
    }

    // --- Atom click -> token highlight ---
    let highlightedAtomIdx = -1;  // currently highlighted token index

    function onAtomClicked(atom) {
        if (!atom) return;
        // atom.serial is 1-based PDB serial -> token index is serial - 1
        const tokenIdx = atom.serial - 1;
        highlightTokenInEditor(tokenIdx);
        highlightAtomInViewer(tokenIdx);
    }

    function highlightTokenInEditor(tokenIdx) {
        const { spans } = parseTokensWithSpans(tokenEditor.value);
        if (tokenIdx < 0 || tokenIdx >= spans.length) return;

        const span = spans[tokenIdx];
        tokenEditor.focus();
        tokenEditor.setSelectionRange(span.start, span.end);

        // Scroll the selection into view
        const textBefore = tokenEditor.value.substring(0, span.start);
        const linesBefore = textBefore.split("\n").length - 1;
        const lineHeight = parseInt(getComputedStyle(tokenEditor).lineHeight) || 21;
        const targetScroll = Math.max(0, linesBefore * lineHeight - tokenEditor.clientHeight / 2);
        tokenEditor.scrollTop = targetScroll;

        highlightedAtomIdx = tokenIdx;
    }

    function highlightAtomInViewer(tokenIdx) {
        if (!viewer || !currentDecodedPdb) return;

        const style = styleSelect.value;

        // Reset all decoded atoms to chain-colored style (model 0)
        applyChainColors(0);

        // Highlight the clicked atom: bright yellow, larger
        if (tokenIdx >= 0) {
            viewer.setStyle(
                { model: 0, serial: tokenIdx + 1 },
                { sphere: { color: "#ffe033", radius: 0.6 } }
            );
            // Add a label
            viewer.removeAllLabels();
            const atoms = viewer.getModel(0).selectedAtoms({ serial: tokenIdx + 1 });
            if (atoms.length > 0) {
                const a = atoms[0];
                const label = `${a.atom} ${a.resn}${a.resi} [token ${tokenIdx}]`;
                viewer.addLabel(label, {
                    position: { x: a.x, y: a.y, z: a.z },
                    fontSize: 12,
                    backgroundColor: "rgba(0,0,0,0.7)",
                    fontColor: "#ffe033",
                    borderRadius: 4,
                    padding: 4,
                });
            }
        }

        // Re-apply original model style if shown
        if (currentOriginalPdb && showOriginalCheckbox.checked) {
            const origSpec = getStyleSpec(style, true);
            viewer.setStyle({ model: 1 }, origSpec);
        }

        viewer.render();
    }

    // --- Rendering ---
    function renderStructures() {
        if (!viewer) return;
        viewer.removeAllModels();
        viewer.removeAllLabels();
        highlightedAtomIdx = -1;

        const style = styleSelect.value;

        if (currentDecodedPdb) {
            const model = viewer.addModel(currentDecodedPdb, "pdb");
            applyChainColors(model.getID());

            // Make decoded atoms clickable
            viewer.setClickable({ model: model.getID() }, true, onAtomClicked);
        }

        if (currentOriginalPdb && showOriginalCheckbox.checked) {
            const model = viewer.addModel(currentOriginalPdb, "pdb");
            const styleSpec = getStyleSpec(style, true);
            viewer.setStyle({ model: model.getID() }, styleSpec);
        }

        viewer.zoomTo();
        viewer.render();
    }

    // Chain color palette -- distinct colors for up to 12 chains
    const CHAIN_COLORS = [
        "#e94560", "#52b788", "#4895ef", "#f9c74f",
        "#f3722c", "#90be6d", "#577590", "#f94144",
        "#43aa8b", "#9b5de5", "#00bbf9", "#fee440",
    ];

    function getStyleSpec(style, isOriginal) {
        const colorSpec = isOriginal
            ? { color: "#888888" }
            : { colorscheme: "chain" };
        const opacity = isOriginal ? 0.5 : 1.0;

        switch (style) {
            case "cartoon":
                return { cartoon: { ...colorSpec, opacity } };
            case "stick":
                return { stick: { ...colorSpec, opacity, radius: 0.15 } };
            case "sphere":
                return { sphere: { ...colorSpec, opacity, scale: 0.3 } };
            case "line":
                return { line: { ...colorSpec, opacity } };
            default:
                return { stick: { ...colorSpec, opacity, radius: 0.15 } };
        }
    }

    function applyChainColors(modelId) {
        // Get unique chains from the model and assign distinct colors
        const atoms = viewer.getModel(modelId).selectedAtoms({});
        const chains = [...new Set(atoms.map(a => a.chain))];
        const style = styleSelect.value;

        chains.forEach((chain, idx) => {
            const color = CHAIN_COLORS[idx % CHAIN_COLORS.length];
            const spec = {};
            switch (style) {
                case "cartoon":
                    spec.cartoon = { color, opacity: 1.0 };
                    break;
                case "stick":
                    spec.stick = { color, opacity: 1.0, radius: 0.15 };
                    break;
                case "sphere":
                    spec.sphere = { color, opacity: 1.0, scale: 0.3 };
                    break;
                case "line":
                    spec.line = { color, opacity: 1.0 };
                    break;
                default:
                    spec.stick = { color, opacity: 1.0, radius: 0.15 };
            }
            viewer.setStyle({ model: modelId, chain }, spec);
        });
    }

    // --- Debounced decode ---
    function scheduleDecode() {
        if (decodeTimer) clearTimeout(decodeTimer);
        decodeTimer = setTimeout(async () => {
            const tokens = parseTokens(tokenEditor.value);

            if (currentTokenizer === "kanzi") {
                tokenCount.textContent = `${tokens.length} tokens (1 per residue)`;
            } else if (currentTokenizer === "apt") {
                tokenCount.textContent = `${tokens.length} / 128 tokens`;
            } else {
                tokenCount.textContent = `${tokens.length} tokens`;
            }

            if (tokens.length === 0) {
                currentDecodedPdb = null;
                renderStructures();
                return;
            }

            if (!modelReady[currentTokenizer]) return;

            statusIndicator.textContent = "Decoding...";
            statusIndicator.className = "status loading";

            try {
                const result = await decodeTokens(tokens);
                currentDecodedPdb = result.pdb_string;
                renderStructures();
                statusIndicator.textContent = `Decoded ${result.num_atoms} atoms`;
                statusIndicator.className = "status ready";
            } catch (e) {
                statusIndicator.textContent = e.message;
                statusIndicator.className = "status error";
            }
        }, 500);
    }

    // --- Event handlers ---
    tokenEditor.addEventListener("input", scheduleDecode);

    // Metadata changes also trigger re-decode
    metaAtomNames.addEventListener("input", scheduleDecode);
    metaResidueTypes.addEventListener("input", scheduleDecode);
    metaResidueIds.addEventListener("input", scheduleDecode);
    metaClear.addEventListener("click", () => {
        clearMetadataFields();
        scheduleDecode();
    });

    // Tokenizer switch
    tokenizerSelect.addEventListener("change", (e) => {
        switchTokenizer(e.target.value);
    });

    // APT steps slider
    aptSteps.addEventListener("input", () => {
        aptStepsValue.textContent = aptSteps.value;
    });
    aptSteps.addEventListener("change", () => {
        // Re-decode with new step count
        if (parseTokens(tokenEditor.value).length > 0) {
            scheduleDecode();
        }
    });

    // Kanzi sliders
    kanziSteps.addEventListener("input", () => {
        kanziStepsValue.textContent = kanziSteps.value;
    });
    kanziSteps.addEventListener("change", () => {
        if (parseTokens(tokenEditor.value).length > 0) {
            scheduleDecode();
        }
    });
    kanziCfg.addEventListener("input", () => {
        kanziCfgValue.textContent = parseFloat(kanziCfg.value).toFixed(1);
    });
    kanziCfg.addEventListener("change", () => {
        if (parseTokens(tokenEditor.value).length > 0) {
            scheduleDecode();
        }
    });

    fileInput.addEventListener("change", async (e) => {
        const file = e.target.files[0];
        if (!file) return;

        statusIndicator.textContent = "Encoding...";
        statusIndicator.className = "status loading";

        try {
            const result = await encodeFile(file);
            await handleEncodeResult(result);
            statusIndicator.textContent = `Encoded ${result.num_tokens} tokens`;
            statusIndicator.className = "status ready";
        } catch (e) {
            statusIndicator.textContent = e.message;
            statusIndicator.className = "status error";
        }

        fileInput.value = "";
    });

    async function loadPdbCode(code) {
        if (!code || !modelReady[currentTokenizer]) return;

        statusIndicator.textContent = `Loading ${code.toUpperCase()}...`;
        statusIndicator.className = "status loading";

        try {
            const result = await encodePdbCode(code);
            await handleEncodeResult(result);
            statusIndicator.textContent = `${code.toUpperCase()}: ${result.num_tokens} tokens`;
            statusIndicator.className = "status ready";
        } catch (e) {
            statusIndicator.textContent = e.message;
            statusIndicator.className = "status error";
        }
    }

    pdbCodeLoad.addEventListener("click", () => loadPdbCode(pdbCodeInput.value.trim()));
    pdbCodeInput.addEventListener("keydown", (e) => {
        if (e.key === "Enter") loadPdbCode(pdbCodeInput.value.trim());
    });

    styleSelect.addEventListener("change", renderStructures);
    showOriginalCheckbox.addEventListener("change", renderStructures);

    // --- Divider drag for resizing ---
    let isDragging = false;

    divider.addEventListener("mousedown", (e) => {
        isDragging = true;
        divider.classList.add("active");
        e.preventDefault();
    });

    document.addEventListener("mousemove", (e) => {
        if (!isDragging) return;
        const containerRect = document.querySelector(".split-pane").getBoundingClientRect();
        const pct = ((e.clientX - containerRect.left) / containerRect.width) * 100;
        const clamped = Math.max(15, Math.min(85, pct));
        leftPane.style.width = clamped + "%";
        if (viewer) viewer.resize();
    });

    document.addEventListener("mouseup", () => {
        if (isDragging) {
            isDragging = false;
            divider.classList.remove("active");
            if (viewer) viewer.resize();
        }
    });

    // --- Handle window resize ---
    window.addEventListener("resize", () => {
        if (viewer) viewer.resize();
    });

    // --- Status polling ---
    async function pollStatus() {
        try {
            const res = await fetch("/api/status");
            const data = await res.json();

            modelReady.bio2token = data.bio2token_loaded;
            modelReady.apt = data.apt_loaded;
            modelReady.kanzi = data.kanzi_loaded;

            // Enable/disable tokenizer options based on availability
            for (const opt of tokenizerSelect.options) {
                if (opt.value === "apt") {
                    opt.disabled = !data.apt_loaded;
                }
                if (opt.value === "bio2token") {
                    opt.disabled = !data.bio2token_loaded;
                }
                if (opt.value === "kanzi") {
                    opt.disabled = !data.kanzi_loaded;
                }
            }

            if (modelReady[currentTokenizer]) {
                statusIndicator.textContent = "Ready";
                statusIndicator.className = "status ready";
                // Auto-load default PDB code on first ready
                const code = pdbCodeInput.value.trim();
                if (code && parseTokens(tokenEditor.value).length === 0) {
                    loadPdbCode(code);
                }
                return;
            }
        } catch (e) {
            // Server not ready yet
        }
        setTimeout(pollStatus, 1000);
    }

    // --- Init ---
    initViewer();
    pollStatus();
})();
