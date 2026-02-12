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

    // Metadata fields
    const metaAtomNames = document.getElementById("meta-atom-names");
    const metaResidueTypes = document.getElementById("meta-residue-types");
    const metaResidueIds = document.getElementById("meta-residue-ids");
    const metaClear = document.getElementById("meta-clear");
    const metaStatus = document.getElementById("meta-status");

    // --- State ---
    let viewer = null;
    let currentDecodedPdb = null;
    let currentOriginalPdb = null;
    let storedMetadata = null;  // metadata from encode, used in decode
    let decodeTimer = null;
    let modelReady = false;

    // --- 3Dmol viewer init ---
    function initViewer() {
        viewer = $3Dmol.createViewer(viewerContainer, {
            backgroundColor: "#1a1a2e",
        });
        viewer.resize();
    }

    // --- Token parsing ---
    function parseTokens(text) {
        text = text.trim();
        if (!text) return [];

        const tokens = [];
        // Match <bNNN> format or plain numbers
        const regex = /<b(\d+)>|(\d+)/g;
        let match;
        while ((match = regex.exec(text)) !== null) {
            const num = parseInt(match[1] || match[2], 10);
            if (!isNaN(num) && num >= 0 && num <= 4095) {
                tokens.push(num);
            }
        }
        return tokens;
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

    // --- API calls ---
    async function decodeTokens(tokenIds) {
        // Determine metadata: UI fields override stored metadata
        const uiMetadata = parseMetadataFields();
        const metadata = uiMetadata || storedMetadata;

        const body = { token_ids: tokenIds };
        if (metadata) {
            if (metadata.atom_names) body.atom_names = metadata.atom_names;
            if (metadata.residue_types) body.residue_types = metadata.residue_types;
            if (metadata.residue_ids) body.residue_ids = metadata.residue_ids;
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

        const res = await fetch("/api/encode", {
            method: "POST",
            body: formData,
        });

        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.detail || "Encode failed");
        }
        return await res.json();
    }

    // --- Rendering ---
    function renderStructures() {
        if (!viewer) return;
        viewer.removeAllModels();
        viewer.removeAllLabels();

        const style = styleSelect.value;

        if (currentDecodedPdb) {
            const model = viewer.addModel(currentDecodedPdb, "pdb");
            const styleSpec = getStyleSpec(style, false);
            viewer.setStyle({ model: model.getID() }, styleSpec);
        }

        if (currentOriginalPdb && showOriginalCheckbox.checked) {
            const model = viewer.addModel(currentOriginalPdb, "pdb");
            const styleSpec = getStyleSpec(style, true);
            viewer.setStyle({ model: model.getID() }, styleSpec);
        }

        viewer.zoomTo();
        viewer.render();
    }

    function getStyleSpec(style, isOriginal) {
        const colorSpec = isOriginal
            ? { color: "#888888" }
            : { colorscheme: "spectral" };
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

    // --- Debounced decode ---
    function scheduleDecode() {
        if (decodeTimer) clearTimeout(decodeTimer);
        decodeTimer = setTimeout(async () => {
            const tokens = parseTokens(tokenEditor.value);
            tokenCount.textContent = `${tokens.length} tokens`;

            if (tokens.length === 0) {
                currentDecodedPdb = null;
                renderStructures();
                return;
            }

            if (!modelReady) return;

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

    fileInput.addEventListener("change", async (e) => {
        const file = e.target.files[0];
        if (!file) return;

        statusIndicator.textContent = "Encoding...";
        statusIndicator.className = "status loading";

        try {
            const result = await encodeFile(file);
            tokenEditor.value = result.token_string;
            tokenCount.textContent = `${result.num_tokens} tokens`;
            currentOriginalPdb = result.gt_pdb_string;

            // Store metadata and populate UI fields
            storedMetadata = {
                atom_names: result.atom_names,
                residue_types: result.residue_types,
                residue_ids: result.residue_ids,
            };
            populateMetadataFields(storedMetadata);

            // Enable "Show original" checkbox
            showOriginalCheckbox.disabled = false;
            showOriginalCheckbox.checked = true;

            // Decode the tokens to get reconstructed structure
            const decoded = await decodeTokens(result.token_ids);
            currentDecodedPdb = decoded.pdb_string;
            renderStructures();

            statusIndicator.textContent = `Encoded ${result.num_tokens} tokens`;
            statusIndicator.className = "status ready";
        } catch (e) {
            statusIndicator.textContent = e.message;
            statusIndicator.className = "status error";
        }

        // Reset file input so re-uploading same file triggers change
        fileInput.value = "";
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
            if (data.model_loaded) {
                modelReady = true;
                statusIndicator.textContent = "Ready";
                statusIndicator.className = "status ready";
                // Trigger decode if there are tokens already in the editor
                const tokens = parseTokens(tokenEditor.value);
                if (tokens.length > 0) {
                    scheduleDecode();
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
