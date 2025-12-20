// State Management
const state = {
    metric: 'NET', // NET, HR01, HR03
    geoMode: 'CH', // CH, KT
    selectedCanton: null, // Code like 'ZH'
    dateRangeMonths: 24, // 0 for max
    data: null,
    dimensions: null
};

// Initialization
document.addEventListener('DOMContentLoaded', async () => {
    initControls();
    await loadData();
    updateStatus();
    render();
});

// UI Controls Initialization
function initControls() {
    // Metric Selector
    const metricSelect = document.getElementById('metricSelect');
    metricSelect.addEventListener('change', (e) => {
        state.metric = e.target.value;
        render();
    });

    // Geo Mode Toggle
    const geoBtns = document.querySelectorAll('.toggle-btn[data-geo]');
    geoBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            setGeoMode(btn.dataset.geo);
        });
    });

    // Canton Selector
    const cantonSelect = document.getElementById('cantonSelect');
    cantonSelect.addEventListener('change', (e) => {
        state.selectedCanton = e.target.value;
        state.geoMode = 'KT'; // Force KT mode when selecting explicitly
        updateControls();
        render();
    });

    // Date Range Buttons
    const rangeBtns = document.querySelectorAll('.range-btn');
    rangeBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            rangeBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            state.dateRangeMonths = parseInt(btn.dataset.months);
            render();
        });
    });
}

function setGeoMode(mode) {
    state.geoMode = mode;
    updateControls();
    render();
}

function updateControls() {
    // Update Geo Buttons
    document.querySelectorAll('.toggle-btn[data-geo]').forEach(btn => {
        if (btn.dataset.geo === state.geoMode) btn.classList.add('active');
        else btn.classList.remove('active');
    });

    // Show/Hide Canton Select
    const cantonGroup = document.getElementById('cantonControlGroup');
    if (state.geoMode === 'KT') {
        cantonGroup.style.display = 'flex';
    } else {
        cantonGroup.style.display = 'none';
    }

    // Sync Canton Select value
    if (state.selectedCanton) {
        document.getElementById('cantonSelect').value = state.selectedCanton;
    }
}

async function loadData() {
    try {
        // Load Dimensions
        const dimResp = await fetch('static/data/dimensions.json');
        state.dimensions = await dimResp.json();
        populateDropdowns();

        // Load Main Data
        const dataResp = await fetch('static/data/shab_monthly.json');
        state.data = await dataResp.json();

        // Default selection if none
        if (!state.selectedCanton && state.dimensions.cantons.length > 0) {
            state.selectedCanton = state.dimensions.cantons[0];
        }

    } catch (e) {
        console.error("Failed to load data", e);
        alert("Failed to load dashboard data.");
    }
}

async function updateStatus() {
    try {
        const resp = await fetch('api/status');
        if (resp.ok) {
            const status = await resp.json();
            const date = new Date(status.data_updated_at).toLocaleString();
            document.getElementById('statusIndicator').textContent = `Data Updated: ${date} (${status.records} records)`;
        }
    } catch (e) {
        console.warn("Could not fetch status");
    }
}

function populateDropdowns() {
    // Metrics
    const metricSelect = document.getElementById('metricSelect');
    metricSelect.innerHTML = '';
    state.dimensions.metrics.forEach(m => {
        const opt = document.createElement('option');
        opt.value = m;
        opt.textContent = m;
        if (m === state.metric) opt.selected = true;
        metricSelect.appendChild(opt);
    });

    // Cantons
    const cantonSelect = document.getElementById('cantonSelect');
    cantonSelect.innerHTML = '';
    state.dimensions.cantons.forEach(c => {
        const opt = document.createElement('option');
        opt.value = c;
        opt.textContent = c;
        cantonSelect.appendChild(opt);
    });
}

// Data Processing & Filtering
function getFilteredData() {
    if (!state.data) return { months: [], values: [], heatmap: null };

    // 1. Filter by Date Range
    let filtered = state.data;
    const allMonths = state.dimensions.months;

    let startIndex = 0;
    if (state.dateRangeMonths > 0) {
        startIndex = Math.max(0, allMonths.length - state.dateRangeMonths);
    }
    const visibleMonths = allMonths.slice(startIndex);
    const visibleMonthsSet = new Set(visibleMonths);

    filtered = filtered.filter(d => visibleMonthsSet.has(d.month));

    // 2. Prepare Time Series Data
    let tsData = [];
    if (state.geoMode === 'CH') {
        tsData = filtered.filter(d => d.geo === 'CH' && d.hr === state.metric);
    } else {
        tsData = filtered.filter(d => d.geo === 'KT' && d.kanton === state.selectedCanton && d.hr === state.metric);
    }

    // Align TS data to visibleMonths to ensure continuity (fill missing with 0)
    const tsMap = new Map();
    tsData.forEach(d => tsMap.set(d.month, d.count));
    const alignedValues = visibleMonths.map(m => tsMap.get(m) || 0);

    // 3. Prepare Heatmap Data
    // X: visibleMonths
    // Y: cantons (reversed for correct top-down display in Plotly)
    // Z: matrix

    const cantons = [...state.dimensions.cantons].reverse();
    const hmData = filtered.filter(d => d.geo === 'KT' && d.hr === state.metric);

    // Build lookup
    const hmLookup = {}; // key: canton_month
    hmData.forEach(d => {
        hmLookup[`${d.kanton}_${d.month}`] = d.count;
    });

    const z = cantons.map(c => {
        return visibleMonths.map(m => hmLookup[`${c}_${m}`] || 0);
    });

    return {
        months: visibleMonths,
        tsValues: alignedValues,
        hmZ: z,
        hmY: cantons
    };
}

// Rendering
function render() {
    if (!state.data) return;

    const { months, tsValues, hmZ, hmY } = getFilteredData();

    renderTimeSeries(months, tsValues);
    renderHeatmap(months, hmY, hmZ);
}

function renderTimeSeries(months, values) {
    const trace = {
        x: months,
        y: values,
        type: 'scatter',
        mode: 'lines+markers',
        line: { shape: 'spline', color: '#007bff' },
        name: state.metric
    };

    const layout = {
        margin: { t: 10, r: 10, l: 40, b: 40 },
        xaxis: {
            title: '',
            showgrid: false
        },
        yaxis: {
            title: state.metric,
            showgrid: true,
            gridcolor: '#f0f0f0'
        },
        hovermode: 'x unified',
        uirevision: 'ts' // Persist zoom
    };

    const config = { responsive: true, displayModeBar: false };

    Plotly.react('tsChart', [trace], layout, config);
}

function renderHeatmap(months, cantons, z) {
    const trace = {
        x: months,
        y: cantons,
        z: z,
        type: 'heatmap',
        colorscale: 'Blues',
        showscale: false, // minimalist
        hovertemplate: 'Canton: %{y}<br>Month: %{x}<br>Value: %{z}<extra></extra>'
    };

    const layout = {
        margin: { t: 10, r: 10, l: 40, b: 40 },
        xaxis: { showgrid: false },
        yaxis: { showgrid: false, ticksuffix: ' ' },
        uirevision: 'hm'
    };

    const config = { responsive: true, displayModeBar: false };

    Plotly.react('hmChart', [trace], layout, config).then(gd => {
        // Add click handler
        gd.removeAllListeners('plotly_click'); // prevent duplicate listeners
        gd.on('plotly_click', data => {
            const pt = data.points[0];
            const clickedCanton = pt.y;

            // Interaction: Select canton
            if (clickedCanton) {
                state.selectedCanton = clickedCanton;
                setGeoMode('KT'); // Switch to canton view
            }
        });
    });
}
