/**
 * FDC Real-time Alert System - Dashboard JavaScript
 */

// Configuration
const CONFIG = {
    wsUrl: `ws://${window.location.host}/ws/dashboard`,
    maxDataPoints: 100,
    reconnectInterval: 3000,
    thresholds: {
        warning: 0.6,
        critical: 0.8,
        emergency: 0.95
    }
};

// State
let ws = null;
let sensorChart = null;
let isRunning = false;
let reconnectAttempts = 0;

// Sensor data history
const sensorHistory = {
    labels: [],
    current: [],
    effective_load: [],
    peak_load: [],
    position: []
};

// DOM Elements
const elements = {
    startBtn: document.getElementById('startBtn'),
    stopBtn: document.getElementById('stopBtn'),
    statusIndicator: document.getElementById('statusIndicator'),
    scoreValue: document.getElementById('scoreValue'),
    scoreLabel: document.getElementById('scoreLabel'),
    scoreBar: document.getElementById('scoreBar'),
    totalProcessed: document.getElementById('totalProcessed'),
    warningCount: document.getElementById('warningCount'),
    criticalCount: document.getElementById('criticalCount'),
    emergencyCount: document.getElementById('emergencyCount'),
    alertLog: document.getElementById('alertLog'),
    alertCount: document.getElementById('alertCount'),
    connectionStatus: document.getElementById('connectionStatus'),
    lastUpdate: document.getElementById('lastUpdate'),
    zscoreBar: document.getElementById('zscoreBar'),
    zscoreValue: document.getElementById('zscoreValue'),
    cusumBar: document.getElementById('cusumBar'),
    cusumValue: document.getElementById('cusumValue'),
    ifBar: document.getElementById('ifBar'),
    ifValue: document.getElementById('ifValue'),
    lofBar: document.getElementById('lofBar'),
    lofValue: document.getElementById('lofValue')
};

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    initChart();
    initWebSocket();
    initEventListeners();
    fetchInitialStatus();
});

// Initialize Chart.js
function initChart() {
    const ctx = document.getElementById('sensorChart').getContext('2d');

    sensorChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [
                {
                    label: 'CURRENT',
                    data: [],
                    borderColor: '#3498db',
                    backgroundColor: 'rgba(52, 152, 219, 0.1)',
                    borderWidth: 2,
                    tension: 0.4,
                    fill: false
                },
                {
                    label: 'EFF_LOAD',
                    data: [],
                    borderColor: '#e67e22',
                    backgroundColor: 'rgba(230, 126, 34, 0.1)',
                    borderWidth: 2,
                    tension: 0.4,
                    fill: false
                },
                {
                    label: 'PEAK_LOAD',
                    data: [],
                    borderColor: '#2ecc71',
                    backgroundColor: 'rgba(46, 204, 113, 0.1)',
                    borderWidth: 2,
                    tension: 0.4,
                    fill: false
                },
                {
                    label: 'POSITION',
                    data: [],
                    borderColor: '#9b59b6',
                    backgroundColor: 'rgba(155, 89, 182, 0.1)',
                    borderWidth: 2,
                    tension: 0.4,
                    fill: false,
                    yAxisID: 'y1'
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: {
                duration: 0
            },
            interaction: {
                intersect: false,
                mode: 'index'
            },
            plugins: {
                legend: {
                    position: 'top',
                    labels: {
                        color: '#a0a0a0'
                    }
                }
            },
            scales: {
                x: {
                    grid: {
                        color: 'rgba(255, 255, 255, 0.1)'
                    },
                    ticks: {
                        color: '#a0a0a0',
                        maxTicksLimit: 10
                    }
                },
                y: {
                    position: 'left',
                    grid: {
                        color: 'rgba(255, 255, 255, 0.1)'
                    },
                    ticks: {
                        color: '#a0a0a0'
                    },
                    title: {
                        display: true,
                        text: 'Load Ratio / Current',
                        color: '#a0a0a0'
                    }
                },
                y1: {
                    position: 'right',
                    grid: {
                        drawOnChartArea: false
                    },
                    ticks: {
                        color: '#9b59b6'
                    },
                    title: {
                        display: true,
                        text: 'Position',
                        color: '#9b59b6'
                    }
                }
            }
        }
    });
}

// Initialize WebSocket
function initWebSocket() {
    if (ws) {
        ws.close();
    }

    ws = new WebSocket(CONFIG.wsUrl);

    ws.onopen = () => {
        console.log('WebSocket connected');
        elements.connectionStatus.textContent = 'Connected';
        elements.connectionStatus.classList.add('connected');
        reconnectAttempts = 0;
    };

    ws.onclose = () => {
        console.log('WebSocket disconnected');
        elements.connectionStatus.textContent = 'Disconnected';
        elements.connectionStatus.classList.remove('connected');

        // Reconnect
        if (reconnectAttempts < 10) {
            reconnectAttempts++;
            setTimeout(initWebSocket, CONFIG.reconnectInterval);
        }
    };

    ws.onerror = (error) => {
        console.error('WebSocket error:', error);
    };

    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        handleMessage(data);
    };
}

// Handle WebSocket messages
function handleMessage(data) {
    switch (data.type) {
        case 'connected':
            console.log('Connected to channel:', data.channel);
            break;
        case 'measurement':
            updateMeasurement(data.data);
            break;
        case 'detection':
            updateDetection(data.data);
            break;
        case 'alert':
            addAlert(data.data);
            break;
        case 'status':
            updateStatus(data.data);
            break;
    }

    elements.lastUpdate.textContent = `Last update: ${new Date().toLocaleTimeString()}`;
}

// Update measurement display
function updateMeasurement(measurement) {
    const values = measurement.values || {};

    // Add to chart
    const windowIdx = measurement.window_index || sensorHistory.labels.length;
    sensorHistory.labels.push(windowIdx.toString());

    sensorHistory.current.push(values.SVM_Z_CURRENT || 0);
    sensorHistory.effective_load.push(values.SVM_Z_EFFECTIVE_LOAD_RATIO || 0);
    sensorHistory.peak_load.push(values.SVM_Z_PEAK_LOAD_RATIO || 0);
    sensorHistory.position.push(values.SVM_Z_POSITION || 0);

    // Limit data points
    if (sensorHistory.labels.length > CONFIG.maxDataPoints) {
        sensorHistory.labels.shift();
        sensorHistory.current.shift();
        sensorHistory.effective_load.shift();
        sensorHistory.peak_load.shift();
        sensorHistory.position.shift();
    }

    // Update chart
    sensorChart.data.labels = sensorHistory.labels;
    sensorChart.data.datasets[0].data = sensorHistory.current;
    sensorChart.data.datasets[1].data = sensorHistory.effective_load;
    sensorChart.data.datasets[2].data = sensorHistory.peak_load;
    sensorChart.data.datasets[3].data = sensorHistory.position;
    sensorChart.update('none');
}

// Update detection display
function updateDetection(detection) {
    const score = detection.ensemble_score || 0;
    const severity = detection.severity || 'NORMAL';
    const scores = detection.individual_scores || {};

    // Update ensemble score
    elements.scoreValue.textContent = score.toFixed(2);
    elements.scoreBar.style.width = `${score * 100}%`;

    // Update score label
    elements.scoreLabel.textContent = severity;
    elements.scoreLabel.className = 'score-label';
    if (severity !== 'NORMAL') {
        elements.scoreLabel.classList.add(severity.toLowerCase());
    }

    // Update score value color
    if (score >= CONFIG.thresholds.emergency) {
        elements.scoreValue.style.color = '#e74c3c';
    } else if (score >= CONFIG.thresholds.critical) {
        elements.scoreValue.style.color = '#e67e22';
    } else if (score >= CONFIG.thresholds.warning) {
        elements.scoreValue.style.color = '#f1c40f';
    } else {
        elements.scoreValue.style.color = '#2ecc71';
    }

    // Update algorithm scores
    updateAlgorithmScore('zscore', scores.zscore || 0);
    updateAlgorithmScore('cusum', scores.cusum || 0);
    updateAlgorithmScore('if', scores.isolation_forest || 0);
    updateAlgorithmScore('lof', scores.lof || 0);

    // Update processed count
    elements.totalProcessed.textContent = detection.window_index || 0;
}

// Update individual algorithm score
function updateAlgorithmScore(algo, score) {
    const barElement = elements[`${algo}Bar`];
    const valueElement = elements[`${algo}Value`];

    if (barElement && valueElement) {
        barElement.style.width = `${score * 100}%`;
        valueElement.textContent = score.toFixed(2);

        // Color based on score
        if (score >= CONFIG.thresholds.critical) {
            barElement.style.background = '#e74c3c';
        } else if (score >= CONFIG.thresholds.warning) {
            barElement.style.background = '#f1c40f';
        } else {
            barElement.style.background = '#4a90d9';
        }
    }
}

// Add alert to log
function addAlert(alert) {
    const severity = alert.severity || 'WARNING';
    const timestamp = new Date(alert.timestamp).toLocaleTimeString();
    const score = alert.ensemble_score || 0;
    const message = alert.message || '';

    // Remove "no alerts" message
    const noAlerts = elements.alertLog.querySelector('.no-alerts');
    if (noAlerts) {
        noAlerts.remove();
    }

    // Create alert element
    const alertElement = document.createElement('div');
    alertElement.className = `alert-item ${severity.toLowerCase()}`;
    alertElement.innerHTML = `
        <span class="alert-time">${timestamp}</span>
        <span class="alert-severity ${severity.toLowerCase()}">${severity}</span>
        <span class="alert-score">${score.toFixed(3)}</span>
        <span class="alert-message">${message}</span>
    `;

    // Add to top of log
    elements.alertLog.insertBefore(alertElement, elements.alertLog.firstChild);

    // Limit log entries
    while (elements.alertLog.children.length > 50) {
        elements.alertLog.removeChild(elements.alertLog.lastChild);
    }

    // Update alert counts
    updateAlertCounts(severity);
}

// Update alert counts
function updateAlertCounts(severity) {
    const currentCount = parseInt(elements[`${severity.toLowerCase()}Count`].textContent) || 0;
    elements[`${severity.toLowerCase()}Count`].textContent = currentCount + 1;

    // Update total count in header
    const total = elements.alertLog.querySelectorAll('.alert-item').length;
    elements.alertCount.textContent = `(${total})`;
}

// Update status
function updateStatus(status) {
    isRunning = status.is_running;

    elements.startBtn.disabled = isRunning;
    elements.stopBtn.disabled = !isRunning;

    elements.statusIndicator.textContent = isRunning ? 'Running' : 'Stopped';
    elements.statusIndicator.className = `status-indicator ${isRunning ? 'status-running' : 'status-stopped'}`;

    if (status.alerts_by_severity) {
        elements.warningCount.textContent = status.alerts_by_severity.WARNING || 0;
        elements.criticalCount.textContent = status.alerts_by_severity.CRITICAL || 0;
        elements.emergencyCount.textContent = status.alerts_by_severity.EMERGENCY || 0;
    }
}

// Initialize event listeners
function initEventListeners() {
    elements.startBtn.addEventListener('click', startMonitoring);
    elements.stopBtn.addEventListener('click', stopMonitoring);
}

// Start monitoring
async function startMonitoring() {
    try {
        const response = await fetch('/api/control/start', { method: 'POST' });
        const result = await response.json();

        if (response.ok) {
            isRunning = true;
            elements.startBtn.disabled = true;
            elements.stopBtn.disabled = false;
            elements.statusIndicator.textContent = 'Running';
            elements.statusIndicator.className = 'status-indicator status-running';
        } else {
            alert(result.detail || 'Failed to start monitoring');
        }
    } catch (error) {
        console.error('Error starting monitoring:', error);
        alert('Failed to start monitoring');
    }
}

// Stop monitoring
async function stopMonitoring() {
    try {
        const response = await fetch('/api/control/stop', { method: 'POST' });
        const result = await response.json();

        if (response.ok) {
            isRunning = false;
            elements.startBtn.disabled = false;
            elements.stopBtn.disabled = true;
            elements.statusIndicator.textContent = 'Stopped';
            elements.statusIndicator.className = 'status-indicator status-stopped';
        } else {
            alert(result.detail || 'Failed to stop monitoring');
        }
    } catch (error) {
        console.error('Error stopping monitoring:', error);
        alert('Failed to stop monitoring');
    }
}

// Fetch initial status
async function fetchInitialStatus() {
    try {
        const response = await fetch('/api/status');
        if (response.ok) {
            const status = await response.json();
            updateStatus(status);
        }
    } catch (error) {
        console.error('Error fetching status:', error);
    }
}
