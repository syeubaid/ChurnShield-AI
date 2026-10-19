/* ═══════════════════════════════════════════════════════════════════
   ChurnShield — Main Application Script
   ═══════════════════════════════════════════════════════════════════ */

'use strict';

// ─── State ───────────────────────────────────────────────────────────────────
const STATE = {
  allResults: [],
  filteredResults: [],
  currentPage: 1,
  pageSize: 15,
  riskFilter: 'all',
  searchQuery: '',
  sortMode: 'prob-desc',
  modelMeta: null,
  charts: {}
};

// ─── DOM Ready ────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  initTabs();
  initUploadZone();
  initFilters();
  initForm();
  loadModelInfo();
  loadPerformance();
});

// ═══════════════════════════════════════════════════════════════════════════
// TAB NAVIGATION
// ═══════════════════════════════════════════════════════════════════════════
function initTabs() {
  document.querySelectorAll('.nav-item').forEach(item => {
    item.addEventListener('click', e => {
      e.preventDefault();
      switchTab(item.dataset.tab);
    });
  });
}

function switchTab(tabId) {
  document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
  document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));

  const navEl = document.getElementById(`nav-${tabId}`);
  const panelEl = document.getElementById(`tab-${tabId}`);
  if (navEl)   navEl.classList.add('active');
  if (panelEl) panelEl.classList.add('active');

  if (tabId === 'performance') loadPerformance();
  if (tabId === 'results')     renderTable();
}

// ═══════════════════════════════════════════════════════════════════════════
// MODEL INFO & DASHBOARD
// ═══════════════════════════════════════════════════════════════════════════
async function loadModelInfo() {
  try {
    const res = await fetch('/api/model-info');
    if (!res.ok) throw new Error('Model not ready');
    const meta = await res.json();
    STATE.modelMeta = meta;
    renderDashboard(meta);
    setModelStatus('ready', `${meta.model_type} — Ready`);
  } catch (e) {
    setModelStatus('error', 'Model load failed');
    showToast('Could not load model info. Is the server running?', 'error');
  }
}

function setModelStatus(state, text) {
  const dot  = document.querySelector('.status-dot');
  const span = document.querySelector('.model-status span:last-child');
  dot.className = `status-dot ${state}`;
  span.textContent = text;
}

function renderDashboard(meta) {
  const m = meta.metrics;
  const d = meta.class_distribution;

  // KPI Cards
  setText('kpi-total-val',    (meta.training_samples + meta.test_samples).toLocaleString());
  setText('kpi-accuracy-val', pct(m.accuracy));
  setText('kpi-auc-val',      pct(m.roc_auc));
  setText('kpi-churnrate-val', pct(d.churn_rate));

  // Feature Importance Chart
  const fi = meta.feature_importance;
  const fd = meta.feature_descriptions || {};
  const top10Keys = Object.keys(fi).slice(0, 10);
  const top10Vals = top10Keys.map(k => (fi[k] * 100).toFixed(1));
  const top10Labels = top10Keys.map(k => fd[k] || k);

  renderBarChart('featureImportanceChart', top10Labels.reverse(), top10Vals.reverse());

  // Class Distribution Donut
  renderDonut('classDistChart',
    ['Churned', 'Retained'],
    [d.churned, d.retained],
    ['#ef4444', '#10b981']
  );

  // Confusion Matrix
  const cm = meta.confusion_matrix;
  if (cm && cm.length === 2) {
    const tn = cm[0][0], fp = cm[0][1], fn = cm[1][0], tp = cm[1][1];
    document.getElementById('confusion-matrix-display').innerHTML = `
      <div class="cm-cell tp">
        <span class="cm-type text-green">TP</span>
        <div class="cm-val text-green">${tp}</div>
        <div class="cm-label">True Positive</div>
      </div>
      <div class="cm-cell fp">
        <span class="cm-type text-amber">FP</span>
        <div class="cm-val text-amber">${fp}</div>
        <div class="cm-label">False Positive</div>
      </div>
      <div class="cm-cell fn">
        <span class="cm-type text-red">FN</span>
        <div class="cm-val text-red">${fn}</div>
        <div class="cm-label">False Negative</div>
      </div>
      <div class="cm-cell tn">
        <span class="cm-type text-green">TN</span>
        <div class="cm-val text-green">${tn}</div>
        <div class="cm-label">True Negative</div>
      </div>
    `;
  }

  // Metrics Radar
  renderRadar('metricsRadarChart', [
    'Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC'
  ], [
    m.accuracy * 100, m.precision * 100,
    m.recall * 100,   m.f1_score * 100,
    m.roc_auc * 100
  ]);
}

// ═══════════════════════════════════════════════════════════════════════════
// CHART HELPERS
// ═══════════════════════════════════════════════════════════════════════════
const CHART_DEFAULTS = {
  animation: { duration: 800, easing: 'easeInOutQuart' },
  plugins: { legend: { labels: { color: '#94a3b8', font: { family: 'Inter', size: 11 } } } },
  scales: {
    x: { ticks: { color: '#94a3b8', font: { size: 11 } }, grid: { color: 'rgba(255,255,255,0.05)' } },
    y: { ticks: { color: '#94a3b8', font: { size: 11 } }, grid: { color: 'rgba(255,255,255,0.05)' } }
  }
};

function destroyChart(id) {
  if (STATE.charts[id]) { STATE.charts[id].destroy(); delete STATE.charts[id]; }
}

function renderBarChart(id, labels, data) {
  destroyChart(id);
  const ctx = document.getElementById(id)?.getContext('2d');
  if (!ctx) return;
  STATE.charts[id] = new Chart(ctx, {
    type: 'bar',
    data: {
      labels,
      datasets: [{
        data,
        backgroundColor: labels.map((_, i) =>
          `hsla(${240 + i * 12}, 70%, 65%, 0.75)`
        ),
        borderColor: labels.map((_, i) =>
          `hsl(${240 + i * 12}, 70%, 65%)`
        ),
        borderWidth: 1,
        borderRadius: 6,
      }]
    },
    options: {
      ...CHART_DEFAULTS,
      indexAxis: 'y',
      plugins: { legend: { display: false } },
      scales: {
        x: { ticks: { color: '#94a3b8', font: { size: 11 }, callback: v => v + '%' }, grid: { color: 'rgba(255,255,255,0.05)' } },
        y: { ticks: { color: '#94a3b8', font: { size: 11 } }, grid: { display: false } }
      }
    }
  });
}

function renderDonut(id, labels, data, colors) {
  destroyChart(id);
  const ctx = document.getElementById(id)?.getContext('2d');
  if (!ctx) return;
  STATE.charts[id] = new Chart(ctx, {
    type: 'doughnut',
    data: {
      labels,
      datasets: [{
        data,
        backgroundColor: colors.map(c => c + 'aa'),
        borderColor: colors,
        borderWidth: 2,
        hoverOffset: 8,
      }]
    },
    options: {
      ...CHART_DEFAULTS,
      cutout: '68%',
      plugins: {
        legend: {
          position: 'bottom',
          labels: { color: '#94a3b8', padding: 16, font: { family: 'Inter', size: 11 } }
        }
      }
    }
  });
}

function renderRiskDonut(id, counts) {
  destroyChart(id);
  const ctx = document.getElementById(id)?.getContext('2d');
  if (!ctx) return;
  STATE.charts[id] = new Chart(ctx, {
    type: 'doughnut',
    data: {
      labels: ['High Risk', 'Medium Risk', 'Low Risk'],
      datasets: [{
        data: [counts.High || 0, counts.Medium || 0, counts.Low || 0],
        backgroundColor: ['rgba(239,68,68,0.7)', 'rgba(245,158,11,0.7)', 'rgba(16,185,129,0.7)'],
        borderColor: ['#ef4444', '#f59e0b', '#10b981'],
        borderWidth: 2,
        hoverOffset: 8,
      }]
    },
    options: {
      ...CHART_DEFAULTS,
      cutout: '65%',
      plugins: { legend: { display: false } }
    }
  });
}

function renderRadar(id, labels, data) {
  destroyChart(id);
  const ctx = document.getElementById(id)?.getContext('2d');
  if (!ctx) return;
  STATE.charts[id] = new Chart(ctx, {
    type: 'radar',
    data: {
      labels,
      datasets: [{
        label: 'Model Metrics (%)',
        data,
        backgroundColor: 'rgba(99,102,241,0.15)',
        borderColor: '#6366f1',
        borderWidth: 2,
        pointBackgroundColor: '#6366f1',
        pointBorderColor: '#fff',
        pointRadius: 4,
      }]
    },
    options: {
      ...CHART_DEFAULTS,
      scales: {
        r: {
          min: 0, max: 100,
          ticks: { display: false },
          grid:  { color: 'rgba(255,255,255,0.06)' },
          pointLabels: { color: '#94a3b8', font: { size: 11 } }
        }
      },
      plugins: { legend: { display: false } }
    }
  });
}

// ═══════════════════════════════════════════════════════════════════════════
// SINGLE PREDICTION FORM
// ═══════════════════════════════════════════════════════════════════════════
function initForm() {
  document.getElementById('single-predict-form').addEventListener('submit', async e => {
    e.preventDefault();
    const btn = document.getElementById('predict-btn');
    btn.disabled = true;
    btn.innerHTML = '<span class="spinner"></span> Predicting...';

    const formData = new FormData(e.target);
    const payload = {};
    formData.forEach((val, key) => { payload[key] = isNaN(val) || val === '' ? val : Number(val); });

    try {
      const res = await fetch('/api/predict/single', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      const data = await res.json();
      if (data.error) throw new Error(data.error);
      renderSingleResult(data);
      showToast(`Prediction complete for ${data.customer_id}`, 'success');
    } catch (err) {
      showToast(err.message || 'Prediction failed', 'error');
    } finally {
      btn.disabled = false;
      btn.innerHTML = '<i class="fas fa-bolt"></i> Predict Churn';
    }
  });
}

function renderSingleResult(r) {
  const placeholder = document.getElementById('result-placeholder');
  const content     = document.getElementById('result-content');
  placeholder.style.display = 'none';
  content.style.display = 'block';

  const riskColors = { High: '#ef4444', Medium: '#f59e0b', Low: '#10b981' };
  const riskGlows  = { High: 'rgba(239,68,68,0.3)', Medium: 'rgba(245,158,11,0.3)', Low: 'rgba(16,185,129,0.3)' };
  const color      = riskColors[r.risk_level] || '#6366f1';
  const glow       = riskGlows[r.risk_level]  || 'rgba(99,102,241,0.3)';
  const pct_val    = r.churn_probability_pct;

  // Top factors HTML
  const factorsHtml = (r.top_features || []).map(f => `
    <div class="factor-bar">
      <span class="factor-name">${f.description}</span>
      <div class="factor-track"><div class="factor-fill" style="width:${f.importance}%"></div></div>
      <span class="factor-val">${f.importance}%</span>
    </div>
  `).join('');

  // Recommendations HTML
  const recsHtml = (r.recommendations || []).length
    ? r.recommendations.map(rec => `
        <div class="rec-item">
          <div class="rec-priority-dot ${rec.priority}"></div>
          <div>
            <div class="rec-title">${rec.title}</div>
            <div class="rec-action">${rec.action}</div>
          </div>
        </div>
      `).join('')
    : `<p class="text-muted" style="font-size:12px; text-align:center; padding:16px">
         <i class="fas fa-check-circle text-green"></i> Low risk — no urgent actions needed.
       </p>`;

  content.innerHTML = `
    <div class="risk-gauge-section">
      <div class="gauge-wrapper">
        <div class="gauge-ring" style="
          --gauge-color: ${color};
          --gauge-glow: ${glow};
          --gauge-pct: ${pct_val.toFixed(1)}%;
          background: conic-gradient(${color} ${pct_val}%, rgba(255,255,255,0.05) 0);
          box-shadow: 0 0 30px ${glow};
        ">
          <div class="gauge-inner">
            <span class="gauge-pct" style="color:${color}">${pct_val.toFixed(1)}%</span>
            <span class="gauge-sub">Churn Risk</span>
          </div>
        </div>
      </div>
      <div style="margin-bottom:8px">
        <span class="risk-badge-large risk-badge ${r.risk_level}" style="background:${color}20; color:${color}; border:1px solid ${color}55">
          <i class="fas fa-${r.risk_level === 'High' ? 'triangle-exclamation' : r.risk_level === 'Medium' ? 'circle-exclamation' : 'circle-check'}"></i>
          ${r.risk_level} Risk
        </span>
      </div>
      <p style="font-size:12px; color:var(--text-secondary)">Customer: <strong>${r.customer_id}</strong></p>
    </div>

    <div class="result-section">
      <div class="result-section-title">Top Influencing Factors</div>
      ${factorsHtml}
    </div>

    <div class="result-section">
      <div class="result-section-title">
        <i class="fas fa-lightbulb" style="color:var(--amber)"></i>
        Retention Recommendations
      </div>
      ${recsHtml}
    </div>
  `;

  // Add to results list
  addToResults(r);
}

function fillSampleData() {
  const samples = [
    { CustomerID:'DEMO001', Age:28, Gender:'Female', TenureMonths:3, ContractType:'Month-to-Month',
      MonthlyCharges:95, TotalCharges:285, NumProducts:1, HasInternet:'Yes', InternetType:'Fiber Optic',
      OnlineSecurity:'No', TechSupport:'No', StreamingTV:'Yes', StreamingMovies:'Yes',
      PaymentMethod:'Electronic Check', PaperlessBilling:'Yes', NumSupportCalls:7,
      AvgCallDurationMin:22, SatisfactionScore:1, LastInteractionDays:280, ReferralsGiven:0,
      PromotionOffered:'No', LatePayments:4, DeviceCount:2, DataUsageGB:88, PlanUpgrades:0, PlanDowngrades:2 },
    { CustomerID:'DEMO002', Age:54, Gender:'Male', TenureMonths:48, ContractType:'Two Year',
      MonthlyCharges:45, TotalCharges:2160, NumProducts:4, HasInternet:'Yes', InternetType:'DSL',
      OnlineSecurity:'Yes', TechSupport:'Yes', StreamingTV:'No', StreamingMovies:'No',
      PaymentMethod:'Bank Transfer', PaperlessBilling:'No', NumSupportCalls:1,
      AvgCallDurationMin:5, SatisfactionScore:5, LastInteractionDays:14, ReferralsGiven:3,
      PromotionOffered:'Yes', LatePayments:0, DeviceCount:4, DataUsageGB:25, PlanUpgrades:2, PlanDowngrades:0 }
  ];
  const s = samples[Math.floor(Math.random() * samples.length)];
  Object.entries(s).forEach(([key, val]) => {
    const el = document.querySelector(`[name="${key}"]`);
    if (el) el.value = val;
  });
  showToast('Sample data filled!', 'info');
}

// ═══════════════════════════════════════════════════════════════════════════
// BATCH UPLOAD
// ═══════════════════════════════════════════════════════════════════════════
function initUploadZone() {
  const zone  = document.getElementById('upload-zone');
  const input = document.getElementById('csv-file-input');

  zone.addEventListener('click', () => input.click());
  zone.addEventListener('dragover', e => { e.preventDefault(); zone.classList.add('dragging'); });
  zone.addEventListener('dragleave', () => zone.classList.remove('dragging'));
  zone.addEventListener('drop', e => {
    e.preventDefault();
    zone.classList.remove('dragging');
    const file = e.dataTransfer.files[0];
    if (file) handleCSVUpload(file);
  });

  input.addEventListener('change', () => {
    if (input.files[0]) handleCSVUpload(input.files[0]);
    input.value = '';
  });
}

async function handleCSVUpload(file) {
  if (!file.name.endsWith('.csv')) {
    showToast('Please upload a CSV file', 'error');
    return;
  }

  // Show progress card
  document.getElementById('batch-progress-card').style.display = 'block';
  document.getElementById('batch-summary-card').style.display = 'none';
  const progressBar = document.getElementById('batch-progress-bar');
  const progressMsg = document.getElementById('batch-progress-msg');

  // Animate progress
  let prog = 0;
  const progInterval = setInterval(() => {
    prog = Math.min(prog + Math.random() * 8, 88);
    progressBar.style.width = prog + '%';
    if (prog < 30) progressMsg.textContent = 'Uploading file...';
    else if (prog < 60) progressMsg.textContent = 'Preprocessing customer data...';
    else progressMsg.textContent = 'Running ML model inference...';
  }, 150);

  try {
    const formData = new FormData();
    formData.append('file', file);

    const res = await fetch('/api/predict/batch', { method: 'POST', body: formData });
    const data = await res.json();
    clearInterval(progInterval);

    if (data.error) {
      progressBar.style.width = '100%';
      progressBar.style.background = '#ef4444';
      throw new Error(data.error);
    }

    progressBar.style.width = '100%';
    progressMsg.textContent = 'Done! Rendering results...';

    setTimeout(() => {
      document.getElementById('batch-progress-card').style.display = 'none';
      STATE.allResults = data.results || [];
      STATE.filteredResults = [...STATE.allResults];
      updateResultsBadge();
      renderBatchSummary(data.summary, data.results);
      showToast(`Analyzed ${data.results.length} customers successfully!`, 'success');
    }, 600);

  } catch (err) {
    clearInterval(progInterval);
    progressMsg.textContent = '⚠ ' + (err.message || 'Upload failed');
    showToast(err.message || 'Batch prediction failed', 'error');
  }
}

function renderBatchSummary(summary, results) {
  const card = document.getElementById('batch-summary-card');
  card.style.display = 'block';

  card.innerHTML = `
    <div class="summary-header">
      <h3>📊 Batch Analysis Complete</h3>
      <p style="font-size:13px; color:var(--text-secondary)">${summary.total_customers} customers processed</p>
    </div>
    <div class="summary-stats">
      <div class="stat-box">
        <div class="stat-val">${summary.total_customers}</div>
        <div class="stat-label">Total Customers</div>
      </div>
      <div class="stat-box danger">
        <div class="stat-val">${summary.predicted_churn}</div>
        <div class="stat-label">Predicted to Churn</div>
      </div>
      <div class="stat-box success">
        <div class="stat-val">${summary.predicted_retain}</div>
        <div class="stat-label">Predicted to Stay</div>
      </div>
      <div class="stat-box warning">
        <div class="stat-val">${summary.churn_rate_pct}%</div>
        <div class="stat-label">Churn Rate</div>
      </div>
      <div class="stat-box">
        <div class="stat-val">${summary.avg_churn_probability}%</div>
        <div class="stat-label">Avg Churn Probability</div>
      </div>
    </div>

    <div class="risk-donut-row">
      <div style="width:200px; height:200px; position:relative">
        <canvas id="batchRiskDonut"></canvas>
      </div>
      <div class="risk-legend">
        <div class="risk-legend-item">
          <div class="risk-dot" style="background:#ef4444"></div>
          <strong style="color:#ef4444">${summary.risk_distribution.High}</strong>
          <span class="text-muted">High Risk</span>
        </div>
        <div class="risk-legend-item">
          <div class="risk-dot" style="background:#f59e0b"></div>
          <strong style="color:#f59e0b">${summary.risk_distribution.Medium}</strong>
          <span class="text-muted">Medium Risk</span>
        </div>
        <div class="risk-legend-item">
          <div class="risk-dot" style="background:#10b981"></div>
          <strong style="color:#10b981">${summary.risk_distribution.Low}</strong>
          <span class="text-muted">Low Risk</span>
        </div>
      </div>
    </div>

    <div style="margin-top:24px; display:flex; gap:12px; justify-content:flex-end">
      <button class="btn btn-secondary" onclick="switchTab('results')">
        <i class="fas fa-table-list"></i> View Results
      </button>
      <button class="btn btn-primary" onclick="downloadResults()">
        <i class="fas fa-download"></i> Export CSV
      </button>
    </div>
  `;

  setTimeout(() => renderRiskDonut('batchRiskDonut', summary.risk_distribution), 100);
}

function downloadSampleCSV() {
  const rows = [
    ['CustomerID','Age','Gender','TenureMonths','ContractType','MonthlyCharges','TotalCharges',
     'NumProducts','HasInternet','InternetType','OnlineSecurity','TechSupport','StreamingTV',
     'StreamingMovies','PaymentMethod','PaperlessBilling','NumSupportCalls','AvgCallDurationMin',
     'SatisfactionScore','LastInteractionDays','ReferralsGiven','PromotionOffered','LatePayments',
     'DeviceCount','DataUsageGB','PlanUpgrades','PlanDowngrades','Churn'],
    ['CUST0001',35,'Male',24,'One Year',65.00,1560,2,'Yes','Fiber Optic','Yes','Yes','No','No',
     'Credit Card','Yes',2,8.5,4,30,1,'No',1,2,45.5,1,0,0],
    ['CUST0002',22,'Female',2,'Month-to-Month',95.50,191,1,'Yes','DSL','No','No','Yes','Yes',
     'Electronic Check','Yes',8,20.0,1,300,0,'No',4,1,85.0,0,1,1],
  ];
  const csv = rows.map(r => r.join(',')).join('\n');
  triggerDownload(csv, 'sample_churn_data.csv', 'text/csv');
  showToast('Sample CSV downloaded!', 'info');
}

// ═══════════════════════════════════════════════════════════════════════════
// RESULTS TABLE
// ═══════════════════════════════════════════════════════════════════════════
function addToResults(r) {
  const existing = STATE.allResults.findIndex(x => x.customer_id === r.customer_id);
  if (existing >= 0) STATE.allResults[existing] = r;
  else STATE.allResults.unshift(r);
  applyFilters();
  updateResultsBadge();
}

function initFilters() {
  document.getElementById('risk-filter').addEventListener('click', e => {
    if (!e.target.classList.contains('pill')) return;
    document.querySelectorAll('#risk-filter .pill').forEach(p => p.classList.remove('active'));
    e.target.classList.add('active');
    STATE.riskFilter = e.target.dataset.val;
    STATE.currentPage = 1;
    applyFilters();
  });

  document.getElementById('customer-search').addEventListener('input', e => {
    STATE.searchQuery = e.target.value.toLowerCase();
    STATE.currentPage = 1;
    applyFilters();
  });

  document.getElementById('sort-select').addEventListener('change', e => {
    STATE.sortMode = e.target.value;
    applyFilters();
  });
}

function applyFilters() {
  let res = [...STATE.allResults];

  if (STATE.riskFilter !== 'all')
    res = res.filter(r => r.risk_level === STATE.riskFilter);

  if (STATE.searchQuery)
    res = res.filter(r => r.customer_id.toLowerCase().includes(STATE.searchQuery));

  if (STATE.sortMode === 'prob-desc')
    res.sort((a, b) => b.churn_probability - a.churn_probability);
  else if (STATE.sortMode === 'prob-asc')
    res.sort((a, b) => a.churn_probability - b.churn_probability);
  else if (STATE.sortMode === 'id-asc')
    res.sort((a, b) => a.customer_id.localeCompare(b.customer_id));

  STATE.filteredResults = res;
  renderTable();
}

function renderTable() {
  const empty   = document.getElementById('results-empty');
  const wrapper = document.getElementById('results-table-wrapper');
  const tbody   = document.getElementById('results-tbody');

  if (!STATE.allResults.length) {
    empty.style.display = 'flex';
    wrapper.style.display = 'none';
    return;
  }

  empty.style.display = 'none';
  wrapper.style.display = 'block';

  const start = (STATE.currentPage - 1) * STATE.pageSize;
  const page  = STATE.filteredResults.slice(start, start + STATE.pageSize);

  tbody.innerHTML = page.map((r, i) => {
    const color   = { High: '#ef4444', Medium: '#f59e0b', Low: '#10b981' }[r.risk_level];
    const topFact = r.top_features?.[0]?.description || '—';
    const prob    = r.churn_probability_pct;

    return `
      <tr>
        <td><span class="font-mono" style="font-size:12px">${r.customer_id}</span></td>
        <td>
          <div class="prob-bar-wrap">
            <div class="prob-bar">
              <div class="prob-fill" style="width:${prob}%; background:${color}"></div>
            </div>
            <span style="font-size:12px; font-weight:600; color:${color}">${prob.toFixed(1)}%</span>
          </div>
        </td>
        <td><span class="risk-badge ${r.risk_level}">${r.risk_level}</span></td>
        <td>
          <span class="pred-chip ${r.prediction ? 'churn' : 'retain'}">
            <i class="fas fa-${r.prediction ? 'user-minus' : 'user-check'}"></i>
            ${r.prediction ? 'Churn' : 'Retain'}
          </span>
        </td>
        <td>
          <select class="actual-select" onchange="trackOutcome('${r.customer_id}', this.value, ${r.prediction}, ${r.churn_probability})">
            <option value="">— Record —</option>
            <option value="0" ${r.actual_churn === 0 ? 'selected' : ''}>✓ Retained</option>
            <option value="1" ${r.actual_churn === 1 ? 'selected' : ''}>✗ Churned</option>
          </select>
        </td>
        <td style="font-size:12px; color:var(--text-secondary)">${topFact}</td>
        <td>
          <button class="btn btn-secondary btn-sm" onclick="openDetailModal(${start + i})">
            <i class="fas fa-eye"></i> Details
          </button>
        </td>
      </tr>
    `;
  }).join('');

  // Count label
  document.getElementById('table-count').textContent =
    `Showing ${Math.min(start+1, STATE.filteredResults.length)}–${Math.min(start+STATE.pageSize, STATE.filteredResults.length)} of ${STATE.filteredResults.length}`;

  renderPagination();
}

function renderPagination() {
  const total = Math.ceil(STATE.filteredResults.length / STATE.pageSize);
  const pg = document.getElementById('pagination');
  if (total <= 1) { pg.innerHTML = ''; return; }

  let html = '';
  if (STATE.currentPage > 1)
    html += `<button class="page-btn" onclick="goPage(${STATE.currentPage-1})"><i class="fas fa-chevron-left"></i></button>`;

  for (let p = Math.max(1, STATE.currentPage-2); p <= Math.min(total, STATE.currentPage+2); p++)
    html += `<button class="page-btn ${p===STATE.currentPage?'active':''}" onclick="goPage(${p})">${p}</button>`;

  if (STATE.currentPage < total)
    html += `<button class="page-btn" onclick="goPage(${STATE.currentPage+1})"><i class="fas fa-chevron-right"></i></button>`;

  pg.innerHTML = html;
}

function goPage(p) {
  STATE.currentPage = p;
  renderTable();
}

function updateResultsBadge() {
  const badge = document.getElementById('results-badge');
  if (STATE.allResults.length) {
    badge.textContent = STATE.allResults.length;
    badge.style.display = 'inline-block';
  } else {
    badge.style.display = 'none';
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// OUTCOME TRACKING
// ═══════════════════════════════════════════════════════════════════════════
async function trackOutcome(customerId, actualChurn, predictedChurn, churnProb) {
  if (actualChurn === '') return;

  try {
    const res = await fetch('/api/outcomes/track', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        customer_id: customerId,
        actual_churn: parseInt(actualChurn),
        predicted_churn: predictedChurn,
        churn_probability: churnProb
      })
    });
    const data = await res.json();
    showToast(`Outcome recorded for ${customerId}`, 'success');

    // Update state
    const idx = STATE.allResults.findIndex(r => r.customer_id === customerId);
    if (idx >= 0) STATE.allResults[idx].actual_churn = parseInt(actualChurn);

  } catch (e) {
    showToast('Failed to record outcome', 'error');
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// CUSTOMER DETAIL MODAL
// ═══════════════════════════════════════════════════════════════════════════
function openDetailModal(index) {
  const r = STATE.filteredResults[index];
  if (!r) return;

  const color = { High: '#ef4444', Medium: '#f59e0b', Low: '#10b981' }[r.risk_level];

  // Customer data fields
  const fields = r.customer_data || {};
  const fieldHtml = Object.entries(fields).map(([k, v]) => `
    <div class="detail-row">
      <span class="detail-label">${k}</span>
      <span class="detail-val">${v}</span>
    </div>
  `).join('');

  // Recommendations
  const recsHtml = (r.recommendations || []).length
    ? r.recommendations.map(rec => `
        <div class="rec-item">
          <div class="rec-priority-dot ${rec.priority}"></div>
          <div>
            <div class="rec-title">${rec.title}</div>
            <div class="rec-action">${rec.action}</div>
          </div>
        </div>
      `).join('')
    : `<p class="text-muted" style="font-size:12px; text-align:center; padding:12px">No urgent actions needed.</p>`;

  document.getElementById('modal-title').textContent = `Customer ${r.customer_id}`;
  document.getElementById('modal-body').innerHTML = `
    <div style="display:flex; align-items:center; gap:20px; margin-bottom:24px; padding:16px;
      background:rgba(255,255,255,0.03); border-radius:10px; border:1px solid var(--border)">
      <div style="text-align:center">
        <div style="font-size:32px; font-weight:900; color:${color}">${r.churn_probability_pct.toFixed(1)}%</div>
        <div style="font-size:11px; color:var(--text-secondary)">Churn Risk</div>
      </div>
      <div>
        <span class="risk-badge ${r.risk_level}" style="margin-bottom:6px; display:inline-flex">${r.risk_level} Risk</span>
        <div style="font-size:12px; color:var(--text-secondary)">
          ${r.prediction ? '<span class="text-red">⚠ Predicted to Churn</span>' : '<span class="text-green">✓ Predicted to Retain</span>'}
        </div>
      </div>
    </div>

    <h4 style="font-size:12px; font-weight:700; text-transform:uppercase; letter-spacing:0.8px;
      color:var(--text-secondary); margin-bottom:12px">Customer Attributes</h4>
    <div class="customer-detail-grid">${fieldHtml}</div>

    <h4 style="font-size:12px; font-weight:700; text-transform:uppercase; letter-spacing:0.8px;
      color:var(--text-secondary); margin-bottom:12px; margin-top:20px">
      <i class="fas fa-lightbulb" style="color:var(--amber)"></i> Retention Recommendations
    </h4>
    ${recsHtml}
  `;

  document.getElementById('detail-modal-overlay').classList.add('open');
}

function closeModal() {
  document.getElementById('detail-modal-overlay').classList.remove('open');
}

// ═══════════════════════════════════════════════════════════════════════════
// MODEL PERFORMANCE TAB
// ═══════════════════════════════════════════════════════════════════════════
async function loadPerformance() {
  // Training metrics from model info
  if (STATE.modelMeta) {
    const m = STATE.modelMeta.metrics;
    setText('perf-accuracy',  pct(m.accuracy));
    setText('perf-precision', pct(m.precision));
    setText('perf-recall',    pct(m.recall));
    setText('perf-f1',        pct(m.f1_score));
    setText('perf-auc',       pct(m.roc_auc));
    setText('perf-cv',        pct(m.cv_auc_mean));
  } else {
    // Try loading
    await loadModelInfo();
  }

  // Live tracking
  try {
    const res  = await fetch('/api/outcomes/performance');
    const data = await res.json();
    renderLivePerformance(data);
  } catch (e) {
    console.warn('Could not load performance data');
  }
}

function renderLivePerformance(data) {
  const container = document.getElementById('live-perf-content');
  const logDiv    = document.getElementById('outcomes-log-table');

  if (!data.tracked_outcomes || !data.metrics) {
    container.innerHTML = `
      <div class="perf-empty">
        <i class="fas fa-chart-line"></i>
        <p>No outcomes tracked yet.</p>
        <p class="sub">Record actual churn outcomes in the Results tab to measure live accuracy.</p>
      </div>`;
    logDiv.innerHTML = '<p class="text-muted" style="font-size:12px; text-align:center; padding:20px">No outcome logs yet.</p>';
    return;
  }

  const m = data.metrics;
  container.innerHTML = `
    <div class="live-metrics-grid">
      <div class="live-metric">
        <div class="val">${pct(m.accuracy)}</div>
        <div class="lbl">Live Accuracy</div>
      </div>
      <div class="live-metric">
        <div class="val">${pct(m.precision)}</div>
        <div class="lbl">Precision</div>
      </div>
      <div class="live-metric">
        <div class="val">${pct(m.recall)}</div>
        <div class="lbl">Recall</div>
      </div>
      <div class="live-metric">
        <div class="val">${pct(m.f1_score)}</div>
        <div class="lbl">F1 Score</div>
      </div>
    </div>
    <div style="display:grid; grid-template-columns:repeat(4,1fr); gap:8px; font-size:12px; text-align:center">
      <div style="padding:8px; background:rgba(16,185,129,0.1); border-radius:6px; color:var(--green)">
        <strong>${m.true_positives}</strong><br><span class="text-muted">True +</span>
      </div>
      <div style="padding:8px; background:rgba(239,68,68,0.1); border-radius:6px; color:var(--red)">
        <strong>${m.false_positives}</strong><br><span class="text-muted">False +</span>
      </div>
      <div style="padding:8px; background:rgba(16,185,129,0.1); border-radius:6px; color:var(--green)">
        <strong>${m.true_negatives}</strong><br><span class="text-muted">True -</span>
      </div>
      <div style="padding:8px; background:rgba(245,158,11,0.1); border-radius:6px; color:var(--amber)">
        <strong>${m.false_negatives}</strong><br><span class="text-muted">False -</span>
      </div>
    </div>
    <p style="font-size:11px; color:var(--text-secondary); margin-top:12px; text-align:center">
      Based on ${data.tracked_outcomes} tracked outcomes
    </p>
  `;

  // Log table
  if (data.outcomes && data.outcomes.length) {
    logDiv.innerHTML = `
      <table class="results-table">
        <thead>
          <tr>
            <th>Customer ID</th>
            <th>Predicted</th>
            <th>Actual</th>
            <th>Churn Probability</th>
            <th>Correct?</th>
            <th>Recorded</th>
          </tr>
        </thead>
        <tbody>
          ${data.outcomes.map(o => {
            const correct = o.predicted_churn === o.actual_churn;
            return `
              <tr>
                <td class="font-mono" style="font-size:12px">${o.customer_id}</td>
                <td>
                  <span class="pred-chip ${o.predicted_churn ? 'churn' : 'retain'}">
                    ${o.predicted_churn ? 'Churn' : 'Retain'}
                  </span>
                </td>
                <td>
                  <span class="pred-chip ${o.actual_churn ? 'churn' : 'retain'}">
                    ${o.actual_churn ? 'Churned' : 'Retained'}
                  </span>
                </td>
                <td style="font-size:12px">${o.churn_probability ? (o.churn_probability * 100).toFixed(1) + '%' : '—'}</td>
                <td>
                  <span style="font-size:12px; color:${correct ? 'var(--green)' : 'var(--red)'}; font-weight:600">
                    ${correct ? '✓ Yes' : '✗ No'}
                  </span>
                </td>
                <td style="font-size:11px; color:var(--text-secondary)">${formatDate(o.recorded_at)}</td>
              </tr>`;
          }).join('')}
        </tbody>
      </table>`;
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// DOWNLOAD
// ═══════════════════════════════════════════════════════════════════════════
async function downloadResults() {
  try {
    const res = await fetch('/api/download/results');
    if (!res.ok) {
      const data = await res.json();
      throw new Error(data.error || 'Download failed');
    }
    const blob = await res.blob();
    const url  = URL.createObjectURL(blob);
    const a    = document.createElement('a');
    a.href = url;
    a.download = `churn_predictions_${new Date().toISOString().slice(0,10)}.csv`;
    a.click();
    URL.revokeObjectURL(url);
    showToast('Predictions exported!', 'success');
  } catch (e) {
    showToast(e.message, 'error');
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// TOAST NOTIFICATIONS
// ═══════════════════════════════════════════════════════════════════════════
function showToast(message, type = 'info') {
  const icons = { success: 'fa-circle-check', error: 'fa-circle-xmark', info: 'fa-circle-info', warning: 'fa-triangle-exclamation' };
  const container = document.getElementById('toast-container');

  const toast = document.createElement('div');
  toast.className = `toast ${type}`;
  toast.innerHTML = `
    <i class="fas ${icons[type] || icons.info}"></i>
    <span class="toast-msg">${message}</span>
    <button class="toast-close" onclick="this.parentElement.remove()">✕</button>
  `;
  container.appendChild(toast);
  setTimeout(() => { if (toast.parentElement) toast.remove(); }, 4500);
}

// ═══════════════════════════════════════════════════════════════════════════
// UTILITY HELPERS
// ═══════════════════════════════════════════════════════════════════════════
function setText(id, text) {
  const el = document.getElementById(id);
  if (el) el.textContent = text;
}

function pct(val) {
  return (val * 100).toFixed(1) + '%';
}

function formatDate(iso) {
  if (!iso) return '—';
  try {
    return new Date(iso).toLocaleDateString('en-US', { month:'short', day:'numeric', hour:'2-digit', minute:'2-digit' });
  } catch { return iso; }
}

function triggerDownload(content, filename, mimeType) {
  const blob = new Blob([content], { type: mimeType });
  const url  = URL.createObjectURL(blob);
  const a    = document.createElement('a');
  a.href = url; a.download = filename; a.click();
  URL.revokeObjectURL(url);
}

// Close modal on Escape
document.addEventListener('keydown', e => {
  if (e.key === 'Escape') closeModal();
});
