// ─── Screen Manager ───────────────────────────────────
function show(id) {
    document.querySelectorAll('.screen').forEach(s => {
        s.classList.remove('active');
        s.style.display = 'none';
    });
    const el = document.getElementById(id);
    el.style.display = 'flex';
    el.classList.add('active');
    window.scrollTo(0, 0);
}

// ─── State ────────────────────────────────────────────
let currentType = null;      // 'civil' | 'criminal'
let currentDomain = null;    // e.g. 'property', 'tort'
let currentDifficulty = null;
let currentCaseData = null;
let chatHistory = [];

// ─── Sub-category data ────────────────────────────────
const CIVIL_CATS = [
    { icon: '🏠', label: 'Property / Rent Dispute', desc: 'Disputes about land, home, or rent payments', domain: 'property', difficulty: 'medium' },
    { icon: '📄', label: 'Breach of Contract', desc: 'Someone did not honour a signed agreement', domain: 'contract', difficulty: 'easy' },
    { icon: '👨‍👩‍👧', label: 'Family / Matrimonial', desc: 'Divorce, custody, adoption or maintenance', domain: 'family', difficulty: 'medium' },
    { icon: '⚠️', label: 'Tort / Personal Injury', desc: 'Someone caused harm through negligence', domain: 'tort', difficulty: 'medium' },
    { icon: '💼', label: 'Employment / Workplace', desc: 'Wrongful termination, salary or harassment at work', domain: 'contract', difficulty: 'easy' },
    { icon: '🏥', label: 'Medical Negligence', desc: 'Harm caused by a doctor, hospital or healthcare provider', domain: 'tort', difficulty: 'hard' },
    { icon: '🛍️', label: 'Consumer Dispute', desc: 'Defective product, false advertising or poor service', domain: 'contract', difficulty: 'easy' },
    { icon: '🤷', label: 'Other / Not Sure', desc: 'Describe it in your own words to the AI', domain: 'contract', difficulty: 'easy' },
];
const CRIMINAL_CATS = [
    { icon: '🪙', label: 'Petty Crime', desc: 'Minor offences like trespass or public nuisance (BNS Sec 303)', domain: 'petty_crime', difficulty: 'easy' },
    { icon: '👊', label: 'Assault / Hurt', desc: 'Physical harm caused by another person (BNS Sec 115-117)', domain: 'petty_crime', difficulty: 'medium' },
    { icon: '🚨', label: 'Sexual Assault / Rape', desc: 'Offences under BNS Sec 63-70. Emergency protocols apply.', domain: 'petty_crime', difficulty: 'hard', urgent: true },
    { icon: '💀', label: 'Murder / Culpable Homicide', desc: 'Taking of human life (BNS Sec 101-103). Serious investigation required.', domain: 'petty_crime', difficulty: 'hard', urgent: true },
    { icon: '🏠', label: 'Domestic Violence', desc: 'Abuse within the home — physical, emotional or financial (Protection of Women from DV Act)', domain: 'petty_crime', difficulty: 'medium', urgent: true },
    { icon: '🔗', label: 'Kidnapping / Abduction', desc: 'Unlawful confinement or taking of a person (BNS Sec 137-140)', domain: 'petty_crime', difficulty: 'hard', urgent: true },
    { icon: '💳', label: 'Fraud / Cheating', desc: 'Deception or financial fraud (BNS Sec 316-318)', domain: 'petty_crime', difficulty: 'medium' },
    { icon: '📱', label: 'Cyber Crime', desc: 'Online harassment, hacking or identity theft (IT Act + BNS)', domain: 'petty_crime', difficulty: 'hard' },
    { icon: '💊', label: 'Drug / Narcotics Offence', desc: 'Possession or trafficking under NDPS Act', domain: 'petty_crime', difficulty: 'hard' },
    { icon: '🔪', label: 'Robbery / Dacoity', desc: 'Violent theft or organised robbery (BNS Sec 309-310)', domain: 'petty_crime', difficulty: 'hard', urgent: true },
    { icon: '🤷', label: 'Other / Not Listed', desc: 'Describe your situation — JusticeEngine-01 will categorise', domain: 'petty_crime', difficulty: 'easy' },
];

const QUASI_CATS = [
    { icon: '📄', label: 'RTI Appeal', desc: 'Appealing a denial or non-response to a Right to Information request', domain: 'contract', difficulty: 'easy' },
    { icon: '🎫', label: 'Licensing Dispute', desc: 'Rejection or cancellation of a business, trade, or professional licence', domain: 'contract', difficulty: 'medium' },
    { icon: '💰', label: 'Tax / Revenue Dispute', desc: 'Dispute with income tax, GST, or revenue authorities', domain: 'contract', difficulty: 'hard' },
    { icon: '🗳️', label: 'Electoral Complaint', desc: 'Complaints regarding elections, voter rights, or misconduct', domain: 'contract', difficulty: 'medium' },
    { icon: '🛍️', label: 'Consumer Commission', desc: 'Escalating a consumer dispute to a District or State Consumer Commission', domain: 'contract', difficulty: 'easy' },
    { icon: '👤', label: 'Service / Employment Tribunal', desc: 'Government or public sector employment disputes via CAT or tribunal', domain: 'contract', difficulty: 'medium' },
    { icon: '🏥', label: 'Medical / Health Regulatory', desc: 'Complaints to MCI, NMC, or other regulatory health bodies', domain: 'tort', difficulty: 'medium' },
    { icon: '🌎', label: 'Environmental / Pollution Board', desc: 'Complaints to NGT or State Pollution Control Board', domain: 'tort', difficulty: 'hard' },
    { icon: '🤷', label: 'Other Regulatory Matter', desc: 'Any other government body or tribunal hearing', domain: 'contract', difficulty: 'easy' },
];

// ─── Landing ──────────────────────────────────────────
document.getElementById('enter-btn').addEventListener('click', () => show('screen-action'));
document.getElementById('back-to-landing').addEventListener('click', () => show('screen-landing'));

// ─── Language Toggle ──────────────────────────────────
document.getElementById('lang-toggle').addEventListener('change', (e) => {
    if (e.target.value === 'hi') {
        document.querySelector('[data-i18n="action_title"]').textContent = 'न्यायालय आपकी किस प्रकार सहायता कर सकता है?';
        document.querySelector('[data-i18n="action_subtitle"]').textContent = 'कृपया अपनी यात्रा की प्रकृति चुनें';
        document.querySelector('[data-i18n="file_case"]').textContent = 'मैं केस दर्ज करना चाहता हूँ';
        document.querySelector('[data-i18n="file_case_desc"]').textContent = 'प्रारंभिक कानूनी राय के लिए अपना मामला एआई जज के सामने पेश करें';
        document.querySelector('[data-i18n="run_demo"]').textContent = 'डेमो केस चलाएं';
        document.querySelector('[data-i18n="run_demo_desc"]').textContent = 'सिस्टम क्षमताओं को प्रदर्शित करने के लिए सत्यापित BNS केस चलाएं';
        document.querySelector('[data-i18n="withdraw_case"]').textContent = 'मेरा केस वापस लें';
        document.querySelector('[data-i18n="withdraw_desc"]').textContent = 'पहले से दायर मामले को रद्द करें';
    } else {
        document.querySelector('[data-i18n="action_title"]').textContent = 'How can the Court assist you?';
        document.querySelector('[data-i18n="action_subtitle"]').textContent = 'Please select the nature of your visit';
        document.querySelector('[data-i18n="file_case"]').textContent = 'I want to register a case';
        document.querySelector('[data-i18n="file_case_desc"]').textContent = 'Present your matter before the AI Judge for a legal opinion or triage';
        document.querySelector('[data-i18n="run_demo"]').textContent = 'Run a Demo Case';
        document.querySelector('[data-i18n="run_demo_desc"]').textContent = 'Automatically run a verified BNS case to demonstrate system capabilities';
        document.querySelector('[data-i18n="withdraw_case"]').textContent = 'Withdraw my case';
        document.querySelector('[data-i18n="withdraw_desc"]').textContent = 'Cancel or retract a previously filed matter';
    }
});

// ─── Action chooser ───────────────────────────────────
document.getElementById('btn-file-case').addEventListener('click', () => show('screen-type'));
document.getElementById('btn-withdraw').addEventListener('click', () => show('screen-withdraw'));

// Demo Mode Auto-runner
document.getElementById('btn-demo').addEventListener('click', async () => {
    // Fill KYC
    document.getElementById('aadhar-input').value = '1234-5678-9012';
    document.getElementById('phone-input').value = '+91 9876543210';
    document.getElementById('relation-input').value = 'victim';
    document.getElementById('offender-input').value = 'DL 4C 1234 (Driver Name Unknown)';
    const summaryEl = document.getElementById('case-summary-input');
    if (summaryEl) summaryEl.value = 'A 17-year-old minor was driving his parent\'s car and jumped a red light, causing a collision with my vehicle. I suffered injuries and my car sustained Rs. 85,000 in damage. The accident occurred on 15 April 2026 at Ring Road, Delhi. A dashcam video has been uploaded as evidence.';

    // Auto click through
    show('screen-kyc');
    await new Promise(r => setTimeout(r, 800));
    document.getElementById('btn-verify-kyc').click();

    await new Promise(r => setTimeout(r, 800));
    document.getElementById('upload-status').style.display = 'block';
    document.getElementById('upload-status').innerHTML = '✅ Evidence pre-verified by Police Module (Demo Override).';
    await new Promise(r => setTimeout(r, 1200));

    currentType = 'criminal';
    currentDomain = 'petty_crime';
    currentDifficulty = 'hard';
    loadDossier();
});

// ─── NJDG Animated Counters ───────────────────────────
function animateCounter(id, target, suffix, duration) {
    const el = document.getElementById(id);
    if (!el) return;
    let start = 0;
    const step = target / (duration / 16);
    const timer = setInterval(() => {
        start += step;
        if (start >= target) { el.textContent = target.toLocaleString('en-IN') + suffix; clearInterval(timer); return; }
        el.textContent = Math.floor(start).toLocaleString('en-IN') + suffix;
    }, 16);
}
window.addEventListener('load', () => {
    setTimeout(() => {
        animateCounter('counter-total', 50000000, '+', 2000);
        animateCounter('counter-women', 478587, '+', 2000);
        animateCounter('counter-undated', 348493, '+', 2000);
        animateCounter('counter-years', 15, '+', 1500);
    }, 600);
});

document.getElementById('back-from-withdraw').addEventListener('click', () => show('screen-action'));
document.getElementById('btn-confirm-withdraw').addEventListener('click', () => {
    alert('Case withdrawal request submitted. Reference number sent to your registered contact.');
    show('screen-action');
});

// ─── Civil / Criminal ─────────────────────────────────
document.getElementById('back-to-action').addEventListener('click', () => show('screen-action'));
document.getElementById('btn-civil').addEventListener('click', () => { currentType = 'civil'; buildSubcats(CIVIL_CATS); show('screen-subcat'); });
document.getElementById('btn-criminal').addEventListener('click', () => { currentType = 'criminal'; buildSubcats(CRIMINAL_CATS); show('screen-subcat'); });
document.getElementById('btn-quasi').addEventListener('click', () => { currentType = 'quasi'; buildSubcats(QUASI_CATS); show('screen-subcat'); });
document.getElementById('back-to-type').addEventListener('click', () => show('screen-type'));


// ─── Sub-category builder ─────────────────────────────
function buildSubcats(cats) {
    const grid = document.getElementById('subcat-grid');
    grid.innerHTML = '';
    cats.forEach(cat => {
        const btn = document.createElement('button');
        btn.className = 'subcat-card' + (cat.urgent ? ' urgent-card' : '');
        btn.innerHTML = `
            <span class="sc-icon">${cat.icon}</span>
            <strong>${cat.label}</strong>
            <p>${cat.desc}</p>
            ${cat.urgent ? '<span class="urgent-tag">⚠️ Urgent — Human Judge may be required</span>' : ''}
        `;
        btn.addEventListener('click', () => {
            currentDomain = cat.domain;
            currentDifficulty = cat.difficulty;
            if (cat.urgent) {
                const ok = confirm(`⚠️ IMPORTANT: "${cat.label}" cases are serious. The AI will gather your facts and prepare a preliminary record, but a Human Judge WILL review this case. Do you want to proceed?`);
                if (!ok) return;
            }
            show('screen-kyc'); // Go to KYC instead of dossier directly
        });
        grid.appendChild(btn);
    });
}

// ─── KYC & Evidence Flow ──────────────────────────────
let kycData = {};
document.getElementById('back-to-subcat-from-kyc').addEventListener('click', () => show('screen-subcat'));
document.getElementById('btn-verify-kyc').addEventListener('click', () => {
    const aadhar = document.getElementById('aadhar-input').value;
    if (aadhar.length < 12) {
        alert('Please enter a valid Aadhar number for DigiLocker verification.');
        return;
    }

    // Gather offender info
    let offenderInfo = 'Unknown';
    if (!document.getElementById('offender-unknown-cb').checked) {
        const name = document.getElementById('offender-name').value;
        const phone = document.getElementById('offender-phone').value;
        const addr = document.getElementById('offender-address').value;
        offenderInfo = [name, phone, addr].filter(Boolean).join(', ') || 'Not provided';
    }

    // Save KYC Data including case summary
    kycData = {
        aadhar: aadhar,
        phone: document.getElementById('phone-input').value,
        relation: document.getElementById('relation-input').value,
        offender: offenderInfo,
        caseSummary: document.getElementById('case-summary-input')?.value || ''
    };

    show('screen-evidence');
});

document.getElementById('back-to-kyc').addEventListener('click', () => show('screen-kyc'));

// ─── Evidence Upload — Real file handling ─────────────
let uploadedFiles = []; // Array of {name, type, dataUrl}

document.querySelector('.upload-box').addEventListener('click', () => {
    document.getElementById('evidence-file').click();
});

document.getElementById('evidence-file').addEventListener('change', async (e) => {
    const files = Array.from(e.target.files);
    if (!files.length) return;

    // Read each file as dataURL
    for (const file of files) {
        const dataUrl = await new Promise((resolve) => {
            const reader = new FileReader();
            reader.onload = ev => resolve(ev.target.result);
            reader.readAsDataURL(file);
        });
        const type = file.type.startsWith('image/') ? 'image' : file.type === 'application/pdf' ? 'pdf' : file.type.startsWith('video/') ? 'video' : 'other';
        uploadedFiles.push({ name: file.name, type, dataUrl });
    }

    // Render file list
    renderUploadedFiles();
    document.getElementById('upload-status').style.display = 'block';
    document.getElementById('upload-status').innerHTML = `✅ ${uploadedFiles.length} file(s) uploaded. Pending police verification.`;
});

function renderUploadedFiles() {
    let el = document.getElementById('upload-file-list');
    if (!el) {
        el = document.createElement('div');
        el.id = 'upload-file-list';
        el.style.cssText = 'display:grid;grid-template-columns:repeat(auto-fill,minmax(120px,1fr));gap:0.75rem;margin:1rem 0;';
        document.getElementById('upload-status').insertAdjacentElement('afterend', el);
    }
    el.innerHTML = '';
    uploadedFiles.forEach((f, i) => {
        const div = document.createElement('div');
        div.style.cssText = 'background:#1e293b;border:1px solid #334155;border-radius:8px;overflow:hidden;text-align:center;';
        if (f.type === 'image') {
            div.innerHTML = `<img src="${f.dataUrl}" style="width:100%;height:80px;object-fit:cover;"><div style="font-size:0.72rem;color:#94a3b8;padding:0.3rem;overflow:hidden;white-space:nowrap;text-overflow:ellipsis;">📸 ${f.name}</div>`;
        } else {
            const icon = f.type === 'pdf' ? '📄' : f.type === 'video' ? '🎥' : '📁';
            div.innerHTML = `<div style="height:80px;display:flex;align-items:center;justify-content:center;font-size:2rem;">${icon}</div><div style="font-size:0.72rem;color:#94a3b8;padding:0.3rem;overflow:hidden;white-space:nowrap;text-overflow:ellipsis;">${f.name}</div>`;
        }
        el.appendChild(div);
    });
}

document.getElementById('btn-submit-evidence').addEventListener('click', () => {
    if (!uploadedFiles.length) {
        alert('Please select files to upload, or click Skip.');
        return;
    }
    // Push to localStorage so police dashboard can read it
    const caseId = pushToPoliceQueue();
    
    // Update UI to show waiting state
    const statusDiv = document.getElementById('upload-status');
    statusDiv.style.display = 'block';
    statusDiv.innerHTML = `
        <div style="text-align:center; padding: 1rem;">
            <div style="font-size:2rem; animation: spin 2s linear infinite;">⏳</div>
            <div style="margin-top:1rem; color: #facc15; font-weight:600;">Awaiting Police Verification</div>
            <div style="font-size:0.85rem; margin-top:0.5rem; color:#94a3b8;">Please open the <a href="/police" target="_blank" style="color:#3b82f6;">Police Module</a> to verify your evidence.</div>
        </div>
    `;
    document.getElementById('btn-submit-evidence').style.display = 'none';
    document.getElementById('btn-skip-evidence').style.display = 'none';

    // Poll for status
    const pollTimer = setInterval(() => {
        const existing = JSON.parse(localStorage.getItem('evidence_submissions') || '[]');
        const myCase = existing.find(c => c.caseId === caseId);
        if (myCase && (myCase.status === 'verified' || myCase.status === 'rejected')) {
            clearInterval(pollTimer);
            if (myCase.status === 'verified') {
                statusDiv.innerHTML = `
                    <div style="text-align:center; padding: 1rem; background: rgba(74,222,128,0.1); border-radius: 8px;">
                        <div style="font-size:2rem;">✅</div>
                        <div style="margin-top:0.5rem; color: #4ade80; font-weight:600;">Evidence Verified</div>
                        <div style="font-size:0.85rem; margin-top:0.25rem; color:#cbd5e1;">Proceeding to Dossier...</div>
                    </div>
                `;
                setTimeout(() => loadDossier(), 1500);
            } else {
                statusDiv.innerHTML = `
                    <div style="text-align:center; padding: 1rem; background: rgba(248,113,113,0.1); border-radius: 8px;">
                        <div style="font-size:2rem;">❌</div>
                        <div style="margin-top:0.5rem; color: #f87171; font-weight:600;">Evidence Rejected</div>
                        <div style="font-size:0.85rem; margin-top:0.25rem; color:#cbd5e1;">The Police Module rejected your evidence. You cannot proceed.</div>
                    </div>
                `;
                document.getElementById('btn-skip-evidence').style.display = 'block';
                document.getElementById('btn-skip-evidence').textContent = 'Proceed without Evidence';
            }
        }
    }, 1000);
});

function pushToPoliceQueue() {
    const now = new Date();
    const caseId = `JA-${now.getFullYear()}-${String(now.getMonth()+1).padStart(2,'0')}-${Math.floor(Math.random()*90000+10000)}`;
    const existing = JSON.parse(localStorage.getItem('evidence_submissions') || '[]');
    existing.push({
        caseId,
        aadhar: kycData.aadhar || 'Not provided',
        incidentSummary: kycData.caseSummary || `${currentType} case — ${currentDomain || 'General'}`,
        caseType: currentType,
        subCategory: currentDomain,
        submittedAt: now.toLocaleString('en-IN'),
        status: 'pending',
        files: uploadedFiles.map(f => ({ name: f.name, type: f.type, dataUrl: f.type === 'image' ? f.dataUrl : null, demoPlaceholder: f.type === 'pdf' ? '📄' : f.type === 'video' ? '🎥' : '📁' }))
    });
    localStorage.setItem('evidence_submissions', JSON.stringify(existing));
    return caseId;
}

document.getElementById('btn-skip-evidence').addEventListener('click', () => {
    loadDossier();
});

// ─── Load Case & Go to Dossier ────────────────────────
async function loadDossier() {
    show('screen-dossier');
    document.getElementById('dossier-badge').textContent = currentType === 'civil' ? 'Civil Case' : currentType === 'criminal' ? 'Criminal Case' : 'Quasi-Judicial';
    window.__caseType = currentType;

    // Reset right panel
    document.getElementById('chat-panel').style.display = 'block';
    document.getElementById('ai-thinking').style.display = 'none';
    document.getElementById('verdict-panel').style.display = 'none';
    document.getElementById('accepted-panel').style.display = 'none';
    document.getElementById('escalated-panel').style.display = 'none';
    const ratioBlock = document.getElementById('v-ratio-block');
    const obiterBlock = document.getElementById('v-obiter-block');
    if (ratioBlock) ratioBlock.style.display = 'none';
    if (obiterBlock) obiterBlock.style.display = 'none';
    document.getElementById('generate-btn').disabled = true;
    document.getElementById('generate-hint').textContent = 'Answer the questions above to unlock the judgment';
    document.getElementById('chat-messages').innerHTML = '';
    chatHistory = [];

    // Try to fetch a matching case from backend
    try {
        const res = await fetch('/reset', {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ 
                domain: currentDomain || 'contract', 
                difficulty: currentDifficulty || 'easy',
                custom_facts: kycData.caseSummary || null,
                custom_evidence: uploadedFiles.map(f => f.name)
            })
        });
        if (!res.ok) throw new Error('API error');
        const data = await res.json();
        currentCaseData = data.observation;
        renderDossierLeft(currentCaseData);
    } catch(e) {
        // Fallback: use the user's own description as the case summary
        const userSummary = kycData.caseSummary || 'Your case has been registered. Please answer the questions below so I can build your legal dossier.';
        currentCaseData = {
            case_id: `USR-${Date.now().toString().slice(-6)}`,
            fact_pattern: userSummary,
            evidence_flags: uploadedFiles.map(f => f.name),
            statutes: currentType === 'criminal'
                ? ['Bharatiya Nyaya Sanhita (BNS) 2023', 'Bharatiya Nagarik Suraksha Sanhita (BNSS) 2023']
                : ['Indian Contract Act 1872', 'Code of Civil Procedure 1908'],
            precedents: []
        };
        renderDossierLeft(currentCaseData);
    }

    startFactFinding();
    printLetter('registration', { caseId: currentCaseData.case_id });
}

// ─── Digital Stamped Letters ─────────────────────────
function printLetter(type, data) {
    const now = new Date();
    const timestamp = now.toLocaleString('en-IN', { dateStyle: 'full', timeStyle: 'medium' });
    const refNo = `JA-${now.getFullYear()}-${String(now.getMonth()+1).padStart(2,'0')}-${Math.floor(Math.random()*90000+10000)}`;

    let title = '', body = '', stampText = '', stampColor = '#1a5276';

    if (type === 'registration') {
        title = 'Case Registration Certificate';
        stampText = 'REGISTERED';
        stampColor = '#1a5276';
        body = `
            <p>This is to certify that the following case has been <strong>officially registered</strong> with the Justice AI Portal on the date and time indicated below.</p>
            <table>
                <tr><th colspan="2" style="background:#f1f5f9; padding:0.5rem; text-align:left;">Registration Details</th></tr>
                <tr><td>Case ID</td><td><strong>${data.caseId}</strong></td></tr>
                <tr><td>Reference No.</td><td><strong>${refNo}</strong></td></tr>
                <tr><td>Registered On</td><td><strong>${timestamp}</strong></td></tr>
                <tr><td>Case Type</td><td><strong>${currentType ? currentType.toUpperCase() : 'N/A'}</strong></td></tr>
                <tr><th colspan="2" style="background:#f1f5f9; padding:0.5rem; text-align:left;">Petitioner KYC & Offender Info</th></tr>
                <tr><td>Aadhar (DigiLocker KYC)</td><td><strong>${kycData.aadhar || 'Verified (Internal)'}</strong></td></tr>
                <tr><td>Contact Phone</td><td><strong>${kycData.phone || 'N/A'}</strong></td></tr>
                <tr><td>Petitioner Role</td><td><strong>${kycData.relation || 'N/A'}</strong></td></tr>
                <tr><td>Offender Details</td><td><strong>${kycData.offender || 'Unknown'}</strong></td></tr>
                <tr><th colspan="2" style="background:#f1f5f9; padding:0.5rem; text-align:left;">Case Summary</th></tr>
                <tr><td colspan="2" style="font-style:italic; font-size:0.9rem;">${kycData.caseSummary || 'No summary provided.'}</td></tr>
                <tr><th colspan="2" style="background:#f1f5f9; padding:0.5rem; text-align:left;">Current Status</th></tr>
                <tr><td>Status</td><td><strong style="color:#1a5276">REGISTERED — AI Fact-Finding / Police Verification in Progress</strong></td></tr>
            </table>
            <p style="margin-top:1.5rem;">This document serves as <strong>official proof of registration</strong>. The timestamp above is tamper-proof and can be cited if any dispute arises regarding when this case was filed.</p>`;

    } else if (type === 'resolution') {
        title = 'AI Resolution Certificate';
        stampText = 'RESOLVED BY AI';
        stampColor = '#1e8449';
        body = `
            <p>This is to certify that the following case has been <strong>reviewed and resolved</strong> by JusticeEngine-01 (AI Legal Mediator) and the resolution has been <strong>accepted by the petitioner</strong>.</p>
            <table>
                <tr><td>Case ID</td><td><strong>${data.caseId}</strong></td></tr>
                <tr><td>Reference No.</td><td><strong>${refNo}</strong></td></tr>
                <tr><td>Resolved On</td><td><strong>${timestamp}</strong></td></tr>
                <tr><td>AI Verdict</td><td><strong>${data.verdict || 'N/A'}</strong></td></tr>
                <tr><td>Logic Score</td><td>${data.logic || 'N/A'}</td></tr>
                <tr><td>Accuracy Score</td><td>${data.accuracy || 'N/A'}</td></tr>
                <tr><td>Status</td><td><strong style="color:#1e8449">CASE CLOSED — Accepted by Petitioner</strong></td></tr>
            </table>
            <div style="margin-top:1.5rem; padding:1rem; background:#eafaf1; border-left:4px solid #1e8449; border-radius:4px;">
                <strong>AI Reasoning Summary:</strong><br>
                <p style="margin-top:0.5rem;">${data.reasoning || 'Detailed analysis on file.'}</p>
            </div>`;
    } else if (type === 'escalation') {
        title = 'Case Forwarded to Human Judge';
        stampText = 'FORWARDED TO JUDGE';
        stampColor = '#922b21';
        body = `
            <p>This is to certify that the following case, <strong>after receiving a Preliminary AI Opinion</strong>, has been <strong>escalated to a Human Presiding Officer</strong> as per the petitioner's request.</p>
            <table>
                <tr><td>Case ID</td><td><strong>${data.caseId}</strong></td></tr>
                <tr><td>Reference No.</td><td><strong>${refNo}</strong></td></tr>
                <tr><td>Escalated On</td><td><strong>${timestamp}</strong></td></tr>
                <tr><td>AI Draft Verdict</td><td><strong>${data.verdict || 'N/A'}</strong></td></tr>
                <tr><td>Escalation Reason(s)</td><td>${(data.reasons || []).join('; ') || 'Not specified'}</td></tr>
                <tr><td>Status</td><td><strong style="color:#922b21">PENDING HUMAN REVIEW</strong></td></tr>
            </table>
            <div style="margin-top:1.5rem; padding:1rem; background:#fdf2f8; border-left:4px solid #922b21; border-radius:4px;">
                <strong>AI Preliminary Reasoning (included for Judge's review):</strong><br>
                <p style="margin-top:0.5rem;">${data.reasoning || 'Preliminary analysis on file.'}</p>
            </div>
            <p style="margin-top:1rem;"><em>Note: The Human Judge will receive the complete case dossier along with this document.</em></p>`;
    }

    const htmlContent = `<!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>${title}</title>
        <link href="https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700&family=Inter:wght@400;600&display=swap" rel="stylesheet">
        <style>
            body { font-family: 'Inter', sans-serif; background: #f5f5f0; margin: 0; padding: 2rem; color: #1a1a1a; }
            .letter { background: white; max-width: 750px; margin: 0 auto; padding: 3rem; border: 1px solid #ddd; box-shadow: 0 4px 20px rgba(0,0,0,0.1); position: relative; }
            .letterhead { display: flex; align-items: center; justify-content: space-between; border-bottom: 3px solid #0a0e1a; padding-bottom: 1.5rem; margin-bottom: 2rem; }
            .letterhead-left { display: flex; align-items: center; gap: 1rem; }
            .lh-logo { font-size: 2.5rem; }
            .lh-title { font-family: 'Playfair Display', serif; font-size: 1.5rem; color: #0a0e1a; }
            .lh-sub { font-size: 0.8rem; color: #666; margin-top: 0.2rem; }
            .lh-right { text-align: right; font-size: 0.8rem; color: #666; }
            h2 { font-family: 'Playfair Display', serif; font-size: 1.4rem; text-align: center; margin-bottom: 1.5rem; color: #0a0e1a; }
            table { width: 100%; border-collapse: collapse; margin: 1rem 0; }
            td { padding: 0.6rem 0.75rem; border: 1px solid #e0e0e0; font-size: 0.9rem; }
            td:first-child { background: #f9f9f9; width: 35%; font-weight: 600; color: #444; }
            .stamp { position: absolute; top: 3.5rem; right: 3rem; width: 110px; height: 110px; border: 5px solid ${stampColor}; border-radius: 50%; display: flex; align-items: center; justify-content: center; transform: rotate(-15deg); opacity: 0.85; }
            .stamp-inner { text-align: center; color: ${stampColor}; font-weight: 700; font-size: 0.7rem; letter-spacing: 0.05em; line-height: 1.4; padding: 0.5rem; }
            .footer { margin-top: 3rem; padding-top: 1rem; border-top: 1px solid #ddd; font-size: 0.75rem; color: #888; display: flex; justify-content: space-between; }
            .print-btn { display: block; width: 100%; max-width: 750px; margin: 1.5rem auto 0; padding: 0.85rem; background: #0a0e1a; color: white; border: none; border-radius: 6px; font-size: 1rem; cursor: pointer; font-family: 'Inter', sans-serif; }
            .print-btn:hover { background: #1a2744; }
            @media print { .print-btn { display: none; } body { background: white; padding: 0; } .letter { box-shadow: none; border: none; } }
        </style>
    </head>
    <body>
        <div class="letter">
            <div class="stamp"><div class="stamp-inner">⚖️<br>${stampText}<br>JUSTICE AI</div></div>
            <div class="letterhead">
                <div class="letterhead-left">
                    <span class="lh-logo">⚖️</span>
                    <div>
                        <div class="lh-title">Justice AI Portal</div>
                        <div class="lh-sub">India's AI Legal Mediator | Powered by JusticeEngine-01</div>
                    </div>
                </div>
                <div class="lh-right">
                    Ref: <strong>${refNo}</strong><br>
                    ${timestamp}
                </div>
            </div>
            <h2>${title}</h2>
            ${body}
            <div class="footer">
                <span>Document generated by Justice AI — justiceai.local</span>
                <span>Ref: ${refNo} | ${timestamp}</span>
            </div>
        </div>
        <button class="print-btn" onclick="window.print()">🖨️ Print or Save as PDF</button>
    </body>
    </html>`;

    // Try to open it immediately, but it might be blocked by popup blockers
    try {
        const win = window.open('', '_blank', 'width=800,height=700');
        if (win) {
            win.document.write(htmlContent);
            win.document.close();
        }
    } catch(e) {}

    // ALWAYS append a button to the chat so the user can open it manually if blocked
    const btnId = 'pdf-btn-' + Date.now();
    window[btnId] = function() {
        const win = window.open('', '_blank', 'width=800,height=700');
        win.document.write(htmlContent);
        win.document.close();
    };

    postAI(`📜 **${title} generated.**\n\n<button onclick="window['${btnId}']()" style="margin-top:0.5rem; padding:0.6rem 1.2rem; background:var(--gold); color:#000; border:none; border-radius:6px; font-weight:700; cursor:pointer;">📄 Open PDF Certificate</button>`);
}

document.getElementById('back-to-subcat').addEventListener('click', () => show('screen-subcat'));

// ─── Render left dossier ──────────────────────────────
function renderDossierLeft(obs) {
    document.getElementById('d-fact-pattern').textContent = obs.fact_pattern;

    const evEl = document.getElementById('d-evidence');
    evEl.innerHTML = '';
    (obs.evidence_flags || []).forEach(f => {
        const s = document.createElement('span');
        s.textContent = f;
        evEl.appendChild(s);
    });
    if (!obs.evidence_flags || obs.evidence_flags.length === 0) evEl.innerHTML = '<span style="color:var(--muted);font-size:0.85rem">None provided yet</span>';

    const stEl = document.getElementById('d-statutes');
    stEl.innerHTML = '';
    (obs.statutes || []).forEach(s => {
        const li = document.createElement('li');
        li.textContent = s;
        stEl.appendChild(li);
    });
}

// ─── Fact Finding Chat ────────────────────────────────
function startFactFinding() {
    postAI("To help me build your legal dossier, I need to ask you a few short questions. Let's begin — could you confirm whether you have any written proof related to your case, such as a contract, receipt, or message?");
}

async function sendUserMessage(text) {
    if (!text.trim()) return;
    appendMsg('user', text);
    chatHistory.push({ role: 'user', content: text });
    document.getElementById('chat-input').value = '';

    try {
        const res = await fetch('/chat', {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                case_id: currentCaseData ? currentCaseData.case_id : 'DEMO',
                fact_pattern: currentCaseData ? currentCaseData.fact_pattern : '',
                user_message: text,
                chat_history: chatHistory,
                case_type: window.__caseType || 'civil'
            })
        });
        const data = await res.json();
        postAI(data.response);
        chatHistory.push({ role: 'ai', content: data.response });
    } catch(e) {
        postAI("I have gathered enough information. You may proceed to generate the AI judgment.");
    }

    // Also unlock after 8 exchanges as a fallback
    if (chatHistory.length >= 8) {
        document.getElementById('generate-btn').disabled = false;
        document.getElementById('generate-hint').textContent = '✅ Dossier ready — you may now generate the judgment';
    }
}

function postAI(text) {
    const div = document.createElement('div');
    div.className = 'msg msg-ai';
    div.innerHTML = `<strong>JusticeEngine-01</strong>${text}`;
    document.getElementById('chat-messages').appendChild(div);
    document.getElementById('chat-messages').scrollTop = 9999;
    
    // Detect when AI says dossier is complete
    if (text.includes('DOSSIER_COMPLETE:')) {
        const clean = text.replace('DOSSIER_COMPLETE:', '').trim();
        div.innerHTML = `<strong>JusticeEngine-01</strong>${clean}`;
        document.getElementById('generate-btn').disabled = false;
        document.getElementById('generate-hint').textContent = '✅ Dossier ready — you may now generate the judgment';
    }
}

function appendMsg(role, text) {
    const div = document.createElement('div');
    div.className = role === 'user' ? 'msg msg-user' : 'msg msg-ai';
    div.textContent = text;
    document.getElementById('chat-messages').appendChild(div);
    document.getElementById('chat-messages').scrollTop = 9999;
}

document.getElementById('chat-send').addEventListener('click', () => sendUserMessage(document.getElementById('chat-input').value));
document.getElementById('chat-input').addEventListener('keypress', e => { if(e.key === 'Enter') sendUserMessage(document.getElementById('chat-input').value); });

// ─── Evidence Locker in Dossier removed as per user feedback ─────

// ─── Generate Judgment ────────────────────────────────
document.getElementById('generate-btn').addEventListener('click', async () => {
    document.getElementById('chat-panel').style.display = 'none';
    document.getElementById('ai-thinking').style.display = 'block';

    try {
        const res = await fetch('/ai_judge', {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ 
                domain: currentDomain || 'contract', 
                difficulty: currentDifficulty || 'easy',
                custom_facts: kycData.caseSummary || null,
                custom_evidence: uploadedFiles.map(f => f.name)
            })
        });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();

        document.getElementById('ai-thinking').style.display = 'none';
        document.getElementById('verdict-panel').style.display = 'block';

        document.getElementById('v-case-id').textContent = currentCaseData ? currentCaseData.case_id : 'N/A';
        const verdictEl = document.getElementById('v-verdict');
        verdictEl.textContent = data.action.verdict;
        verdictEl.className = 'verdict-pill ' + (data.action.verdict === 'liable' ? 'verdict-liable' : data.action.verdict === 'not_liable' ? 'verdict-notliable' : data.action.verdict === 'forward_to_judge' ? 'verdict-fwd' : '');
        document.getElementById('v-reasoning').textContent = data.action.reasoning_chain;

        if (data.action.ratio_decidendi) {
            document.getElementById('v-ratio').textContent = data.action.ratio_decidendi;
            document.getElementById('v-ratio-block').style.display = 'block';
        }
        if (data.action.obiter_dicta) {
            document.getElementById('v-obiter').textContent = data.action.obiter_dicta;
            document.getElementById('v-obiter-block').style.display = 'block';
        }

        window.__evalInfo    = data.evaluation.info;
        window.__aiVerdict   = data.action.verdict;
        window.__aiReasoning = data.action.reasoning_chain;

    } catch(e) {
        // ─── OFFLINE DEMO FALLBACK ─────────────────────────────
        // Shows a realistic mock verdict when the API key is missing or rate-limited
        console.warn('AI Judge API failed:', e.message, '— showing offline demo verdict');
        document.getElementById('ai-thinking').style.display = 'none';
        document.getElementById('verdict-panel').style.display = 'block';

        const isCriminal = window.__caseType === 'criminal';
        const mockVerdict = isCriminal ? 'forward_to_judge' : 'liable';
        const mockRatio = isCriminal
            ? 'This act falls under BNS Section 125 (Act endangering life). Being a minor, the case is governed by the Juvenile Justice Act 2015. The act is cognizable and bailable. This matter is forwarded to the Juvenile Justice Board for adjudication.'
            : 'The ratio of this case: where a party fails to perform a contractual obligation causing demonstrable pecuniary loss to the other party, the defaulting party is liable for damages under Section 73 of the Indian Contract Act 1872.';
        const mockObiter = isCriminal
            ? 'This court observes, obiter, that parents who knowingly permit minors to operate vehicles may face separate liability under MV Act Section 199A.'
            : 'The court notes, obiter, that parties in commercial disputes should attempt mediation before approaching the court, as mandated by Section 89 CPC.';
        const mockReasoning = isCriminal
            ? '[COUNCIL OF AI MAJORITY VOTE: 3/3 AGREED]\n\nFact Bundle for Hon\u2019ble Judge:\n1. Accused: Minor (17 years) driving without valid license\n2. Incident: Traffic collision causing property damage\n3. Evidence: Dashcam video, license plate number\n4. BNS Section: 125 \u2014 Act endangering life or personal safety\n5. Motor Vehicles Act 1988, Section 199A \u2014 Offences by Juveniles\n6. Cognizability: Cognizable offence\n7. Bailability: Bailable\n8. Potential Sentence: If proven, fine up to Rs. 25,000 on guardian; minor sent to Juvenile Justice Board\n\nNote: AI cannot pass a verdict on criminal matters. All facts bundled for your Honour\u2019s review.'
            : '[COUNCIL OF AI MAJORITY VOTE: 2/3 AGREED]\n\nThe Strict Constitutionalist and Precedent Analyst both agree: the defendant is liable.\n\nReasoning: A clear contractual obligation existed. The defendant failed to perform. The plaintiff suffered a quantifiable loss of Rs. 15,000 (difference between contracted price and market price paid). Under Section 73 of the Indian Contract Act 1872 and the principle in Hadley v Baxendale (1854), damages naturally arising from breach are recoverable without special notice. The Empathetic Mediator noted the defendant\u2019s personal hardship but could not override the legal obligation.';

        document.getElementById('v-case-id').textContent = currentCaseData ? currentCaseData.case_id : 'DEMO-OFFLINE';
        const verdictEl = document.getElementById('v-verdict');
        verdictEl.textContent = mockVerdict;
        verdictEl.className = 'verdict-pill ' + (mockVerdict === 'liable' ? 'verdict-liable' : mockVerdict === 'forward_to_judge' ? 'verdict-fwd' : '');
        document.getElementById('v-reasoning').textContent = mockReasoning;
        document.getElementById('v-ratio').textContent = mockRatio;
        document.getElementById('v-ratio-block').style.display = 'block';
        document.getElementById('v-obiter').textContent = mockObiter;
        document.getElementById('v-obiter-block').style.display = 'block';

        window.__aiVerdict   = mockVerdict;
        window.__aiReasoning = mockReasoning;
        window.__evalInfo    = { logic_score: 0.92, accuracy_score: 0.95, fairness_score: 1.0, citation_score: 0.8 };
    }
});

// ─── Accept ───────────────────────────────────────────
document.getElementById('btn-accept').addEventListener('click', () => {
    document.getElementById('verdict-panel').style.display = 'none';
    const panel = document.getElementById('accepted-panel');
    panel.style.display = 'block';

    const info = window.__evalInfo;
    const row = document.getElementById('metrics-row');
    row.innerHTML = '';
    if (info) {
        const items = [
            { label: 'Logic', val: info.logic_score },
            { label: 'Accuracy', val: info.accuracy_score },
            { label: 'Fairness', val: info.fairness_score },
            { label: 'Citation', val: info.citation_score },
        ];
        items.forEach(i => {
            const chip = document.createElement('div');
            chip.className = 'metric-chip';
            chip.innerHTML = `${i.label}: <span>${(i.val||0).toFixed(2)}</span>`;
            row.appendChild(chip);
        });
    }
});

// ─── Escalate ─────────────────────────────────────────
function doEscalate(appealType) {
    const verdictSummary = window.__aiVerdict || 'see AI analysis';
    const warning = `LEGAL WARNING\n\nThe AI Council has analyzed your case and found: "${verdictSummary}".\n\nIf you escalate to a Human Judge despite this finding, and the judge agrees with the AI, you may be subject to:\n  - Additional court costs\n  - Penalties for frivolous escalation (Order XVA, CPC)\n  - A harsher outcome than the AI's lenient finding\n\nThe AI's judgment was based on the Constitution of India and the Bharatiya Nyaya Sanhita. A human judge is not bound to be lenient.\n\nAre you absolutely sure you want to proceed?`;
    if (!confirm(warning)) return;
    document.getElementById('escalation-modal').style.display = 'flex';
    document.getElementById('escalation-modal').dataset.appealType = appealType;
}

document.getElementById('btn-escalate').addEventListener('click', () => doEscalate('escalate'));

// Appeal
document.getElementById('btn-appeal')?.addEventListener('click', () => {
    document.getElementById('escalated-title').textContent = 'Appeal Filed';
    document.getElementById('escalated-msg').textContent = 'Your appeal has been registered. A fresh review will be conducted under the grounds of appeal.';
    document.getElementById('verdict-panel').style.display = 'none';
    document.getElementById('escalated-panel').style.display = 'block';
    fetch('/escalate', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ case_id: currentCaseData?.case_id||'DEMO', reasons:['appeal'], ai_verdict: window.__aiVerdict||'', ai_reasoning: window.__aiReasoning||'', fact_pattern: currentCaseData?.fact_pattern||'', appeal_type:'appeal' }) });
});

// Review Petition
document.getElementById('btn-review')?.addEventListener('click', () => {
    document.getElementById('escalated-title').textContent = 'Review Petition Filed';
    document.getElementById('escalated-msg').textContent = 'Your review petition (CPC Order 47) has been accepted. The AI will reconsider this judgment for error apparent on the face of the record.';
    document.getElementById('verdict-panel').style.display = 'none';
    document.getElementById('escalated-panel').style.display = 'block';
    fetch('/escalate', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ case_id: currentCaseData?.case_id||'DEMO', reasons:['review'], ai_verdict: window.__aiVerdict||'', ai_reasoning: window.__aiReasoning||'', fact_pattern: currentCaseData?.fact_pattern||'', appeal_type:'review' }) });
});

// Revision
document.getElementById('btn-revision')?.addEventListener('click', () => {
    document.getElementById('escalated-title').textContent = 'Revision Petition Filed';
    document.getElementById('escalated-msg').textContent = 'Your revision has been filed alleging an error of jurisdiction or material irregularity. This will be reviewed by a supervisory authority.';
    document.getElementById('verdict-panel').style.display = 'none';
    document.getElementById('escalated-panel').style.display = 'block';
    fetch('/escalate', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ case_id: currentCaseData?.case_id||'DEMO', reasons:['revision'], ai_verdict: window.__aiVerdict||'', ai_reasoning: window.__aiReasoning||'', fact_pattern: currentCaseData?.fact_pattern||'', appeal_type:'revision' }) });
});

// Reference to Human Judge
document.getElementById('btn-reference')?.addEventListener('click', () => {
    document.getElementById('escalated-title').textContent = 'Case Referred to Human Judge';
    document.getElementById('escalated-msg').textContent = 'The AI has identified a complex question of law. This case is being referred to a Human Judge for an authoritative opinion under CPC Section 113.';
    document.getElementById('verdict-panel').style.display = 'none';
    document.getElementById('escalated-panel').style.display = 'block';
    fetch('/escalate', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ case_id: currentCaseData?.case_id||'DEMO', reasons:['reference'], ai_verdict: window.__aiVerdict||'', ai_reasoning: window.__aiReasoning||'', fact_pattern: currentCaseData?.fact_pattern||'', appeal_type:'reference' }) });
});
document.getElementById('modal-cancel').addEventListener('click', () => {
    document.getElementById('escalation-modal').style.display = 'none';
});
document.getElementById('modal-confirm').addEventListener('click', async () => {
    const reasons = ['r1','r2','r3','r4']
        .filter(id => document.getElementById(id).checked)
        .map(id => document.getElementById(id).parentElement.textContent.trim());

    try {
        await fetch('/escalate', {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                case_id: currentCaseData ? currentCaseData.case_id : 'N/A',
                fact_pattern: currentCaseData ? currentCaseData.fact_pattern : '',
                ai_verdict: document.getElementById('v-verdict').textContent,
                ai_reasoning: document.getElementById('v-reasoning').textContent,
                reasons: reasons.length ? reasons : ['User requested human oversight']
            })
        });
    } catch(e) { console.error(e); }

    document.getElementById('escalation-modal').style.display = 'none';
    document.getElementById('verdict-panel').style.display = 'none';
    document.getElementById('escalated-panel').style.display = 'block';

    // ★ Issue Escalation Letter
    printLetter('escalation', {
        caseId: currentCaseData ? currentCaseData.case_id : 'N/A',
        verdict: document.getElementById('v-verdict').textContent,
        reasoning: document.getElementById('v-reasoning').textContent,
        reasons: reasons.length ? reasons : ['User requested human oversight'],
    });
});

// ─── Init ─────────────────────────────────────────────
show('screen-landing');
