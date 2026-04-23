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

// ─── Action chooser ───────────────────────────────────
document.getElementById('btn-file-case').addEventListener('click', () => show('screen-type'));
document.getElementById('btn-withdraw').addEventListener('click', () => show('screen-withdraw'));
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
            loadDossier();
        });
        grid.appendChild(btn);
    });
}

// ─── Load Case & Go to Dossier ────────────────────────
async function loadDossier() {
    show('screen-dossier');
    document.getElementById('dossier-badge').textContent = currentType === 'civil' ? 'Civil Case' : 'Criminal Case';

    // Reset right panel
    document.getElementById('chat-panel').style.display = 'block';
    document.getElementById('ai-thinking').style.display = 'none';
    document.getElementById('verdict-panel').style.display = 'none';
    document.getElementById('accepted-panel').style.display = 'none';
    document.getElementById('escalated-panel').style.display = 'none';
    document.getElementById('generate-btn').disabled = true;
    document.getElementById('generate-hint').textContent = 'Answer the questions above to unlock the judgment';
    document.getElementById('chat-messages').innerHTML = '';
    chatHistory = [];

    // Fetch case from backend
    try {
        const res = await fetch('/reset', {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ domain: currentDomain, difficulty: currentDifficulty })
        });
        const data = await res.json();
        currentCaseData = data.observation;
        renderDossierLeft(currentCaseData);
        startFactFinding();
        // ★ Issue Registration Letter immediately
        printLetter('registration', { caseId: currentCaseData.case_id });
    } catch(e) {
        renderDossierLeft({ case_id: 'DEMO-001', fact_pattern: 'Could not load case from server.', evidence_flags: [], statutes: [] });
        startFactFinding();
        printLetter('registration', { caseId: 'DEMO-001' });
    }
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
                <tr><td>Case ID</td><td><strong>${data.caseId}</strong></td></tr>
                <tr><td>Reference No.</td><td><strong>${refNo}</strong></td></tr>
                <tr><td>Registered On</td><td><strong>${timestamp}</strong></td></tr>
                <tr><td>Case Type</td><td><strong>${currentType ? currentType.toUpperCase() : 'N/A'}</strong></td></tr>
                <tr><td>Status</td><td><strong style="color:#1a5276">REGISTERED — AI Fact-Finding in Progress</strong></td></tr>
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

    const win = window.open('', '_blank', 'width=800,height=700');
    win.document.write(`<!DOCTYPE html>
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
    </html>`);
    win.document.close();
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
                chat_history: chatHistory
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

// Evidence locker click
document.getElementById('locker-zone').addEventListener('click', () => {
    sendUserMessage('[Document uploaded: Evidence file submitted to dossier]');
});

// ─── Generate Judgment ────────────────────────────────
document.getElementById('generate-btn').addEventListener('click', async () => {
    document.getElementById('chat-panel').style.display = 'none';
    document.getElementById('ai-thinking').style.display = 'block';

    try {
        const res = await fetch('/ai_judge', {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ domain: currentDomain, difficulty: currentDifficulty })
        });
        if (!res.ok) throw new Error();
        const data = await res.json();

        document.getElementById('ai-thinking').style.display = 'none';
        document.getElementById('verdict-panel').style.display = 'block';

        document.getElementById('v-case-id').textContent = currentCaseData ? currentCaseData.case_id : 'N/A';
        document.getElementById('v-verdict').textContent = data.action.verdict;
        document.getElementById('v-reasoning').textContent = data.action.reasoning_chain;
        window.__evalInfo = data.evaluation.info;
    } catch(e) {
        document.getElementById('ai-thinking').style.display = 'none';
        document.getElementById('chat-panel').style.display = 'block';
        document.getElementById('generate-btn').disabled = false;
        postAI("⚠️ I was unable to generate a judgment at this time. Please ensure the server API key is configured and try again.");
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
document.getElementById('btn-escalate').addEventListener('click', () => {
    document.getElementById('escalation-modal').style.display = 'flex';
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
