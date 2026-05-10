const API_BASE_URL = 'http://localhost:8001';

function selectCategory(category) {
    // Update active card
    document.querySelectorAll('.feature-item').forEach(card => {
        card.classList.remove('active');
    });
    document.querySelector(`[data-category="${category}"]`).classList.add('active');

    // Show/Hide sections
    document.querySelectorAll('.content-section').forEach(section => {
        section.classList.remove('active');
    });
    document.getElementById(category).classList.add('active');

    // Reset results and buttons if needed
    const resultEl = document.getElementById(`${category}-result`);
    resultEl.classList.remove('show');
}

function handleFileSelect(category) {
    const input = document.getElementById(`${category}-upload`);
    const textEl = document.getElementById(`${category}-filename`);
    
    if (input.files && input.files[0]) {
        textEl.textContent = `Selected: ${input.files[0].name}`;
        textEl.style.color = 'var(--primary)';
        textEl.style.fontWeight = '600';
    }
}

async function analyze(category) {
    const input = document.getElementById(`${category}-upload`);
    const btn = document.getElementById(`btn-${category}`);
    const resultEl = document.getElementById(`${category}-result`);
    
    if (!input.files || !input.files[0]) {
        alert("Please select a file first.");
        return;
    }

    // Prepare UI state
    btn.classList.add('loading');
    btn.disabled = true;
    resultEl.classList.remove('show');

    const formData = new FormData();
    formData.append('file', input.files[0]);

    try {
        const response = await fetch(`${API_BASE_URL}/analyze/${category}`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const errorText = await response.text();
            let errorMessage = 'Analysis failed';
            try {
                const errorData = JSON.parse(errorText);
                errorMessage = errorData.detail || errorData.message || errorMessage;
            } catch (e) {
                errorMessage = errorText || errorMessage;
            }
            throw new Error(errorMessage);
        }

        const data = await response.json();
        displayResult(category, data);
    } catch (error) {
        console.error(error);
        
        // Display error in the result area
        resultEl.innerHTML = `
            <div style="background: #fff5f5; border-left: 5px solid #fc8181; padding: 25px; border-radius: 10px;">
                <div style="color: #c53030; font-weight: 700; font-size: 18px; margin-bottom: 10px;">
                    <i class="fas fa-exclamation-triangle" style="margin-right: 10px;"></i> Analysis Error
                </div>
                <p style="color: #742a2a; font-size: 15px; line-height: 1.5;">${error.message}</p>
                <button class="btn btn-outline" style="margin-top: 15px; padding: 8px 15px; font-size: 12px; color: #c53030; border-color: #feb2b2;" onclick="location.reload()">
                    Reset Dashboard
                </button>
            </div>
        `;
        resultEl.classList.add('show');
    } finally {
        btn.classList.remove('loading');
        btn.disabled = false;
    }
}

function displayResult(category, data) {
    const resultEl = document.getElementById(`${category}-result`);
    
    let metricsHtml = '';
    for (const [key, value] of Object.entries(data.metrics)) {
        metricsHtml += `
            <div style="margin-bottom: 0.8rem; display: flex; justify-content: space-between; border-bottom: 1px solid #eee; padding-bottom: 8px;">
                <span style="color: var(--text-light); font-size: 14px;">${key}</span>
                <span style="font-weight: 700; color: var(--secondary);">${value}</span>
            </div>
        `;
    }

    // Advanced Radiology Info
    let advancedHtml = '';
    if (category === 'radiology' && data.topMatches) {
        let matchesHtml = '';
        data.topMatches.forEach(match => {
            matchesHtml += `
                <div style="text-align: center; background: #f8f9fa; padding: 10px; border-radius: 8px;">
                    <img src="${API_BASE_URL}/dataset_images/${match.path}" style="width: 100%; height: 100px; object-fit: cover; border-radius: 4px; margin-bottom: 8px;">
                    <div style="font-size: 11px; font-weight: 600; color: var(--secondary); text-transform: uppercase;">${match.label}</div>
                </div>
            `;
        });

        let capsHtml = data.predictedCaptions.map(c => `
            <div style="font-size: 13px; padding: 10px 15px; background: #f0faf8; border-radius: 8px; margin-bottom: 8px; border-left: 3px solid var(--primary); color: var(--secondary);">
                <i class="fas fa-robot" style="margin-right: 10px; color: var(--primary);"></i> ${c}
            </div>
        `).join('');

        advancedHtml = `
            <div style="margin-top: 30px; display: grid; grid-template-columns: 1fr; gap: 20px;">
                <div>
                    <div style="font-size: 14px; font-weight: 700; color: var(--secondary); margin-bottom: 15px; display: flex; align-items: center; gap: 10px;">
                        <i class="fas fa-layer-group"></i> SIMILAR CASE RETRIEVAL (FAISS)
                    </div>
                    <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px;">${matchesHtml}</div>
                </div>
                <div>
                    <div style="font-size: 14px; font-weight: 700; color: var(--secondary); margin-bottom: 15px; display: flex; align-items: center; gap: 10px;">
                        <i class="fas fa-comment-medical"></i> AI GENERATED CAPTIONS
                    </div>
                    ${capsHtml}
                </div>
            </div>
        `;
    }

    resultEl.innerHTML = `
        <div class="result-header">
            <h3 style="font-weight: 700; font-size: 22px; color: var(--secondary);">Diagnostic Summary</h3>
            <span class="badge">${data.modality}</span>
        </div>
        
        <div style="background: #f8f9fa; padding: 20px; border-radius: 10px; margin-bottom: 25px;">
            <div style="font-size: 18px; color: var(--primary); font-weight: 700; margin-bottom: 10px;">
                <i class="fas fa-check-circle"></i> ${data.diagnosis}
            </div>
            <p style="font-size: 14px; color: var(--text-main); line-height: 1.6;">
                ${data.findings}
            </p>
        </div>
        
        <div style="margin-bottom: 20px;">
            <div style="font-size: 14px; font-weight: 700; color: var(--secondary); margin-bottom: 15px; display: flex; align-items: center; gap: 10px;">
                <i class="fas fa-chart-line"></i> PERFORMANCE METRICS
            </div>
            ${metricsHtml}
        </div>

        ${advancedHtml}
        
        <div style="margin-top: 30px; border-top: 1px solid #eee; padding-top: 20px; text-align: center;">
            <button class="btn btn-outline" style="color: var(--secondary); border-color: #ddd; font-size: 12px; padding: 10px 20px;" onclick="window.print()">
                <i class="fas fa-print" style="margin-right: 8px;"></i> Print Medical Report
            </button>
        </div>
    `;
    
    resultEl.classList.add('show');
}
