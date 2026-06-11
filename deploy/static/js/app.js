// ===== Global State =====
let currentUser = null;
let authToken = localStorage.getItem('token');
let featuresConfig = [];
let currentPrediction = null;
let priceChart = null;
let districtChart = null;
let roomsChart = null;
let authMode = 'login'; // 'login' or 'register'

// ===== API Client =====
const API_BASE = '';

async function api(endpoint, options = {}) {
    const url = `${API_BASE}${endpoint}`;
    const config = {
        headers: {
            'Content-Type': 'application/json',
            ...options.headers
        },
        ...options
    };

    if (authToken) {
        config.headers['Authorization'] = `Bearer ${authToken}`;
    }

    if (config.body && typeof config.body === 'object') {
        config.body = JSON.stringify(config.body);
    }

    try {
        const response = await fetch(url, config);
        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.detail || 'Произошла ошибка');
        }

        return data;
    } catch (error) {
        console.error('API Error:', error);
        throw error;
    }
}

// ===== Initialization =====
document.addEventListener('DOMContentLoaded', async () => {
    await loadFeatures();
    setupNavigation();
    setupForms();

    if (authToken) {
        await loadCurrentUser();
    }
});

async function loadFeatures() {
    try {
        const data = await api('/api/features');
        featuresConfig = data.features;
        renderFormFields();
        populateFilterDistricts();
    } catch (error) {
        showToast('Ошибка загрузки конфигурации', 'error');
    }
}

async function loadCurrentUser() {
    try {
        const user = await api('/api/auth/me');
        currentUser = user;
        updateUIForAuth();
    } catch (error) {
        // Токен недействителен
        authToken = null;
        localStorage.removeItem('token');
    }
}

// ===== UI Updates =====
function updateUIForAuth() {
    const navAuth = document.getElementById('navAuth');
    const navUser = document.getElementById('navUser');
    const username = document.getElementById('username');

    if (currentUser) {
        navAuth.classList.add('hidden');
        navUser.classList.remove('hidden');
        username.textContent = currentUser.username;
    } else {
        navAuth.classList.remove('hidden');
        navUser.classList.add('hidden');
    }
}

// ===== Navigation =====
function setupNavigation() {
    const navLinks = document.querySelectorAll('.nav-link');

    navLinks.forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            const section = link.dataset.section;
            showSection(section);

            // Update active state
            navLinks.forEach(l => l.classList.remove('active'));
            link.classList.add('active');
        });
    });
}

function showSection(sectionName) {
    const sections = document.querySelectorAll('.section');
    sections.forEach(s => s.classList.remove('active'));
    document.getElementById(sectionName).classList.add('active');

    // Load data for specific sections
    if (sectionName === 'history' && currentUser) {
        loadHistory();
    } else if (sectionName === 'stats' && currentUser) {
        loadStatistics();
    }
}

// ===== Form Rendering =====
function renderFormFields() {
    const container = document.getElementById('formFields');
    container.innerHTML = '';

    featuresConfig.forEach(feature => {
        const group = document.createElement('div');
        group.className = 'form-group';

        const label = document.createElement('label');
        label.textContent = feature.name + (feature.unit ? ` (${feature.unit})` : '');
        group.appendChild(label);

        let input;
        if (feature.type === 'select') {
            input = document.createElement('select');
            input.className = 'form-control';
            input.required = feature.required;

            const defaultOption = document.createElement('option');
            defaultOption.value = '';
            defaultOption.textContent = 'Выберите...';
            input.appendChild(defaultOption);

            feature.options.forEach(opt => {
                const option = document.createElement('option');
                option.value = opt;
                option.textContent = opt;
                input.appendChild(option);
            });
        } else {
            input = document.createElement('input');
            input.type = 'number';
            if (feature.type === 'float') {
                input.step = '0.01'  //feature.step;
            }
            input.className = 'form-control';
            input.required = feature.required;
            if (feature.min !== undefined) input.min = feature.min;
            if (feature.max !== undefined) input.max = feature.max;
            input.placeholder = feature.unit || '';
        }

        input.name = feature.key;
        input.id = `field_${feature.key}`;
        group.appendChild(input);
        container.appendChild(group);
    });
}

function populateFilterDistricts() {
    const select = document.getElementById('filterDistrict');
    const districtFeature = featuresConfig.find(f => f.key === 'district');

    if (districtFeature) {
        districtFeature.options.forEach(opt => {
            const option = document.createElement('option');
            option.value = opt;
            option.textContent = opt;
            select.appendChild(option);
        });
    }
}

// ===== Forms =====
function setupForms() {
    // Prediction form
    document.getElementById('predictionForm').addEventListener('submit', handlePrediction);

    // Auth form
    document.getElementById('authForm').addEventListener('submit', handleAuth);

    // Edit form
    document.getElementById('editForm').addEventListener('submit', handleEdit);
}

async function handlePrediction(e) {
    e.preventDefault();

    if (!currentUser) {
        showAuthModal('login');
        showToast('Для предсказания необходимо войти', 'info');
        return;
    }

    const formData = new FormData(e.target);
    const data = {};

    featuresConfig.forEach(feature => {
        let value = formData.get(feature.key);
        if (feature.type === 'number') {
            value = parseFloat(value);
        }
        data[feature.key] = value;
    });

    try {
        const result = await api('/api/predict', {
            method: 'POST',
            body: data
        });

        currentPrediction = result;
        displayResult(result);
        showToast('Предсказание успешно выполнено!', 'success');
    } catch (error) {
        showToast(error.message, 'error');
    }
}

function displayResult(result) {
    const resultCard = document.getElementById('resultCard');
    resultCard.classList.remove('hidden');

    // Format price
    const formatPrice = (price) => {
        return new Intl.NumberFormat('ru-RU', {
            style: 'currency',
            currency: 'RUB',
            maximumFractionDigits: 0
        }).format(price);
    };

    document.getElementById('predictedPrice').textContent = formatPrice(result.predicted_price);
    document.getElementById('priceRange').textContent =
        `${formatPrice(result.lower_bound)} - ${formatPrice(result.upper_bound)}`;
    document.getElementById('confidence').textContent = result.confidence;
    document.getElementById('errorMargin').textContent = result.error_margin;

    // Render chart
    renderPriceChart(result);
}

function renderPriceChart(result) {
    const ctx = document.getElementById('priceChart').getContext('2d');

    if (priceChart) {
        priceChart.destroy();
    }

    priceChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['Минимум', 'Прогноз', 'Максимум'],
            datasets: [{
                label: 'Цена (₽)',
                data: [result.lower_bound, result.predicted_price, result.upper_bound],
                backgroundColor: [
                    'rgba(99, 102, 241, 0.5)',
                    'rgba(99, 102, 241, 0.8)',
                    'rgba(99, 102, 241, 0.5)'
                ],
                borderColor: [
                    'rgba(99, 102, 241, 1)',
                    'rgba(99, 102, 241, 1)',
                    'rgba(99, 102, 241, 1)'
                ],
                borderWidth: 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    ticks: {
                        callback: function(value) {
                            return (value / 1000000).toFixed(1) + ' млн';
                        }
                    }
                }
            }
        }
    });
}

// ===== Authentication =====
function showAuthModal(mode) {
    authMode = mode;
    const modal = document.getElementById('authModal');
    const title = document.getElementById('authTitle');
    const emailGroup = document.getElementById('emailGroup');
    const submitText = document.getElementById('authSubmitText');
    const switchText = document.getElementById('authSwitchText');
    const switchBtn = document.getElementById('authSwitchBtn');
    const error = document.getElementById('authError');

    error.classList.add('hidden');
    modal.classList.remove('hidden');

    if (mode === 'login') {
        title.textContent = 'Вход';
        emailGroup.style.display = 'none';
        submitText.textContent = 'Войти';
        switchText.textContent = 'Нет аккаунта?';
        switchBtn.textContent = 'Зарегистрироваться';
    } else {
        title.textContent = 'Регистрация';
        emailGroup.style.display = 'block';
        submitText.textContent = 'Зарегистрироваться';
        switchText.textContent = 'Уже есть аккаунт?';
        switchBtn.textContent = 'Войти';
    }
}

function closeAuthModal() {
    document.getElementById('authModal').classList.add('hidden');
    document.getElementById('authForm').reset();
}

function switchAuthMode() {
    showAuthModal(authMode === 'login' ? 'register' : 'login');
}

async function handleAuth(e) {
    e.preventDefault();

    const username = document.getElementById('authUsername').value;
    const password = document.getElementById('authPassword').value;
    const errorDiv = document.getElementById('authError');

    try {
        if (authMode === 'login') {
            const formData = new URLSearchParams();
            formData.append('username', username);
            formData.append('password', password);

            const response = await fetch('/api/auth/login', {
                method: 'POST',
                headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
                body: formData
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.detail || 'Ошибка входа');
            }

            authToken = data.access_token;
            localStorage.setItem('token', authToken);
            await loadCurrentUser();
            closeAuthModal();
            showToast('Добро пожаловать!', 'success');
        } else {
            const email = document.getElementById('authEmail').value;

            await api('/api/auth/register', {
                method: 'POST',
                body: { username, email, password }
            });

            showToast('Регистрация успешна! Теперь войдите.', 'success');
            showAuthModal('login');
        }
    } catch (error) {
        errorDiv.textContent = error.message;
        errorDiv.classList.remove('hidden');
    }
}

async function logout() {
    currentUser = null;
    authToken = null;
    localStorage.removeItem('token');
    updateUIForAuth();
    showToast('Вы вышли из системы', 'info');
}

// ===== History =====
async function loadHistory() {
    if (!currentUser) return;

    try {
        const params = new URLSearchParams();

        const dateFrom = document.getElementById('filterDateFrom').value;
        const dateTo = document.getElementById('filterDateTo').value;
        const district = document.getElementById('filterDistrict').value;
        const rooms = document.getElementById('filterRooms').value;
        const minPrice = document.getElementById('filterMinPrice').value;
        const maxPrice = document.getElementById('filterMaxPrice').value;

        if (dateFrom) params.append('date_from', new Date(dateFrom).toISOString());
        if (dateTo) params.append('date_to', new Date(dateTo).toISOString());
        if (district) params.append('district', district);
        if (rooms) params.append('rooms', rooms);
        if (minPrice) params.append('min_price', minPrice);
        if (maxPrice) params.append('max_price', maxPrice);

        const predictions = await api(`/api/predictions?${params}`);
        renderHistoryTable(predictions);
    } catch (error) {
        showToast('Ошибка загрузки истории', 'error');
    }
}

function renderHistoryTable(predictions) {
    const tbody = document.getElementById('historyTableBody');
    const emptyState = document.getElementById('emptyState');
    const table = document.getElementById('historyTable');

    if (predictions.length === 0) {
        tbody.innerHTML = '';
        emptyState.classList.remove('hidden');
        table.classList.add('hidden');
        return;
    }

    emptyState.classList.add('hidden');
    table.classList.remove('hidden');

    const formatPrice = (price) => {
        return new Intl.NumberFormat('ru-RU', {
            style: 'currency',
            currency: 'RUB',
            maximumFractionDigits: 0
        }).format(price);
    };

    const formatDate = (dateStr) => {
        return new Date(dateStr).toLocaleString('ru-RU');
    };

    tbody.innerHTML = predictions.map(p => `
        <tr>
            <td>${formatDate(p.created_at)}</td>
            <td>${p.district || '-'}</td>
            <td>${p.area || '-'} м²</td>
            <td>${p.rooms || '-'}</td>
            <td><strong>${formatPrice(p.predicted_price)}</strong></td>
            <td>${formatPrice(p.lower_bound)} - ${formatPrice(p.upper_bound)}</td>
            <td class="actions">
                <button class="btn btn-sm btn-outline" onclick="showEditModal(${p.id}, '${encodeURIComponent(JSON.stringify(p.input_data))}')">
                    <i class="fas fa-edit"></i>
                </button>
                <button class="btn btn-sm btn-danger" onclick="deletePrediction(${p.id})">
                    <i class="fas fa-trash"></i>
                </button>
            </td>
        </tr>
    `).join('');
}

function applyFilters() {
    loadHistory();
}

function resetFilters() {
    document.getElementById('filterDateFrom').value = '';
    document.getElementById('filterDateTo').value = '';
    document.getElementById('filterDistrict').value = '';
    document.getElementById('filterRooms').value = '';
    document.getElementById('filterMinPrice').value = '';
    document.getElementById('filterMaxPrice').value = '';
    loadHistory();
}

// ===== Edit =====
function showEditModal(id, inputDataEncoded) {
    const inputData = JSON.parse(decodeURIComponent(inputDataEncoded));

    document.getElementById('editId').value = id;

    // Render edit form fields
    const container = document.getElementById('editFormFields');
    container.innerHTML = '';

    featuresConfig.forEach(feature => {
        const group = document.createElement('div');
        group.className = 'form-group';

        const label = document.createElement('label');
        label.textContent = feature.name + (feature.unit ? ` (${feature.unit})` : '');
        group.appendChild(label);

        let input;
        if (feature.type === 'select') {
            input = document.createElement('select');
            input.className = 'form-control';

            feature.options.forEach(opt => {
                const option = document.createElement('option');
                option.value = opt;
                option.textContent = opt;
                if (inputData[feature.key] === opt) option.selected = true;
                input.appendChild(option);
            });
        } else {
            input = document.createElement('input');
            input.type = 'number';
            if (feature.type === 'float') {
                input.step = '0.01'  //feature.step;
            }
            input.className = 'form-control';
            input.value = inputData[feature.key] || '';
            if (feature.min !== undefined) input.min = feature.min;
            if (feature.max !== undefined) input.max = feature.max;
        }

        input.name = feature.key;
        group.appendChild(input);
        container.appendChild(group);
    });

    document.getElementById('editModal').classList.remove('hidden');
}

function closeEditModal() {
    document.getElementById('editModal').classList.add('hidden');
}

async function handleEdit(e) {
    e.preventDefault();

    const id = document.getElementById('editId').value;
    const formData = new FormData(e.target);
    const data = {};

    featuresConfig.forEach(feature => {
        let value = formData.get(feature.key);
        if (feature.type === 'number') {
            value = parseFloat(value);
        }
        data[feature.key] = value;
    });

    try {
        await api(`/api/predictions/${id}`, {
            method: 'PUT',
            body: { input_data: data }
        });

        closeEditModal();
        loadHistory();
        showToast('Предсказание обновлено!', 'success');
    } catch (error) {
        showToast(error.message, 'error');
    }
}

async function deletePrediction(id) {
    if (!confirm('Вы уверены, что хотите удалить это предсказание?')) {
        return;
    }

    try {
        await api(`/api/predictions/${id}`, {
            method: 'DELETE'
        });

        loadHistory();
        showToast('Предсказание удалено', 'success');
    } catch (error) {
        showToast(error.message, 'error');
    }
}

// ===== Statistics =====
async function loadStatistics() {
    if (!currentUser) return;

    try {
        const stats = await api('/api/statistics');
        renderStatistics(stats);
    } catch (error) {
        showToast('Ошибка загрузки статистики', 'error');
    }
}

function renderStatistics(stats) {
    const formatPrice = (price) => {
        if (price >= 1000000) {
            return (price / 1000000).toFixed(1) + ' млн ₽';
        }
        return new Intl.NumberFormat('ru-RU', {
            style: 'currency',
            currency: 'RUB',
            maximumFractionDigits: 0
        }).format(price);
    };

    document.getElementById('statTotal').textContent = stats.total_predictions;
    document.getElementById('statAvg').textContent = formatPrice(stats.avg_price);
    document.getElementById('statMin').textContent = formatPrice(stats.min_price);
    document.getElementById('statMax').textContent = formatPrice(stats.max_price);

    // District chart
    const districtCtx = document.getElementById('districtChart').getContext('2d');
    if (districtChart) districtChart.destroy();

    const districtLabels = Object.keys(stats.by_district);
    const districtData = districtLabels.map(d => stats.by_district[d].avg_price);

    districtChart = new Chart(districtCtx, {
        type: 'bar',
        data: {
            labels: districtLabels,
            datasets: [{
                label: 'Средняя цена',
                data: districtData,
                backgroundColor: 'rgba(99, 102, 241, 0.7)',
                borderColor: 'rgba(99, 102, 241, 1)',
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                y: {
                    ticks: {
                        callback: (v) => (v / 1000000).toFixed(1) + ' млн'
                    }
                }
            }
        }
    });

    // Rooms chart
    const roomsCtx = document.getElementById('roomsChart').getContext('2d');
    if (roomsChart) roomsChart.destroy();

    const roomsLabels = Object.keys(stats.by_rooms).map(r => r + ' комн.');
    const roomsData = Object.keys(stats.by_rooms).map(r => stats.by_rooms[r].avg_price);

    roomsChart = new Chart(roomsCtx, {
        type: 'doughnut',
        data: {
            labels: roomsLabels,
            datasets: [{
                data: roomsData,
                backgroundColor: [
                    'rgba(99, 102, 241, 0.7)',
                    'rgba(16, 185, 129, 0.7)',
                    'rgba(245, 158, 11, 0.7)',
                    'rgba(239, 68, 68, 0.7)',
                    'rgba(59, 130, 246, 0.7)',
                    'rgba(139, 92, 246, 0.7)'
                ]
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                tooltip: {
                    callbacks: {
                        label: (context) => {
                            const value = context.raw;
                            return new Intl.NumberFormat('ru-RU', {
                                style: 'currency',
                                currency: 'RUB',
                                maximumFractionDigits: 0
                            }).format(value);
                        }
                    }
                }
            }
        }
    });
}

// ===== Toast Notifications =====
function showToast(message, type = 'info') {
    const container = document.getElementById('toastContainer');
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;

    const icon = type === 'success' ? 'check-circle' :
                 type === 'error' ? 'exclamation-circle' : 'info-circle';

    toast.innerHTML = `<i class="fas fa-${icon}"></i> ${message}`;
    container.appendChild(toast);

    setTimeout(() => {
        toast.style.opacity = '0';
        toast.style.transform = 'translateX(100%)';
        setTimeout(() => toast.remove(), 300);
    }, 3000);
}