// app.js - Catto-Lingo Frontend Logic

const API_URL = 'http://127.0.0.1:8001';

// State Management
let currentUser = null;
let authToken = null;

// Initialize App
document.addEventListener('DOMContentLoaded', () => {
    // Check if user is logged in
    const storedToken = localStorage.getItem('authToken');
    const storedUser = localStorage.getItem('currentUser');
    
    if (storedToken && storedUser) {
        authToken = storedToken;
        currentUser = JSON.parse(storedUser);
        showDashboard();
    } else {
        showPage('login');
    }
});

// Page Navigation
function showPage(pageName) {
    // Hide all pages
    document.getElementById('loginPage').classList.add('hidden');
    document.getElementById('registerPage').classList.add('hidden');
    document.getElementById('dashboardPage').classList.add('hidden');
    
    // Show selected page
    document.getElementById(`${pageName}Page`).classList.remove('hidden');
    
    // Update nav menu visibility
    if (pageName === 'dashboard') {
        document.getElementById('navMenu').classList.add('hidden');
        document.getElementById('userMenu').classList.remove('hidden');
        document.getElementById('userName').textContent = currentUser?.username || '';
    } else {
        document.getElementById('navMenu').classList.remove('hidden');
        document.getElementById('userMenu').classList.add('hidden');
    }
}

// Authentication Functions
async function handleLogin(event) {
    event.preventDefault();
    
    const username = document.getElementById('loginUsername').value;
    const password = document.getElementById('loginPassword').value;
    
    showLoading(true);
    
    try {
        // Create form data for OAuth2PasswordRequestForm
        const formData = new URLSearchParams();
        formData.append('username', username);
        formData.append('password', password);
        
        const response = await fetch(`${API_URL}/auth/login`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/x-www-form-urlencoded',
            },
            body: formData
        });
        
        const data = await response.json();
        
        if (response.ok) {
            authToken = data.access_token;
            localStorage.setItem('authToken', authToken);
            
            // Get user info
            const userResponse = await fetch(`${API_URL}/auth/me`, {
                headers: {
                    'Authorization': `Bearer ${authToken}`
                }
            });
            
            currentUser = await userResponse.json();
            localStorage.setItem('currentUser', JSON.stringify(currentUser));
            
            console.log('User logged in:', currentUser);
            console.log('User ID:', currentUser.user_id);
            
            showNotification('تم تسجيل الدخول بنجاح!', 'success');
            showDashboard();
        } else {
            showNotification(data.detail || 'خطأ في تسجيل الدخول', 'error');
        }
    } catch (error) {
        console.error('Login error:', error);
        showNotification('حدث خطأ في الاتصال', 'error');
    } finally {
        showLoading(false);
    }
}

async function handleRegister(event) {
    event.preventDefault();
    
    const username = document.getElementById('registerUsername').value;
    const email = document.getElementById('registerEmail').value;
    const password = document.getElementById('registerPassword').value;
    
    showLoading(true);
    
    try {
        const response = await fetch(`${API_URL}/auth/register`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ username, email, password })
        });
        
        const data = await response.json();
        
        if (response.ok) {
            showNotification('تم إنشاء الحساب بنجاح! يمكنك تسجيل الدخول الآن', 'success');
            showPage('login');
        } else {
            showNotification(data.detail || 'خطأ في التسجيل', 'error');
        }
    } catch (error) {
        console.error('Register error:', error);
        showNotification('حدث خطأ في الاتصال', 'error');
    } finally {
        showLoading(false);
    }
}

function logout() {
    localStorage.removeItem('authToken');
    localStorage.removeItem('currentUser');
    authToken = null;
    currentUser = null;
    showPage('login');
    showNotification('تم تسجيل الخروج بنجاح', 'success');
}

async function refreshUserData() {
    if (!authToken) return;
    
    try {
        const response = await fetch(`${API_URL}/auth/me`, {
            headers: {
                'Authorization': `Bearer ${authToken}`
            }
        });
        
        if (response.ok) {
            const userData = await response.json();
            currentUser = userData;
            localStorage.setItem('currentUser', JSON.stringify(userData));
            console.log('User data refreshed:', userData);
        } else {
            console.error('Failed to refresh user data');
        }
    } catch (error) {
        console.error('Error refreshing user data:', error);
    }
}

function showDashboard() {
    showPage('dashboard');
}

// Prediction Forms
function showPredictionForm(type) {
    const container = document.getElementById('predictionFormContainer');
    container.classList.remove('hidden');
    
    let formHTML = '';
    
    switch(type) {
        case 'text':
            formHTML = `
                <div class="bg-white rounded-xl shadow-lg p-8 fade-in">
                    <h3 class="text-2xl font-bold text-gray-800 mb-6">
                        <i class="fas fa-comment-dots text-blue-600 ml-2"></i>
                        تحليل النص
                    </h3>
                    <form onsubmit="handleTextPrediction(event)">
                        <textarea id="textInput" required
                                  class="w-full px-4 py-3 rounded-lg border border-gray-300 focus:border-purple-500 focus:ring-2 focus:ring-purple-200 transition"
                                  rows="5" placeholder="اكتب وصف لحالة قطتك هنا..."></textarea>
                        <div class="mt-4 flex space-x-4 space-x-reverse">
                            <button type="submit" class="flex-1 gradient-bg text-white py-3 rounded-lg font-bold hover:opacity-90 transition">
                                <i class="fas fa-search ml-2"></i> تحليل
                            </button>
                            <button type="button" onclick="hidePredictionForm()" 
                                    class="px-6 bg-gray-300 text-gray-700 py-3 rounded-lg font-bold hover:bg-gray-400 transition">
                                إلغاء
                            </button>
                        </div>
                    </form>
                </div>
            `;
            break;
            
        case 'audio':
            formHTML = `
                <div class="bg-white rounded-xl shadow-lg p-8 fade-in">
                    <h3 class="text-2xl font-bold text-gray-800 mb-6">
                        <i class="fas fa-microphone text-green-600 ml-2"></i>
                        تحليل الصوت
                    </h3>
                    <form onsubmit="handleAudioPrediction(event)">
                        <div class="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center">
                            <i class="fas fa-file-audio text-6xl text-gray-400 mb-4"></i>
                            <p class="text-gray-600 mb-4">اختر ملف صوتي (.wav, .mp3)</p>
                            <input type="file" id="audioFile" accept="audio/*" required
                                   class="w-full px-4 py-3 rounded-lg border border-gray-300">
                        </div>
                        <div class="mt-4 flex space-x-4 space-x-reverse">
                            <button type="submit" class="flex-1 gradient-bg text-white py-3 rounded-lg font-bold hover:opacity-90 transition">
                                <i class="fas fa-search ml-2"></i> تحليل
                            </button>
                            <button type="button" onclick="hidePredictionForm()" 
                                    class="px-6 bg-gray-300 text-gray-700 py-3 rounded-lg font-bold hover:bg-gray-400 transition">
                                إلغاء
                            </button>
                        </div>
                    </form>
                </div>
            `;
            break;
            
        case 'image':
            formHTML = `
                <div class="bg-white rounded-xl shadow-lg p-8 fade-in">
                    <h3 class="text-2xl font-bold text-gray-800 mb-6">
                        <i class="fas fa-image text-yellow-600 ml-2"></i>
                        تحليل الصورة
                    </h3>
                    <form onsubmit="handleImagePrediction(event)">
                        <div class="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center">
                            <i class="fas fa-file-image text-6xl text-gray-400 mb-4"></i>
                            <p class="text-gray-600 mb-4">اختر صورة قطتك (.jpg, .png)</p>
                            <input type="file" id="imageFile" accept="image/*" required
                                   class="w-full px-4 py-3 rounded-lg border border-gray-300">
                            <div id="imagePreview" class="mt-4"></div>
                        </div>
                        <div class="mt-4 flex space-x-4 space-x-reverse">
                            <button type="submit" class="flex-1 gradient-bg text-white py-3 rounded-lg font-bold hover:opacity-90 transition">
                                <i class="fas fa-search ml-2"></i> تحليل
                            </button>
                            <button type="button" onclick="hidePredictionForm()" 
                                    class="px-6 bg-gray-300 text-gray-700 py-3 rounded-lg font-bold hover:bg-gray-400 transition">
                                إلغاء
                            </button>
                        </div>
                    </form>
                </div>
            `;
            break;
            
        case 'video':
            formHTML = `
                <div class="bg-white rounded-xl shadow-lg p-8 fade-in">
                    <h3 class="text-2xl font-bold text-gray-800 mb-6">
                        <i class="fas fa-video text-red-600 ml-2"></i>
                        تحليل الفيديو
                    </h3>
                    <form onsubmit="handleVideoPrediction(event)">
                        <div class="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center">
                            <i class="fas fa-file-video text-6xl text-gray-400 mb-4"></i>
                            <p class="text-gray-600 mb-4">اختر فيديو قطتك (.mp4, .avi)</p>
                            <input type="file" id="videoFile" accept="video/*" required
                                   class="w-full px-4 py-3 rounded-lg border border-gray-300">
                        </div>
                        <div class="mt-4 flex space-x-4 space-x-reverse">
                            <button type="submit" class="flex-1 gradient-bg text-white py-3 rounded-lg font-bold hover:opacity-90 transition">
                                <i class="fas fa-search ml-2"></i> تحليل
                            </button>
                            <button type="button" onclick="hidePredictionForm()" 
                                    class="px-6 bg-gray-300 text-gray-700 py-3 rounded-lg font-bold hover:bg-gray-400 transition">
                                إلغاء
                            </button>
                        </div>
                    </form>
                </div>
            `;
            break;
    }
    
    container.innerHTML = formHTML;
    container.scrollIntoView({ behavior: 'smooth' });
}

function hidePredictionForm() {
    document.getElementById('predictionFormContainer').classList.add('hidden');
    document.getElementById('resultsContainer').classList.add('hidden');
}

// Prediction Handlers
async function handleTextPrediction(event) {
    event.preventDefault();
    
    // التحقق من تسجيل الدخول
    if (!currentUser || !currentUser.user_id) {
        showNotification('جاري تحديث بيانات المستخدم...', 'info');
        // إعادة تحميل بيانات المستخدم
        await refreshUserData();
        if (!currentUser || !currentUser.user_id) {
            showNotification('يرجى تسجيل الدخول مرة أخرى', 'error');
            logout();
            return;
        }
    }
    
    const text = document.getElementById('textInput').value;
    
    showLoading(true);
    
    try {
        const response = await fetch(`${API_URL}/predict/text`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${authToken}`
            },
            body: JSON.stringify({
                text: text,
                user_id: parseInt(currentUser.user_id)
            })
        });
        
        const data = await response.json();
        
        if (response.ok) {
            displayResults(data, 'text');
        } else {
            const errorMsg = typeof data.detail === 'string' ? data.detail : JSON.stringify(data.detail) || data.message || 'خطأ في التحليل';
            showNotification(errorMsg, 'error');
        }
    } catch (error) {
        console.error('Prediction error:', error);
        showNotification(`خطأ: ${error.message || 'حدث خطأ في الاتصال'}`, 'error');
    } finally {
        showLoading(false);
    }
}

async function handleAudioPrediction(event) {
    event.preventDefault();
    
    // التحقق من تسجيل الدخول
    if (!currentUser || !currentUser.user_id) {
        showNotification('جاري تحديث بيانات المستخدم...', 'info');
        await refreshUserData();
        if (!currentUser || !currentUser.user_id) {
            showNotification('يرجى تسجيل الدخول مرة أخرى', 'error');
            logout();
            return;
        }
    }
    
    const fileInput = document.getElementById('audioFile');
    const file = fileInput.files[0];
    
    if (!file) {
        showNotification('يرجى اختيار ملف صوتي', 'error');
        return;
    }
    
    showLoading(true);
    
    const formData = new FormData();
    formData.append('file', file);
    
    try {
        const response = await fetch(`${API_URL}/predict/audio?user_id=${parseInt(currentUser.user_id)}`, {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${authToken}`
            },
            body: formData
        });
        
        const data = await response.json();
        
        if (response.ok) {
            displayResults(data, 'audio');
        } else {
            const errorMsg = typeof data.detail === 'string' ? data.detail : JSON.stringify(data.detail) || data.message || 'خطأ في التحليل';
            showNotification(errorMsg, 'error');
        }
    } catch (error) {
        console.error('Prediction error:', error);
        showNotification(`خطأ: ${error.message || 'حدث خطأ في الاتصال'}`, 'error');
    } finally {
        showLoading(false);
    }
}

async function handleImagePrediction(event) {
    event.preventDefault();
    
    // التحقق من تسجيل الدخول
    if (!currentUser || !currentUser.user_id) {
        showNotification('جاري تحديث بيانات المستخدم...', 'info');
        await refreshUserData();
        if (!currentUser || !currentUser.user_id) {
            showNotification('يرجى تسجيل الدخول مرة أخرى', 'error');
            logout();
            return;
        }
    }
    
    const fileInput = document.getElementById('imageFile');
    const file = fileInput.files[0];
    
    if (!file) {
        showNotification('يرجى اختيار صورة', 'error');
        return;
    }
    
    showLoading(true);
    
    const formData = new FormData();
    formData.append('file', file);
    
    try {
        const response = await fetch(`${API_URL}/predict/image?user_id=${parseInt(currentUser.user_id)}`, {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${authToken}`
            },
            body: formData
        });
        
        const data = await response.json();
        
        if (response.ok) {
            displayResults(data, 'image');
        } else {
            const errorMsg = typeof data.detail === 'string' ? data.detail : JSON.stringify(data.detail) || data.message || 'خطأ في التحليل';
            showNotification(errorMsg, 'error');
        }
    } catch (error) {
        console.error('Prediction error:', error);
        showNotification(`خطأ: ${error.message || 'حدث خطأ في الاتصال'}`, 'error');
    } finally {
        showLoading(false);
    }
}

async function handleVideoPrediction(event) {
    event.preventDefault();
    
    // التحقق من تسجيل الدخول
    if (!currentUser || !currentUser.user_id) {
        showNotification('جاري تحديث بيانات المستخدم...', 'info');
        await refreshUserData();
        if (!currentUser || !currentUser.user_id) {
            showNotification('يرجى تسجيل الدخول مرة أخرى', 'error');
            logout();
            return;
        }
    }
    
    const fileInput = document.getElementById('videoFile');
    const file = fileInput.files[0];
    
    if (!file) {
        showNotification('يرجى اختيار فيديو', 'error');
        return;
    }
    
    showLoading(true);
    
    const formData = new FormData();
    formData.append('file', file);
    
    try {
        const response = await fetch(`${API_URL}/predict/video?user_id=${parseInt(currentUser.user_id)}`, {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${authToken}`
            },
            body: formData
        });
        
        const data = await response.json();
        
        if (response.ok) {
            displayResults(data, 'video');
        } else {
            const errorMsg = typeof data.detail === 'string' ? data.detail : JSON.stringify(data.detail) || data.message || 'خطأ في التحليل';
            showNotification(errorMsg, 'error');
        }
    } catch (error) {
        console.error('Prediction error:', error);
        showNotification(`خطأ: ${error.message || 'حدث خطأ في الاتصال'}`, 'error');
    } finally {
        showLoading(false);
    }
}

// Display Results
function displayResults(data, type) {
    const container = document.getElementById('resultsContainer');
    container.classList.remove('hidden');
    
    const emotionEmojis = {
        'happy': '😸',
        'normal': '😺',
        'angry': '😾',
        'sad': '😿',
        'scared': '🙀',
        'relaxed': '😌',
        'surprised': '😲',
        'uncomfortable': '😰'
    };
    
    const emotionColors = {
        'happy': 'bg-yellow-100 text-yellow-800',
        'normal': 'bg-green-100 text-green-800',
        'angry': 'bg-red-100 text-red-800',
        'sad': 'bg-blue-100 text-blue-800',
        'scared': 'bg-purple-100 text-purple-800',
        'relaxed': 'bg-teal-100 text-teal-800',
        'surprised': 'bg-pink-100 text-pink-800',
        'uncomfortable': 'bg-orange-100 text-orange-800'
    };
    
    const emotion = data.emotion || data.dominant_emotion || 'غير معروف';
    const confidence = data.confidence || data.overall_confidence || 0;
    const emoji = emotionEmojis[emotion.toLowerCase()] || '🐱';
    const colorClass = emotionColors[emotion.toLowerCase()] || 'bg-gray-100 text-gray-800';
    
    let resultHTML = `
        <div class="bg-white rounded-xl shadow-lg p-8 mt-8 fade-in">
            <h3 class="text-2xl font-bold text-gray-800 mb-6 text-center">
                <i class="fas fa-chart-line text-purple-600 ml-2"></i>
                نتائج التحليل
            </h3>
            
            <div class="text-center mb-6">
                <div class="text-8xl mb-4">${emoji}</div>
                <div class="inline-block px-6 py-3 rounded-full ${colorClass} font-bold text-2xl emotion-badge">
                    ${emotion}
                </div>
                <div class="mt-4">
                    <div class="text-gray-600 font-semibold mb-2">مستوى الثقة</div>
                    <div class="relative pt-1">
                        <div class="overflow-hidden h-4 mb-4 text-xs flex rounded-full bg-gray-200">
                            <div style="width:${(confidence * 100).toFixed(2)}%" 
                                 class="shadow-none flex flex-col text-center whitespace-nowrap text-white justify-center gradient-bg transition-all duration-500">
                            </div>
                        </div>
                        <span class="text-2xl font-bold text-purple-600">${(confidence * 100).toFixed(2)}%</span>
                    </div>
                </div>
            </div>
    `;
    
    // Add AI advice if available
    if (data.agent_response || data.advice) {
        const advice = data.agent_response || data.advice;
        resultHTML += `
            <div class="mt-6 bg-purple-50 border-r-4 border-purple-500 p-6 rounded-lg">
                <div class="flex items-start">
                    <i class="fas fa-robot text-3xl text-purple-600 ml-4 mt-1"></i>
                    <div class="flex-1">
                        <h4 class="font-bold text-gray-800 mb-2 text-lg">نصيحة من Catto-Lingo AI</h4>
                        <p class="text-gray-700 leading-relaxed whitespace-pre-line">${advice}</p>
                    </div>
                </div>
            </div>
        `;
    }
    
    resultHTML += `
            <div class="mt-6 text-center">
                <button onclick="hidePredictionForm()" 
                        class="gradient-bg text-white px-8 py-3 rounded-lg font-bold hover:opacity-90 transition">
                    <i class="fas fa-redo ml-2"></i> تحليل جديد
                </button>
            </div>
        </div>
    `;
    
    container.innerHTML = resultHTML;
    container.scrollIntoView({ behavior: 'smooth' });
}

// Utility Functions
function showLoading(show) {
    const overlay = document.getElementById('loadingOverlay');
    if (show) {
        overlay.classList.remove('hidden');
    } else {
        overlay.classList.add('hidden');
    }
}

function showNotification(message, type = 'info') {
    // Create notification element
    const notification = document.createElement('div');
    notification.className = `fixed top-4 left-1/2 transform -translate-x-1/2 px-6 py-4 rounded-lg shadow-lg z-50 fade-in`;
    
    if (type === 'success') {
        notification.className += ' bg-green-500 text-white';
        notification.innerHTML = `<i class="fas fa-check-circle ml-2"></i> ${message}`;
    } else if (type === 'error') {
        notification.className += ' bg-red-500 text-white';
        notification.innerHTML = `<i class="fas fa-exclamation-circle ml-2"></i> ${message}`;
    } else {
        notification.className += ' bg-blue-500 text-white';
        notification.innerHTML = `<i class="fas fa-info-circle ml-2"></i> ${message}`;
    }
    
    document.body.appendChild(notification);
    
    // Remove after 3 seconds
    setTimeout(() => {
        notification.remove();
    }, 3000);
}
