let uploadedVideos = {}; // {index: filename}
let activeStreams = new Set();
let queryImages = [];
let statusInterval = null;
let paramUpdateTimer = null;

const panelCount = 4;

function initVideoPanels() {
    const grid = document.getElementById('videoGrid');
    const template = document.getElementById('videoPanelTemplate').content;
    for (let i = 0; i < panelCount; i++) {
        const clone = template.cloneNode(true);
        clone.querySelector('.panel-title').textContent = `通道 ${i + 1}`;
        clone.querySelector('.panel-status').id = `panelStatus-${i}`;
        clone.querySelector('button[title="清除"]').setAttribute('onclick', `clearVideoSlot(${i})`);
        clone.querySelector(`#placeholder-EVENT_SLOT`).id = `placeholder-${i}`;
        clone.querySelector(`#videoFrame-EVENT_SLOT`).id = `videoFrame-${i}`;
        grid.appendChild(clone);
    }
}

document.addEventListener('DOMContentLoaded', () => {
    initVideoPanels();
    fetchStatus();
    statusInterval = setInterval(fetchStatus, 3000);
    bindParamControls();
});

function bindParamControls() {
    ['confThres', 'nmsThres', 'distThres', 'boxSize', 'trailLen'].forEach(id => {
        const el = document.getElementById(id);
        if (el) {
            el.addEventListener('input', scheduleParamUpdate);
        }
    });
    const showAllToggle = document.getElementById('showAll');
    if (showAllToggle) {
        showAllToggle.addEventListener('change', scheduleParamUpdate);
    }
}

function getCurrentParams() {
    return {
        conf: parseFloat(document.getElementById('confThres').value),
        iou: parseFloat(document.getElementById('nmsThres').value),
        threshold: parseFloat(document.getElementById('distThres').value),
        min_box_size: parseInt(document.getElementById('boxSize').value, 10),
        trajectory_length: parseInt(document.getElementById('trailLen').value, 10),
        show_all: document.getElementById('showAll').checked
    };
}

function scheduleParamUpdate() {
    if (paramUpdateTimer) {
        clearTimeout(paramUpdateTimer);
    }
    paramUpdateTimer = setTimeout(pushParamUpdate, 400);
}

async function pushParamUpdate() {
    paramUpdateTimer = null;
    const params = getCurrentParams();
    try {
        await fetch('/api/update_params', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(params)
        });
    } catch (err) {
        console.error('参数更新失败', err);
    }
}

async function fetchStatus() {
    try {
        const res = await fetch('/api/status');
        const data = await res.json();
        updateModelIndicator(data.models_loaded, data.models_loading);
        updateQueryUI(data.query_images || []);
        updateVideoStateFromStatus(data.video_details || []);
    } catch (err) {
        console.error('状态刷新失败', err);
    }
}

function updateModelIndicator(loaded, loading) {
    const indicator = document.getElementById('modelIndicator');
    const btn = document.getElementById('initBtn');
    if (loaded) {
        indicator.textContent = '模型已就绪';
        indicator.className = 'status-badge status-live';
        btn.disabled = true;
        btn.classList.add('opacity-60', 'cursor-not-allowed');
    } else if (loading) {
        indicator.textContent = '模型加载中…';
        indicator.className = 'status-badge status-idle';
        btn.disabled = true;
        btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin mr-2"></i>加载中';
    } else {
        indicator.textContent = '模型未加载';
        indicator.className = 'status-badge status-idle';
        btn.disabled = false;
        btn.innerHTML = '初始化模型';
    }
}

function updateQueryUI(images) {
    queryImages = images;
    document.getElementById('queryCount').textContent = images.length;
    const container = document.getElementById('queryList');
    container.innerHTML = '';
    images.forEach(img => {
        const card = document.createElement('div');
        card.className = 'relative group border border-slate-600 rounded overflow-hidden';
        card.innerHTML = `
            <img src="${img.thumbnail}" class="w-full h-24 object-cover" alt="query">
            <button class="absolute top-1 right-1 text-xs bg-black/60 px-2 py-0.5 rounded hidden group-hover:block" onclick="deleteQuery('${img.id}')">删除</button>
        `;
        container.appendChild(card);
    });
}

function updateVideoStateFromStatus(videoDetails) {
    const latestIndices = new Set();
    videoDetails.forEach(item => {
        latestIndices.add(item.index);
        uploadedVideos[item.index] = item.filename;
        showVideoReady(item.index, item.filename, item.active);
    });

    Object.keys(uploadedVideos).forEach(idx => {
        if (!latestIndices.has(Number(idx))) {
            resetVideoSlot(Number(idx));
            delete uploadedVideos[idx];
        }
    });

    updateVideoList();
}

async function initializeModels() {
    document.getElementById('initBtn').innerHTML = '<i class="fa-solid fa-spinner fa-spin mr-2"></i>请求中';
    const res = await fetch('/api/initialize_models', { method: 'POST' });
    const data = await res.json();
    alert(data.message);
    fetchStatus();
}

async function uploadQuery(event) {
    const file = event.target.files[0];
    if (!file) return;
    const form = new FormData();
    form.append('query_image', file);
    form.append('width', document.getElementById('resizeWidth').value);
    form.append('height', document.getElementById('resizeHeight').value);
    try {
        const res = await fetch('/api/upload_query', { method: 'POST', body: form });
        const data = await res.json();
        if (!data.success) throw new Error(data.message);
        updateQueryUI(data.query_images);
    } catch (err) {
        alert('上传失败: ' + err.message);
    }
    event.target.value = '';
}

async function loadDefaultQueries() {
    try {
        const res = await fetch('/api/load_default_queries');
        const data = await res.json();
        if (!data.success) throw new Error(data.message);
        updateQueryUI(data.query_images);
    } catch (err) {
        alert('加载失败: ' + err.message);
    }
}

async function deleteQuery(id) {
    if (!confirm('删除该查询图片？')) return;
    const res = await fetch(`/api/delete_query/${id}`, { method: 'DELETE' });
    const data = await res.json();
    if (!data.success) {
        alert(data.message);
        return;
    }
    updateQueryUI(data.query_images);
}

async function uploadVideo(event) {
    const file = event.target.files[0];
    if (!file) return;
    const slot = findAvailableSlot();
    if (slot === -1) {
        alert('通道已满，请先删除一个视频');
        return;
    }
    const form = new FormData();
    form.append('video', file);
    form.append('video_index', slot);
    try {
        const res = await fetch('/api/upload_video', { method: 'POST', body: form });
        const data = await res.json();
        if (!data.success) throw new Error(data.message);
        uploadedVideos[slot] = data.filename;
        showVideoReady(slot, data.filename, false);
        updateVideoList();
    } catch (err) {
        alert('上传失败: ' + err.message);
    }
    event.target.value = '';
}

function findAvailableSlot() {
    for (let i = 0; i < panelCount; i++) {
        if (!(i in uploadedVideos)) return i;
    }
    return -1;
}

function showVideoReady(index, filename, active) {
    const placeholder = document.getElementById(`placeholder-${index}`);
    const frame = document.getElementById(`videoFrame-${index}`);
    const status = document.getElementById(`panelStatus-${index}`);
    placeholder.style.display = 'none';
    frame.style.display = 'block';
    if (active) {
        if (!activeStreams.has(index) || !frame.src) {
            frame.src = `/api/video_feed/${index}?t=${Date.now()}`;
        }
        status.textContent = '检测中';
        status.className = 'status-badge status-live';
        activeStreams.add(index);
    } else {
        frame.src = '';
        status.textContent = '已上传';
        status.className = 'status-badge status-idle';
        activeStreams.delete(index);
    }
}

function resetVideoSlot(index) {
    const placeholder = document.getElementById(`placeholder-${index}`);
    const frame = document.getElementById(`videoFrame-${index}`);
    const status = document.getElementById(`panelStatus-${index}`);
    placeholder.style.display = 'flex';
    frame.style.display = 'none';
    frame.src = '';
    status.textContent = '等待';
    status.className = 'status-badge status-idle';
    activeStreams.delete(index);
}

function updateVideoList() {
    const list = document.getElementById('videoList');
    list.innerHTML = '';
    Object.entries(uploadedVideos).forEach(([idx, name]) => {
        const row = document.createElement('div');
        row.className = 'flex items-center justify-between text-xs bg-slate-800 px-3 py-2 rounded-lg';
        row.innerHTML = `<span>通道 ${Number(idx) + 1}: ${name}</span>`;
        list.appendChild(row);
    });
}

async function clearVideoSlot(index) {
    if (!(index in uploadedVideos)) return;
    await fetch(`/api/delete_video/${index}`, { method: 'DELETE' });
    delete uploadedVideos[index];
    resetVideoSlot(index);
    updateVideoList();
}

async function startProcessing() {
    if (Object.keys(uploadedVideos).length === 0) {
        alert('请先上传视频');
        return;
    }
    const params = getCurrentParams();

    for (const idx of Object.keys(uploadedVideos)) {
        const res = await fetch(`/api/start_detection/${idx}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(params)
        });
        const data = await res.json();
        if (!data.success) {
            alert(`通道${Number(idx)+1} 启动失败: ${data.message}`);
            continue;
        }
        startStream(Number(idx));
    }

    document.getElementById('startBtn').classList.add('hidden');
    document.getElementById('stopBtn').classList.remove('hidden');
}

function startStream(index) {
    const frame = document.getElementById(`videoFrame-${index}`);
    frame.style.display = 'block';
    frame.src = `/api/video_feed/${index}?t=${Date.now()}`;
    document.getElementById(`panelStatus-${index}`).textContent = '检测中';
    document.getElementById(`panelStatus-${index}`).className = 'status-badge status-live';
    activeStreams.add(index);
}

async function stopProcessing() {
    for (const idx of Array.from(activeStreams)) {
        await fetch(`/api/stop_detection/${idx}`, { method: 'POST' });
        resetVideoSlot(idx);
    }
    document.getElementById('startBtn').classList.remove('hidden');
    document.getElementById('stopBtn').classList.add('hidden');
}
