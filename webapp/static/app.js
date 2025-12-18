let videoSources = {}; // {index: {label, type, source}}
let activeStreams = new Set();
let queryImages = [];
let statusInterval = null;
let paramUpdateTimer = null;
let availableCameras = [];

const panelCount = 4;

// 折叠面板切换
function toggleCollapse(header) {
    const content = header.nextElementSibling;
    const icon = header.querySelector('.collapse-icon');
    content.classList.toggle('open');
    icon.classList.toggle('open');
}

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
    fetchCameras();
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
    document.getElementById('queryCount').textContent = `(${images.length})`;
    const container = document.getElementById('queryList');
    container.innerHTML = '';
    images.forEach(img => {
        const card = document.createElement('div');
        card.className = 'relative group border border-slate-200 rounded overflow-hidden shadow-sm';
        card.innerHTML = `
            <img src="${img.thumbnail}" class="w-full h-12 object-cover" alt="query">
            <button class="absolute top-0 right-0 text-[10px] bg-white/90 px-1 rounded-bl hidden group-hover:block text-red-500 shadow" onclick="deleteQuery('${img.id}')">×</button>
        `;
        container.appendChild(card);
    });
}

function updateVideoStateFromStatus(videoDetails) {
    const latestIndices = new Set();
    videoDetails.forEach(item => {
        latestIndices.add(item.index);
        videoSources[item.index] = {
            label: item.label || item.filename || `通道 ${item.index + 1}`,
            type: item.type || 'file',
            source: item.source
        };
        showVideoReady(item.index, videoSources[item.index].label, item.active);
    });

    Object.keys(videoSources).forEach(idx => {
        if (!latestIndices.has(Number(idx))) {
            resetVideoSlot(Number(idx));
            delete videoSources[idx];
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

async function uploadVideos(event) {
    const files = Array.from(event.target.files);
    if (!files.length) return;

    const availableSlots = [];
    for (let i = 0; i < panelCount; i++) {
        if (!(i in videoSources)) availableSlots.push(i);
    }

    if (availableSlots.length === 0) {
        alert('通道已满，请先删除一个视频');
        event.target.value = '';
        return;
    }

    const filesToUpload = files.slice(0, availableSlots.length);
    if (files.length > availableSlots.length) {
        alert(`只有 ${availableSlots.length} 个空闲通道，将上传前 ${availableSlots.length} 个视频`);
    }

    for (let i = 0; i < filesToUpload.length; i++) {
        const file = filesToUpload[i];
        const slot = availableSlots[i];
        const form = new FormData();
        form.append('video', file);
        form.append('video_index', slot);
        try {
            const res = await fetch('/api/upload_video', { method: 'POST', body: form });
            const data = await res.json();
            if (!data.success) throw new Error(data.message);
            videoSources[slot] = { label: data.label, type: data.type, source: null };
            showVideoReady(slot, data.label, false);
            updateVideoList();
        } catch (err) {
            alert(`上传 ${file.name} 失败: ` + err.message);
        }
    }
    event.target.value = '';
}

async function fetchCameras() {
    try {
        const res = await fetch('/api/list_cameras');
        const data = await res.json();
        if (!data.success) throw new Error(data.message || '获取失败');
        availableCameras = data.cameras || [];
        renderCameraOptions();
    } catch (err) {
        console.error('摄像头列表获取失败', err);
    }
}

function renderCameraOptions() {
    const select = document.getElementById('cameraSelect');
    if (!select) return;
    select.innerHTML = '';
    availableCameras.forEach(cam => {
        const option = document.createElement('option');
        option.value = cam.id;
        option.textContent = cam.name;
        select.appendChild(option);
    });
    const customOption = document.createElement('option');
    customOption.value = 'custom';
    customOption.textContent = '自定义源...';
    select.appendChild(customOption);
}

function refreshCameras() {
    fetchCameras();
}

async function assignCamera() {
    const slot = findAvailableSlot();
    if (slot === -1) {
        alert('通道已满，请先删除一个视频或摄像头');
        return;
    }
    const select = document.getElementById('cameraSelect');
    if (!select) {
        alert('摄像头选择器不存在');
        return;
    }

    let cameraId = select.value;
    if (!cameraId) {
        alert('请选择摄像头');
        return;
    }

    let labelInput = document.getElementById('cameraLabel');
    let customInput = document.getElementById('cameraCustom');
    let label = labelInput ? labelInput.value.trim() : '';

    if (cameraId === 'custom') {
        const customValue = customInput ? customInput.value.trim() : '';
        if (!customValue) {
            alert('请输入自定义摄像头 URI 或管线');
            return;
        }
        cameraId = customValue;
        if (!label) label = '自定义摄像头';
    } else {
        const selectedOption = select.selectedOptions[0];
        if (!label && selectedOption) {
            label = selectedOption.textContent;
        }
    }

    const payload = { camera_id: cameraId, video_index: slot, label };

    try {
        const res = await fetch('/api/assign_camera', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        const data = await res.json();
        if (!data.success) throw new Error(data.message);
        videoSources[slot] = { label: data.label, type: data.type, source: cameraId };
        showVideoReady(slot, data.label, false);
        updateVideoList();
        if (labelInput) labelInput.value = '';
        if (customInput) customInput.value = '';
    } catch (err) {
        alert('绑定摄像头失败: ' + err.message);
    }
}

function findAvailableSlot() {
    for (let i = 0; i < panelCount; i++) {
        if (!(i in videoSources)) return i;
    }
    return -1;
}

function showVideoReady(index, label, active) {
    const placeholder = document.getElementById(`placeholder-${index}`);
    const frame = document.getElementById(`videoFrame-${index}`);
    const status = document.getElementById(`panelStatus-${index}`);
    placeholder.style.display = 'none';
    frame.style.display = 'block';
    frame.alt = label;
    if (active) {
        if (!activeStreams.has(index) || !frame.src) {
            frame.src = `/api/video_feed/${index}?t=${Date.now()}`;
        }
        status.textContent = '检测中';
        status.className = 'status-badge status-live';
        activeStreams.add(index);
    } else {
        frame.src = '';
        status.textContent = '已就绪';
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
    Object.entries(videoSources).forEach(([idx, info]) => {
        const typeIcon = info.type === 'camera' ? 'fa-camera' : 'fa-file-video';
        const row = document.createElement('div');
        row.className = 'flex items-center justify-between bg-slate-100 border border-slate-200 px-2 py-1 rounded text-slate-600';
        row.innerHTML = `
            <span class="truncate flex-1"><i class="fa-solid ${typeIcon} mr-1 text-slate-400"></i>通道${Number(idx) + 1}: ${info.label}</span>
            <button class="text-slate-400 hover:text-red-500 ml-1" onclick="clearVideoSlot(${idx})"><i class="fa-solid fa-xmark"></i></button>
        `;
        list.appendChild(row);
    });
}

async function clearVideoSlot(index) {
    if (!(index in videoSources)) return;
    await fetch(`/api/delete_video/${index}`, { method: 'DELETE' });
    delete videoSources[index];
    resetVideoSlot(index);
    updateVideoList();
}

async function startProcessing() {
    if (Object.keys(videoSources).length === 0) {
        alert('请先上传视频或绑定摄像头');
        return;
    }
    const params = getCurrentParams();

    for (const idx of Object.keys(videoSources)) {
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
