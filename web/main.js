import { fileToSquare, paint } from './image.js';
import { normalizeStory } from '../src/document.js';
import { toTitle } from '../src/title.js';

const $ = (id) => document.getElementById(id);
const appBase = new URL('./', document.baseURI).href;

// --- workers ----------------------------------------------------------------

// Vite only bundles workers created with this exact `new Worker(new URL())` form.
function connect(w) {
  w.postMessage({ type: 'init', appBase });
  const pending = new Map();
  let next = 0;
  w.onmessage = ({ data }) => {
    if (data.type === 'download') return showDownload(data);
    const job = pending.get(data.id);
    if (!job) return;
    if (data.type === 'done') {
      pending.delete(data.id);
      job.resolve(data);
    } else if (data.type === 'error') {
      pending.delete(data.id);
      job.reject(new Error(data.message));
    } else {
      job.onEvent?.(data);
    }
  };
  w.onerror = (e) => {
    for (const job of pending.values()) job.reject(new Error(e.message || 'worker crashed'));
    pending.clear();
  };
  return (msg, onEvent, transfer = []) =>
    new Promise((resolve, reject) => {
      const id = next++;
      pending.set(id, { resolve, reject, onEvent });
      w.postMessage({ ...msg, id }, transfer);
    });
}

const lmJob = connect(new Worker(new URL('./lm-worker.js', import.meta.url), { type: 'module' }));
let captionJob;
const caption = (blob, onEvent) => {
  captionJob ??= connect(new Worker(new URL('./caption-worker.js', import.meta.url), { type: 'module' }));
  return captionJob({ type: 'caption', blob }, onEvent);
};

// The captioning progress bar under the subject field: the model download
// (first visit only), then an indeterminate bar while it writes the caption.
let captionDownloadHook = null;
function captionMeter(state, fraction = 0) {
  $('caption-meter-wrap').hidden = state === 'off';
  $('caption-meter-wrap').classList.toggle('indeterminate', state === 'busy');
  $('caption-meter').style.width = `${100 * fraction}%`;
}

// --- model download panel -------------------------------------------------

const downloads = new Map();
function showDownload({ name, loaded, total }) {
  if (name === 'Captioner') captionDownloadHook?.(loaded, total);
  let row = downloads.get(name);
  if (!row) {
    row = document.createElement('li');
    row.innerHTML = `<span></span><div class="meter"><div class="meter-fill"></div></div><span class="amount"></span>`;
    row.firstChild.textContent = name;
    $('download-list').append(row);
    downloads.set(name, row);
  }
  const mb = (n) => (n / 1e6).toFixed(0);
  row.querySelector('.meter-fill').style.width = total ? `${(100 * loaded) / total}%` : '0';
  row.querySelector('.amount').textContent = total ? `${mb(loaded)} / ${mb(total)} MB` : `${mb(loaded)} MB`;
  const done = total && loaded >= total;
  row.dataset.done = done ? '1' : '';
  const anyActive = [...downloads.values()].some((r) => !r.dataset.done);
  $('downloads').hidden = !anyActive;
}

// --- tabs ---------------------------------------------------------------------

function selectTab(name) {
  for (const t of ['hide', 'reveal']) {
    $(`tab-${t}`).setAttribute('aria-selected', String(t === name));
    $(t).hidden = t !== name;
  }
}
$('tab-hide').onclick = () => selectTab('hide');
$('tab-reveal').onclick = () => selectTab('reveal');

// --- hide a photo --------------------------------------------------------------

let photo = null; // { rgb, file }
let captionToken = 0;

function showError(id, err) {
  $(id).textContent = err ? err.message || String(err) : '';
  $(id).hidden = !err;
}

async function choosePhoto(file) {
  if (!file || !file.type.startsWith('image/')) return;
  showError('hide-error', null);
  $('story').hidden = true;
  try {
    const rgb = await fileToSquare(file);
    paint($('preview'), rgb);
    photo = { rgb, file };
  } catch (e) {
    showError('hide-error', new Error(`Couldn't read that image (${e.message}).`));
    return;
  }
  $('drop').hidden = true;
  $('setup').hidden = false;
  $('title').value = '';
  $('title').placeholder = 'Captioning the photo…';
  $('title-hint').textContent = 'A small vision model is describing the photo.';
  $('write').disabled = true;

  const token = ++captionToken;
  captionMeter('busy');
  captionDownloadHook = (loaded, total) => {
    if (token !== captionToken) return;
    if (total && loaded < total) {
      captionMeter('download', loaded / total);
      $('title-hint').textContent = `Downloading the captioning model (first time only): ${Math.round(
        (100 * loaded) / total,
      )}%`;
    } else {
      captionMeter('busy');
    }
  };
  try {
    const { title } = await caption(file, (ev) => {
      if (token !== captionToken || ev.type !== 'status') return;
      if (ev.stage === 'captioning') {
        captionMeter('busy');
        $('title-hint').textContent = 'A small vision model is describing the photo.';
      }
    });
    if (token !== captionToken) return;
    captionMeter('off');
    if (!$('title').value) $('title').value = title;
    $('title-hint').textContent = "This only steers what the story is about; it isn't needed to reveal the photo.";
  } catch (e) {
    if (token !== captionToken) return;
    captionMeter('off');
    $('title-hint').textContent = `Captioning failed (${e.message}). Describe the photo yourself.`;
  }
  $('title').placeholder = 'Describe the photo';
  updateWrite();
}

function updateWrite() {
  $('write').disabled = !photo || !toTitle($('title').value);
}
$('title').oninput = updateWrite;

$('file').onchange = (e) => choosePhoto(e.target.files[0]);
const drop = $('drop');
drop.ondragover = (e) => {
  e.preventDefault();
  drop.classList.add('over');
};
drop.ondragleave = () => drop.classList.remove('over');
drop.ondrop = (e) => {
  e.preventDefault();
  drop.classList.remove('over');
  choosePhoto(e.dataTransfer.files[0]);
};
$('another').onclick = () => {
  captionToken++;
  captionMeter('off');
  photo = null;
  $('file').value = '';
  $('setup').hidden = true;
  $('story').hidden = true;
  $('drop').hidden = false;
};

let lastDoc = '';
$('write').onclick = async () => {
  const subject = toTitle($('title').value);
  $('title').value = subject;
  $('write').disabled = true;
  showError('hide-error', null);
  $('story').hidden = false;
  $('story-actions').hidden = true;
  $('story-text').textContent = '';
  $('hide-meter').style.width = '0';
  $('hide-status').textContent = 'Starting…';
  try {
    const rgb = photo.rgb.slice();
    const { story } = await lmJob({ type: 'encode', rgb, subject }, (ev) => {
      if (ev.type === 'status') $('hide-status').textContent = ev.text;
      if (ev.type === 'token') {
        $('story-text').textContent = ev.text;
        $('hide-meter').style.width = `${(100 * ev.bits) / ev.totalBits}%`;
        if (ev.bits) {
          const pct = Math.round((100 * ev.bits) / ev.totalBits);
          $('hide-status').textContent = `Converting your photo to words… ${pct}%`;
        }
      }
    }, [rgb.buffer]);
    $('story-text').textContent = story;
    const words = story.split(/\s+/).length;
    $('hide-status').textContent = `Done: the photo is hidden in these ${words} words.`;
    lastDoc = story;
    $('story-actions').hidden = false;
  } catch (e) {
    $('hide-status').textContent = '';
    showError('hide-error', e);
  }
  updateWrite();
};

$('copy').onclick = async () => {
  await navigator.clipboard.writeText(lastDoc);
  $('copy').classList.add('done');
  setTimeout(() => $('copy').classList.remove('done'), 1500);
};
$('download').onclick = () => {
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([lastDoc + '\n'], { type: 'text/plain' }));
  a.download = 'story.txt';
  a.click();
  URL.revokeObjectURL(a.href);
};
$('try-reveal').onclick = () => {
  $('doc').value = lastDoc;
  selectTab('reveal');
  reveal();
};

// --- reveal a photo ---------------------------------------------------------------

async function reveal() {
  const story = normalizeStory($('doc').value);
  showError('reveal-error', null);
  $('result').hidden = true;
  if (!story) {
    showError('reveal-error', new Error('Paste a story first.'));
    return;
  }
  $('reveal-btn').disabled = true;
  $('reveal-meter-wrap').hidden = false;
  $('reveal-meter').style.width = '0';
  $('reveal-status').textContent = 'Starting…';
  try {
    const { image } = await lmJob({ type: 'decode', story }, (ev) => {
      if (ev.type === 'status') $('reveal-status').textContent = ev.text;
      if (ev.type === 'progress') {
        $('reveal-meter').style.width = `${(100 * ev.done) / ev.total}%`;
        $('reveal-status').textContent = `Reading the story: ${Math.round((100 * ev.done) / ev.total)}%`;
      }
    });
    paint($('out'), image);
    $('result').hidden = false;
    $('reveal-status').textContent = '';
    $('reveal-meter-wrap').hidden = true;
  } catch (e) {
    $('reveal-status').textContent = '';
    $('reveal-meter-wrap').hidden = true;
    showError('reveal-error', e);
  }
  $('reveal-btn').disabled = false;
}
$('reveal-btn').onclick = reveal;
