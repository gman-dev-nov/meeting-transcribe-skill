#!/usr/bin/env node
'use strict';

// Explicit experiment against Type's private native ABI; no Type GUI or profile.
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { execFileSync } = require('node:child_process');

const SAMPLE_RATE = 16000;
const ENGINE = 'gigaam-duo-type-experimental';
const KNOWN_RELEASE = 'type.v1.019.20260917';
const KNOWN_MODEL = 'asr.gigaam_duo.v1.3';
const REQUEST_TIMEOUT_MS = 120000;
const PAUSE_CONFIG = Object.freeze({
  sampleRate: SAMPLE_RATE,
  windowSamples: 320,
  windowMs: 20,
  thresholdDbfs: -40,
  minPauseSamples: 4000,
  minPauseMs: 250,
  maxChunkSeconds: 25,
  readBufferSamples: 20480,
});
const HELP = `Usage: node duo_transcribe.cjs /absolute/input --output-dir /absolute/dir
  [--app /Applications/Type.app] [--chunk-seconds 20] [--segmentation pause|fixed]

Experimental local GigaAM Duo adapter for Type; chunk-seconds must be 1..25.
Default segmentation: pause (local low-energy intervals, not speech VAD).
No downloads, GUI, Type profile, LLM postprocessing, or word timestamps.
`;

function parseArgs(argv) {
  const args = { app: '/Applications/Type.app', chunkSeconds: 20, segmentation: 'pause' };
  const seen = new Set();
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    if (arg === '--help' || arg === '-h') return { help: true };
    const key = {
      '--app': 'app', '--output-dir': 'outputDir', '--chunk-seconds': 'chunkSeconds',
      '--segmentation': 'segmentation',
    }[arg];
    if (key) {
      if (seen.has(key)) throw new Error(`Duplicate option: ${arg}`);
      seen.add(key);
      const value = argv[++i];
      if (!value || value.startsWith('--')) throw new Error(`Missing value: ${arg}`);
      args[key] = key === 'chunkSeconds' ? Number(value) : value;
    } else if (arg.startsWith('-')) {
      throw new Error(`Unknown option: ${arg}`);
    } else if (args.input) {
      throw new Error('Exactly one input file is required');
    } else {
      args.input = arg;
    }
  }
  for (const key of ['input', 'outputDir', 'app']) {
    if (typeof args[key] !== 'string' || !path.isAbsolute(args[key])) {
      throw new Error(`${key} must be an absolute path`);
    }
    args[key] = path.resolve(args[key]);
  }
  if (!Number.isFinite(args.chunkSeconds) || args.chunkSeconds < 1 || args.chunkSeconds > 25) {
    throw new Error('--chunk-seconds must be a finite number in 1..25');
  }
  if (!['pause', 'fixed'].includes(args.segmentation)) throw new Error('--segmentation must be pause or fixed');
  return args;
}

function chunkRanges(totalSamples, chunkSeconds) {
  if (!Number.isSafeInteger(totalSamples) || totalSamples < 0) throw new Error('Invalid PCM sample count');
  if (!Number.isFinite(chunkSeconds) || chunkSeconds < 1 || chunkSeconds > 25) throw new Error('Invalid chunk duration');
  const step = Math.floor(chunkSeconds * SAMPLE_RATE);
  const chunks = [];
  for (let startSample = 0; startSample < totalSamples; startSample += step) {
    const endSample = Math.min(startSample + step, totalSamples);
    chunks.push({ startSample, endSample, start: startSample / SAMPLE_RATE, end: endSample / SAMPLE_RATE });
  }
  return chunks;
}

function readPcmBytes(pcmFd, buffer, sampleOffset, sampleCount) {
  const size = sampleCount * 4;
  let read = 0;
  while (read < size) {
    const received = fs.readSync(pcmFd, buffer, read, size - read, sampleOffset * 4 + read);
    if (received === 0) throw new Error('Short PCM read');
    read += received;
  }
}

function detectPauses(pcmFd, totalSamples) {
  if (!Number.isSafeInteger(totalSamples) || totalSamples < 0) throw new Error('Invalid PCM sample count');
  const { windowSamples, readBufferSamples, minPauseSamples, thresholdDbfs } = PAUSE_CONFIG;
  const thresholdPower = 10 ** (thresholdDbfs / 10);
  const buffer = Buffer.alloc(readBufferSamples * 4);
  const pauses = [];
  let lowStart = null;
  const finishPause = endSample => {
    if (lowStart !== null && endSample - lowStart >= minPauseSamples) {
      pauses.push({ startSample: lowStart, endSample });
    }
    lowStart = null;
  };
  // Buffer size is a multiple of the 20ms window; low spans carry across reads.
  for (let start = 0; start < totalSamples; start += readBufferSamples) {
    const samplesRead = Math.min(readBufferSamples, totalSamples - start);
    readPcmBytes(pcmFd, buffer, start, samplesRead);
    for (let offset = 0; offset < samplesRead; offset += windowSamples) {
      const length = Math.min(windowSamples, samplesRead - offset);
      let power = 0;
      for (let i = 0; i < length; i++) {
        const sample = buffer.readFloatLE((offset + i) * 4);
        if (!Number.isFinite(sample)) throw new Error(`Non-finite PCM sample at ${start + offset + i}`);
        power += sample * sample;
      }
      if (power / length <= thresholdPower) {
        if (lowStart === null) lowStart = start + offset;
      } else {
        finishPause(start + offset);
      }
    }
  }
  finishPause(totalSamples);
  return pauses;
}

function pauseChunkRanges(totalSamples, chunkSeconds, pauses = []) {
  if (!Number.isSafeInteger(totalSamples) || totalSamples < 0) throw new Error('Invalid PCM sample count');
  if (!Number.isFinite(chunkSeconds) || chunkSeconds < 1 || chunkSeconds > PAUSE_CONFIG.maxChunkSeconds) {
    throw new Error('Invalid chunk duration');
  }
  if (!Array.isArray(pauses)) throw new Error('Pauses must be an array');
  let previousEnd = 0;
  for (const pause of pauses) {
    if (!pause || typeof pause !== 'object' || Array.isArray(pause) ||
        !Number.isSafeInteger(pause.startSample) || !Number.isSafeInteger(pause.endSample) ||
        pause.startSample < previousEnd || pause.startSample < 0 || pause.endSample <= pause.startSample ||
        pause.endSample > totalSamples) {
      throw new Error('Invalid, unsorted, or overlapping pause span');
    }
    previousEnd = pause.endSample;
  }
  const targetSamples = Math.floor(chunkSeconds * SAMPLE_RATE);
  const lowerSamples = Math.ceil(Math.max(1, 0.6 * chunkSeconds) * SAMPLE_RATE);
  const upperSamples = Math.min(PAUSE_CONFIG.maxChunkSeconds * SAMPLE_RATE, targetSamples + 5 * SAMPLE_RATE);
  const chunks = [];
  let pauseIndex = 0;
  for (let startSample = 0; startSample < totalSamples;) {
    const remaining = totalSamples - startSample;
    let endSample;
    let boundaryReason;
    // Preserve the whole final tail, even when it exceeds the nominal target.
    if (remaining <= upperSamples) {
      endSample = totalSamples;
      boundaryReason = 'final';
    } else {
      const minimumCut = startSample + lowerSamples;
      const maximumCut = startSample + upperSamples;
      const targetCut = startSample + targetSamples;
      while (pauseIndex < pauses.length && pauses[pauseIndex].endSample <= minimumCut) pauseIndex++;
      let candidateCut = null;
      let distance = Infinity;
      for (let i = pauseIndex; i < pauses.length && pauses[i].startSample < maximumCut; i++) {
        const overlapStart = Math.max(minimumCut, pauses[i].startSample);
        const overlapEnd = Math.min(maximumCut, pauses[i].endSample);
        if (overlapStart >= overlapEnd) continue;
        // Use the midpoint of the accessible part of a long pause. A 100s
        // silence must not force cuts merely because its global midpoint is far away.
        const midpoint = overlapStart + Math.floor((overlapEnd - overlapStart) / 2);
        const fromTarget = Math.abs(midpoint - targetCut);
        if (fromTarget < distance || (fromTarget === distance && midpoint < candidateCut)) {
          candidateCut = midpoint;
          distance = fromTarget;
        }
      }
      endSample = candidateCut ?? targetCut;
      boundaryReason = candidateCut === null ? 'forced' : 'pause';
    }
    chunks.push({ startSample, endSample, start: startSample / SAMPLE_RATE, end: endSample / SAMPLE_RATE, boundaryReason });
    startSample = endSample;
  }
  return chunks;
}

function existsEvenSymlink(file) {
  try { fs.lstatSync(file); return true; } catch (error) {
    if (error.code === 'ENOENT') return false;
    throw error;
  }
}

function outputPaths(input, outputDir) {
  const name = path.basename(input); // Keep extension: meeting.wav and meeting.mp4 cannot collide.
  const result = {
    json: path.join(outputDir, `${name}.duo.transcript.json`),
    markdown: path.join(outputDir, `${name}.duo.transcript.md`),
  };
  for (const file of Object.values(result)) {
    if (path.resolve(file) === path.resolve(input)) throw new Error('Output collides with input');
    if (existsEvenSymlink(file)) throw new Error(`Refusing to overwrite existing output: ${file}`);
  }
  return result;
}

function timestamp(seconds) {
  const value = Math.max(0, Math.floor(seconds));
  return [Math.floor(value / 3600), Math.floor(value / 60) % 60, value % 60]
    .map(n => String(n).padStart(2, '0')).join(':');
}

function writeOutputs(paths, document) {
  const parent = path.dirname(paths.json);
  if (path.dirname(paths.markdown) !== parent) throw new Error('Output pair must share a directory');
  for (const file of Object.values(paths)) {
    if (existsEvenSymlink(file)) throw new Error(`Refusing to overwrite existing output: ${file}`);
  }
  fs.mkdirSync(parent, { recursive: true });
  const staging = fs.mkdtempSync(path.join(parent, '.duo-output-'));
  const sourceJson = path.join(staging, 'transcript.json');
  const sourceMarkdown = path.join(staging, 'transcript.md');
  let publishedJson = false;
  try {
    const meta = document.metadata || {};
    const lines = [
      '# GigaAM Duo — экспериментальная расшифровка', '',
      `- Движок: ${meta.engine || ENGINE}`,
      `- Модель: ${meta.model || 'unknown'}; Type: ${meta.app_version || 'unknown'}`,
      '- LLM: выключена. Временные метки относятся к чанкам, а не к словам.',
      `- Разбиение: ${meta.segmentation?.method || 'fixed'}; срезов без паузы: ${meta.segmentation?.forced_cuts ?? 'unknown'}.`,
      '- На границах чанков возможны обрывы и ошибки.', '',
      ...document.segments.map(s => `[${timestamp(s.start)}–${timestamp(s.end)}] ${s.text}`), '',
    ];
    fs.writeFileSync(sourceJson, JSON.stringify(document, null, 2) + '\n', { flag: 'wx', mode: 0o600 });
    fs.writeFileSync(sourceMarkdown, lines.join('\n'), { flag: 'wx', mode: 0o600 });
    // Hard links publish complete files and fail if a concurrent process created the target.
    fs.linkSync(sourceJson, paths.json);
    publishedJson = true;
    fs.linkSync(sourceMarkdown, paths.markdown);
  } catch (error) {
    if (publishedJson) {
      try {
        const ours = fs.statSync(sourceJson);
        const current = fs.lstatSync(paths.json);
        if (ours.ino === current.ino && ours.dev === current.dev) fs.unlinkSync(paths.json);
      } catch (rollbackError) {
        if (rollbackError.code !== 'ENOENT') error.message += `; output rollback failed: ${rollbackError.message}`;
      }
    }
    throw error;
  } finally {
    fs.rmSync(staging, { recursive: true, force: true });
  }
}

function bundledPath(root, relative, kind) {
  if (typeof relative !== 'string' || !relative || path.isAbsolute(relative)) throw new Error('Invalid bundled resource path');
  const candidate = fs.realpathSync(path.resolve(root, relative));
  if (!candidate.startsWith(root + path.sep)) throw new Error('Bundled resource escapes Type resources');
  const stat = fs.statSync(candidate);
  if (kind === 'file' ? !stat.isFile() : !stat.isDirectory()) throw new Error(`Invalid bundled ${kind}: ${candidate}`);
  return candidate;
}

function readJson(file) {
  const value = JSON.parse(fs.readFileSync(file, 'utf8'));
  if (!value || typeof value !== 'object' || Array.isArray(value)) throw new Error(`Invalid metadata: ${file}`);
  return value;
}

function prepareModelRoot(resources) {
  const bundle = bundledPath(resources, 'gigatype-model', 'directory');
  const meta = readJson(bundledPath(bundle, 'meta.json', 'file'));
  const asr = bundledPath(bundle, meta.parts?.asr, 'directory');
  const passthrough = bundledPath(bundle, meta.parts?.llm_passthrough, 'directory');
  const core = bundledPath(bundle, meta.core, 'file');
  const asrMeta = readJson(bundledPath(asr, 'meta.json', 'file'));
  const passthroughMeta = readJson(bundledPath(passthrough, 'meta.json', 'file'));
  if (meta.release_id !== KNOWN_RELEASE || Object.keys(asrMeta.versions || {}).join() !== KNOWN_MODEL ||
      asrMeta.versions[KNOWN_MODEL] !== 'config.yaml' ||
      Object.keys(passthroughMeta.versions || {}).join() !== 'llm.passthrough' ||
      passthroughMeta.versions['llm.passthrough'] !== 'config.yaml') {
    throw new Error('Unverified Type model release/layout; this experiment supports only ' + KNOWN_RELEASE + ' / ' + KNOWN_MODEL);
  }
  const tempRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'meeting-duo-'));
  try {
    const parts = { asr, llm_passthrough: passthrough };
    // Native open enumerates these versions even with llm_off. Metadata only;
    // deliberately no model.tar, no links into Type's user model cache.
    for (const [directory, version] of [
      ['giga3b_v1.020_qat', 'llm.giga3b.v1.020.qat'],
      ['giga600m_v1.020_qat', 'llm.giga600m.v1.020.qat'],
    ]) {
      fs.mkdirSync(path.join(tempRoot, directory));
      fs.writeFileSync(path.join(tempRoot, directory, 'meta.json'), JSON.stringify({ versions: { [version]: 'config.yaml' } }));
      parts[directory] = directory;
    }
    fs.writeFileSync(path.join(tempRoot, 'meta.json'), JSON.stringify({ release_id: meta.release_id, core, parts }));
    return { tempRoot, release: meta.release_id, model: KNOWN_MODEL };
  } catch (error) {
    fs.rmSync(tempRoot, { recursive: true, force: true });
    throw error;
  }
}

function executable(name, resources) {
  const candidates = (process.env.PATH || '').split(path.delimiter).filter(Boolean).map(p => path.join(p, name));
  candidates.push(path.join(resources, 'bin', name));
  for (const candidate of candidates) {
    try {
      fs.accessSync(candidate, fs.constants.X_OK);
      if (fs.statSync(candidate).isFile()) return candidate;
    } catch { /* Try the next explicit local executable. */ }
  }
  throw new Error(`${name} not found in PATH or Type's bundled bin directory`);
}

function withTimeout(promise, milliseconds, label) {
  let timer;
  return Promise.race([
    promise,
    new Promise((_, reject) => { timer = setTimeout(() => reject(new Error(`${label} timed out`)), milliseconds); }),
  ]).finally(() => clearTimeout(timer));
}

function createResultCallbacks() {
  let pending = null;
  function fail(error) {
    const request = pending;
    pending = null;
    if (request) request.reject(error);
  }
  function settle() {
    if (!pending || pending.raw === null || pending.text === null) return;
    const request = pending;
    pending = null;
    request.resolve(request.raw); // llm_off: preserve actual ASR callback, never a rewritten fallback.
  }
  return {
    onAsrText(text) {
      if (!pending) return;
      if (typeof text !== 'string') { fail(new Error('Invalid onAsrText result')); return; }
      pending.raw = text;
      settle();
    },
    onText(text, answer) {
      if (!pending) return;
      if (answer?.error) { fail(new Error(`Native ASR failed: ${String(answer.error)}`)); return; }
      if (answer?.complete === false) { fail(new Error('Native ASR returned an incomplete result')); return; }
      if (typeof text !== 'string') { fail(new Error('Invalid onText result')); return; }
      pending.text = text;
      settle();
    },
    request(call, label, timeoutMs = REQUEST_TIMEOUT_MS) {
      if (pending) throw new Error('A native ASR request is already pending');
      const result = new Promise((resolve, reject) => {
        pending = { resolve, reject, raw: null, text: null };
        try { call(); } catch (error) { fail(error); }
      });
      return withTimeout(result, timeoutMs, label).finally(() => { pending = null; });
    },
    cancel() { pending = null; },
  };
}

async function main(argv = process.argv.slice(2)) {
  const args = parseArgs(argv);
  if (args.help) { process.stdout.write(HELP); return; }
  const input = fs.realpathSync(args.input);
  if (!fs.statSync(input).isFile()) throw new Error('Input must be a regular file');
  const outputs = outputPaths(input, args.outputDir);
  const appRoot = fs.realpathSync(args.app);
  const resources = bundledPath(appRoot, 'Contents/Resources', 'directory');
  const addonPath = bundledPath(resources, 'bin/gigatype_runtime.node', 'file');
  const infoPlist = bundledPath(appRoot, 'Contents/Info.plist', 'file');
  const appVersion = execFileSync('/usr/bin/plutil', ['-extract', 'CFBundleShortVersionString', 'raw', '-o', '-', infoPlist],
    { encoding: 'utf8', timeout: 10000 }).trim();
  if (!/^\d+(?:\.\d+){1,3}(?:[-+][\w.-]+)?$/.test(appVersion)) throw new Error('Invalid Type app version');
  const ffmpeg = executable('ffmpeg', resources);
  const ffprobe = executable('ffprobe', resources);
  const duration = Number(execFileSync(ffprobe, ['-v', 'error', '-show_entries', 'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', input],
    { encoding: 'utf8', timeout: 30000, maxBuffer: 1024 * 1024 }).trim());
  if (!Number.isFinite(duration) || duration <= 0) throw new Error('ffprobe returned an invalid audio duration');
  process.stderr.write(`WARNING: experimental Type native adapter; LLM disabled; no word timestamps. Segmentation=${args.segmentation}; energy pauses are not speech VAD and boundaries may cut words.\n`);
  let model;
  let runtime;
  const callbacks = createResultCallbacks();
  let pcmFd;
  let decodedDuration;
  let segmentation;
  const started = Date.now();
  const segments = [];
  try {
    model = prepareModelRoot(resources);
    const pcmPath = path.join(model.tempRoot, 'input.f32le');
    execFileSync(ffmpeg, ['-nostdin', '-v', 'error', '-i', input, '-map', '0:a:0', '-ac', '1', '-ar', String(SAMPLE_RATE), '-f', 'f32le', '-y', pcmPath],
      { timeout: Math.max(120000, Math.ceil(duration * 2000)), maxBuffer: 1024 * 1024, stdio: ['ignore', 'ignore', 'pipe'] });
    const pcmBytes = fs.statSync(pcmPath).size;
    if (!pcmBytes || pcmBytes % 4) throw new Error('ffmpeg produced invalid float32 PCM');
    const totalSamples = pcmBytes / 4;
    decodedDuration = totalSamples / SAMPLE_RATE;
    pcmFd = fs.openSync(pcmPath, 'r');
    const pauses = args.segmentation === 'pause' ? detectPauses(pcmFd, totalSamples) : [];
    const ranges = args.segmentation === 'pause'
      ? pauseChunkRanges(totalSamples, args.chunkSeconds, pauses)
      : chunkRanges(totalSamples, args.chunkSeconds).map((chunk, index, all) => ({
        ...chunk, boundaryReason: index === all.length - 1 ? 'final' : 'forced',
      }));
    segmentation = {
      method: args.segmentation,
      window_ms: PAUSE_CONFIG.windowMs,
      threshold_dbfs: PAUSE_CONFIG.thresholdDbfs,
      min_pause_ms: PAUSE_CONFIG.minPauseMs,
      target_chunk_seconds: args.chunkSeconds,
      max_chunk_seconds: PAUSE_CONFIG.maxChunkSeconds,
      detected_pauses: args.segmentation === 'pause' ? pauses.length : null,
      forced_cuts: ranges.filter(chunk => chunk.boundaryReason === 'forced').length,
      boundaries: ranges.map(chunk => ({ end_sample: chunk.endSample, end: chunk.end, reason: chunk.boundaryReason })),
    };
    const addon = require(addonPath);
    const openPromise = Promise.resolve().then(() => addon.open({
      modelRoot: model.tempRoot,
      features: ['llm_off'],
      onAsrText: callbacks.onAsrText,
      onText: callbacks.onText,
    }));
    try { runtime = await withTimeout(openPromise, REQUEST_TIMEOUT_MS, 'Native runtime open'); }
    catch (error) { openPromise.then(r => r.close()).catch(() => {}); throw error; }
    if (!runtime.features().includes('llm_off')) throw new Error('Native runtime did not confirm llm_off');
    for (const [index, chunk] of ranges.entries()) {
      const bytes = Buffer.alloc((chunk.endSample - chunk.startSample) * 4);
      readPcmBytes(pcmFd, bytes, chunk.startSample, bytes.length / 4);
      // Decode explicitly as little endian, independent of Buffer alignment.
      const samples = new Float32Array(bytes.length / 4);
      for (let i = 0; i < samples.length; i++) {
        samples[i] = bytes.readFloatLE(i * 4);
        if (!Number.isFinite(samples[i])) throw new Error(`Non-finite PCM sample at ${chunk.startSample + i}`);
      }
      if (samples.length > PAUSE_CONFIG.maxChunkSeconds * SAMPLE_RATE) throw new Error('ASR chunk exceeds 25 seconds');
      const text = await callbacks.request(() => {
        if (!runtime.startRecording()) throw new Error('Native startRecording failed');
        const paddingSamples = Math.min(8000, Math.floor((PAUSE_CONFIG.maxChunkSeconds * SAMPLE_RATE - samples.length) / 2));
        if (paddingSamples) runtime.pushAudio(new Float32Array(paddingSamples));
        for (let i = 0; i < samples.length; i += 1600) runtime.pushAudio(samples.subarray(i, i + 1600));
        if (paddingSamples) runtime.pushAudio(new Float32Array(paddingSamples));
        runtime.stopRecording();
      }, `ASR chunk ${index + 1}`);
      if (typeof text !== 'string') throw new Error(`Invalid ASR result for chunk ${index + 1}`);
      segments.push({ start: chunk.start, end: chunk.end, text: text.trim() });
      process.stderr.write(`Duo chunk ${index + 1}/${ranges.length} complete\n`);
    }
  } finally {
    callbacks.cancel();
    try { if (pcmFd !== undefined) fs.closeSync(pcmFd); }
    finally {
      try { if (runtime) await withTimeout(Promise.resolve().then(() => runtime.close()), 30000, 'Native runtime close'); }
      finally { if (model) fs.rmSync(model.tempRoot, { recursive: true, force: true }); }
    }
  }
  const document = {
    metadata: {
      engine: ENGINE, model: model.model, app_version: appVersion, release: model.release,
      llm_enabled: false, timing_precision: 'segment', source: input,
      text_source: 'onAsrText',
      duration_seconds: decodedDuration, source_duration_seconds: duration,
      chunk_seconds: args.chunkSeconds, segments_count: segments.length,
      segmentation,
      processing_seconds: (Date.now() - started) / 1000,
    },
    segments,
  };
  writeOutputs(outputs, document);
  process.stdout.write(JSON.stringify(outputs) + '\n');
}

module.exports = { parseArgs, chunkRanges, detectPauses, pauseChunkRanges, PAUSE_CONFIG, outputPaths, writeOutputs, createResultCallbacks };
if (require.main === module) main().catch(error => {
  process.stderr.write(`Duo experiment failed: ${error.message}\n`);
  // A timed-out native close may retain its worker handles; cleanup was attempted
  // above, and a failed standalone experiment must not wait forever on those.
  process.exit(1);
});
