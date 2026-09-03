# Camera monitor -- "Pilot View" layout, live or replay.
#
# LIVE (default): serves an MJPEG stream per camera plus a web page showing one
# feed large with the others as thumbnails. Opens the cameras read-only; never
# touches the base or arms.
#
#   python camera_monitor.py                       # cameras from record_video_config
#   python camera_monitor.py --port 5000
#   python camera_monitor.py --cameras base_image=logitech:DAA051BE
#
# REPLAY: plays back a recorded episode instead of the live cameras -- no
# hardware needed. Same Pilot View, plus a timeline scrubber and play/pause.
#
#   python camera_monitor.py --replay                       # most recent recording
#   python camera_monitor.py --replay data/demos/20260902T155014617269
#
# Then from any machine on the same network:  http://sixsevensupremacy.local:8000

import argparse
import json
import time
from pathlib import Path

import cv2 as cv
from flask import Flask, Response, jsonify, abort

from cameras import _build_camera
from constants import CAMERAS
from episode_storage import EpisodeReader, default_data_dir
from record_video_config import RECORD_HZ

# Display metadata (obs_key -> (location label, device name, capture mode)).
CAMERA_INFO = {
    'base_image':        ('Base · front', 'Logitech C930e', '640×360'),
    'left_wrist_image':  ('Left wrist',   'RealSense D405', '640×480'),
    'right_wrist_image': ('Right wrist',  'RealSense D405', '640×480'),
}

app = Flask(__name__)

MODE = 'live'                 # 'live' or 'replay'
CAMERAS_OPEN = {}            # live: obs_key -> {'cam', 'backend', 'serial', 'ok', 'err'}
REPLAY = None               # replay: {'reader', 'dir', 'keys', 'num_frames', 'fps'}
STREAM_FPS = 20
JPEG_QUALITY = 80


# --------------------------------------------------------------------------- live

def open_cameras(roster):
    for key, (backend, identifier) in roster.items():
        try:
            cam = _build_camera(backend, identifier)
            CAMERAS_OPEN[key] = {'cam': cam, 'backend': backend, 'serial': identifier, 'ok': True, 'err': None}
            print(f'[camera_monitor] {key}: open ({backend} {identifier})')
        except Exception as e:  # noqa: BLE001 - keep going if one camera is missing
            CAMERAS_OPEN[key] = {'cam': None, 'backend': backend, 'serial': identifier, 'ok': False, 'err': str(e)}
            print(f'[camera_monitor] {key}: FAILED - {e}')


def close_cameras():
    for entry in CAMERAS_OPEN.values():
        if entry['cam'] is not None:
            try:
                entry['cam'].close()
            except Exception:  # noqa: BLE001
                pass


def mjpeg_frames(entry):
    period = 1.0 / STREAM_FPS
    encode_params = [cv.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
    while True:
        t0 = time.time()
        img = entry['cam'].get_image() if entry['cam'] is not None else None
        if img is not None:
            ok, buf = cv.imencode('.jpg', cv.cvtColor(img, cv.COLOR_RGB2BGR), encode_params)
            if ok:
                payload = buf.tobytes()
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n'
                       b'Content-Length: ' + str(len(payload)).encode() + b'\r\n\r\n'
                       + payload + b'\r\n')
        dt = time.time() - t0
        if dt < period:
            time.sleep(period - dt)


@app.route('/stream/<key>')
def stream(key):
    if MODE != 'live':
        abort(404)
    entry = CAMERAS_OPEN.get(key)
    if entry is None:
        return f'unknown camera: {key}', 404
    return Response(mjpeg_frames(entry),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


# ------------------------------------------------------------------------- replay

def latest_episode(root):
    root = Path(root).expanduser()
    if not root.is_dir():
        return None
    dirs = sorted((d for d in root.iterdir() if d.is_dir() and (d / 'data.pkl').exists()),
                  key=lambda d: d.name)
    return dirs[-1] if dirs else None


def load_replay(spec):
    """spec: '__latest__' or a path to an episode directory."""
    if spec == '__latest__':
        ep = latest_episode(default_data_dir())
        if ep is None:
            raise SystemExit(f'[camera_monitor] no recordings found under {default_data_dir()}')
    else:
        ep = Path(spec).expanduser()
        if not (ep / 'data.pkl').exists():
            raise SystemExit(f'[camera_monitor] not an episode directory (no data.pkl): {ep}')

    reader = EpisodeReader(ep)
    keys = [k for k, v in reader.observations[0].items() if getattr(v, 'ndim', 0) == 3]
    ts = reader.timestamps
    span = (ts[-1] - ts[0]) if len(ts) > 1 else 0.0
    fps = (len(ts) - 1) / span if span > 0.1 else float(RECORD_HZ)
    return {'reader': reader, 'dir': ep, 'keys': keys, 'num_frames': len(reader), 'fps': fps}


@app.route('/replay/frame/<key>/<int:idx>')
def replay_frame(key, idx):
    if MODE != 'replay' or key not in REPLAY['keys']:
        abort(404)
    idx = max(0, min(REPLAY['num_frames'] - 1, idx))
    img = REPLAY['reader'].observations[idx][key]        # HWC uint8 RGB
    ok, buf = cv.imencode('.jpg', cv.cvtColor(img, cv.COLOR_RGB2BGR),
                          [cv.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ok:
        abort(500)
    resp = Response(buf.tobytes(), mimetype='image/jpeg')
    resp.headers['Cache-Control'] = 'public, max-age=3600'   # frames never change
    return resp


# --------------------------------------------------------------------------- page

def page_config():
    if MODE == 'replay':
        cams = [{'key': k,
                 'location': CAMERA_INFO.get(k, (k, '', ''))[0],
                 'device': CAMERA_INFO.get(k, ('', '', ''))[1],
                 'mode': CAMERA_INFO.get(k, ('', '', ''))[2]}
                for k in REPLAY['keys']]
        return {'mode': 'replay', 'cameras': cams,
                'episode': {'path': str(REPLAY['dir']),
                            'name': REPLAY['dir'].name,
                            'num_frames': REPLAY['num_frames'],
                            'fps': round(REPLAY['fps'], 3)}}
    cams = [{'key': k,
             'location': CAMERA_INFO.get(k, (k, '', ''))[0],
             'device': CAMERA_INFO.get(k, ('', '', ''))[1],
             'mode': CAMERA_INFO.get(k, ('', '', ''))[2],
             'serial': e['serial'], 'online': e['ok'], 'error': e['err']}
            for k, e in CAMERAS_OPEN.items()]
    return {'mode': 'live', 'cameras': cams, 'episode': None}


@app.route('/status')
def status():
    return jsonify(page_config())


@app.route('/')
def index():
    return PAGE.replace('/*CONFIG_JSON*/', json.dumps(page_config()))


PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Camera Monitor</title>
<style>
  :root {
    --ground:#101317; --surface:#191d23; --surface-2:#21262d;
    --ink:#e8eae6; --muted:#8b949d; --line:rgba(255,255,255,.12);
    --line-strong:rgba(255,255,255,.24); --accent:#e39244; --accent-tint:rgba(227,146,68,.16);
    --live:#48b978;
  }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--ground); color:var(--ink);
    font-family:"IBM Plex Sans",system-ui,-apple-system,sans-serif; }
  .mono { font-family:"IBM Plex Mono",ui-monospace,SFMono-Regular,Menlo,monospace; }
  button { font:inherit; color:inherit; }

  header { display:flex; align-items:center; gap:10px; padding:12px 18px;
    border-bottom:1px solid var(--line); flex-wrap:wrap; }
  header .dot { width:9px; height:9px; border-radius:50%; background:var(--accent);
    box-shadow:0 0 0 3px var(--accent-tint); }
  header b { font-weight:600; letter-spacing:-.01em; }
  header .badge { font-family:"IBM Plex Mono",ui-monospace,monospace; font-size:10.5px;
    letter-spacing:.1em; padding:2px 8px; border-radius:5px; background:var(--accent-tint);
    color:var(--accent); }
  header .epath { font-family:"IBM Plex Mono",ui-monospace,monospace; font-size:11px;
    color:var(--muted); overflow:hidden; text-overflow:ellipsis; white-space:nowrap;
    max-width:52vw; }
  header .clock { margin-left:auto; font-size:12px; color:var(--muted);
    font-family:"IBM Plex Mono",ui-monospace,monospace; }

  main { max-width:1200px; margin:0 auto; padding:18px; }
  .grid { display:grid; grid-template-columns:2.1fr 1fr; gap:14px; }
  @media (max-width:820px){ .grid { grid-template-columns:1fr; } }

  .primary { border:1px solid var(--line-strong); border-radius:12px; overflow:hidden;
    background:var(--surface); }
  .frame { position:relative; background:#0b0d10; aspect-ratio:16/10; display:flex;
    align-items:center; justify-content:center; }
  .frame img { width:100%; height:100%; object-fit:contain; display:block; }
  .frame .offline { color:var(--muted); font-size:13px; padding:20px; text-align:center; }
  .pill { position:absolute; top:11px; right:11px; display:inline-flex; align-items:center;
    gap:6px; font-size:10.5px; letter-spacing:.08em; padding:3px 8px; border-radius:6px;
    background:rgba(10,12,14,.66); color:var(--live);
    font-family:"IBM Plex Mono",ui-monospace,monospace; }
  .pill::before { content:""; width:7px; height:7px; border-radius:50%; background:var(--live);
    animation:pulse 2s ease-in-out infinite; }
  .pill.down { color:var(--muted); } .pill.down::before { background:var(--muted); animation:none; }
  .pill.past { color:var(--accent); } .pill.past::before { background:var(--accent); animation:none; }
  @keyframes pulse { 0%,100%{opacity:1;} 50%{opacity:.35;} }
  @media (prefers-reduced-motion:reduce){ .pill::before{ animation:none; } }

  .caption { display:flex; align-items:center; gap:10px; flex-wrap:wrap;
    padding:11px 14px; border-top:1px solid var(--line); font-size:13px; }
  .caption .loc { font-weight:600; }
  .caption .sep { color:var(--line-strong); }
  .caption .mono { font-size:12.5px; color:var(--muted); }

  .rail { display:flex; flex-direction:column; gap:14px; }
  @media (max-width:820px){ .rail { flex-direction:row; } .rail button { flex:1; } }
  .thumb { border:1px solid var(--line); border-radius:12px; overflow:hidden; background:var(--surface);
    cursor:pointer; padding:0; text-align:left; transition:border-color .15s, box-shadow .15s; }
  .thumb:hover { border-color:var(--accent); }
  .thumb:focus-visible { outline:2px solid var(--accent); outline-offset:2px; }
  .thumb .frame { aspect-ratio:16/10; }
  .thumb .tmeta { padding:8px 10px; display:flex; justify-content:space-between; align-items:center; gap:8px; }
  .thumb .tmeta b { font-size:12.5px; font-weight:600; }
  .thumb .tmeta span { font-size:10.5px; color:var(--muted);
    font-family:"IBM Plex Mono",ui-monospace,monospace; }
  .thumb .chip { position:absolute; top:7px; left:7px; font-size:10px; padding:2px 6px;
    border-radius:5px; background:rgba(10,12,14,.66); color:#f4f4f2;
    font-family:"IBM Plex Mono",ui-monospace,monospace; }

  /* transport (replay only) */
  .transport { margin-top:14px; border:1px solid var(--line); border-radius:12px;
    background:var(--surface); padding:12px 14px; display:flex; align-items:center; gap:14px; }
  .transport button.pp {
    flex-shrink:0; width:40px; height:40px; border-radius:50%; border:1px solid var(--line-strong);
    background:var(--accent); color:#1a1207; font-size:13px; font-weight:600; cursor:pointer;
    display:flex; align-items:center; justify-content:center;
  }
  .transport button.pp:hover { filter:brightness(1.08); }
  .transport input[type=range] {
    flex:1; accent-color:var(--accent); height:4px; cursor:pointer;
  }
  .transport .time { flex-shrink:0; font-family:"IBM Plex Mono",ui-monospace,monospace;
    font-size:12px; color:var(--muted); font-variant-numeric:tabular-nums; min-width:150px; text-align:right; }
  .transport .buf { flex-shrink:0; font-family:"IBM Plex Mono",ui-monospace,monospace;
    font-size:11px; color:var(--accent); }

  footer { max-width:1200px; margin:0 auto; padding:14px 18px; color:var(--muted); font-size:12px;
    word-break:break-all; }
</style>
</head>
<body>
<header>
  <span class="dot" id="dot"></span>
  <b>Camera Monitor</b>
  <span class="badge" id="badge" hidden>REPLAY</span>
  <span class="epath" id="epath" hidden></span>
  <span class="clock" id="clock"></span>
</header>
<main>
  <div class="grid">
    <div class="primary">
      <div class="frame" id="primary-frame"></div>
      <div class="caption" id="primary-caption"></div>
    </div>
    <div class="rail" id="rail"></div>
  </div>
  <div class="transport" id="transport" hidden>
    <button class="pp" id="pp" aria-label="Play / pause">Play</button>
    <input type="range" id="scrub" min="0" max="0" value="0" step="1" aria-label="Timeline">
    <span class="buf" id="buf" hidden></span>
    <span class="time" id="time">0.0 / 0.0 s</span>
  </div>
</main>
<footer id="foot"></footer>

<script>
  const CONFIG = /*CONFIG_JSON*/;
  const CAMS = CONFIG.cameras;
  const REPLAY = CONFIG.mode === 'replay';
  let selected = (CAMS.find(c => c.online !== false) || CAMS[0] || {}).key;

  // ---- per-camera image / offline nodes (persistent; moving them in the DOM
  //      does not reload them) --------------------------------------------------
  const nodes = {};
  for (const cam of CAMS) {
    if (!REPLAY && cam.online === false) {
      const d = document.createElement('div');
      d.className = 'offline';
      d.textContent = 'offline — ' + (cam.error || 'not connected');
      nodes[cam.key] = d;
    } else {
      const img = document.createElement('img');
      img.alt = cam.location;
      if (!REPLAY) img.src = '/stream/' + cam.key;   // live: MJPEG stream
      nodes[cam.key] = img;
    }
  }

  // ---- replay playback (requestAnimationFrame wall-clock player) --------------
  // The displayed frame is always floor(elapsed * fps), so after any stall
  // (backgrounded tab, slow network) it shows the CORRECT current frame instead
  // of replaying a queued backlog. Frames are preloaded into the browser cache,
  // so showFrame() never waits on the network.
  const NF = REPLAY ? CONFIG.episode.num_frames : 0;
  const FPS = REPLAY ? (CONFIG.episode.fps || 10) : 0;
  const frameURL = (k, i) => '/replay/frame/' + k + '/' + i;
  let frame = -1, playing = false, raf = 0, startWall = 0, startFrame = 0;

  function showFrame(i) {
    i = Math.max(0, Math.min(NF - 1, Math.round(i)));
    if (i === frame) return;
    frame = i;
    for (const cam of CAMS) {
      const n = nodes[cam.key];
      if (n.tagName === 'IMG') n.src = frameURL(cam.key, i);   // cached -> instant
    }
    const scrub = document.getElementById('scrub');
    if (+scrub.value !== i) scrub.value = i;
    const total = (NF - 1) / FPS;
    document.getElementById('time').textContent =
      (i / FPS).toFixed(1) + ' / ' + total.toFixed(1) + ' s   ·   ' + (i + 1) + ' / ' + NF;
  }
  function loop() {
    if (!playing) return;
    let target = startFrame + Math.floor((performance.now() - startWall) / 1000 * FPS);
    if (target >= NF) { startWall = performance.now(); startFrame = 0; target = 0; }  // loop
    showFrame(target);
    raf = requestAnimationFrame(loop);
  }
  function play() {
    if (playing || NF === 0) return;
    playing = true;
    document.getElementById('pp').textContent = 'Pause';
    startWall = performance.now();
    startFrame = (frame < 0 || frame >= NF - 1) ? 0 : frame;
    raf = requestAnimationFrame(loop);
  }
  function pause() {
    if (!playing) return;
    playing = false;
    document.getElementById('pp').textContent = 'Play';
    cancelAnimationFrame(raf);
  }
  function seek(i) { pause(); showFrame(i); }

  function preload() {
    const buf = document.getElementById('buf');
    const total = NF * CAMS.length;
    let done = 0;
    const bump = () => {
      done++;
      if (done >= total) buf.hidden = true;
      else { buf.hidden = false; buf.textContent = 'buffering ' + Math.round(100 * done / total) + '%'; }
    };
    for (const cam of CAMS)
      for (let i = 0; i < NF; i++) { const im = new Image(); im.onload = im.onerror = bump; im.src = frameURL(cam.key, i); }
  }

  // ---- render (Pilot View: one big + thumbnails) -----------------------------
  function render() {
    const cam = CAMS.find(c => c.key === selected) || CAMS[0];

    const pf = document.getElementById('primary-frame');
    pf.replaceChildren(nodes[cam.key]);
    const pill = document.createElement('span');
    if (REPLAY) { pill.className = 'pill past'; pill.textContent = 'RECORDING'; }
    else if (cam.online === false) { pill.className = 'pill down'; pill.textContent = 'DOWN'; }
    else { pill.className = 'pill'; pill.textContent = 'LIVE'; }
    pf.appendChild(pill);

    const cap = document.getElementById('primary-caption');
    cap.innerHTML = '<span class="loc"></span><span class="sep">—</span><span class="mono d"></span>'
      + '<span class="sep">·</span><span class="mono s"></span>';
    cap.querySelector('.loc').textContent = cam.location;
    cap.querySelector('.d').textContent = cam.device;
    cap.querySelector('.s').textContent = REPLAY ? cam.key : cam.serial;

    const rail = document.getElementById('rail');
    rail.replaceChildren();
    CAMS.filter(c => c.key !== cam.key).forEach(c => {
      const b = document.createElement('button');
      b.className = 'thumb';
      b.onclick = () => { selected = c.key; render(); };
      const fr = document.createElement('div');
      fr.className = 'frame';
      fr.appendChild(nodes[c.key]);
      const chip = document.createElement('span');
      chip.className = 'chip';
      chip.textContent = c.location;
      fr.appendChild(chip);
      const meta = document.createElement('div');
      meta.className = 'tmeta';
      const bb = document.createElement('b'); bb.textContent = c.location;
      const sp = document.createElement('span'); sp.textContent = REPLAY ? c.key : c.serial;
      meta.append(bb, sp);
      b.append(fr, meta);
      rail.appendChild(b);
    });
  }

  // ---- wire up --------------------------------------------------------------
  if (REPLAY) {
    document.getElementById('dot').style.background = 'var(--accent)';
    document.getElementById('badge').hidden = false;
    const ep = document.getElementById('epath');
    ep.hidden = false; ep.textContent = CONFIG.episode.path; ep.title = CONFIG.episode.path;
    document.getElementById('clock').textContent =
      NF + ' frames · ' + FPS.toFixed(1) + ' fps';

    const t = document.getElementById('transport');
    t.hidden = false;
    const scrub = document.getElementById('scrub');
    scrub.max = String(NF - 1);
    document.getElementById('pp').onclick = () => (playing ? pause() : play());
    scrub.addEventListener('input', () => seek(+scrub.value));
    document.addEventListener('keydown', e => {
      if (e.key === ' ') { e.preventDefault(); playing ? pause() : play(); }
      else if (e.key === 'ArrowRight') seek(frame + 1);
      else if (e.key === 'ArrowLeft') seek(frame - 1);
    });
    // tab was hidden -> rAF paused -> re-anchor so playback resumes from the
    // frame on screen, not wherever wall-clock drifted to
    document.addEventListener('visibilitychange', () => {
      if (!document.hidden && playing) { startWall = performance.now(); startFrame = frame; }
    });

    render();
    showFrame(0);
    preload();
    play();
  } else {
    document.getElementById('foot').textContent =
      CAMS.length + ' camera(s) · ' + CAMS.filter(c => c.online !== false).length
      + ' online · click a thumbnail to enlarge it';
    setInterval(() => {
      document.getElementById('clock').textContent = new Date().toLocaleTimeString();
    }, 1000);
    render();
  }
</script>
</body>
</html>
"""


def parse_camera_specs(specs):
    roster = {}
    for spec in specs:
        key, rest = spec.split('=', 1)
        backend, identifier = rest.split(':', 1)
        roster[key] = (backend, identifier)
    return roster


def main():
    global STREAM_FPS, JPEG_QUALITY, MODE, REPLAY
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--fps', type=int, default=STREAM_FPS, help='max live stream fps per camera')
    parser.add_argument('--quality', type=int, default=JPEG_QUALITY, help='JPEG quality 1-100')
    parser.add_argument('--cameras', nargs='*', default=None,
                        help='live only: override configured cameras, e.g. base_image=logitech:DAA051BE')
    parser.add_argument('--replay', nargs='?', const='__latest__', default=None, metavar='EPISODE_DIR',
                        help='replay a recording instead of the live cameras; '
                             'pass an episode directory, or omit for the most recent recording')
    args = parser.parse_args()

    STREAM_FPS, JPEG_QUALITY = args.fps, args.quality

    if args.replay is not None:
        MODE = 'replay'
        REPLAY = load_replay(args.replay)
        print(f'[camera_monitor] replay: {REPLAY["dir"]}')
        print(f'[camera_monitor]   {REPLAY["num_frames"]} frames, {REPLAY["fps"]:.1f} fps, '
              f'cameras: {REPLAY["keys"]}')
    else:
        roster = parse_camera_specs(args.cameras) if args.cameras else CAMERAS
        open_cameras(roster)
        online = [k for k, e in CAMERAS_OPEN.items() if e['ok']]
        print(f'[camera_monitor] live: {len(online)}/{len(CAMERAS_OPEN)} cameras online: {online}')

    print(f'[camera_monitor] open http://<this-host>:{args.port}  '
          f'(e.g. sixsevensupremacy.local:{args.port})')

    try:
        app.run(host=args.host, port=args.port, threaded=True)
    finally:
        if MODE == 'live':
            close_cameras()


if __name__ == '__main__':
    main()
