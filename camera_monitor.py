# Live camera monitor -- "Pilot View" layout.
#
# Serves an MJPEG stream per camera plus a web page that shows one feed large
# with the others as thumbnails (click a thumbnail to promote it). Read-only:
# opens the cameras and streams frames, never touches the base or arms.
#
#   python camera_monitor.py                       # all cameras in constants.CAMERAS
#   python camera_monitor.py --port 5000
#   python camera_monitor.py --cameras base_image=logitech:DAA051BE
#
# Then from any machine on the same network:  http://sixsevensupremacy.local:8000

import argparse
import time

import cv2 as cv
from flask import Flask, Response, jsonify

from cameras import _build_camera
from constants import CAMERAS

# Display metadata (obs_key -> (location label, device name, capture mode)).
# Serial comes from the CAMERAS roster; this is just what the page shows.
CAMERA_INFO = {
    'base_image':        ('Base · front', 'Logitech C930e', '640×360 · 30 fps'),
    'left_wrist_image':  ('Left wrist',        'RealSense D405',  '640×480 · 30 fps'),
    'right_wrist_image': ('Right wrist',       'RealSense D405',  '640×480 · 30 fps'),
}

app = Flask(__name__)
CAMERAS_OPEN = {}   # obs_key -> {'cam', 'backend', 'serial', 'ok', 'err'}
STREAM_FPS = 20
JPEG_QUALITY = 80


def open_cameras(roster):
    for key, (backend, identifier) in roster.items():
        try:
            cam = _build_camera(backend, identifier)
            CAMERAS_OPEN[key] = {'cam': cam, 'backend': backend, 'serial': identifier, 'ok': True, 'err': None}
            print(f'[camera_monitor] {key}: open ({backend} {identifier})')
        except Exception as e:  # noqa: BLE001 - want to keep going if one camera is missing
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
    entry = CAMERAS_OPEN.get(key)
    if entry is None:
        return f'unknown camera: {key}', 404
    return Response(mjpeg_frames(entry),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/status')
def status():
    return jsonify({
        key: {'location': CAMERA_INFO.get(key, (key, '', ''))[0],
              'device': CAMERA_INFO.get(key, ('', '', ''))[1],
              'mode': CAMERA_INFO.get(key, ('', '', ''))[2],
              'serial': e['serial'], 'backend': e['backend'],
              'online': e['ok'], 'error': e['err']}
        for key, e in CAMERAS_OPEN.items()
    })


@app.route('/')
def index():
    cams = [
        {'key': key,
         'location': CAMERA_INFO.get(key, (key, '', ''))[0],
         'device': CAMERA_INFO.get(key, ('', '', ''))[1],
         'mode': CAMERA_INFO.get(key, ('', '', ''))[2],
         'serial': e['serial'], 'backend': e['backend'],
         'online': e['ok'], 'error': e['err']}
        for key, e in CAMERAS_OPEN.items()
    ]
    import json
    return PAGE.replace('/*CAMERAS_JSON*/', json.dumps(cams))


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
  header { display:flex; align-items:center; gap:10px; padding:12px 18px;
    border-bottom:1px solid var(--line); }
  header .rec { width:9px; height:9px; border-radius:50%; background:var(--accent);
    box-shadow:0 0 0 3px var(--accent-tint); }
  header b { font-weight:600; letter-spacing:-.01em; }
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
    cursor:pointer; padding:0; font:inherit; color:inherit; text-align:left;
    transition:border-color .15s, box-shadow .15s; }
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

  footer { max-width:1200px; margin:0 auto; padding:14px 18px; color:var(--muted); font-size:12px; }
</style>
</head>
<body>
<header>
  <span class="rec"></span><b>Camera Monitor</b>
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
</main>
<footer id="foot"></footer>

<script>
  const CAMS = /*CAMERAS_JSON*/;
  let selected = (CAMS.find(c => c.online) || CAMS[0] || {}).key;

  function streamNode(cam) {
    if (!cam.online) {
      const d = document.createElement('div');
      d.className = 'offline';
      d.textContent = 'offline — ' + (cam.error || 'not connected');
      return d;
    }
    const img = document.createElement('img');
    img.alt = cam.location;
    img.src = '/stream/' + cam.key + '?t=' + Date.now();
    return img;
  }

  function render() {
    const cam = CAMS.find(c => c.key === selected) || CAMS[0];

    // primary
    const pf = document.getElementById('primary-frame');
    pf.innerHTML = '';
    pf.appendChild(streamNode(cam));
    const pill = document.createElement('span');
    pill.className = 'pill' + (cam.online ? '' : ' down');
    pill.textContent = cam.online ? 'LIVE' : 'DOWN';
    pf.appendChild(pill);

    document.getElementById('primary-caption').innerHTML =
      '<span class="loc">' + cam.location + '</span>' +
      '<span class="sep">—</span>' +
      '<span class="mono">' + cam.device + '</span>' +
      '<span class="sep">·</span>' +
      '<span class="mono">' + cam.serial + '</span>' +
      '<span class="sep">·</span>' +
      '<span class="mono">' + cam.mode + '</span>';

    // rail: the other cameras
    const rail = document.getElementById('rail');
    rail.innerHTML = '';
    CAMS.filter(c => c.key !== cam.key).forEach(c => {
      const b = document.createElement('button');
      b.className = 'thumb';
      b.onclick = () => { selected = c.key; render(); };
      const fr = document.createElement('div');
      fr.className = 'frame';
      fr.appendChild(streamNode(c));
      const chip = document.createElement('span');
      chip.className = 'chip';
      chip.textContent = c.location;
      fr.appendChild(chip);
      const meta = document.createElement('div');
      meta.className = 'tmeta';
      meta.innerHTML = '<b>' + c.location + '</b><span>' + c.serial + '</span>';
      b.appendChild(fr); b.appendChild(meta);
      rail.appendChild(b);
    });
  }

  document.getElementById('foot').textContent =
    CAMS.length + ' camera(s) · ' + CAMS.filter(c => c.online).length + ' online · '
    + 'click a thumbnail to enlarge it';

  setInterval(() => {
    document.getElementById('clock').textContent = new Date().toLocaleTimeString();
  }, 1000);

  render();
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
    global STREAM_FPS, JPEG_QUALITY
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--fps', type=int, default=STREAM_FPS, help='max stream fps per camera')
    parser.add_argument('--quality', type=int, default=JPEG_QUALITY, help='JPEG quality 1-100')
    parser.add_argument('--cameras', nargs='*', default=None,
                        help='override constants.CAMERAS, e.g. base_image=logitech:DAA051BE')
    args = parser.parse_args()

    STREAM_FPS, JPEG_QUALITY = args.fps, args.quality

    roster = parse_camera_specs(args.cameras) if args.cameras else CAMERAS
    open_cameras(roster)
    online = [k for k, e in CAMERAS_OPEN.items() if e['ok']]
    print(f'[camera_monitor] {len(online)}/{len(CAMERAS_OPEN)} cameras online: {online}')
    print(f'[camera_monitor] open http://<this-host>:{args.port}  (e.g. sixsevensupremacy.local:{args.port})')

    try:
        app.run(host=args.host, port=args.port, threaded=True)
    finally:
        close_cameras()


if __name__ == '__main__':
    main()
