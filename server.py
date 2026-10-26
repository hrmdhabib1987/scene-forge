#!/usr/bin/env python3
"""Scene-by-scene AI video studio backend.
Python 3.9+, needs requests. Optional: edge-tts, ffmpeg.
Free experimental video: public Hugging Face Wan 2.2 Space.
"""
import os, json, time, uuid, threading, subprocess, shutil, base64, mimetypes, re, urllib.parse
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from pathlib import Path
import requests

ROOT = Path(__file__).parent
if (ROOT/'.env').exists():
    for l in (ROOT/'.env').read_text().splitlines():
        if '=' in l and not l.strip().startswith('#'):
            k, v = l.split('=', 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"\''))
PROJ = ROOT/'projects'
PROJ.mkdir(exist_ok=True)
K = lambda n: os.environ.get(n, '').strip()
JOBS = {}

class ProviderError(Exception):
    pass

def safe(s):
    return re.sub(r'[^a-zA-Z0-9_-]', '', str(s))[:40] or 'x'

def pdir(p):
    d = PROJ/safe(p)
    d.mkdir(exist_ok=True)
    return d

def setp(job, msg, pct=None):
    job['msg'] = msg
    job['pct'] = pct if pct is not None else job.get('pct', 0)

def http(method, url, **kw):
    kw.setdefault('timeout', 120)
    try:
        r = requests.request(method, url, **kw)
    except requests.RequestException as e:
        raise ProviderError(f'Network error contacting {urllib.parse.urlparse(url).netloc}: {e}')
    if r.status_code >= 400:
        raise ProviderError(f'{urllib.parse.urlparse(url).netloc} returned HTTP {r.status_code}: {r.text[:300]}')
    return r

def data_uri(path):
    return 'data:image/png;base64,' + base64.b64encode(Path(path).read_bytes()).decode()

# ---------------- IMAGE ----------------
def img_pollinations(p, job):
    q = urllib.parse.quote(p['prompt'][:1500])
    url = f"https://image.pollinations.ai/prompt/{q}?width={p['w']}&height={p['h']}&seed={p['seed']}&model=flux&nologo=true"
    r = http('GET', url, timeout=180)
    if not r.headers.get('content-type', '').startswith('image'):
        raise ProviderError('Pollinations did not return an image: ' + r.text[:200])
    return r.content

def img_hf(p, job):
    if not K('HF_TOKEN'):
        raise ProviderError('HF_TOKEN is not set on the server (free token from huggingface.co/settings/tokens).')
    r = http(
        'POST',
        'https://router.huggingface.co/hf-inference/models/black-forest-labs/FLUX.1-schnell',
        headers={'Authorization': 'Bearer ' + K('HF_TOKEN')},
        json={'inputs': p['prompt'], 'parameters': {
            'width': p['w'], 'height': p['h'], 'seed': p['seed']
        }},
        timeout=180
    )
    if not r.headers.get('content-type', '').startswith('image'):
        raise ProviderError('Hugging Face did not return an image: ' + r.text[:200])
    return r.content

def fal_queue(model, payload, job, timeout=900):
    if not K('FAL_KEY'):
        raise ProviderError('FAL_KEY is not set on the server (fal.ai is pay-per-use / credits).')
    h = {'Authorization': 'Key ' + K('FAL_KEY')}
    s = http('POST', 'https://queue.fal.run/' + model, headers=h, json=payload).json()
    t0 = time.time()
    while True:
        st = http('GET', s['status_url'], headers=h).json()
        if st.get('status') == 'COMPLETED':
            break
        if st.get('status') in ('FAILED', 'ERROR'):
            raise ProviderError('fal.ai job failed: ' + json.dumps(st)[:300])
        if time.time()-t0 > timeout:
            raise ProviderError('fal.ai timed out')
        setp(job, f"fal.ai {st.get('status','queued')}… {int(time.time()-t0)}s")
        time.sleep(3)
    return http('GET', s['response_url'], headers=h).json()

def img_fal(p, job):
    j = fal_queue('fal-ai/flux/schnell', {
        'prompt': p['prompt'],
        'image_size': {'width': p['w'], 'height': p['h']},
        'seed': p['seed']
    }, job)
    return http('GET', j['images'][0]['url']).content

IMG = {'pollinations': img_pollinations, 'huggingface': img_hf, 'fal': img_fal}

# ---------------- VIDEO ----------------
FAL_VIDEO = {
    'ltx': ('fal-ai/ltx-video/image-to-video', 'fal-ai/ltx-video'),
    'kling': ('fal-ai/kling-video/v1.6/standard/image-to-video',
              'fal-ai/kling-video/v1.6/standard/text-to-video'),
    'wan': ('fal-ai/wan-i2v', 'fal-ai/wan-t2v')
}

def vid_fal(p, job, img):
    i2v, t2v = FAL_VIDEO[p.get('model', 'ltx')]
    pl = {'prompt': p['prompt']}
    if p.get('model') == 'kling':
        pl.update(duration=str(5 if p['duration'] <= 5 else 10),
                  aspect_ratio=p['aspect'])
    if p['mode'] == 'i2v':
        if not img or not Path(img).exists():
            raise ProviderError('Image-to-video needs a generated image for this scene first.')
        pl['image_url'] = data_uri(img)
        model = i2v
    else:
        model = t2v
    j = fal_queue(model, pl, job)
    return http('GET', j['video']['url'], timeout=300).content

def vid_replicate(p, job, img):
    if not K('REPLICATE_API_TOKEN'):
        raise ProviderError('REPLICATE_API_TOKEN is not set on the server (Replicate is pay-per-use).')
    h = {
        'Authorization': 'Bearer ' + K('REPLICATE_API_TOKEN'),
        'Prefer': 'wait'
    }
    inp = {'prompt': p['prompt']}
    if p['mode'] == 'i2v':
        if not img or not Path(img).exists():
            raise ProviderError('Image-to-video needs a generated image first.')
        inp['image'] = data_uri(img)
    r = http(
        'POST',
        'https://api.replicate.com/v1/models/lightricks/ltx-video/predictions',
        headers=h, json={'input': inp}, timeout=180
    ).json()
    t0 = time.time()
    while r.get('status') not in ('succeeded', 'failed', 'canceled'):
        if time.time()-t0 > 900:
            raise ProviderError('Replicate timed out')
        setp(job, f"Replicate {r.get('status')}… {int(time.time()-t0)}s")
        time.sleep(3)
        r = http('GET', r['urls']['get'], headers=h).json()
    if r['status'] != 'succeeded':
        raise ProviderError('Replicate failed: ' + str(r.get('error'))[:300])
    out = r['output']
    out = out[0] if isinstance(out, list) else out
    return http('GET', out, timeout=300).content

# Free experimental provider. Uses a public Hugging Face Space,
# not the paid Hugging Face Inference API. No token or paid API is used.
HF_VIDEO_SPACE = 'https://cbensimon-wan2-2-fp8da-aoti-preview2.hf.space'

def _hf_video_file(obj):
    """Find a video file reference in Gradio's nested response."""
    if isinstance(obj, dict):
        for key in ('video', 'path', 'url'):
            val = obj.get(key)
            if isinstance(val, str) and (
                val.lower().split('?')[0].endswith(('.mp4', '.webm', '.mov'))
                or key in ('path', 'url') and (
                    '/gradio_api/file=' in val or val.startswith('http')
                )
            ):
                return val
        for val in obj.values():
            found = _hf_video_file(val)
            if found:
                return found
    elif isinstance(obj, list):
        for val in obj:
            found = _hf_video_file(val)
            if found:
                return found
    return None

def vid_huggingface_free(p, job, img):
    """Call the public Wan 2.2 Space through its official Gradio client."""
    if p.get('mode') != 'i2v':
        raise ProviderError('The free Wan 2.2 option requires Image → video mode.')
    if not img or not Path(img).exists():
        raise ProviderError('Generate this scene image through the app first.')

    try:
        from gradio_client import Client, handle_file
    except ImportError:
        raise ProviderError('Missing gradio_client dependency. Add gradio_client to requirements.txt and redeploy Render.')

    duration = max(0.5, min(float(p.get('duration', 3)), 10.0))
    prompt = str(p.get('prompt', 'A gentle cinematic movement'))[:1500]
    setp(job, 'Connecting to free Wan 2.2 GPU…', 5)
    try:
        # Gradio Client handles image upload, queue events, and output file download.
        client = Client('cbensimon/wan2-2-fp8da-aoti-preview2', verbose=False)
        setp(job, 'Submitting image to free Wan 2.2…', 15)
        result_data = client.predict(
            handle_file(str(Path(img).resolve())),  # input_image: Gradio ImageData upload
            None,                      # last_image
            prompt,                    # prompt
            4,                         # steps
            'static image, blurry, low quality, subtitles, watermark',
            duration,                  # duration_seconds (Space supports up to 10s)
            1,                         # guidance_scale
            1,                         # guidance_scale_2
            42,                        # seed
            True,                      # randomize_seed
            6,                         # quality
            'UniPCMultistep',           # scheduler
            3.0,                       # flow_shift
            24,                        # frame_multiplier
            True,                      # safe_mode (requests extra GPU time)
            True,                      # video_component
            api_name='/generate_video'
        )
        setp(job, 'Downloading generated video…', 90)

        # The endpoint returns (video preview, downloadable file, seed). Prefer the
        # downloadable file output, but accept the preview output as a fallback.
        candidates = []
        if isinstance(result_data, (list, tuple)):
            if len(result_data) > 1:
                candidates.append(result_data[1])
            if result_data:
                candidates.append(result_data[0])
        else:
            candidates.append(result_data)

        video_ref = None
        for candidate in candidates:
            video_ref = _hf_video_file(candidate)
            if not video_ref and isinstance(candidate, str) and candidate.lower().split('?')[0].endswith(('.mp4', '.webm', '.mov')):
                video_ref = candidate
            if video_ref:
                break
        if not video_ref:
            raise ProviderError('Wan 2.2 finished but returned no readable video file. Check the Render logs for the Gradio response.')

        local = Path(video_ref)
        if local.exists() and local.is_file():
            content = local.read_bytes()
        elif video_ref.startswith(('http://', 'https://')):
            content = http('GET', video_ref, timeout=300).content
        else:
            video_url = HF_VIDEO_SPACE + '/gradio_api/file=' + urllib.parse.quote(video_ref, safe='/')
            content = http('GET', video_url, timeout=300).content

        if not content or (len(content) < 1000):
            raise ProviderError('Wan 2.2 returned an empty or incomplete video file.')
        if b'ftyp' not in content[:64]:
            raise ProviderError('Wan 2.2 result is not a valid MP4 video.')
        return content
    except ProviderError:
        raise
    except Exception as e:
        raise ProviderError('Free Wan 2.2 connection failed: ' + f'{type(e).__name__}: {e}'[:300])

VID = {
    'fal': vid_fal,
    'replicate': vid_replicate,
    'huggingface-free': vid_huggingface_free
}

# ---------------- AUDIO ----------------
def run(cmd, **kw):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, **kw)
    except FileNotFoundError:
        raise ProviderError(f'{cmd[0]} is not installed on the server.')
    if r.returncode:
        raise ProviderError(f'{cmd[0]} failed: {r.stderr[-400:]}')
    return r

def tts(p, out):
    eng = p.get('engine', 'edge')
    if eng == 'edge':
        try:
            run([
                'python3', '-m', 'edge_tts', '--voice', p['voice'],
                '--text', p['text'], '--write-media', str(out)
            ], timeout=120)
        except ProviderError as e:
            raise ProviderError('Edge-TTS unavailable (pip install edge-tts, needs internet; unofficial free service): ' + str(e)[:200])
    else:
        wav = out.with_suffix('.wav')
        run(['espeak-ng', '-v', p.get('lang', 'en'), '-w', str(wav), p['text']], timeout=120)
        run(['ffmpeg', '-y', '-i', str(wav), str(out)])
        wav.unlink(missing_ok=True)

def music_hf(p, out):
    if not K('HF_TOKEN'):
        raise ProviderError('HF_TOKEN not set.')
    r = http(
        'POST',
        'https://router.huggingface.co/hf-inference/models/facebook/musicgen-small',
        headers={'Authorization': 'Bearer ' + K('HF_TOKEN')},
        json={'inputs': p['prompt']}, timeout=240
    )
    if not r.headers.get('content-type', '').startswith('audio'):
        raise ProviderError('MusicGen not available on HF free inference right now: ' + r.text[:200])
    out.write_bytes(r.content)

def dur(path):
    r = run([
        'ffprobe', '-v', 'error', '-show_entries', 'format=duration',
        '-of', 'csv=p=0', str(path)
    ])
    try:
        return float(r.stdout.strip())
    except ValueError:
        return 0.0

def has_audio(path):
    return bool(run([
        'ffprobe', '-v', 'error', '-select_streams', 'a',
        '-show_entries', 'stream=index', '-of', 'csv=p=0', str(path)
    ]).stdout.strip())

def render(p, job):
    d = pdir(p['project'])
    W, H = p['w'], p['h']
    n = p['count']
    missing = [
        i+1 for i in range(n)
        if not (d/f'scene_{i+1:03d}.mp4').exists()
    ]
    if missing and not p.get('skip_missing'):
        raise ProviderError(
            f'{len(missing)} scene(s) have no generated video: {missing[:30]}. '
            'Generate them (or tick "skip scenes without video") — stills are never substituted.'
        )
    parts = []
    idxs = [i for i in range(1, n+1) if i not in missing]
    if not idxs:
        raise ProviderError('No scene videos exist yet.')
    for k, i in enumerate(idxs):
        setp(job, f'Encoding scene {i}/{n}', int(80*k/len(idxs)))
        v = d/f'scene_{i:03d}.mp4'
        a = d/f'scene_{i:03d}.mp3'
        o = d/f'_part{i:03d}.mp4'
        vd = dur(v)
        ad = dur(a) if a.exists() else 0
        T = max(vd, ad, 0.5)
        vf = f'scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},fps=24,tpad=stop_mode=clone:stop_duration={max(0, T-vd):.2f},format=yuv420p'
        cmd = ['ffmpeg', '-y', '-i', str(v)]
        cmd += ['-i', str(a)] if a.exists() else ['-f', 'lavfi', '-i', 'anullsrc=r=44100:cl=stereo']
        cmd += [
            '-vf', vf, '-af', 'aresample=44100,apad', '-ac', '2',
            '-map', '0:v', '-map', '1:a', '-t', f'{T:.2f}',
            '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '21',
            '-c:a', 'aac', '-b:a', '160k', str(o)
        ]
        run(cmd)
        parts.append(o)
    lst = d/'_list.txt'
    lst.write_text(''.join(f"file '{x.name}'\n" for x in parts))
    setp(job, 'Joining scenes', 85)
    joined = d/'_joined.mp4'
    run(['ffmpeg', '-y', '-f', 'concat', '-safe', '0', '-i', str(lst), '-c', 'copy', str(joined)])
    final = d/'final.mp4'
    mus = d/'music.mp3'
    sfx = d/'sfx.mp3'
    ins = ['-i', str(joined)]
    fl = ['[0:a]volume=1.0[a0]']
    labels = ['[a0]']
    if mus.exists() and p.get('music', True):
        ins += ['-stream_loop', '-1', '-i', str(mus)]
        fl.append(f'[{len(labels)}:a]volume={p.get("music_vol",0.15)}[am]')
        labels.append('[am]')
    if sfx.exists() and p.get('sfx', True):
        ins += ['-i', str(sfx)]
        fl.append(f'[{len(labels)}:a]volume=0.5[as]')
        labels.append('[as]')
    if len(labels) > 1:
        setp(job, 'Mixing music / SFX', 92)
        fl.append(''.join(labels) + f'amix=inputs={len(labels)}:duration=first:normalize=0[aout]')
        run([
            'ffmpeg', '-y'
        ] + ins + [
            '-filter_complex', ';'.join(fl), '-map', '0:v',
            '-map', '[aout]', '-c:v', 'copy', '-c:a', 'aac',
            '-shortest', str(final)
        ])
    else:
        shutil.copy(joined, final)
    for x in parts + [lst, joined]:
        x.unlink(missing_ok=True)
    return {
        'url': f'/files/{safe(p["project"])}/final.mp4',
        'skipped': missing,
        'seconds': round(dur(final), 1)
    }

def work(job, typ, p):
    d = pdir(p['project'])
    i = p.get('index', 0)
    if typ == 'image':
        fn = IMG.get(p['provider'])
        if not fn:
            raise ProviderError('Unknown image provider')
        setp(job, 'Generating image…', 10)
        b = fn(p, job)
        f = d/f'scene_{i:03d}.png'
        f.write_bytes(b)
        return {'url': f'/files/{safe(p["project"])}/{f.name}?t={int(time.time())}'}
    if typ == 'video':
        fn = VID.get(p['provider'])
        if not fn:
            raise ProviderError('Unknown video provider')
        setp(job, 'Submitting video job…', 10)
        b = fn(p, job, d/f'scene_{i:03d}.png')
        f = d/f'scene_{i:03d}.mp4'
        f.write_bytes(b)
        return {'url': f'/files/{safe(p["project"])}/{f.name}?t={int(time.time())}'}
    if typ == 'tts':
        f = d/f'scene_{i:03d}.mp3'
        setp(job, 'Synthesising voice…', 20)
        tts(p, f)
        return {
            'url': f'/files/{safe(p["project"])}/{f.name}?t={int(time.time())}',
            'seconds': round(dur(f), 1) if shutil.which('ffprobe') else None
        }
    if typ == 'music':
        f = d/'music.mp3'
        setp(job, 'Generating music…', 20)
        music_hf(p, f)
        return {'url': f'/files/{safe(p["project"])}/music.mp3'}
    if typ == 'render':
        return render(p, job)
    raise ProviderError('Unknown job type')

def start(typ, p):
    jid = uuid.uuid4().hex[:10]
    job = JOBS[jid] = {'status': 'running', 'msg': 'Queued', 'pct': 0}
    def t():
        try:
            job['result'] = work(job, typ, p)
            job['status'] = 'done'
            job['pct'] = 100
        except ProviderError as e:
            job['status'] = 'error'
            job['error'] = str(e)
        except Exception as e:
            job['status'] = 'error'
            job['error'] = f'{type(e).__name__}: {e}'
    threading.Thread(target=t, daemon=True).start()
    return jid

def health():
    edge = subprocess.run(
        ['python3', '-c', 'import edge_tts'], capture_output=True
    ).returncode == 0
    return {
        'ok': True,
        'ffmpeg': bool(shutil.which('ffmpeg')),
        'edge_tts': edge,
        'espeak': bool(shutil.which('espeak-ng')),
        'keys': {
            'HF_TOKEN': bool(K('HF_TOKEN')),
            'FAL_KEY': bool(K('FAL_KEY')),
            'REPLICATE_API_TOKEN': bool(K('REPLICATE_API_TOKEN'))
        }
    }

class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def send(self, code, body, ct='application/json'):
        if isinstance(body, (dict, list)):
            body = json.dumps(body).encode()
        self.send_response(code)
        self.send_header('Content-Type', ct)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Accept-Ranges', 'bytes')
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urllib.parse.urlparse(self.path)
        path = u.path
        if path in ('/', '/index.html'):
            return self.send(200, (ROOT/'index.html').read_bytes(), 'text/html; charset=utf-8')
        if path == '/api/health':
            return self.send(200, health())
        if path.startswith('/api/job/'):
            j = JOBS.get(path.split('/')[-1])
            return self.send(200 if j else 404, j or {'error': 'unknown job'})
        if path.startswith('/files/'):
            parts = path.split('/')[2:]
            if len(parts) == 2:
                f = PROJ/safe(parts[0])/Path(parts[1]).name
                if f.exists():
                    data = f.read_bytes()
                    ct = mimetypes.guess_type(f.name)[0] or 'application/octet-stream'
                    rng = self.headers.get('Range')
                    if rng and (m := re.match(r'bytes=(\d*)-(\d*)', rng)):
                        s = int(m.group(1) or 0)
                        e = int(m.group(2)) if m.group(2) else len(data)-1
                        chunk = data[s:e+1]
                        self.send_response(206)
                        self.send_header('Content-Type', ct)
                        self.send_header('Content-Range', f'bytes {s}-{e}/{len(data)}')
                        self.send_header('Content-Length', str(len(chunk)))
                        self.send_header('Accept-Ranges', 'bytes')
                        self.end_headers()
                        self.wfile.write(chunk)
                        return
                    return self.send(200, data, ct)
        self.send(404, {'error': 'not found'})

    def do_POST(self):
        u = urllib.parse.urlparse(self.path)
        n = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(n)
        if u.path.startswith('/api/upload/'):
            _, _, _, proj, kind = u.path.split('/')[:5]
            if kind not in ('music', 'sfx'):
                return self.send(400, {'error': 'bad kind'})
            f = pdir(proj)/f'{kind}.mp3'
            tmp = pdir(proj)/f'_up_{kind}'
            tmp.write_bytes(body)
            try:
                run(['ffmpeg', '-y', '-i', str(tmp), str(f)])
            except ProviderError as e:
                return self.send(400, {'error': str(e)})
            finally:
                tmp.unlink(missing_ok=True)
            return self.send(200, {'url': f'/files/{safe(proj)}/{kind}.mp3'})
        if u.path.startswith('/api/job/'):
            try:
                p = json.loads(body or b'{}')
            except ValueError:
                return self.send(400, {'error': 'bad json'})
            return self.send(200, {'id': start(u.path.split('/')[-1], p)})
        self.send(404, {'error': 'not found'})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    print(f'Open http://localhost:{port}')
    ThreadingHTTPServer(('0.0.0.0', port), H).serve_forever()
