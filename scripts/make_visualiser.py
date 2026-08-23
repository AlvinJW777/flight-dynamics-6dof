"""Generate a self-contained interactive visualiser of the flight modes.

Every frame comes from the real simulator: each mode is excited, propagated with
the same RK4 integrator the tests verify, and the resulting attitude quaternions
are exported. Nothing here is a canned animation.

    python scripts/make_visualiser.py

Writes ``visualiser.html`` — one file, no external dependencies, openable
anywhere.

ON EXAGGERATION
---------------
Phugoid attitude excursions are a couple of degrees and dutch-roll bank a few
more; at true scale the aircraft would barely appear to move. Each mode therefore
carries a visual gain, and **the page states the gain on screen** rather than
quietly overstating the motion. The plotted traces and all numbers are unscaled.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from flightdyn.aerodynamics import AeroModel, Propulsion  # noqa: E402
from flightdyn.aircraft.b747 import (  # noqa: E402
    FC2, FC2_BODY, FC2_LATERAL, FC2_LONGITUDINAL, FC2_MODES, GEOMETRY,
)
from flightdyn.analysis.linear import (  # noqa: E402
    LATERAL, LONGITUDINAL, euler_derivative, jacobian, lateral_modes,
    longitudinal_modes, quat_state_to_euler, submatrix,
)
from flightdyn.atmosphere import isa  # noqa: E402
from flightdyn.dynamics import IDX_RATE, IDX_VEL, propagate  # noqa: E402
from flightdyn.frames import airdata, quat_to_euler  # noqa: E402
from flightdyn.trim import trim_straight_level, trimmed_derivative  # noqa: E402
from flightdyn.units import ft_to_m, lbf_to_n  # noqa: E402

FRAMES = 400


def simulate_modes():
    prop = Propulsion(lbf_to_n(4 * 46000.0), math.radians(2.5), ft_to_m(10.0))
    model = AeroModel(
        GEOMETRY, FC2_LONGITUDINAL, FC2_LATERAL, FC2.alpha_trim_rad, prop,
        mach_trim=FC2.mach, speed_of_sound_m_s=isa(FC2.altitude_m).speed_of_sound_m_s,
    )
    tr = trim_straight_level(model, FC2_BODY, FC2.true_airspeed_m_s, FC2.altitude_m)
    d = trimmed_derivative(model, FC2_BODY, FC2.altitude_m, tr.controls)
    A = jacobian(euler_derivative(d), quat_state_to_euler(tr.state))
    lon = longitudinal_modes(submatrix(A, LONGITUDINAL))
    lat = lateral_modes(submatrix(A, LATERAL))

    # Each entry: how to excite it, how long to watch, and the visual gain.
    recipes = [
        ("short_period", "Short period", lon["short_period"], 30.0, 2.5,
         lambda x: _set(x, IDX_RATE, [0.0, 0.045, 0.0]),
         "Fast pitch rotation at nearly constant speed. Damps out in a few seconds."),
        ("phugoid", "Phugoid", lon["phugoid"], 260.0, 8.0,
         lambda x: _scale(x, IDX_VEL, 1.05),
         "Slow trade of speed for height. Only drag removes energy, so it barely damps."),
        ("dutch_roll", "Dutch roll", lat["dutch_roll"], 45.0, 2.5,
         lambda x: _set(x, IDX_RATE, [0.0, 0.0, 0.05]),
         "Coupled yaw and roll. Dihedral rolls it level while the fin yaws it back."),
        ("roll_subsidence", "Roll subsidence", lat["roll_subsidence"], 12.0, 1.0,
         lambda x: _set(x, IDX_RATE, [0.35, 0.0, 0.0]),
         "Pure roll damping. The fastest mode — a real root, no oscillation."),
        ("spiral", "Spiral", lat["spiral"], 200.0, 2.5,
         lambda x: _bank(x, math.radians(6.0)),
         "Very slow bank divergence or convergence. Here it slowly returns to level."),
    ]

    modes = []
    for key, label, mode, seconds, gain, excite, blurb in recipes:
        dt = seconds / FRAMES
        x0 = excite(tr.state.copy())
        traj = propagate(d, x0, dt=dt, n_steps=FRAMES)

        att, trace = [], []
        for s in traj:
            phi, theta, psi = quat_to_euler(s[6:10])
            V, alpha, beta = airdata(s[IDX_VEL])
            att.append([round(phi, 6), round(theta, 6), round(psi, 6)])
            trace.append([round(math.degrees(alpha), 4), round(V - FC2.true_airspeed_m_s, 4),
                          round(math.degrees(phi), 4)])

        modes.append({
            "key": key, "label": label, "blurb": blurb,
            "dt": dt, "gain": gain, "seconds": seconds,
            "attitude": att, "trace": trace,
            "trim": [round(0.0, 6), round(tr.alpha_rad, 6), 0.0],
            "zeta": round(mode.zeta, 4),
            "omega": round(mode.omega_n, 4),
            "period": None if not mode.is_oscillatory else round(mode.period_s, 2),
            "tau": None if mode.is_oscillatory else round(mode.time_constant_s, 2),
            "oscillatory": bool(mode.is_oscillatory),
            "eig": [round(mode.eigenvalue.real, 5), round(abs(mode.eigenvalue.imag), 5)],
        })

    published = {
        k: {"zeta": v.zeta, "omega": v.omega_n_rad_s} for k, v in FC2_MODES.items()
    }
    return modes, published, tr


def _set(x, sl, vals):
    x[sl] = vals
    return x


def _scale(x, sl, f):
    x[sl] = x[sl] * f
    return x


def _bank(x, phi):
    from flightdyn.frames import euler_to_quat
    _, theta, _ = quat_to_euler(x[6:10])
    x[6:10] = euler_to_quat(phi, theta, 0.0)
    return x


def main() -> int:
    print("Simulating modes...")
    modes, published, tr = simulate_modes()
    for m in modes:
        print(f"  {m['label']:<18} zeta={m['zeta']:+.4f}  {m['seconds']:.0f} s  gain x{m['gain']}")

    payload = json.dumps({"modes": modes, "published": published,
                          "trimAlphaDeg": round(math.degrees(tr.alpha_rad), 3),
                          "trimSpeed": round(FC2.true_airspeed_m_s, 2)},
                         separators=(",", ":"))

    html = TEMPLATE.replace("/*__DATA__*/", payload)
    out = REPO / "visualiser.html"
    out.write_text(html, encoding="utf-8")
    print(f"\n  wrote {out.name}  ({len(html)/1024:.0f} KB)")
    return 0


TEMPLATE = r"""<title>747 Flight Modes</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap">
<style>
:root{
  --ground:#0d1117;--panel:#161c24;--panel2:#1d2530;--ink:#e6edf3;--ink2:#9aa7b4;--ink3:#6b7784;
  --rule:#263039;--accent:#4aa8d8;--sky1:#1b3a52;--sky2:#0e1a26;
  --c1:#4aa8d8;--c2:#e06c5a;--c3:#5fb98c;--c4:#d8a24a;--c5:#a97fd0;
}
:root[data-theme="light"]{
  --ground:#eef2f5;--panel:#ffffff;--panel2:#f3f6f8;--ink:#111820;--ink2:#4a5661;--ink3:#78848f;
  --rule:#d6dee5;--accent:#1b6f9c;--sky1:#bcd6e8;--sky2:#e8f0f6;
}
@media (prefers-color-scheme: light){:root:not([data-theme="dark"]){
  --ground:#eef2f5;--panel:#ffffff;--panel2:#f3f6f8;--ink:#111820;--ink2:#4a5661;--ink3:#78848f;
  --rule:#d6dee5;--accent:#1b6f9c;--sky1:#bcd6e8;--sky2:#e8f0f6;
}}
*{box-sizing:border-box}
body{margin:0;background:var(--ground);color:var(--ink);
  font-family:"IBM Plex Sans",system-ui,sans-serif;font-size:15px;line-height:1.55}
.wrap{max-width:1120px;margin:0 auto;padding:26px 20px 70px}
header{border-bottom:1px solid var(--rule);padding-bottom:16px;margin-bottom:20px}
h1{font-size:clamp(22px,3.4vw,31px);font-weight:700;letter-spacing:-.02em;margin:0 0 5px}
.sub{color:var(--ink2);font-size:14.5px;margin:0}
.tag{display:inline-block;font-family:"IBM Plex Mono",monospace;font-size:10.5px;letter-spacing:.09em;
  text-transform:uppercase;color:var(--accent);border:1px solid var(--rule);border-radius:3px;
  padding:2px 7px;margin-bottom:9px}
.grid{display:grid;grid-template-columns:1fr 300px;gap:16px}
@media (max-width:860px){.grid{grid-template-columns:1fr}}
.card{background:var(--panel);border:1px solid var(--rule);border-radius:8px;padding:14px}
canvas{display:block;width:100%;border-radius:6px}
#view{background:linear-gradient(180deg,var(--sky1),var(--sky2))}
.modes{display:flex;flex-wrap:wrap;gap:6px;margin-bottom:14px}
button.mode{font-family:inherit;font-size:13px;font-weight:500;color:var(--ink2);
  background:var(--panel2);border:1px solid var(--rule);border-radius:5px;padding:7px 12px;cursor:pointer}
button.mode:hover{color:var(--ink)}
button.mode[aria-pressed="true"]{background:var(--accent);border-color:var(--accent);color:#fff}
button.mode:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
.rowbtns{display:flex;gap:6px;align-items:center;margin-top:10px}
button.ctl{font-family:inherit;font-size:12.5px;color:var(--ink2);background:var(--panel2);
  border:1px solid var(--rule);border-radius:5px;padding:5px 11px;cursor:pointer}
button.ctl:hover{color:var(--ink)}
button.ctl:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
.scrub{flex:1;accent-color:var(--accent)}
.blurb{color:var(--ink2);font-size:13.5px;margin:10px 0 0}
dl.read{display:grid;grid-template-columns:auto 1fr;gap:5px 12px;margin:0;
  font-family:"IBM Plex Mono",monospace;font-size:12.5px}
dl.read dt{color:var(--ink3)}
dl.read dd{margin:0;text-align:right;font-variant-numeric:tabular-nums}
h2{font-size:12px;letter-spacing:.09em;text-transform:uppercase;color:var(--ink3);
  font-weight:600;margin:0 0 10px}
.card + .card{margin-top:14px}
.val{display:flex;justify-content:space-between;font-family:"IBM Plex Mono",monospace;
  font-size:12.5px;padding:3px 0;border-bottom:1px solid var(--rule)}
.val:last-child{border-bottom:0}
.val .d{color:var(--c3)}
.note{color:var(--ink3);font-size:12px;margin-top:12px;line-height:1.5}
footer{margin-top:26px;padding-top:16px;border-top:1px solid var(--rule);
  color:var(--ink3);font-size:12.5px}
</style>

<div class="wrap">
<header>
  <span class="tag">NASA CR-2144 &middot; Boeing 747 &middot; Flight Condition 2</span>
  <h1>Flight modes of a Boeing 747</h1>
  <p class="sub">Every frame is computed by the simulator, not animated by hand &mdash; the same
  RK4 integrator and validated aerodynamic model the test suite checks.</p>
</header>

<div class="grid">
  <div>
    <div class="card">
      <div class="modes" id="modes"></div>
      <canvas id="view" height="380"></canvas>
      <div class="rowbtns">
        <button class="ctl" id="play">Pause</button>
        <input class="scrub" id="scrub" type="range" min="0" max="399" value="0" aria-label="time">
        <span id="clock" style="font-family:'IBM Plex Mono',monospace;font-size:12.5px;color:var(--ink2);min-width:74px;text-align:right"></span>
      </div>
      <p class="blurb" id="blurb"></p>
    </div>
    <div class="card">
      <h2>Response</h2>
      <canvas id="trace" height="150"></canvas>
    </div>
  </div>

  <div>
    <div class="card">
      <h2>This mode</h2>
      <dl class="read" id="readout"></dl>
    </div>
    <div class="card">
      <h2>Against CR-2144</h2>
      <div id="compare"></div>
      <p class="note" id="comparenote"></p>
    </div>
    <div class="card">
      <h2>Live state</h2>
      <dl class="read" id="live"></dl>
    </div>
  </div>
</div>

<footer>
  Attitude is exaggerated on screen by the gain shown, because true phugoid and spiral
  excursions are a couple of degrees and would be invisible. <strong>All numbers and the
  response trace are unscaled.</strong>
</footer>
</div>

<script>
const DATA = /*__DATA__*/;
const view = document.getElementById('view'), tc = view.getContext('2d');
const trace = document.getElementById('trace'), gc = trace.getContext('2d');
let mi = 0, fi = 0, playing = true;

const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const COLS = ['--c1','--c2','--c3','--c4','--c5'];

/* ---- 747 geometry, body axes: x fwd, y right, z DOWN ---- */
function geom(){
  const F=[]; const q=(a,b,c,d,s)=>F.push({v:[a,b,c,d],s});
  // fuselage: octagonal tube, nose at +x
  const N=8, xs=[-4.6,-3.4,-1.0,2.2,3.9,4.6], rr=[0.10,0.34,0.40,0.40,0.30,0.10];
  for(let k=0;k<xs.length-1;k++)for(let i=0;i<N;i++){
    const a1=i/N*2*Math.PI, a2=(i+1)/N*2*Math.PI;
    q([xs[k],rr[k]*Math.cos(a1),rr[k]*Math.sin(a1)],
      [xs[k],rr[k]*Math.cos(a2),rr[k]*Math.sin(a2)],
      [xs[k+1],rr[k+1]*Math.cos(a2),rr[k+1]*Math.sin(a2)],
      [xs[k+1],rr[k+1]*Math.cos(a1),rr[k+1]*Math.sin(a1)], 0.55+0.10*Math.cos(a1));
  }
  // upper deck hump
  q([1.3,-0.30,-0.34],[2.9,-0.30,-0.34],[2.9,0.30,-0.34],[1.3,0.30,-0.34],0.78);
  for(const sgn of [1,-1]){
    // wing: swept, root at x 1.1..-0.6, tip at x -1.9..-2.7
    q([1.1,sgn*0.35,0.10],[-0.6,sgn*0.35,0.10],[-2.7,sgn*4.1,0.02],[-1.9,sgn*4.1,0.02],0.86);
    // engine pylons + nacelles
    for(const [yy,xx] of [[1.5,0.10],[2.9,-0.55]]){
      q([xx-0.1,sgn*yy-0.02,0.12],[xx+0.5,sgn*yy-0.02,0.12],[xx+0.5,sgn*yy+0.02,0.40],[xx-0.1,sgn*yy+0.02,0.40],0.5);
      for(let i=0;i<6;i++){
        const a1=i/6*2*Math.PI,a2=(i+1)/6*2*Math.PI,r=0.19;
        q([xx-0.25,sgn*yy+r*Math.cos(a1),0.42+r*Math.sin(a1)],
          [xx-0.25,sgn*yy+r*Math.cos(a2),0.42+r*Math.sin(a2)],
          [xx+0.45,sgn*yy+r*Math.cos(a2),0.42+r*Math.sin(a2)],
          [xx+0.45,sgn*yy+r*Math.cos(a1),0.42+r*Math.sin(a1)],0.44);
      }
    }
    // tailplane
    q([-3.5,sgn*0.25,-0.05],[-4.3,sgn*0.25,-0.05],[-4.6,sgn*1.7,-0.12],[-4.0,sgn*1.7,-0.12],0.80);
  }
  // vertical fin: up is NEGATIVE z
  q([-3.3,0,-0.30],[-4.4,0,-0.30],[-4.7,0,-1.75],[-4.0,0,-1.75],0.72);
  return F;
}
const MESH = geom();

function rot(phi,th,psi){ // body -> world (NED)
  const cp=Math.cos(phi),sp=Math.sin(phi),ct=Math.cos(th),st=Math.sin(th),cy=Math.cos(psi),sy=Math.sin(psi);
  return [[ct*cy, sp*st*cy-cp*sy, cp*st*cy+sp*sy],
          [ct*sy, sp*st*sy+cp*cy, cp*st*sy-sp*cy],
          [-st,   sp*ct,          cp*ct        ]];
}
const ap=(R,v)=>[R[0][0]*v[0]+R[0][1]*v[1]+R[0][2]*v[2],
                 R[1][0]*v[0]+R[1][1]*v[1]+R[1][2]*v[2],
                 R[2][0]*v[0]+R[2][1]*v[1]+R[2][2]*v[2]];

/* camera: behind, above and slightly right, looking along the flight path.
   rows are screen-x (right), screen-y (down), depth. */
const AZ=0.62, EL=0.16;
const CAM=[[-Math.sin(AZ), Math.cos(AZ), 0],
           [-Math.sin(EL)*Math.cos(AZ), -Math.sin(EL)*Math.sin(AZ), Math.cos(EL)],
           [ Math.cos(EL)*Math.cos(AZ),  Math.cos(EL)*Math.sin(AZ), Math.sin(EL)]];

function drawAircraft(phi,th,psi){
  const w=view.width,h=view.height, S=Math.min(w,h)/10.5;
  tc.clearRect(0,0,w,h);

  // artificial-horizon backdrop, rolled and pitched for spatial context
  tc.save(); tc.translate(w/2,h/2); tc.rotate(-phi);
  const hy=th*h*1.0;
  const g=tc.createLinearGradient(0,hy-h,0,hy+h);
  g.addColorStop(0,css('--sky1')); g.addColorStop(0.49,css('--sky2'));
  g.addColorStop(0.51,'#2b2419'); g.addColorStop(1,'#171208');
  tc.fillStyle=g; tc.fillRect(-w,hy-h,2*w,2*h);
  tc.strokeStyle='rgba(255,255,255,.22)'; tc.lineWidth=1.5;
  tc.beginPath(); tc.moveTo(-w,hy); tc.lineTo(w,hy); tc.stroke();
  for(let d=-30;d<=30;d+=10){ if(!d) continue;
    const y=hy+d*h/60; tc.strokeStyle='rgba(255,255,255,.10)'; tc.lineWidth=1;
    tc.beginPath(); tc.moveTo(-60,y); tc.lineTo(60,y); tc.stroke(); }
  tc.restore();

  const R=rot(phi,th,psi);
  const faces=MESH.map(f=>{
    const pts=f.v.map(p=>ap(CAM,ap(R,p)));
    return {pts,z:pts.reduce((s,p)=>s+p[2],0)/4,s:f.s};
  }).sort((a,b)=>a.z-b.z);

  const base=css('--accent');
  for(const f of faces){
    tc.beginPath();
    f.pts.forEach((p,i)=>{const X=w/2+p[0]*S,Y=h/2+p[1]*S; i?tc.lineTo(X,Y):tc.moveTo(X,Y);});
    tc.closePath();
    tc.fillStyle=shade(base,Math.max(0.28,Math.min(1,f.s))); tc.fill();
    tc.strokeStyle='rgba(0,0,0,.45)'; tc.lineWidth=0.7; tc.stroke();
  }
}
function shade(hex,f){
  const n=parseInt(hex.slice(1),16);
  return `rgb(${Math.round(((n>>16)&255)*f)},${Math.round(((n>>8)&255)*f)},${Math.round((n&255)*f)})`;
}

function drawTrace(){
  const m=DATA.modes[mi], w=trace.width, h=trace.height;
  gc.clearRect(0,0,w,h);
  const series = m.key==='dutch_roll'||m.key==='roll_subsidence'||m.key==='spiral'
      ? {i:2,label:'bank angle (deg)'} : (m.key==='phugoid'
      ? {i:1,label:'speed change (m/s)'} : {i:0,label:'angle of attack (deg)'});
  const ys=m.trace.map(t=>t[series.i]);
  const lo=Math.min(...ys), hi=Math.max(...ys), pad=(hi-lo)*0.15+1e-6;
  const Y=v=>h-8-((v-lo+pad)/((hi-lo)+2*pad))*(h-22);
  gc.strokeStyle=css('--rule'); gc.lineWidth=1;
  gc.beginPath(); gc.moveTo(0,Y(ys[0])); gc.lineTo(w,Y(ys[0])); gc.stroke();
  gc.strokeStyle=css(COLS[mi]); gc.lineWidth=1.8; gc.beginPath();
  ys.forEach((v,i)=>{ const X=i/(ys.length-1)*w; i?gc.lineTo(X,Y(v)):gc.moveTo(X,Y(v)); });
  gc.stroke();
  const X=fi/(ys.length-1)*w;
  gc.strokeStyle=css('--ink3'); gc.lineWidth=1; gc.beginPath(); gc.moveTo(X,0); gc.lineTo(X,h); gc.stroke();
  gc.fillStyle=css('--ink3'); gc.font='11px "IBM Plex Mono", monospace';
  gc.fillText(series.label,6,13);
}

function fmt(x,n){ return x.toFixed(n); }
function renderPanels(){
  const m=DATA.modes[mi];
  document.getElementById('readout').innerHTML =
    (m.oscillatory
      ? `<dt>damping &zeta;</dt><dd>${fmt(m.zeta,4)}</dd>`+
        `<dt>frequency &omega;<sub>n</sub></dt><dd>${fmt(m.omega,4)} rad/s</dd>`
      : `<dt>type</dt><dd>real root</dd>`+
        `<dt>rate</dt><dd>${fmt(m.eig[0],4)} /s</dd>`)+
    (m.oscillatory?`<dt>period</dt><dd>${fmt(m.period,1)} s</dd>`
                  :`<dt>time constant</dt><dd>${fmt(m.tau,2)} s</dd>`)+
    `<dt>eigenvalue</dt><dd>${m.eig[0]>=0?'+':''}${fmt(m.eig[0],4)}${m.oscillatory?` &plusmn; ${fmt(m.eig[1],4)}j`:''}</dd>`+
    `<dt>visual gain</dt><dd>&times;${m.gain}</dd>`;

  const pub=DATA.published[m.key];
  const cmp=document.getElementById('compare');
  if(pub){
    const dz=100*(m.zeta/pub.zeta-1), dw=100*(m.omega/pub.omega-1);
    cmp.innerHTML =
      `<div class="val"><span>&zeta; computed</span><span>${fmt(m.zeta,4)}</span></div>`+
      `<div class="val"><span>&zeta; CR-2144</span><span>${fmt(pub.zeta,4)}</span></div>`+
      `<div class="val"><span>difference</span><span class="d">${dz>=0?'+':''}${fmt(dz,1)}%</span></div>`+
      `<div class="val" style="margin-top:8px"><span>&omega;<sub>n</sub> computed</span><span>${fmt(m.omega,4)}</span></div>`+
      `<div class="val"><span>&omega;<sub>n</sub> CR-2144</span><span>${fmt(pub.omega,4)}</span></div>`+
      `<div class="val"><span>difference</span><span class="d">${dw>=0?'+':''}${fmt(dw,1)}%</span></div>`;
    document.getElementById('comparenote').textContent =
      'Published transfer-function factors, CR-2144 Table IX-5.';
  } else {
    cmp.innerHTML = '<div class="val"><span>no published value</span><span>&mdash;</span></div>';
    document.getElementById('comparenote').textContent =
      'CR-2144’s lateral transfer functions were not transcribed, so this mode is checked for physical character only.';
  }
  document.getElementById('blurb').textContent = m.blurb;
}

function frame(){
  const m=DATA.modes[mi], a=m.attitude[fi], t=m.trim, g=m.gain;
  drawAircraft(t[0]+(a[0]-t[0])*g, t[1]+(a[1]-t[1])*g, (a[2]-t[2])*g);
  drawTrace();
  const tr=m.trace[fi];
  document.getElementById('live').innerHTML =
    `<dt>time</dt><dd>${fmt(fi*m.dt,1)} s</dd>`+
    `<dt>angle of attack</dt><dd>${fmt(tr[0],2)}&deg;</dd>`+
    `<dt>bank</dt><dd>${fmt(tr[2],2)}&deg;</dd>`+
    `<dt>speed change</dt><dd>${tr[1]>=0?'+':''}${fmt(tr[1],2)} m/s</dd>`;
  document.getElementById('clock').textContent = `${fmt(fi*m.dt,1)} / ${fmt(m.seconds,0)} s`;
  document.getElementById('scrub').value = fi;
}

function resize(){
  for(const c of [view,trace]){
    const r=c.getBoundingClientRect(), d=window.devicePixelRatio||1;
    c.width=r.width*d; c.height=(c===view?380:150)*d;
    c.getContext('2d').setTransform(d,0,0,d,0,0);
    c.width=r.width; c.height=(c===view?380:150);
  }
  frame();
}

const mb=document.getElementById('modes');
DATA.modes.forEach((m,i)=>{
  const b=document.createElement('button');
  b.className='mode'; b.textContent=m.label; b.setAttribute('aria-pressed', i===0?'true':'false');
  b.onclick=()=>{ mi=i; fi=0;
    [...mb.children].forEach((c,j)=>c.setAttribute('aria-pressed', j===i?'true':'false'));
    document.getElementById('scrub').max = m.attitude.length-1;
    renderPanels(); frame(); };
  mb.appendChild(b);
});
document.getElementById('play').onclick=e=>{ playing=!playing; e.target.textContent=playing?'Pause':'Play'; };
document.getElementById('scrub').oninput=e=>{ fi=+e.target.value; playing=false;
  document.getElementById('play').textContent='Play'; frame(); };

let last=0;
function loop(ts){
  if(playing && ts-last>28){ last=ts; fi=(fi+1)%DATA.modes[mi].attitude.length; frame(); }
  requestAnimationFrame(loop);
}
window.addEventListener('resize',resize);
renderPanels(); resize(); requestAnimationFrame(loop);
</script>
"""

if __name__ == "__main__":
    raise SystemExit(main())
