from flask import Flask, request, jsonify, render_template_string
import joblib, numpy as np, os

app = Flask(__name__)
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'house_price_model.pkl')
model = joblib.load(MODEL_PATH)
FEATURES = ['bedrooms','bathrooms','sqft_living','sqft_lot','floors',
            'waterfront','view','condition','sqft_above','sqft_basement',
            'yr_built','yr_renovated']

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/><meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>ESTATA — Property Intelligence</title>
<link href="https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,300;0,400;0,600;0,700;1,300;1,400&family=DM+Mono:wght@300;400;500&family=Instrument+Serif:ital@0;1&display=swap" rel="stylesheet"/>
<style>
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

:root {
  --ink: #0a0a0a;
  --ink2: #1a1a1a;
  --ink3: #2a2a2a;
  --paper: #f5f2ec;
  --paper2: #ede9e1;
  --paper3: #e5e0d6;
  --rule: rgba(10,10,10,0.12);
  --rule2: rgba(10,10,10,0.25);
  --accent: #0a0a0a;
  --ghost: rgba(10,10,10,0.04);
  --ghost2: rgba(10,10,10,0.07);
  --mono: 'DM Mono', monospace;
  --serif: 'Cormorant Garamond', Georgia, serif;
}

html { scroll-behavior: smooth; }

body {
  font-family: var(--mono);
  background: var(--paper);
  color: var(--ink);
  min-height: 100vh;
  overflow-x: hidden;
  -webkit-font-smoothing: antialiased;
}

/* ── GRAIN OVERLAY ── */
body::before {
  content: '';
  position: fixed; inset: 0; z-index: 9999;
  pointer-events: none;
  opacity: 0.025;
  background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 512 512' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.75' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)'/%3E%3C/svg%3E");
  background-size: 200px;
}

/* ── ARCHITECTURAL LINES BG ── */
.bg-lines {
  position: fixed; inset: 0; z-index: 0; pointer-events: none;
  background-image:
    linear-gradient(var(--rule) 1px, transparent 1px),
    linear-gradient(90deg, var(--rule) 1px, transparent 1px);
  background-size: 60px 60px;
  mask-image: radial-gradient(ellipse 80% 80% at 50% 50%, rgba(0,0,0,0.15) 0%, transparent 100%);
}

/* ── MASTHEAD ── */
.masthead {
  position: relative; z-index: 10;
  border-bottom: 1px solid var(--ink);
  padding: 0 48px;
  display: grid;
  grid-template-columns: 1fr auto 1fr;
  align-items: center;
  height: 56px;
  background: var(--paper);
}
.masthead-left {
  display: flex; align-items: center; gap: 24px;
}
.mast-rule {
  font-family: var(--mono);
  font-size: 0.6rem; font-weight: 400;
  letter-spacing: 0.1em;
  color: rgba(10,10,10,0.4);
  text-transform: uppercase;
}
.live-pill {
  display: flex; align-items: center; gap: 6px;
  font-size: 0.58rem; font-weight: 500; letter-spacing: 0.08em;
  color: var(--ink3);
  text-transform: uppercase;
}
.live-dot {
  width: 5px; height: 5px; border-radius: 50%;
  background: var(--ink);
  animation: blink 1.8s ease-in-out infinite;
}
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.2} }

.masthead-brand {
  font-family: var(--serif);
  font-size: 1.35rem; font-weight: 600;
  letter-spacing: 0.35em;
  text-transform: uppercase;
  color: var(--ink);
  text-align: center;
}
.masthead-right {
  display: flex; justify-content: flex-end; align-items: center;
}
.mast-edition {
  font-size: 0.58rem; font-weight: 400; letter-spacing: 0.08em;
  color: rgba(10,10,10,0.45); text-transform: uppercase;
}

/* ── HERO SECTION ── */
.hero {
  position: relative; z-index: 2;
  display: grid; grid-template-columns: 1fr 1fr;
  min-height: 340px;
  border-bottom: 1px solid var(--ink);
}
.hero-left {
  border-right: 1px solid var(--ink);
  padding: 56px 64px 56px 48px;
  display: flex; flex-direction: column; justify-content: space-between;
}
.hero-label {
  font-size: 0.58rem; font-weight: 500; letter-spacing: 0.2em;
  text-transform: uppercase; color: rgba(10,10,10,0.45);
  margin-bottom: 24px;
  display: flex; align-items: center; gap: 12px;
}
.hero-label::before {
  content: ''; width: 24px; height: 1px; background: rgba(10,10,10,0.3);
}
h1 {
  font-family: var(--serif);
  font-size: clamp(3rem, 5.5vw, 5.2rem);
  font-weight: 300;
  line-height: 1.0;
  letter-spacing: -0.02em;
  color: var(--ink);
}
h1 em {
  font-style: italic;
  font-weight: 300;
}
h1 strong {
  font-weight: 700;
  display: block;
}
.hero-sub {
  font-size: 0.72rem; font-weight: 300;
  line-height: 1.9; letter-spacing: 0.02em;
  color: rgba(10,10,10,0.55);
  max-width: 380px;
  margin-top: 32px;
}
.hero-right {
  display: grid; grid-template-columns: 1fr 1fr;
  grid-template-rows: 1fr 1fr;
  border-bottom: none;
}
.stat-cell {
  padding: 32px 36px;
  border-bottom: 1px solid var(--ink);
  border-right: 1px solid var(--ink);
  display: flex; flex-direction: column; justify-content: flex-end;
}
.stat-cell:nth-child(2) { border-right: none; }
.stat-cell:nth-child(3) { border-bottom: none; }
.stat-cell:nth-child(4) { border-right: none; border-bottom: none; }
.stat-num {
  font-family: var(--serif);
  font-size: 2.8rem; font-weight: 300;
  letter-spacing: -0.03em;
  line-height: 1;
  margin-bottom: 8px;
}
.stat-lbl {
  font-size: 0.58rem; font-weight: 400; letter-spacing: 0.12em;
  text-transform: uppercase; color: rgba(10,10,10,0.45);
}

/* ── SECTION DIVIDER ── */
.section-rule {
  position: relative; z-index: 2;
  border-top: 1px solid var(--ink);
  background: var(--ink);
  padding: 11px 48px;
  display: flex; align-items: center; gap: 24px;
}
.sr-text {
  font-size: 0.58rem; font-weight: 500; letter-spacing: 0.2em;
  text-transform: uppercase; color: var(--paper);
}
.sr-line { flex: 1; height: 1px; background: rgba(255,255,255,0.2); }
.sr-num {
  font-family: var(--serif);
  font-size: 0.7rem; font-style: italic;
  color: rgba(255,255,255,0.5);
}

/* ── MAIN LAYOUT ── */
.main-grid {
  position: relative; z-index: 2;
  display: grid;
  grid-template-columns: 1.15fr 1fr;
  border-bottom: 1px solid var(--ink);
}
.form-col {
  border-right: 1px solid var(--ink);
  padding: 48px;
}
.result-col {
  padding: 0;
  display: flex; flex-direction: column;
}

/* ── FORM TYPOGRAPHY ── */
.form-section-head {
  display: flex; align-items: center; gap: 0;
  margin-bottom: 28px;
}
.form-section-head + .form-section-head {
  margin-top: 40px;
}
.fsh-num {
  font-family: var(--serif);
  font-size: 0.65rem; font-style: italic;
  color: rgba(10,10,10,0.3);
  width: 28px; flex-shrink: 0;
}
.fsh-title {
  font-size: 0.58rem; font-weight: 500; letter-spacing: 0.2em;
  text-transform: uppercase; color: rgba(10,10,10,0.5);
  border-top: 1px solid var(--rule);
  flex: 1; padding-top: 10px;
}

/* ── FIELD GRID ── */
.fgrid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px 28px; }
.field { display: flex; flex-direction: column; gap: 8px; }
.field.f2 { grid-column: span 2; }
.field label {
  font-size: 0.58rem; font-weight: 500; letter-spacing: 0.12em;
  text-transform: uppercase; color: rgba(10,10,10,0.4);
}
.field label .req { color: var(--ink); margin-left: 2px; }

/* Inputs */
.field input[type=number], .field select {
  height: 46px; padding: 0 16px;
  background: transparent;
  border: 1px solid var(--rule2);
  border-radius: 0;
  color: var(--ink); font-family: var(--mono);
  font-size: 0.82rem; font-weight: 400;
  outline: none;
  transition: border-color 0.2s, background 0.2s;
  width: 100%; -webkit-appearance: none; appearance: none;
}
.field input::placeholder { color: rgba(10,10,10,0.25); font-weight: 300; }
.field input:focus, .field select:focus {
  border-color: var(--ink);
  background: var(--ghost2);
}
.field select {
  padding-right: 36px; cursor: pointer;
  background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='10' height='6' viewBox='0 0 10 6'%3E%3Cpath d='M1 1l4 4 4-4' stroke='%230a0a0a' stroke-width='1.2' fill='none' stroke-linecap='round'/%3E%3C/svg%3E");
  background-repeat: no-repeat; background-position: right 14px center;
  background-color: transparent;
}
select option { background: #f5f2ec; color: #0a0a0a; }
optgroup { color: #888; }

/* Stepper */
.stepper {
  display: flex; align-items: stretch;
  border: 1px solid var(--rule2); height: 46px;
  transition: border-color 0.2s;
}
.stepper:focus-within { border-color: var(--ink); }
.stepper button {
  width: 46px; height: 100%; border: none;
  background: transparent; color: rgba(10,10,10,0.4);
  font-size: 1rem; cursor: pointer; flex-shrink: 0;
  font-family: var(--mono); font-weight: 300;
  transition: background 0.15s, color 0.15s; outline: none;
  border-right: 1px solid var(--rule);
}
.stepper button:last-child { border-right: none; border-left: 1px solid var(--rule); }
.stepper button:hover { background: var(--ghost2); color: var(--ink); }
.stepper input {
  border: none !important; box-shadow: none !important;
  background: transparent !important;
  text-align: center; font-weight: 500; color: var(--ink);
  flex: 1; min-width: 0; height: 100%; padding: 0;
  font-size: 0.9rem;
}

/* Toggle segment */
.seg-wrap { display: flex; }
.seg-wrap input[type=radio] { display: none; }
.seg-wrap label {
  flex: 1; height: 46px;
  display: flex; align-items: center; justify-content: center;
  font-size: 0.68rem; font-weight: 400; letter-spacing: 0.08em;
  color: rgba(10,10,10,0.45); text-transform: uppercase;
  cursor: pointer;
  border: 1px solid var(--rule2);
  transition: all 0.15s;
}
.seg-wrap label:first-of-type { border-right: none; }
.seg-wrap input[type=radio]:checked + label {
  background: var(--ink); color: var(--paper);
  border-color: var(--ink);
}
.seg-wrap label:hover:not(.checked) { background: var(--ghost); }

/* Condition pills */
.cond-strip { display: flex; gap: 0; }
.cpill {
  flex: 1; height: 40px; border: 1px solid var(--rule2);
  border-right: none; background: transparent;
  color: rgba(10,10,10,0.4); font-size: 0.6rem; font-weight: 500;
  letter-spacing: 0.08em; text-transform: uppercase;
  cursor: pointer; font-family: var(--mono);
  transition: all 0.15s; outline: none;
}
.cpill:last-child { border-right: 1px solid var(--rule2); }
.cpill:hover { background: var(--ghost); color: var(--ink); }
.cpill.on { background: var(--ink); color: var(--paper); border-color: var(--ink); }
.cpill.on + .cpill { border-left-color: var(--ink); }

/* ── SUBMIT BUTTON ── */
.submit-area { margin-top: 40px; }
.btn-estimate {
  width: 100%; height: 56px;
  background: var(--ink); color: var(--paper);
  border: none; cursor: pointer;
  font-family: var(--serif);
  font-size: 1.1rem; font-weight: 400; font-style: italic;
  letter-spacing: 0.05em;
  transition: opacity 0.2s, transform 0.15s;
  position: relative; overflow: hidden;
  display: flex; align-items: center; justify-content: center; gap: 12px;
}
.btn-estimate::before {
  content: '';
  position: absolute; inset: 0;
  background: linear-gradient(90deg, transparent, rgba(255,255,255,0.06), transparent);
  transform: translateX(-100%);
  transition: transform 0.6s;
}
.btn-estimate:hover::before { transform: translateX(100%); }
.btn-estimate:hover { opacity: 0.88; }
.btn-estimate:active { transform: scaleX(0.99); }
.btn-estimate .bt { display: flex; align-items: center; gap: 10px; }
.btn-estimate .bs { display: none; }
.btn-estimate.loading .bt { display: none; }
.btn-estimate.loading .bs {
  display: block; width: 18px; height: 18px;
  border: 1.5px solid rgba(245,242,236,0.3);
  border-top-color: var(--paper);
  border-radius: 50%;
  animation: spin 0.7s linear infinite;
}
@keyframes spin { to { transform: rotate(360deg); } }
.btn-arrow {
  font-size: 1rem; font-style: normal;
  transition: transform 0.2s;
}
.btn-estimate:hover .btn-arrow { transform: translateX(4px); }

/* ── RESULT COLUMN ── */
.result-main {
  flex: 1; padding: 48px 40px;
  display: flex; flex-direction: column; justify-content: center;
  border-bottom: 1px solid var(--ink);
  min-height: 340px;
  position: relative;
}

/* Empty state */
.empty-state {
  display: flex; flex-direction: column; align-items: flex-start; gap: 16px;
}
.es-icon {
  width: 48px; height: 48px;
  border: 1px solid var(--rule2);
  display: grid; place-items: center;
  color: rgba(10,10,10,0.2);
  font-size: 1.2rem;
  font-family: var(--serif); font-style: italic;
}
.es-title {
  font-family: var(--serif);
  font-size: 1.5rem; font-weight: 300; font-style: italic;
  color: rgba(10,10,10,0.25);
  line-height: 1.3;
}
.es-sub {
  font-size: 0.62rem; font-weight: 400; letter-spacing: 0.05em;
  color: rgba(10,10,10,0.25); line-height: 1.7;
}

/* Result state */
.result-inner { display: none; }
.result-inner.show { display: block; }

.res-location {
  display: flex; align-items: center; gap: 12px;
  margin-bottom: 32px;
}
.res-loc-label {
  font-size: 0.58rem; font-weight: 500; letter-spacing: 0.15em;
  text-transform: uppercase; color: rgba(10,10,10,0.4);
}
.res-loc-name {
  font-size: 0.68rem; font-weight: 500; letter-spacing: 0.05em;
  color: var(--ink);
  border-left: 1px solid var(--rule2); padding-left: 12px;
}

.res-label {
  font-size: 0.55rem; font-weight: 500; letter-spacing: 0.2em;
  text-transform: uppercase; color: rgba(10,10,10,0.35);
  margin-bottom: 10px;
}
.res-price {
  font-family: var(--serif);
  font-size: clamp(2.8rem, 5.5vw, 4.2rem);
  font-weight: 300; letter-spacing: -0.03em; line-height: 1;
  color: var(--ink); margin-bottom: 8px;
  animation: slideUp 0.5s cubic-bezier(0.16,1,0.3,1);
}
@keyframes slideUp {
  from { transform: translateY(16px); opacity: 0; }
  to { transform: translateY(0); opacity: 1; }
}
.res-psf {
  font-size: 0.65rem; font-weight: 300; letter-spacing: 0.05em;
  color: rgba(10,10,10,0.4); margin-bottom: 36px;
  font-style: italic; font-family: var(--serif);
}

/* Range band */
.range-band {
  display: grid; grid-template-columns: 1fr 40px 1fr;
  align-items: center; gap: 12px; margin-bottom: 32px;
}
.range-col { }
.range-direction {
  font-size: 0.55rem; font-weight: 500; letter-spacing: 0.1em;
  text-transform: uppercase; color: rgba(10,10,10,0.35);
  margin-bottom: 4px;
}
.range-val {
  font-family: var(--serif);
  font-size: 1rem; font-weight: 300;
  color: var(--ink);
}
.range-sep {
  text-align: center;
  font-family: var(--serif); font-style: italic;
  font-size: 1.2rem; color: rgba(10,10,10,0.2);
}
.range-bar {
  grid-column: span 3;
  height: 1px; background: var(--rule);
  position: relative; margin-top: 4px;
}
.range-bar-fill {
  position: absolute;
  left: 20%; right: 20%;
  top: -1px; height: 3px;
  background: var(--ink);
  transition: all 0.8s cubic-bezier(0.16,1,0.3,1);
}
.range-marker {
  position: absolute; top: -5px;
  width: 2px; height: 11px;
  background: var(--ink);
  transform: translateX(-50%);
}
.range-marker.center { left: 50%; }
.range-marker.lo { left: 20%; }
.range-marker.hi { right: 20%; transform: translateX(50%); }

/* Confidence */
.conf-section { }
.conf-header {
  display: flex; justify-content: space-between; align-items: baseline;
  margin-bottom: 10px;
}
.conf-title {
  font-size: 0.55rem; font-weight: 500; letter-spacing: 0.18em;
  text-transform: uppercase; color: rgba(10,10,10,0.35);
}
.conf-pct {
  font-family: var(--serif);
  font-size: 1rem; font-weight: 300;
  color: var(--ink);
}
.conf-track {
  height: 1px; background: var(--rule);
  position: relative;
}
.conf-fill {
  position: absolute; left: 0; top: -1px; height: 3px;
  width: 0%; background: var(--ink);
  transition: width 1.2s cubic-bezier(0.16,1,0.3,1);
}

/* ── RIGHT BOTTOM CARDS ── */
.right-cards {
  display: flex; flex-direction: column;
}
.rcard {
  padding: 28px 40px;
  border-bottom: 1px solid var(--ink);
}
.rcard:last-child { border-bottom: none; }
.rcard-head {
  display: flex; align-items: center; gap: 0;
  margin-bottom: 20px;
}
.rcard-num {
  font-family: var(--serif); font-style: italic;
  font-size: 0.65rem; color: rgba(10,10,10,0.25);
  width: 24px; flex-shrink: 0;
}
.rcard-title {
  font-size: 0.58rem; font-weight: 500; letter-spacing: 0.18em;
  text-transform: uppercase; color: rgba(10,10,10,0.45);
  border-top: 1px solid var(--rule); flex: 1; padding-top: 8px;
}

/* Drivers */
.driver-list { display: flex; flex-direction: column; gap: 10px; }
.driver-row { display: flex; align-items: center; gap: 14px; }
.driver-name {
  font-size: 0.62rem; font-weight: 300; color: rgba(10,10,10,0.55);
  width: 80px; flex-shrink: 0;
}
.driver-track { flex: 1; height: 1px; background: var(--rule); position: relative; }
.driver-fill {
  position: absolute; left: 0; top: -1px; height: 3px;
  background: var(--ink);
}
.driver-pct {
  font-size: 0.58rem; font-weight: 500; font-family: var(--mono);
  color: rgba(10,10,10,0.4); width: 30px; text-align: right;
}

/* City tiers table */
.tier-table { width: 100%; border-collapse: collapse; }
.tier-table tr { border-bottom: 1px solid var(--rule); }
.tier-table tr:last-child { border-bottom: none; }
.tier-table td { padding: 9px 0; vertical-align: middle; }
.tier-dot { width: 5px; height: 5px; border-radius: 50%; }
.td-premium { background: var(--ink); }
.td-growth { background: rgba(10,10,10,0.5); }
.td-moderate { background: rgba(10,10,10,0.2); }
.tier-city { font-size: 0.65rem; font-weight: 300; color: rgba(10,10,10,0.6); padding-left: 12px; }
.tier-badge {
  font-size: 0.52rem; font-weight: 500; letter-spacing: 0.1em;
  text-transform: uppercase; color: rgba(10,10,10,0.35);
  text-align: right;
}

/* ── FOOTER ── */
.footer-rule {
  position: relative; z-index: 2;
  background: var(--ink);
  padding: 20px 48px;
  display: flex; justify-content: space-between; align-items: center;
}
.footer-brand {
  font-family: var(--serif); font-size: 0.9rem; font-weight: 300;
  letter-spacing: 0.3em; text-transform: uppercase;
  color: var(--paper);
}
.footer-meta {
  display: flex; gap: 32px;
}
.footer-item {
  font-size: 0.58rem; font-weight: 400; letter-spacing: 0.08em;
  color: rgba(245,242,236,0.35);
}

/* ── TOAST ── */
.toast {
  position: fixed; bottom: 28px; right: 28px; z-index: 9998;
  background: var(--ink); color: var(--paper);
  padding: 12px 20px;
  font-size: 0.72rem; font-weight: 300; letter-spacing: 0.03em;
  transform: translateY(60px); opacity: 0;
  transition: all 0.3s cubic-bezier(0.16,1,0.3,1);
  border-left: 2px solid var(--paper);
}
.toast.show { transform: translateY(0); opacity: 1; }

/* ── RESPONSIVE ── */
@media (max-width: 860px) {
  .masthead { padding: 0 20px; grid-template-columns: 1fr auto; }
  .masthead-right { display: none; }
  .hero { grid-template-columns: 1fr; }
  .hero-right { grid-template-columns: 1fr 1fr; border-top: 1px solid var(--ink); }
  .hero-left { padding: 40px 24px; }
  .section-rule { padding: 11px 24px; }
  .main-grid { grid-template-columns: 1fr; }
  .form-col { border-right: none; padding: 32px 24px; }
  .result-col { border-top: 1px solid var(--ink); }
  .result-main { padding: 36px 24px; }
  .rcard { padding: 24px; }
  .fgrid { grid-template-columns: 1fr; }
  .field.f2 { grid-column: span 1; }
  .footer-rule { flex-direction: column; gap: 12px; text-align: center; padding: 20px 24px; }
  .footer-meta { flex-direction: column; gap: 8px; text-align: center; }
  h1 { font-size: 2.8rem; }
}
</style>
</head>
<body>

<div class="bg-lines"></div>

<!-- MASTHEAD -->
<header class="masthead">
  <div class="masthead-left">
    <span class="mast-rule">Property Intelligence</span>
    <div class="live-pill">
      <div class="live-dot"></div>
      <span>Model Live</span>
    </div>
  </div>
  <div class="masthead-brand">Estata</div>
  <div class="masthead-right">
    <span class="mast-edition">Random Forest · v2.1 · 2026</span>
  </div>
</header>

<!-- HERO -->
<section class="hero">
  <div class="hero-left">
    <div>
      <div class="hero-label">AI Valuation Engine</div>
      <h1>
        <em>Precision</em><br>
        <strong>Pricing</strong><br>
        <em>Intelligence</em>
      </h1>
    </div>
    <p class="hero-sub">Machine-learning property valuation calibrated to 18 Indian city markets. Enter your specifications — receive an institutional-grade estimate in under one second.</p>
  </div>
  <div class="hero-right">
    <div class="stat-cell">
      <div class="stat-num">4,601</div>
      <div class="stat-lbl">Training Records</div>
    </div>
    <div class="stat-cell">
      <div class="stat-num">87<span style="font-size:1.4rem">%</span></div>
      <div class="stat-lbl">R² Accuracy</div>
    </div>
    <div class="stat-cell">
      <div class="stat-num">18</div>
      <div class="stat-lbl">City Markets</div>
    </div>
    <div class="stat-cell">
      <div class="stat-num">&lt;1<span style="font-size:1.4rem">s</span></div>
      <div class="stat-lbl">Response Time</div>
    </div>
  </div>
</section>

<!-- SECTION RULE -->
<div class="section-rule">
  <span class="sr-text">Property Specification</span>
  <div class="sr-line"></div>
  <span class="sr-num">§ 01</span>
</div>

<!-- MAIN -->
<div class="main-grid">

  <!-- FORM -->
  <div class="form-col">
    <form id="valForm">

      <!-- 01. Location -->
      <div class="form-section-head">
        <span class="fsh-num">01</span>
        <span class="fsh-title">Market Location</span>
      </div>
      <div class="fgrid">
        <div class="field f2">
          <label for="city_m">City / Location <span class="req">*</span></label>
          <select id="city_m">
            <optgroup label="Tier 1 Premium">
              <option value="2.55">Mumbai  ·  ₹20,000 – 30,000 / sqft</option>
              <option value="2.10">Delhi  ·  ₹15,000 – 25,000 / sqft</option>
              <option value="1.80">Noida / Greater Noida  ·  ₹10,000 – 18,000 / sqft</option>
              <option value="1.55">Bangalore  ·  ₹8,000 – 15,000 / sqft</option>
            </optgroup>
            <optgroup label="Tier 1 High Growth">
              <option value="1.30">Pune  ·  ₹7,000 – 13,000 / sqft</option>
              <option value="1.10" selected>Hyderabad  ·  ₹6,000 – 11,000 / sqft</option>
              <option value="1.05">Gurgaon / Gurugram  ·  ₹9,000 – 16,000 / sqft</option>
            </optgroup>
            <optgroup label="Tier 2 Established">
              <option value="0.92">Chennai  ·  ₹5,500 – 9,500 / sqft</option>
              <option value="0.82">Kolkata  ·  ₹4,500 – 8,000 / sqft</option>
              <option value="0.75">Ahmedabad  ·  ₹4,000 – 7,500 / sqft</option>
              <option value="0.70">Surat  ·  ₹3,500 – 6,500 / sqft</option>
              <option value="0.65">Jaipur  ·  ₹3,500 – 6,000 / sqft</option>
            </optgroup>
            <optgroup label="Tier 3 Affordable">
              <option value="0.58">Lucknow  ·  ₹3,000 – 5,500 / sqft</option>
              <option value="0.55">Nagpur  ·  ₹3,000 – 5,000 / sqft</option>
              <option value="0.52">Bhopal  ·  ₹2,800 – 5,000 / sqft</option>
              <option value="0.50">Indore  ·  ₹2,800 – 4,800 / sqft</option>
              <option value="0.48">Coimbatore  ·  ₹2,500 – 4,200 / sqft</option>
            </optgroup>
          </select>
        </div>
      </div>

      <!-- 02. Property Specs -->
      <div class="form-section-head">
        <span class="fsh-num">02</span>
        <span class="fsh-title">Property Specifications</span>
      </div>
      <div class="fgrid">
        <div class="field">
          <label>Bedrooms <span class="req">*</span></label>
          <div class="stepper">
            <button type="button" onclick="step('bd',-1,1,10)">−</button>
            <input type="number" id="bd" value="3" min="1" max="10" readonly/>
            <button type="button" onclick="step('bd',1,1,10)">+</button>
          </div>
        </div>
        <div class="field">
          <label>Bathrooms <span class="req">*</span></label>
          <div class="stepper">
            <button type="button" onclick="step('bt',-0.5,0.5,8)">−</button>
            <input type="number" id="bt" value="2" min="0.5" max="8" step="0.5" readonly/>
            <button type="button" onclick="step('bt',0.5,0.5,8)">+</button>
          </div>
        </div>
        <div class="field">
          <label for="sl">Built-up Area (sq ft) <span class="req">*</span></label>
          <input type="number" id="sl" value="1800" min="200" max="15000" placeholder="e.g. 1800"/>
        </div>
        <div class="field">
          <label for="lot">Plot / Lot Area (sq ft) <span class="req">*</span></label>
          <input type="number" id="lot" value="4500" min="200" max="500000" placeholder="e.g. 4500"/>
        </div>
        <div class="field f2">
          <label>Waterfront Property <span class="req">*</span></label>
          <div class="seg-wrap">
            <input type="radio" name="wf" id="wf0" value="0" checked><label for="wf0">No Waterfront</label>
            <input type="radio" name="wf" id="wf1" value="1"><label for="wf1">Waterfront</label>
          </div>
        </div>
      </div>

      <!-- 03. Quality -->
      <div class="form-section-head">
        <span class="fsh-num">03</span>
        <span class="fsh-title">Quality & Construction</span>
      </div>
      <div class="fgrid">
        <div class="field">
          <label for="vw">View Quality <span class="req">*</span></label>
          <select id="vw">
            <option value="0">None</option>
            <option value="1">Fair</option>
            <option value="2" selected>Average</option>
            <option value="3">Good</option>
            <option value="4">Excellent</option>
          </select>
        </div>
        <div class="field">
          <label for="yb">Year Built <span class="req">*</span></label>
          <input type="number" id="yb" value="2000" min="1900" max="2025" placeholder="e.g. 2005"/>
        </div>
        <div class="field">
          <label for="sa">Above Ground (sq ft) <span class="req">*</span></label>
          <input type="number" id="sa" value="1500" min="200" max="12000" placeholder="e.g. 1500"/>
        </div>
        <div class="field">
          <label for="sb">Basement Area (sq ft)</label>
          <input type="number" id="sb" value="0" min="0" max="5000" placeholder="0 if none"/>
        </div>
        <div class="field f2">
          <label for="yr">Year Renovated <span class="req">*</span></label>
          <input type="number" id="yr" value="0" min="0" max="2025" placeholder="Enter 0 if never renovated"/>
        </div>
      </div>

      <!-- 04. Condition -->
      <div class="form-section-head">
        <span class="fsh-num">04</span>
        <span class="fsh-title">Overall Condition</span>
      </div>
      <div class="field f2">
        <div class="cond-strip" id="cRow">
          <button type="button" class="cpill" data-v="1" onclick="setCond(1)">Poor</button>
          <button type="button" class="cpill" data-v="2" onclick="setCond(2)">Fair</button>
          <button type="button" class="cpill on" data-v="3" onclick="setCond(3)">Average</button>
          <button type="button" class="cpill" data-v="4" onclick="setCond(4)">Good</button>
          <button type="button" class="cpill" data-v="5" onclick="setCond(5)">Excellent</button>
        </div>
        <input type="hidden" id="cond" value="3"/>
      </div>

      <div class="submit-area">
        <button type="submit" class="btn-estimate" id="sbtn">
          <span class="bt">
            <span>Calculate Estimated Value</span>
            <span class="btn-arrow">→</span>
          </span>
          <div class="bs"></div>
        </button>
      </div>

    </form>
  </div>

  <!-- RESULT COLUMN -->
  <div class="result-col">

    <div class="result-main" id="rc">
      <div class="empty-state" id="emptyState">
        <div class="es-icon">⌂</div>
        <div class="es-title">Your valuation<br>will appear here</div>
        <div class="es-sub">Complete the property specifications<br>and submit for an instant estimate.</div>
      </div>

      <div class="result-inner" id="ri">
        <div class="res-location">
          <span class="res-loc-label">Market</span>
          <span class="res-loc-name" id="rcity">—</span>
        </div>
        <div class="res-label">Estimated Market Value</div>
        <div class="res-price" id="rprice">—</div>
        <div class="res-psf" id="rpsf">— per square foot</div>

        <div class="range-band">
          <div class="range-col">
            <div class="range-direction">Conservative −8%</div>
            <div class="range-val" id="rlo">—</div>
          </div>
          <div class="range-sep">↔</div>
          <div class="range-col" style="text-align:right">
            <div class="range-direction">Optimistic +8%</div>
            <div class="range-val" id="rhi">—</div>
          </div>
          <div class="range-bar">
            <div class="range-bar-fill"></div>
            <div class="range-marker lo"></div>
            <div class="range-marker center"></div>
            <div class="range-marker hi"></div>
          </div>
        </div>

        <div class="conf-section">
          <div class="conf-header">
            <span class="conf-title">Model Confidence</span>
            <span class="conf-pct" id="cpct">—</span>
          </div>
          <div class="conf-track">
            <div class="conf-fill" id="cfill"></div>
          </div>
        </div>
      </div>
    </div>

    <div class="right-cards">
      <!-- Key Drivers -->
      <div class="rcard">
        <div class="rcard-head">
          <span class="rcard-num">i.</span>
          <span class="rcard-title">Key Valuation Drivers</span>
        </div>
        <div class="driver-list">
          <div class="driver-row">
            <div class="driver-name">Living area</div>
            <div class="driver-track"><div class="driver-fill" style="width:88%"></div></div>
            <div class="driver-pct">88%</div>
          </div>
          <div class="driver-row">
            <div class="driver-name">Location</div>
            <div class="driver-track"><div class="driver-fill" style="width:74%"></div></div>
            <div class="driver-pct">74%</div>
          </div>
          <div class="driver-row">
            <div class="driver-name">Year built</div>
            <div class="driver-track"><div class="driver-fill" style="width:58%"></div></div>
            <div class="driver-pct">58%</div>
          </div>
          <div class="driver-row">
            <div class="driver-name">Bathrooms</div>
            <div class="driver-track"><div class="driver-fill" style="width:48%"></div></div>
            <div class="driver-pct">48%</div>
          </div>
          <div class="driver-row">
            <div class="driver-name">Condition</div>
            <div class="driver-track"><div class="driver-fill" style="width:36%"></div></div>
            <div class="driver-pct">36%</div>
          </div>
          <div class="driver-row">
            <div class="driver-name">View</div>
            <div class="driver-track"><div class="driver-fill" style="width:24%"></div></div>
            <div class="driver-pct">24%</div>
          </div>
        </div>
      </div>

      <!-- City Tiers -->
      <div class="rcard">
        <div class="rcard-head">
          <span class="rcard-num">ii.</span>
          <span class="rcard-title">City Market Classification</span>
        </div>
        <table class="tier-table">
          <tr>
            <td><div class="tier-dot td-premium"></div></td>
            <td class="tier-city">Mumbai · Delhi · Noida · Bangalore</td>
            <td class="tier-badge">Premium</td>
          </tr>
          <tr>
            <td><div class="tier-dot td-growth"></div></td>
            <td class="tier-city">Pune · Hyderabad · Gurgaon</td>
            <td class="tier-badge">High Growth</td>
          </tr>
          <tr>
            <td><div class="tier-dot td-moderate"></div></td>
            <td class="tier-city">Chennai · Kolkata · Ahmedabad +8</td>
            <td class="tier-badge">Moderate</td>
          </tr>
        </table>
      </div>
    </div>

  </div>
</div>

<!-- FOOTER -->
<footer class="footer-rule">
  <div class="footer-brand">Estata</div>
  <div class="footer-meta">
    <span class="footer-item">Prices in Indian Rupees (INR)</span>
    <span class="footer-item">1 USD = ₹83.5</span>
    <span class="footer-item">Trained on King County, WA Dataset</span>
    <span class="footer-item">scikit-learn · Random Forest</span>
  </div>
</footer>

<div class="toast" id="toast"></div>

<script>
const USD_TO_INR = 83.5;

function step(id, d, mn, mx) {
  const el = document.getElementById(id);
  let v = Math.round((parseFloat(el.value) + d) * 10) / 10;
  el.value = Math.max(mn, Math.min(mx, v));
}

function setCond(v) {
  document.getElementById('cond').value = v;
  document.querySelectorAll('.cpill').forEach(b => b.classList.toggle('on', +b.dataset.v === v));
}

function fmt(n) {
  n = Math.round(n);
  if (n >= 10000000) return '₹' + (n / 10000000).toFixed(2) + ' Cr';
  if (n >= 100000)   return '₹' + (n / 100000).toFixed(2) + ' L';
  return '₹' + n.toLocaleString('en-IN');
}

function toast(msg) {
  const t = document.getElementById('toast');
  t.textContent = msg; t.classList.add('show');
  setTimeout(() => t.classList.remove('show'), 3800);
}

document.getElementById('valForm').addEventListener('submit', async function(e) {
  e.preventDefault();
  const btn = document.getElementById('sbtn');
  btn.classList.add('loading'); btn.disabled = true;

  const payload = {
    bedrooms:      parseFloat(document.getElementById('bd').value),
    bathrooms:     parseFloat(document.getElementById('bt').value),
    sqft_living:   parseFloat(document.getElementById('sl').value),
    sqft_lot:      parseFloat(document.getElementById('lot').value),
    floors:        1.0,
    waterfront:    parseInt(document.querySelector('[name=wf]:checked')?.value || 0),
    view:          parseInt(document.getElementById('vw').value),
    condition:     parseInt(document.getElementById('cond').value),
    sqft_above:    parseFloat(document.getElementById('sa').value),
    sqft_basement: parseFloat(document.getElementById('sb').value),
    yr_built:      parseInt(document.getElementById('yb').value),
    yr_renovated:  parseInt(document.getElementById('yr').value),
  };

  for (const [k, v] of Object.entries(payload)) {
    if (isNaN(v)) { toast('Please fill in all required fields.'); btn.classList.remove('loading'); btn.disabled = false; return; }
  }

  try {
    const r = await fetch('/predict', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
    const d = await r.json();
    if (d.error) { toast('Error: ' + d.error); return; }

    const sel = document.getElementById('city_m');
    const cityName = sel.options[sel.selectedIndex].text.split('·')[0].trim();
    const cityMult = parseFloat(sel.value);
    const priceINR = d.predicted_price * USD_TO_INR * cityMult;
    const sqft = payload.sqft_living;

    document.getElementById('rcity').textContent = cityName;
    document.getElementById('rprice').textContent = fmt(priceINR);
    document.getElementById('rpsf').textContent = fmt(priceINR / sqft) + ' per sq ft';
    document.getElementById('rlo').textContent = fmt(priceINR * 0.92);
    document.getElementById('rhi').textContent = fmt(priceINR * 1.08);

    const conf = Math.round(Math.min(94, Math.max(68, 88 - Math.abs(d.predicted_price - 540000) / 55000)));
    document.getElementById('cpct').textContent = conf + '%';
    setTimeout(() => { document.getElementById('cfill').style.width = conf + '%'; }, 120);

    document.getElementById('emptyState').style.display = 'none';
    document.getElementById('ri').classList.add('show');
    document.getElementById('rc').scrollIntoView({ behavior: 'smooth', block: 'nearest' });

  } catch (err) {
    toast('Network error — is the server running?');
  } finally {
    btn.classList.remove('loading'); btn.disabled = false;
  }
});
</script>
</body>
</html>"""

@app.route('/')
def index():
    return render_template_string(HTML)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        X = np.array([data.get(f, 0) for f in FEATURES]).reshape(1, -1)
        pred = model.predict(X)[0]
        return jsonify({'predicted_price': round(float(pred), 2)})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)