"use client";

import { useMemo, useState } from "react";
import type { PlayInput, AllResponse } from "../types";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000";

type Status = 'idle' | 'loading' | 'done' | 'error';

export default function Page() {
  const [qtr, setQtr] = useState(2);
  const [min, setMin] = useState(10);
  const [sec, setSec] = useState(0);
  const [yardline100, setYardline100] = useState(60);
  const [ydstogo, setYdstogo] = useState(4);
  const [scoreDiff, setScoreDiff] = useState(0);
  const [postTO, setPostTO] = useState(3);
  const [defTO, setDefTO] = useState(3);
  const [seasonType, setSeasonType] = useState<'REG'|'POST'>('REG');
  const [roof, setRoof] = useState('open_air');
  const [surface, setSurface] = useState('natural');
  const [tempF, setTempF] = useState<number | ''>('');
  const [windMph, setWindMph] = useState<number | ''>('');

  const [status, setStatus] = useState<Status>('idle');
  const [error, setError] = useState<string | null>(null);
  const [resp, setResp] = useState<AllResponse | null>(null);

  const quarterSeconds = useMemo(() => (Math.max(0, Math.min(14, min)) * 60) + Math.max(0, Math.min(59, sec)), [min, sec]);

  const onSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setStatus('loading');
    setError(null);
    setResp(null);
    try {
      const payload: PlayInput = {
        qtr,
        quarter_seconds_remaining: quarterSeconds,
        yardline_100: Number(yardline100),
        ydstogo: Number(ydstogo),
        score_differential: Number(scoreDiff),
        posteam_timeouts_remaining: Number(postTO),
        defteam_timeouts_remaining: Number(defTO),
        season_type: seasonType,
        roof,
        surface,
        temp_f: tempF === '' ? undefined : Number(tempF),
        wind_mph: windMph === '' ? undefined : Number(windMph),
      };
      const r = await fetch(`${API_BASE}/predict/all`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      const data = await r.json();
      setResp(data as AllResponse);
      setStatus('done');
    } catch (err: any) {
      setError(err?.message || 'Request failed');
      setStatus('error');
    }
  };

  return (
    <div>
      <form onSubmit={onSubmit} style={{ display: 'grid', gap: 12, gridTemplateColumns: 'repeat(2, minmax(0, 1fr))' }}>
        <fieldset style={{ gridColumn: '1 / -1' }}>
          <legend>Game State</legend>
          <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap' }}>
            <Labelled label="Quarter">
              <input type="number" min={1} max={5} value={qtr} onChange={e => setQtr(Number(e.target.value))} />
            </Labelled>
            <Labelled label="Clock (MM:SS)">
              <input type="number" min={0} max={14} value={min} onChange={e => setMin(Number(e.target.value))} style={{ width: 60 }} /> :
              <input type="number" min={0} max={59} value={sec} onChange={e => setSec(Number(e.target.value))} style={{ width: 60 }} />
            </Labelled>
            <Labelled label="Yardline 100">
              <input type="number" min={1} max={99} value={yardline100} onChange={e => setYardline100(Number(e.target.value))} />
            </Labelled>
            <Labelled label="Yards To Go">
              <input type="number" min={1} value={ydstogo} onChange={e => setYdstogo(Number(e.target.value))} />
            </Labelled>
            <Labelled label="Score Diff (O - D)">
              <input type="number" value={scoreDiff} onChange={e => setScoreDiff(Number(e.target.value))} />
            </Labelled>
            <Labelled label="Off TOs">
              <input type="number" min={0} max={3} value={postTO} onChange={e => setPostTO(Number(e.target.value))} />
            </Labelled>
            <Labelled label="Def TOs">
              <input type="number" min={0} max={3} value={defTO} onChange={e => setDefTO(Number(e.target.value))} />
            </Labelled>
          </div>
        </fieldset>

        <fieldset>
          <legend>Context</legend>
          <div style={{ display: 'grid', gap: 8 }}>
            <label>
              <span style={{ display: 'inline-block', width: 140 }}>Season Type</span>
              <select value={seasonType} onChange={e => setSeasonType(e.target.value as any)}>
                <option value="REG">REG</option>
                <option value="POST">POST</option>
              </select>
            </label>
            <label>
              <span style={{ display: 'inline-block', width: 140 }}>Roof</span>
              <select value={roof} onChange={e => setRoof(e.target.value)}>
                <option value="open_air">open_air</option>
                <option value="indoor">indoor</option>
              </select>
            </label>
            <label>
              <span style={{ display: 'inline-block', width: 140 }}>Surface</span>
              <select value={surface} onChange={e => setSurface(e.target.value)}>
                <option value="natural">natural</option>
                <option value="artificial">artificial</option>
              </select>
            </label>
          </div>
        </fieldset>

        <fieldset>
          <legend>Weather</legend>
          <div style={{ display: 'grid', gap: 8 }}>
            <label>
              <span style={{ display: 'inline-block', width: 140 }}>Temp (°F)</span>
              <input type="number" value={tempF as number | undefined} onChange={e => setTempF(e.target.value === '' ? '' : Number(e.target.value))} placeholder="e.g. 68" />
            </label>
            <label>
              <span style={{ display: 'inline-block', width: 140 }}>Wind (mph)</span>
              <input type="number" value={windMph as number | undefined} onChange={e => setWindMph(e.target.value === '' ? '' : Number(e.target.value))} placeholder="e.g. 5" />
            </label>
          </div>
        </fieldset>

        <div style={{ gridColumn: '1 / -1', display: 'flex', gap: 12, alignItems: 'center' }}>
          <button type="submit" disabled={status==='loading'}>Predict All</button>
          {status==='loading' && <span>Running…</span>}
          {status==='error' && <span style={{ color: 'crimson' }}>{error}</span>}
        </div>
      </form>

      {resp && (
        <div style={{ marginTop: 24, display: 'grid', gap: 16, gridTemplateColumns: 'repeat(3, minmax(0, 1fr))' }}>
          <Card title="Win Probabilities">
            <p>GO: {(resp.wp.go_wp*100).toFixed(1)}%</p>
            <p>FG: {(resp.wp.fg_wp*100).toFixed(1)}%</p>
            <p>PUNT: {(resp.wp.punt_wp*100).toFixed(1)}%</p>
            <p><b>Best:</b> {resp.wp.best_action}</p>
          </Card>
          <Card title="Components">
            <p>FG Make: {(resp.comp.fg_make_prob*100).toFixed(1)}%</p>
            <p>1st Down (GO): {(resp.comp.first_down_prob*100).toFixed(1)}%</p>
          </Card>
          <Card title="Coach Policy">
            <p><b>Policy:</b> {resp.coach.policy}</p>
            <small style={{ color: '#666' }}>GO: {(resp.coach.probs.GO*100).toFixed(1)}% · FG: {(resp.coach.probs['FIELD_GOAL']*100).toFixed(1)}% · PUNT: {(resp.coach.probs.PUNT*100).toFixed(1)}%</small>
          </Card>
        </div>
      )}
    </div>
  );
}

function Labelled({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <label style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
      <span style={{ width: 140 }}>{label}</span>
      {children}
    </label>
  );
}

function Card({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div style={{ border: '1px solid #ddd', borderRadius: 8, padding: 12 }}>
      <h3 style={{ marginTop: 0 }}>{title}</h3>
      <div>{children}</div>
    </div>
  );
}

