import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { supabase } from "@/integrations/supabase/client";
import { Camera, CircleStop } from "lucide-react";
import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid, Legend } from "recharts";

const INFER_BASE = (import.meta as any).env.VITE_INFER_BASE || "http://127.0.0.1:5001";

type InferResp = {
  emotion: { label: string; confidence: number; probs: number[] };
  engagement: { probs: number[] };
  stress: { probs: number[] };
  latency_ms: number;
};

function toScore(probs: number[], anchors: number[]): number {
  if (!probs?.length) return 0;
  let s = 0;
  for (let i = 0; i < Math.min(probs.length, anchors.length); i++) s += probs[i] * anchors[i];
  return Math.max(0, Math.min(100, s));
}

export default function MonitoringPanel({ sessionId }: { sessionId?: string }) {
  const [active, setActive] = useState(false);
  const [last, setLast] = useState<InferResp | null>(null);
  const [series, setSeries] = useState<{ t: number; stress: number; attention: number }[]>([]);
  const [status, setStatus] = useState<string>("idle");
  const [effSessionId, setEffSessionId] = useState<string | null>(sessionId || null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const timerRef = useRef<number | null>(null);
  const insertThrottleRef = useRef<number>(0);

  // Resolve latest session if not provided
  useEffect(() => {
    (async () => {
      if (effSessionId) return;
      const { data: auth } = await supabase.auth.getUser();
      const uid = auth?.user?.id;
      if (!uid) return;
      const { data } = await supabase
        .from("analytics_sessions")
        .select("id, started_at")
        .order("started_at", { ascending: false })
        .limit(1)
        .maybeSingle();
      if (data?.id) setEffSessionId(data.id);
    })();
  }, [effSessionId]);

  const stressScore = useMemo(() => {
    const probs = last?.stress?.probs || [];
    // 4-class stress → map to 0..100 via anchors
    return toScore(probs, [10, 35, 65, 90]);
  }, [last]);
  const attentionScore = useMemo(() => {
    const probs = last?.engagement?.probs || [];
    // 4-class engagement → map to 0..100
    return toScore(probs, [25, 50, 75, 95]);
  }, [last]);

  const start = useCallback(async () => {
    if (active) return;
    try {
      setStatus("requesting camera...");
      const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 320, height: 240 }, audio: false });
      if (!videoRef.current) return;
      videoRef.current.srcObject = stream;
      await videoRef.current.play();
      setActive(true);
      setStatus("running");
      // start loop ~5 fps
      const loop = async () => {
        if (!videoRef.current || !canvasRef.current) return;
        const v = videoRef.current;
        const c = canvasRef.current;
        const ctx = c.getContext("2d");
        if (!ctx) return;
        c.width = 224;
        c.height = 224;
        ctx.drawImage(v, 0, 0, c.width, c.height);
        c.toBlob(async (blob) => {
          if (!blob) return;
          try {
            const form = new FormData();
            form.append("file", new File([blob], "frame.jpg", { type: "image/jpeg" }));
            const res = await fetch(`${INFER_BASE}/infer`, { method: "POST", body: form });
            if (res.ok) {
              const j = (await res.json()) as InferResp;
              setLast(j);
              const now = Date.now();
              setSeries((prev) => {
                const next = [...prev, { t: now, stress: toScore(j.stress.probs, [10, 35, 65, 90]), attention: toScore(j.engagement.probs, [25, 50, 75, 95]) }];
                return next.slice(-60); // keep last 60 points
              });
              // Throttle DB inserts to ~1 per 2s
              if (effSessionId && (now - insertThrottleRef.current) > 2000) {
                insertThrottleRef.current = now;
                void supabase.from("analytics_data").insert({
                  session_id: effSessionId,
                  timestamp: new Date().toISOString(),
                  stress_score: toScore(j.stress.probs, [10, 35, 65, 90]),
                  attention_score: toScore(j.engagement.probs, [25, 50, 75, 95]),
                  emotion_state: j.emotion.label,
                });
              }
            } else {
              setStatus("infer error");
            }
          } catch (e) {
            setStatus("infer error");
          }
        }, "image/jpeg", 0.8);
      };
      timerRef.current = window.setInterval(loop, 200);
    } catch (e) {
      setStatus("camera error");
    }
  }, [active, effSessionId]);

  const stop = useCallback(() => {
    setActive(false);
    setStatus("stopped");
    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }
    if (videoRef.current?.srcObject) {
      (videoRef.current.srcObject as MediaStream).getTracks().forEach((t) => t.stop());
      videoRef.current.srcObject = null;
    }
  }, []);

  useEffect(() => () => stop(), [stop]);

  return (
    <Card className="shadow-soft">
      <CardHeader className="flex flex-col gap-2 md:flex-row md:items-center md:justify-between">
        <div>
          <CardTitle className="flex items-center gap-2"><Camera className="h-5 w-5" /> Monitoring</CardTitle>
          <CardDescription>Start/Stop camera and view live emotion, engagement, stress</CardDescription>
        </div>
        <div className="flex gap-2">
          {!active ? (
            <Button onClick={start} className="gap-2"> <Camera className="h-4 w-4" /> Start Monitoring</Button>
          ) : (
            <Button variant="destructive" onClick={stop} className="gap-2"> <CircleStop className="h-4 w-4" /> Stop</Button>
          )}
        </div>
      </CardHeader>
      <CardContent>
        <div className="grid md:grid-cols-3 gap-4">
          <div className="md:col-span-1">
            <div className="relative aspect-video overflow-hidden rounded border">
              <video ref={videoRef} muted playsInline className="h-full w-full object-cover bg-black" />
              <canvas ref={canvasRef} className="hidden" />
            </div>
            <div className="mt-2 text-xs text-muted-foreground">Status: {status}{last ? ` • ${last.latency_ms} ms` : ""}</div>
            {last && (
              <div className="mt-3 space-y-2 text-sm">
                <div className="flex items-center justify-between"><span>Emotion</span><span className="font-medium">{last.emotion.label} ({Math.round(last.emotion.confidence*100)}%)</span></div>
                <div className="flex items-center justify-between"><span>Attention</span><span className="text-primary font-semibold">{Math.round(attentionScore)}</span></div>
                <div className="flex items-center justify-between"><span>Stress</span><span className="text-accent font-semibold">{Math.round(stressScore)}</span></div>
              </div>
            )}
          </div>
          <div className="md:col-span-2">
            <div className="h-52">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={series.map((d, i) => ({ i, stress: d.stress, attention: d.attention }))}>
                  <CartesianGrid strokeDasharray="3 3" opacity={0.2} />
                  <XAxis dataKey="i" tick={false} />
                  <YAxis domain={[0, 100]} />
                  <Tooltip />
                  <Legend />
                  <Line type="monotone" dataKey="attention" stroke="hsl(var(--primary))" dot={false} name="Attention" />
                  <Line type="monotone" dataKey="stress" stroke="hsl(var(--accent))" dot={false} name="Stress" />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
