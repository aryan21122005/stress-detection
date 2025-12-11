import { useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { supabase } from "@/integrations/supabase/client";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { useToast } from "@/components/ui/use-toast";
import { LogOut, Activity, Brain, Eye, Heart, Key, Plus } from "lucide-react";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, AreaChart, Area } from "recharts";
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog";
import MonitoringPanel from "@/components/MonitoringPanel";

interface Session {
  id: string;
  session_name: string | null;
  started_at: string;
  overall_stress_score: number | null;
  overall_attention_score: number | null;
}

interface AnalyticsData {
  timestamp: string;
  emotion_state: string | null;
  stress_score: number | null;
  attention_score: number | null;
  heart_rate: number | null;
}

interface EyeClosureEvent {
  timestamp: string;
  duration_seconds: number;
}

const ParticipantDashboard = () => {
  const [user, setUser] = useState<any>(null);
  const [sessions, setSessions] = useState<Session[]>([]);
  const [analyticsData, setAnalyticsData] = useState<AnalyticsData[]>([]);
  const [eyeClosureEvents, setEyeClosureEvents] = useState<EyeClosureEvent[]>([]);
  const [loading, setLoading] = useState(true);
  const [hostKey, setHostKey] = useState("");
  const [sessionName, setSessionName] = useState("");
  const [dialogOpen, setDialogOpen] = useState(false);
  const [joiningSession, setJoiningSession] = useState(false);
  const navigate = useNavigate();
  const { toast } = useToast();

  useEffect(() => {
    checkAuth();
    fetchSessions();
    fetchAnalytics();
    fetchEyeClosureEvents();
  }, []);

  const checkAuth = async () => {
    const { data: { user } } = await supabase.auth.getUser();
    if (!user) {
      navigate("/auth");
      return;
    }

    const { data: profile } = await supabase
      .from("profiles")
      .select("role")
      .eq("id", user.id)
      .single();

    if (profile?.role !== "participant") {
      navigate("/host-dashboard");
      return;
    }

    setUser(user);
    setLoading(false);
  };

  const fetchSessions = async () => {
    const { data, error } = await supabase
      .from("analytics_sessions")
      .select("*")
      .order("started_at", { ascending: false });

    if (error) {
      console.error("Error fetching sessions:", error);
      return;
    }

    setSessions(data || []);
  };

  const fetchAnalytics = async () => {
    if (sessions.length === 0) return;

    const { data, error } = await supabase
      .from("analytics_data")
      .select("*")
      .eq("session_id", sessions[0]?.id)
      .order("timestamp", { ascending: true });

    if (error) {
      console.error("Error fetching analytics:", error);
      return;
    }

    setAnalyticsData(data || []);
  };

  const fetchEyeClosureEvents = async () => {
    if (sessions.length === 0) return;

    const { data, error } = await supabase
      .from("eye_closure_events")
      .select("*")
      .eq("session_id", sessions[0]?.id)
      .order("timestamp", { ascending: true });

    if (error) {
      console.error("Error fetching eye closure events:", error);
      return;
    }

    setEyeClosureEvents(data || []);
  };

  const handleJoinSession = async () => {
    if (!hostKey.trim()) {
      toast({
        title: "Error",
        description: "Please enter a host key",
        variant: "destructive",
      });
      return;
    }

    setJoiningSession(true);

    try {
      // Verify the host key exists and is active
      const { data: keyData, error: keyError } = await supabase
        .from("host_keys")
        .select("*")
        .eq("key_code", hostKey.trim().toUpperCase())
        .eq("is_active", true)
        .single();

      if (keyError || !keyData) {
        toast({
          title: "Invalid Key",
          description: "The host key you entered is invalid or inactive",
          variant: "destructive",
        });
        return;
      }

      // Create a new analytics session
      const { data: sessionData, error: sessionError } = await supabase
        .from("analytics_sessions")
        .insert({
          participant_id: user.id,
          host_key_id: keyData.id,
          session_name: sessionName.trim() || null,
        })
        .select()
        .single();

      if (sessionError) {
        toast({
          title: "Error",
          description: "Failed to join session",
          variant: "destructive",
        });
        return;
      }

      toast({
        title: "Success!",
        description: "You've joined the session. The host can now see your analytics.",
      });

      setHostKey("");
      setSessionName("");
      setDialogOpen(false);
      fetchSessions();
    } catch (error: any) {
      toast({
        title: "Error",
        description: error.message,
        variant: "destructive",
      });
    } finally {
      setJoiningSession(false);
    }
  };

  const handleSignOut = async () => {
    await supabase.auth.signOut();
    navigate("/auth");
  };

  // Build live chart rows from analytics_data (latest session)
  const stressData = useMemo(() => {
    return analyticsData.map(d => ({
      time: new Date(d.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      stress: Math.round(d.stress_score || 0),
      attention: Math.round(d.attention_score || 0),
    }));
  }, [analyticsData]);

  const emotionDist = useMemo(() => {
    const counts: Record<string, number> = {};
    for (const d of analyticsData) {
      const e = d.emotion_state || 'Unknown';
      counts[e] = (counts[e] || 0) + 1;
    }
    const entries = Object.entries(counts);
    if (entries.length === 0) return [] as { emotion: string; count: number }[];
    return entries.map(([emotion, count]) => ({ emotion, count }));
  }, [analyticsData]);

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary mx-auto"></div>
          <p className="mt-4 text-muted-foreground">Loading...</p>
        </div>
      </div>
    );
  }

  const latestSession = sessions[0];
  const avgStress = stressData.length ? (stressData.reduce((sum, d) => sum + d.stress, 0) / stressData.length) : 0;
  const avgAttention = stressData.length ? (stressData.reduce((sum, d) => sum + d.attention, 0) / stressData.length) : 0;

  return (
    <div className="min-h-screen bg-gradient-hero p-4 md:p-8">
      <div className="max-w-7xl mx-auto space-y-8">
        {/* Header */}
        <div className="flex justify-between items-center">
          <div>
            <h1 className="text-3xl font-bold text-foreground">Your Analytics</h1>
            <p className="text-muted-foreground">Track your stress and attention during meetings</p>
          </div>
          <div className="flex gap-2">
            <Dialog open={dialogOpen} onOpenChange={setDialogOpen}>
              <DialogTrigger asChild>
                <Button>
                  <Plus className="mr-2 h-4 w-4" />
                  Join Session
                </Button>
              </DialogTrigger>
              <DialogContent>
                <DialogHeader>
                  <DialogTitle>Join a Host Session</DialogTitle>
                  <DialogDescription>
                    Enter the host key provided by your meeting organizer
                  </DialogDescription>
                </DialogHeader>
                <div className="space-y-4">
                  <div className="space-y-2">
                    <Label htmlFor="sessionName">Session Name (Optional)</Label>
                    <Input
                      id="sessionName"
                      placeholder="e.g., Morning Standup, Team Review"
                      value={sessionName}
                      onChange={(e) => setSessionName(e.target.value)}
                    />
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="hostKey">Host Key *</Label>
                    <Input
                      id="hostKey"
                      placeholder="Enter host key (e.g., KEY-ABC123)"
                      value={hostKey}
                      onChange={(e) => setHostKey(e.target.value)}
                      className="font-mono"
                    />
                  </div>
                  <Button 
                    onClick={handleJoinSession} 
                    className="w-full"
                    disabled={joiningSession}
                  >
                    {joiningSession ? "Joining..." : "Join Session"}
                  </Button>
                </div>
              </DialogContent>
            </Dialog>
            <Button variant="outline" onClick={handleSignOut}>
              <LogOut className="mr-2 h-4 w-4" />
              Sign Out
            </Button>
          </div>
        </div>

        {/* Monitoring (Start/Stop Camera) */}
        <MonitoringPanel />

        {/* Active Sessions */}
        {sessions.length > 0 && (
          <Card className="shadow-soft border-primary/20">
            <CardHeader>
              <div className="flex items-center gap-2">
                <Key className="h-5 w-5 text-primary" />
                <CardTitle>Active Sessions</CardTitle>
              </div>
              <CardDescription>You are currently being monitored in {sessions.length} session(s)</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-2">
                {sessions.slice(0, 3).map((session) => (
                  <div key={session.id} className="flex items-center justify-between p-3 border rounded-lg bg-card">
                    <div>
                      <p className="font-medium text-sm">
                        {session.session_name || "Unnamed Session"}
                      </p>
                      <p className="text-xs text-muted-foreground">
                        Started {new Date(session.started_at).toLocaleString()}
                      </p>
                    </div>
                    <div className="text-xs bg-success/10 text-success px-2 py-1 rounded">
                      Active
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        )}

        {/* Stats Cards */}
        <div className="grid md:grid-cols-4 gap-6">
          <Card className="shadow-soft">
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Avg Stress</CardTitle>
              <Activity className="h-4 w-4 text-accent" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-accent">{avgStress.toFixed(1)}</div>
              <p className="text-xs text-muted-foreground">Current session</p>
            </CardContent>
          </Card>

          <Card className="shadow-soft">
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Attention</CardTitle>
              <Brain className="h-4 w-4 text-primary" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-primary">{avgAttention.toFixed(1)}</div>
              <p className="text-xs text-muted-foreground">Average score</p>
            </CardContent>
          </Card>

          <Card className="shadow-soft">
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Eye Closures</CardTitle>
              <Eye className="h-4 w-4 text-warning" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">12</div>
              <p className="text-xs text-muted-foreground">Times detected</p>
            </CardContent>
          </Card>

          <Card className="shadow-soft">
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Heart Rate</CardTitle>
              <Heart className="h-4 w-4 text-destructive" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">78</div>
              <p className="text-xs text-muted-foreground">BPM average</p>
            </CardContent>
          </Card>
        </div>

        {/* Stress & Attention Chart */}
        <Card className="shadow-soft">
          <CardHeader>
            <CardTitle>Stress & Attention Over Time</CardTitle>
            <CardDescription>Real-time monitoring during your session</CardDescription>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={300}>
              <AreaChart data={stressData}>
                <defs>
                  <linearGradient id="colorStress" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="hsl(var(--accent))" stopOpacity={0.3} />
                    <stop offset="95%" stopColor="hsl(var(--accent))" stopOpacity={0} />
                  </linearGradient>
                  <linearGradient id="colorAttention" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="hsl(var(--primary))" stopOpacity={0.3} />
                    <stop offset="95%" stopColor="hsl(var(--primary))" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
                <XAxis dataKey="time" />
                <YAxis />
                <Tooltip />
                <Legend />
                <Area
                  type="monotone"
                  dataKey="stress"
                  stroke="hsl(var(--accent))"
                  fillOpacity={1}
                  fill="url(#colorStress)"
                  name="Stress Level"
                />
                <Area
                  type="monotone"
                  dataKey="attention"
                  stroke="hsl(var(--primary))"
                  fillOpacity={1}
                  fill="url(#colorAttention)"
                  name="Attention Score"
                />
              </AreaChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>

        {/* Eye Closure Timeline */}
        <Card className="shadow-soft">
          <CardHeader>
            <CardTitle>Eye Closure Events</CardTitle>
            <CardDescription>Timestamps and duration of detected eye closures</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {[
                { time: "10:15:32", duration: 2.3 },
                { time: "10:23:18", duration: 1.8 },
                { time: "10:31:45", duration: 3.1 },
                { time: "10:42:09", duration: 2.0 },
                { time: "10:58:23", duration: 4.2 },
              ].map((event, index) => (
                <div
                  key={index}
                  className="flex items-center justify-between p-3 border rounded-lg bg-card"
                >
                  <div className="flex items-center gap-3">
                    <Eye className="h-4 w-4 text-warning" />
                    <div>
                      <p className="font-medium text-sm">{event.time}</p>
                      <p className="text-xs text-muted-foreground">Event #{index + 1}</p>
                    </div>
                  </div>
                  <div className="text-right">
                    <p className="font-semibold">{event.duration}s</p>
                    <p className="text-xs text-muted-foreground">Duration</p>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>

        {/* Emotion State Distribution */}
        <Card className="shadow-soft">
          <CardHeader>
            <CardTitle>Emotion State Distribution</CardTitle>
            <CardDescription>Breakdown of your emotional states during the session</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {emotionDist.map((emotion) => {
                const total = emotionDist.reduce((sum, e) => sum + e.count, 0);
                const percentage = (emotion.count / total) * 100;
                
                return (
                  <div key={emotion.emotion} className="space-y-2">
                    <div className="flex justify-between text-sm">
                      <span className="font-medium">{emotion.emotion}</span>
                      <span className="text-muted-foreground">{percentage.toFixed(0)}%</span>
                    </div>
                    <div className="w-full bg-secondary rounded-full h-2">
                      <div
                        className="bg-primary h-2 rounded-full transition-all"
                        style={{ width: `${percentage}%` }}
                      />
                    </div>
                  </div>
                );
              })}
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
};

export default ParticipantDashboard;
