import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { supabase } from "@/integrations/supabase/client";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { useToast } from "@/components/ui/use-toast";
import { LogOut, Plus, Key, TrendingUp, Users, Trash2, Download } from "lucide-react";
import { AlertDialog, AlertDialogAction, AlertDialogCancel, AlertDialogContent, AlertDialogDescription, AlertDialogFooter, AlertDialogHeader, AlertDialogTitle, AlertDialogTrigger } from "@/components/ui/alert-dialog";
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";

interface HostKey {
  id: string;
  key_code: string;
  name: string;
  is_active: boolean;
  created_at: string;
}

interface ParticipantSession {
  id: string;
  participant_id: string;
  session_name: string | null;
  started_at: string;
  overall_stress_score: number | null;
  overall_attention_score: number | null;
  profiles: {
    full_name: string | null;
    email: string;
  };
}

const HostDashboard = () => {
  const [user, setUser] = useState<any>(null);
  const [hostKeys, setHostKeys] = useState<HostKey[]>([]);
  const [sessions, setSessions] = useState<ParticipantSession[]>([]);
  const [loading, setLoading] = useState(true);
  const [newKeyName, setNewKeyName] = useState("");
  const [dialogOpen, setDialogOpen] = useState(false);
  const navigate = useNavigate();
  const { toast } = useToast();

  useEffect(() => {
    checkAuth();
    fetchHostKeys();
    fetchSessions();
    // Subscribe to realtime changes for live updates
    const ch = supabase
      .channel("host-realtime")
      .on("postgres_changes", { event: "INSERT", schema: "public", table: "analytics_sessions" }, () => {
        fetchSessions();
      })
      .on("postgres_changes", { event: "INSERT", schema: "public", table: "analytics_data" }, () => {
        // Light refresh to reflect latest metrics
        fetchSessions();
      })
      .subscribe();
    return () => {
      try { supabase.removeChannel(ch); } catch {}
    };
  }, []);

  const checkAuth = async () => {
    const { data: { user } } = await supabase.auth.getUser();
    if (!user) {
      navigate("/auth");
      return;
    }

    // Check if user is a host
    const { data: profile } = await supabase
      .from("profiles")
      .select("role")
      .eq("id", user.id)
      .single();

    if (profile?.role !== "host") {
      navigate("/participant-dashboard");
      return;
    }

    setUser(user);
    setLoading(false);
  };

  const fetchHostKeys = async () => {
    const { data, error } = await supabase
      .from("host_keys")
      .select("*")
      .order("created_at", { ascending: false });

    if (error) {
      toast({
        title: "Error",
        description: "Failed to fetch host keys",
        variant: "destructive",
      });
      return;
    }

    setHostKeys(data || []);
  };

  const fetchSessions = async () => {
    const { data, error } = await supabase
      .from("analytics_sessions")
      .select(`
        *,
        profiles:participant_id (
          full_name,
          email
        )
      `)
      .order("started_at", { ascending: false });

    if (error) {
      toast({
        title: "Error",
        description: "Failed to fetch sessions",
        variant: "destructive",
      });
      return;
    }

    setSessions(data || []);
  };

  const generateHostKey = async () => {
    if (!newKeyName.trim()) {
      toast({
        title: "Error",
        description: "Please enter a name for the key",
        variant: "destructive",
      });
      return;
    }

    const keyCode = `KEY-${Math.random().toString(36).substring(2, 10).toUpperCase()}`;

    const { error } = await supabase
      .from("host_keys")
      .insert({
        host_id: user.id,
        key_code: keyCode,
        name: newKeyName,
      });

    if (error) {
      toast({
        title: "Error",
        description: "Failed to generate key",
        variant: "destructive",
      });
      return;
    }

    toast({
      title: "Success",
      description: "New host key generated",
    });

    setNewKeyName("");
    setDialogOpen(false);
    fetchHostKeys();
  };

  const exportSessionCSV = async (sessionId: string) => {
    const { data, error } = await supabase
      .from("analytics_data")
      .select("*")
      .eq("session_id", sessionId)
      .order("timestamp", { ascending: true });
    if (error) {
      toast({ title: "Export failed", description: "Could not fetch analytics_data", variant: "destructive" });
      return;
    }
    const rows = data || [];
    const header = ["timestamp","emotion_state","stress_score","attention_score","heart_rate","notes"];
    const csv = [header.join(",")]
      .concat(rows.map((r: any) => [r.timestamp, r.emotion_state ?? "", r.stress_score ?? "", r.attention_score ?? "", r.heart_rate ?? "", String(r.notes ?? "").replace(/[\n\r,]+/g, " ")].join(",")))
      .join("\n");
    const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `session_${sessionId}.csv`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  };

  const deleteHostKey = async (keyId: string) => {
    const { error } = await supabase
      .from("host_keys")
      .delete()
      .eq("id", keyId);

    if (error) {
      toast({
        title: "Error",
        description: "Failed to delete host key",
        variant: "destructive",
      });
      return;
    }

    toast({
      title: "Success",
      description: "Host key deleted",
    });

    fetchHostKeys();
  };

  const handleSignOut = async () => {
    await supabase.auth.signOut();
    navigate("/auth");
  };

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

  return (
    <div className="min-h-screen bg-gradient-hero p-4 md:p-8">
      <div className="max-w-7xl mx-auto space-y-8">
        {/* Header */}
        <div className="flex justify-between items-center">
          <div>
            <h1 className="text-3xl font-bold text-foreground">Host Dashboard</h1>
            <p className="text-muted-foreground">Monitor and manage participant sessions</p>
          </div>
          <Button variant="outline" onClick={handleSignOut}>
            <LogOut className="mr-2 h-4 w-4" />
            Sign Out
          </Button>
        </div>

        {/* Stats Cards */}
        <div className="grid md:grid-cols-3 gap-6">
          <Card className="shadow-soft">
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Active Keys</CardTitle>
              <Key className="h-4 w-4 text-primary" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{hostKeys.filter(k => k.is_active).length}</div>
              <p className="text-xs text-muted-foreground">Host keys generated</p>
            </CardContent>
          </Card>

          <Card className="shadow-soft">
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Total Sessions</CardTitle>
              <Users className="h-4 w-4 text-primary" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{sessions.length}</div>
              <p className="text-xs text-muted-foreground">Participant sessions tracked</p>
            </CardContent>
          </Card>

          <Card className="shadow-soft">
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Avg Stress</CardTitle>
              <TrendingUp className="h-4 w-4 text-accent" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">
                {sessions.length > 0
                  ? (sessions.reduce((sum, s) => sum + (s.overall_stress_score || 0), 0) / sessions.length).toFixed(1)
                  : "0.0"}
              </div>
              <p className="text-xs text-muted-foreground">Across all sessions</p>
            </CardContent>
          </Card>
        </div>

        {/* Host Keys Section */}
        <Card className="shadow-soft">
          <CardHeader>
            <div className="flex justify-between items-center">
              <div>
                <CardTitle>Your Host Keys</CardTitle>
                <CardDescription>Generate and manage keys for participants</CardDescription>
              </div>
              <Dialog open={dialogOpen} onOpenChange={setDialogOpen}>
                <DialogTrigger asChild>
                  <Button>
                    <Plus className="mr-2 h-4 w-4" />
                    Generate Key
                  </Button>
                </DialogTrigger>
                <DialogContent>
                  <DialogHeader>
                    <DialogTitle>Generate New Host Key</DialogTitle>
                    <DialogDescription>
                      Create a new key for participants to join your sessions
                    </DialogDescription>
                  </DialogHeader>
                  <div className="space-y-4">
                    <div className="space-y-2">
                      <Label htmlFor="keyName">Key Name</Label>
                      <Input
                        id="keyName"
                        placeholder="e.g., Team Meeting, Workshop 2024"
                        value={newKeyName}
                        onChange={(e) => setNewKeyName(e.target.value)}
                      />
                    </div>
                    <Button onClick={generateHostKey} className="w-full">
                      Generate Key
                    </Button>
                  </div>
                </DialogContent>
              </Dialog>
            </div>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {hostKeys.length === 0 ? (
                <p className="text-center text-muted-foreground py-8">
                  No host keys yet. Generate your first key to get started!
                </p>
              ) : (
                hostKeys.map((key) => (
                  <div
                    key={key.id}
                    className="flex items-center justify-between p-4 border rounded-lg bg-card"
                  >
                    <div className="space-y-1">
                      <p className="font-medium">{key.name}</p>
                      <p className="text-sm text-muted-foreground font-mono">{key.key_code}</p>
                      <p className="text-xs text-muted-foreground">
                        Created {new Date(key.created_at).toLocaleDateString()}
                      </p>
                    </div>
                    <div className="flex items-center gap-3">
                      {key.is_active ? (
                        <span className="text-xs bg-success/10 text-success px-2 py-1 rounded">Active</span>
                      ) : (
                        <span className="text-xs bg-muted text-muted-foreground px-2 py-1 rounded">Inactive</span>
                      )}
                      <AlertDialog>
                        <AlertDialogTrigger asChild>
                          <Button variant="ghost" size="icon" className="h-8 w-8 text-destructive hover:text-destructive">
                            <Trash2 className="h-4 w-4" />
                          </Button>
                        </AlertDialogTrigger>
                        <AlertDialogContent>
                          <AlertDialogHeader>
                            <AlertDialogTitle>Delete Host Key</AlertDialogTitle>
                            <AlertDialogDescription>
                              Are you sure you want to delete "{key.name}"? This will remove all participant sessions associated with this key.
                            </AlertDialogDescription>
                          </AlertDialogHeader>
                          <AlertDialogFooter>
                            <AlertDialogCancel>Cancel</AlertDialogCancel>
                            <AlertDialogAction onClick={() => deleteHostKey(key.id)} className="bg-destructive text-destructive-foreground hover:bg-destructive/90">
                              Delete
                            </AlertDialogAction>
                          </AlertDialogFooter>
                        </AlertDialogContent>
                      </AlertDialog>
                    </div>
                  </div>
                ))
              )}
            </div>
          </CardContent>
        </Card>

        {/* Participants Leaderboard */}
        <Card className="shadow-soft">
          <CardHeader>
            <CardTitle>Participant Sessions</CardTitle>
            <CardDescription>View detailed analytics for each participant</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {sessions.length === 0 ? (
                <p className="text-center text-muted-foreground py-8">
                  No participant sessions yet
                </p>
              ) : (
                sessions.map((session, index) => (
                  <div
                    key={session.id}
                    className="flex items-center justify-between p-4 border rounded-lg bg-card hover:bg-accent/5 transition-colors cursor-pointer"
                    onClick={() => navigate(`/session/${session.id}`)}
                  >
                    <div className="flex items-center gap-4">
                      <div className="w-8 h-8 rounded-full bg-primary/10 flex items-center justify-center text-primary font-semibold">
                        {index + 1}
                      </div>
                      <div>
                        <p className="font-medium">{session.profiles?.full_name || "Unknown"}</p>
                        <p className="text-sm text-muted-foreground">{session.profiles?.email}</p>
                        <p className="text-xs text-muted-foreground">
                          {new Date(session.started_at).toLocaleString()}
                        </p>
                      </div>
                    </div>
                    <div className="flex items-center gap-4">
                      <button
                        className="h-8 w-8 flex items-center justify-center rounded border hover:bg-accent/10"
                        title="Export CSV"
                        onClick={(e) => { e.stopPropagation(); exportSessionCSV(session.id); }}
                      >
                        <Download className="h-4 w-4" />
                      </button>
                      <div className="flex gap-8 text-center">
                        <div>
                          <p className="text-2xl font-bold text-accent">
                            {session.overall_stress_score?.toFixed(1) || "N/A"}
                          </p>
                          <p className="text-xs text-muted-foreground">Stress</p>
                        </div>
                        <div>
                          <p className="text-2xl font-bold text-primary">
                            {session.overall_attention_score?.toFixed(1) || "N/A"}
                          </p>
                          <p className="text-xs text-muted-foreground">Attention</p>
                        </div>
                      </div>
                    </div>
                  </div>
                ))
              )}
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
};

export default HostDashboard;
