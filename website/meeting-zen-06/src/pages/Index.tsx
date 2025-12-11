import { useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Brain, Users, Activity, Eye, TrendingUp, Shield } from "lucide-react";

const Index = () => {
  const navigate = useNavigate();

  const features = [
    {
      icon: Brain,
      title: "AI-Powered Detection",
      description: "Advanced machine learning algorithms monitor stress levels in real-time during virtual meetings.",
    },
    {
      icon: Activity,
      title: "Real-Time Analytics",
      description: "Track attention scores, emotion states, and stress levels with live data visualization.",
    },
    {
      icon: Eye,
      title: "Eye Tracking",
      description: "Monitor eye closure events and patterns to detect fatigue and attention lapses.",
    },
    {
      icon: TrendingUp,
      title: "Detailed Insights",
      description: "Comprehensive reports with graphs, timestamps, and actionable recommendations.",
    },
    {
      icon: Users,
      title: "Host Management",
      description: "Hosts can monitor all participants and view aggregated analytics in a leaderboard format.",
    },
    {
      icon: Shield,
      title: "Privacy First",
      description: "Your data is secure and private. Hosts see analytics, not personal information.",
    },
  ];

  return (
    <div className="min-h-screen bg-gradient-hero">
      {/* Hero Section */}
      <div className="container mx-auto px-4 py-20 text-center">
        <div className="max-w-4xl mx-auto space-y-8">
          <div className="inline-flex items-center gap-2 bg-primary/10 text-primary px-4 py-2 rounded-full text-sm font-medium">
            <Activity className="w-4 h-4" />
            AI-Powered Stress Detection
          </div>
          
          <h1 className="text-5xl md:text-6xl font-bold text-foreground leading-tight">
            Monitor Stress & Attention
            <br />
            <span className="text-primary">During Virtual Meetings</span>
          </h1>
          
          <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
            A revolutionary platform that uses AI to detect stress, track attention, and provide
            actionable insights to improve meeting wellness and productivity.
          </p>

          <div className="flex gap-4 justify-center pt-4">
            <Button
              size="lg"
              className="shadow-glow"
              onClick={() => navigate("/auth")}
            >
              <Users className="mr-2 h-5 w-5" />
              Join as Participant
            </Button>
            <Button
              size="lg"
              variant="outline"
              onClick={() => navigate("/auth")}
            >
              <Brain className="mr-2 h-5 w-5" />
              Host a Session
            </Button>
          </div>
        </div>
      </div>

      {/* Features Grid */}
      <div className="container mx-auto px-4 py-20">
        <div className="text-center mb-16">
          <h2 className="text-3xl md:text-4xl font-bold mb-4">Powerful Features</h2>
          <p className="text-muted-foreground text-lg">
            Everything you need to understand and improve meeting wellness
          </p>
        </div>

        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8 max-w-6xl mx-auto">
          {features.map((feature) => (
            <Card key={feature.title} className="shadow-soft hover:shadow-glow transition-shadow">
              <CardHeader>
                <div className="w-12 h-12 rounded-lg bg-primary/10 flex items-center justify-center mb-4">
                  <feature.icon className="w-6 h-6 text-primary" />
                </div>
                <CardTitle className="text-xl">{feature.title}</CardTitle>
              </CardHeader>
              <CardContent>
                <CardDescription className="text-base">{feature.description}</CardDescription>
              </CardContent>
            </Card>
          ))}
        </div>
      </div>

      {/* How It Works */}
      <div className="container mx-auto px-4 py-20">
        <div className="text-center mb-16">
          <h2 className="text-3xl md:text-4xl font-bold mb-4">How It Works</h2>
          <p className="text-muted-foreground text-lg">
            Get started in three simple steps
          </p>
        </div>

        <div className="max-w-4xl mx-auto grid md:grid-cols-3 gap-8">
          <div className="text-center space-y-4">
            <div className="w-16 h-16 rounded-full bg-primary text-primary-foreground flex items-center justify-center text-2xl font-bold mx-auto">
              1
            </div>
            <h3 className="text-xl font-semibold">Sign Up</h3>
            <p className="text-muted-foreground">
              Choose your role - Host or Participant - and create your account in seconds.
            </p>
          </div>

          <div className="text-center space-y-4">
            <div className="w-16 h-16 rounded-full bg-primary text-primary-foreground flex items-center justify-center text-2xl font-bold mx-auto">
              2
            </div>
            <h3 className="text-xl font-semibold">Join Session</h3>
            <p className="text-muted-foreground">
              Hosts generate keys, participants enter the key to join monitoring sessions.
            </p>
          </div>

          <div className="text-center space-y-4">
            <div className="w-16 h-16 rounded-full bg-primary text-primary-foreground flex items-center justify-center text-2xl font-bold mx-auto">
              3
            </div>
            <h3 className="text-xl font-semibold">View Analytics</h3>
            <p className="text-muted-foreground">
              Access detailed dashboards with real-time stress, attention, and wellness metrics.
            </p>
          </div>
        </div>
      </div>

      {/* CTA Section */}
      <div className="container mx-auto px-4 py-20">
        <Card className="max-w-4xl mx-auto shadow-glow bg-gradient-calm text-primary-foreground">
          <CardContent className="p-12 text-center space-y-6">
            <h2 className="text-3xl md:text-4xl font-bold">
              Ready to Transform Your Virtual Meetings?
            </h2>
            <p className="text-lg opacity-90 max-w-2xl mx-auto">
              Join hundreds of teams already using our platform to create healthier,
              more productive virtual meeting environments.
            </p>
            <Button
              size="lg"
              variant="secondary"
              className="mt-4"
              onClick={() => navigate("/auth")}
            >
              Get Started Now
            </Button>
          </CardContent>
        </Card>
      </div>
    </div>
  );
};

export default Index;
