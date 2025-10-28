def suggest(emotion, stress_score, focus_score):
    out = []
    if stress_score >= 0.7:
        out.append("Take 3 slow breaths and a 2–3 minute break.")
    if focus_score < 0.4:
        out.append("Silence notifications and try a 25-minute focus sprint.")
    if emotion in ("sad", "fear"):
        out.append("Play calm music or take a short walk.")
    if emotion == "angry":
        out.append("Pause before acting and re-evaluate the task priority.")
    if not out:
        out.append("Keep going. Maintain your current pace.")
    return out[:3]
