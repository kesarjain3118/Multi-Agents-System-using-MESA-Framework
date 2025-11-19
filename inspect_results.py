# analyze_results.py
import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv("combined_with_dynamic_cleaned.csv")

# Basic counts
print("Rows:", len(df))
print(df.head())

# Per-model summary: success rate, avg steps (all), median steps, avg steps (successful only), avg path length
summary = df.groupby("model").agg(
    Success_Rate = ("reached", lambda s: 100*s.mean()),
    Avg_Steps_All = ("steps", "mean"),
    Median_Steps_All = ("steps", "median"),
    Avg_Steps_Success = ("steps", lambda s: s[df.loc[s.index, "reached"]].mean() if s[df.loc[s.index, "reached"]].any() else float("nan")),
    Avg_Path = ("path_length", "mean"),
    Count = ("agent_id", "count")
).sort_values("Success_Rate", ascending=False)

print(summary)

# Plot success rates
summary["Success_Rate"].plot(kind="bar", title="Success Rate by Model", ylabel="Success %", rot=45)
plt.tight_layout()
plt.savefig("compare_plot_success_rate.png")
plt.close()

# Plot median steps (less sensitive to timeouts)
summary["Median_Steps_All"].plot(kind="bar", title="Median Steps by Model", ylabel="Steps", rot=45)
plt.tight_layout()
plt.savefig("compare_plot_median_steps.png")
plt.close()

print("Saved compare_plot_success_rate.png and compare_plot_median_steps.png")
