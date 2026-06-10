"""Generate report figures: the local-proxy vs leaderboard disconnect."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Local proxy FID (500 gen vs aligned held-out ref) across fine-tuning steps
steps = [0, 1000, 2000, 3000, 4000, 5000, 6000]
local_fid = [76.6, 71.8, 68.1, 67.1, 64.1, 67.7, 67.0]

# Leaderboard FID per submitted variant
labels = ["prior\nbest", "v1\ncenter", "v2\naligned", "v3\nGFPGAN", "v4\nblend"]
lb_fid = [45.43, 69.56, 69.39, 79.84, 81.27]
colors = ["#2a9d8f", "#e76f51", "#e76f51", "#e76f51", "#e76f51"]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 3.5))

ax1.plot(steps, local_fid, "o-", color="#264653", lw=2, ms=6)
ax1.annotate("base", (steps[0], local_fid[0]), textcoords="offset points",
             xytext=(6, 6), fontsize=9)
ax1.annotate("best ckpt\n(step 4000)", (steps[4], local_fid[4]),
             textcoords="offset points", xytext=(-2, -32), fontsize=8, ha="center")
ax1.set_xlabel("fine-tuning steps")
ax1.set_ylabel("local proxy FID")
ax1.set_title("(a) Local proxy: fine-tuning 'helps'", fontsize=10)
ax1.grid(alpha=0.3)
ax1.invert_yaxis()  # lower is better -> up

bars = ax2.bar(range(len(labels)), lb_fid, color=colors)
ax2.axhline(45.43, color="#2a9d8f", ls="--", lw=1.5)
ax2.text(3.4, 47, "prior best 45.4", color="#2a9d8f", fontsize=8)
for i, v in enumerate(lb_fid):
    ax2.text(i, v + 1.2, f"{v:.0f}", ha="center", fontsize=8)
ax2.set_xticks(range(len(labels)))
ax2.set_xticklabels(labels, fontsize=8)
ax2.set_ylabel("leaderboard FID")
ax2.set_title("(b) Leaderboard: every fine-tune is worse", fontsize=10)
ax2.set_ylim(0, 92)
ax2.grid(axis="y", alpha=0.3)

fig.tight_layout()
fig.savefig("report/figures/disconnect.pdf", bbox_inches="tight")
fig.savefig("report/figures/disconnect.png", dpi=150, bbox_inches="tight")
print("saved report/figures/disconnect.{pdf,png}")
