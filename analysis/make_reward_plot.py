from __future__ import annotations

import csv
from pathlib import Path

root = Path(__file__).resolve().parent
rows = list(csv.DictReader((root / "reward_block_means.csv").open(encoding="utf-8")))
series: dict[str, list[tuple[float, float]]] = {"twinkle": [], "verl": []}

for row in rows:
    midpoint = (float(row["start_step"]) + float(row["end_step"])) / 2
    series[row["framework"]].append((midpoint, float(row["mean_reward"])))

width, height = 1000, 600
left, right, top, bottom = 85, 35, 60, 75
plot_w, plot_h = width - left - right, height - top - bottom

def sx(step: float) -> float:
    return left + (step - 1) / 124 * plot_w

def sy(reward: float) -> float:
    return top + (1 - reward) * plot_h

parts = [
    f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
    '<rect width="100%" height="100%" fill="white"/>',
    '<text x="500" y="32" text-anchor="middle" font-family="sans-serif" font-size="22" font-weight="bold">Twinkle vs VERL Reward Trend (10-step block means)</text>',
]

for i in range(11):
    reward = i / 10
    y = sy(reward)
    parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" stroke="#e5e7eb"/>')
    parts.append(f'<text x="{left-12}" y="{y+5:.1f}" text-anchor="end" font-family="sans-serif" font-size="13">{reward:.1f}</text>')

for step in (1, 25, 50, 75, 100, 125):
    x = sx(step)
    parts.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{height-bottom}" stroke="#f1f5f9"/>')
    parts.append(f'<text x="{x:.1f}" y="{height-bottom+25}" text-anchor="middle" font-family="sans-serif" font-size="13">{step}</text>')

parts.extend([
    f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#111827"/>',
    f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#111827"/>',
    f'<text x="{left + plot_w/2:.1f}" y="{height-22}" text-anchor="middle" font-family="sans-serif" font-size="15">Training step</text>',
    f'<text x="22" y="{top + plot_h/2:.1f}" text-anchor="middle" transform="rotate(-90 22 {top + plot_h/2:.1f})" font-family="sans-serif" font-size="15">Mean reward</text>',
])

colors = {"twinkle": "#2563eb", "verl": "#dc2626"}
labels = {"twinkle": "Twinkle", "verl": "VERL"}

for index, name in enumerate(("twinkle", "verl")):
    points = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in series[name])
    parts.append(f'<polyline points="{points}" fill="none" stroke="{colors[name]}" stroke-width="4" stroke-linejoin="round" stroke-linecap="round"/>')
    for x, y in series[name]:
        parts.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="4.5" fill="white" stroke="{colors[name]}" stroke-width="3"><title>{labels[name]} steps around {x:.1f}: {y:.6f}</title></circle>')
    legend_x = 730
    legend_y = 78 + index * 26
    parts.append(f'<line x1="{legend_x}" y1="{legend_y}" x2="{legend_x+35}" y2="{legend_y}" stroke="{colors[name]}" stroke-width="4"/>')
    parts.append(f'<text x="{legend_x+45}" y="{legend_y+5}" font-family="sans-serif" font-size="14">{labels[name]}</text>')

parts.append('</svg>')
(root / "reward_curves.svg").write_text("\n".join(parts), encoding="utf-8")
print(root / "reward_curves.svg")
