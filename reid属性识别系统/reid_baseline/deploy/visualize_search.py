import argparse
import html
from pathlib import Path

from reid_baseline.deploy.inferencer import GalleryIndex, ReIDInferencer


def parse_args():
    parser = argparse.ArgumentParser(description="Create an HTML report for ReID search results.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--index", required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--output", default="deploy/search_report.html")
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def image_uri(path):
    return Path(path).expanduser().resolve().as_uri()


def render_html(query_path, results):
    cards = []
    for item in results:
        path = item.get("path")
        image = f'<img src="{image_uri(path)}" alt="{html.escape(item["id"])}">' if path else ""
        cards.append(f"""
        <article class="card">
          <div class="rank">Rank-{item["rank"]}</div>
          {image}
          <div class="score">score: {item["score"]:.4f}</div>
          <div class="id">{html.escape(item["id"])}</div>
          <div class="path">{html.escape(path or "")}</div>
        </article>
        """)

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>ReID Search Report</title>
  <style>
    body {{
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #f5f7f9;
      color: #17202a;
    }}
    header {{
      padding: 24px 32px 12px;
      border-bottom: 1px solid #d8dee6;
      background: #ffffff;
    }}
    h1 {{
      margin: 0;
      font-size: 24px;
      font-weight: 700;
      letter-spacing: 0;
    }}
    main {{
      padding: 24px 32px 40px;
    }}
    .query {{
      display: flex;
      gap: 20px;
      align-items: flex-start;
      margin-bottom: 28px;
      padding-bottom: 24px;
      border-bottom: 1px solid #d8dee6;
    }}
    .query img, .card img {{
      width: 128px;
      height: 256px;
      object-fit: cover;
      background: #e5e9ef;
      border: 1px solid #ccd3dc;
    }}
    .query-meta {{
      max-width: 760px;
      font-size: 14px;
      line-height: 1.6;
      word-break: break-all;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(190px, 1fr));
      gap: 16px;
    }}
    .card {{
      background: #ffffff;
      border: 1px solid #d8dee6;
      border-radius: 8px;
      padding: 12px;
      min-width: 0;
    }}
    .rank {{
      font-size: 15px;
      font-weight: 700;
      margin-bottom: 10px;
    }}
    .score {{
      margin-top: 10px;
      font-size: 14px;
      font-weight: 700;
    }}
    .id, .path {{
      margin-top: 6px;
      font-size: 12px;
      color: #475569;
      overflow-wrap: anywhere;
    }}
  </style>
</head>
<body>
  <header><h1>ReID Search Report</h1></header>
  <main>
    <section class="query">
      <img src="{image_uri(query_path)}" alt="query">
      <div class="query-meta">
        <strong>Query</strong><br>
        {html.escape(str(Path(query_path).resolve()))}
      </div>
    </section>
    <section class="grid">
      {"".join(cards)}
    </section>
  </main>
</body>
</html>
"""


def main():
    args = parse_args()
    inferencer = ReIDInferencer(args.checkpoint, device=args.device)
    index = GalleryIndex.load(args.index)
    query_feature = inferencer.extract_image(args.query)
    results = index.search(query_feature, topk=args.topk, include_path=True)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_html(args.query, results), encoding="utf-8")
    print(f"可视化报告已生成: {output.resolve()}")


if __name__ == "__main__":
    main()

