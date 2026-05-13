#!/usr/bin/env python3
"""
推荐系统技术动态抓取脚本
每晚22点运行，检查各博主/公司博客是否有新文章，汇总到 UPDATES.md
"""

import json
import os
import re
import sys
import time
import urllib.request
import urllib.parse
from datetime import datetime, timedelta, timezone

# ── 时区 ──────────────────────────────────────────────────────────────────────
CST = timezone(timedelta(hours=8))

def now_cst():
    return datetime.now(CST)

def today_str():
    return now_cst().strftime("%Y-%m-%d")

# ── 信息源配置 ─────────────────────────────────────────────────────────────────
SOURCES = [
    # --- 国内知乎专栏 (RSS) ---
    {
        "name": "王喆的机器学习笔记",
        "type": "zhihu_rss",
        "url": "https://www.zhihu.com/rss/column/wangzhenotes",
        "rss": "https://rsshub.app/zhihu/zhuanlan/wangzhenotes",
        "category": "国内博主",
        "tags": ["精排", "召回", "LLM", "生成式推荐"],
    },
    {
        "name": "石塔西",
        "type": "zhihu_rss",
        "url": "https://www.zhihu.com/people/si-ta-xi/posts",
        "rss": "https://rsshub.app/zhihu/people/si-ta-xi/activities",
        "category": "国内博主",
        "tags": ["召回", "冷启动", "特征工程"],
    },
    # --- 海外博客 (RSS) ---
    {
        "name": "Eugene Yan",
        "type": "rss",
        "url": "https://eugeneyan.com/",
        "rss": "https://eugeneyan.com/rss.xml",
        "category": "海外博主",
        "tags": ["LLM", "推荐系统", "工程实践"],
    },
    # --- 大厂技术博客 (RSS) ---
    {
        "name": "Netflix TechBlog",
        "type": "rss",
        "url": "https://netflixtechblog.com/tagged/recommendation-system",
        "rss": "https://netflixtechblog.com/feed",
        "category": "海外公司",
        "tags": ["精排", "召回", "基础模型"],
    },
    {
        "name": "美团技术团队",
        "type": "rss",
        "url": "https://tech.meituan.com/",
        "rss": "https://tech.meituan.com/feed/",
        "category": "国内大厂",
        "tags": ["搜索", "推荐", "广告"],
    },
    {
        "name": "Pinterest Engineering",
        "type": "rss",
        "url": "https://medium.com/pinterest-engineering",
        "rss": "https://medium.com/feed/pinterest-engineering",
        "category": "海外公司",
        "tags": ["GNN", "多模态", "召回"],
    },
    {
        "name": "Spotify Engineering",
        "type": "rss",
        "url": "https://engineering.atspotify.com/",
        "rss": "https://engineering.atspotify.com/feed",
        "category": "海外公司",
        "tags": ["音乐推荐", "用户兴趣"],
    },
    # --- GitHub 仓库动态 ---
    {
        "name": "Algorithm-Practice-in-Industry (Doragd)",
        "type": "github_commits",
        "url": "https://github.com/Doragd/Algorithm-Practice-in-Industry",
        "api": "https://api.github.com/repos/Doragd/Algorithm-Practice-in-Industry/commits?per_page=5",
        "category": "GitHub",
        "tags": ["工业实践", "搜广推"],
    },
    {
        "name": "LLM4Rec-Awesome-Papers (WLiK)",
        "type": "github_commits",
        "url": "https://github.com/WLiK/LLM4Rec-Awesome-Papers",
        "api": "https://api.github.com/repos/WLiK/LLM4Rec-Awesome-Papers/commits?per_page=5",
        "category": "GitHub",
        "tags": ["LLM", "推荐系统"],
    },
    {
        "name": "microsoft/recommenders",
        "type": "github_commits",
        "url": "https://github.com/microsoft/recommenders",
        "api": "https://api.github.com/repos/microsoft/recommenders/commits?per_page=5",
        "category": "GitHub",
        "tags": ["工具库", "算法实现"],
    },
]

# ── 工具函数 ───────────────────────────────────────────────────────────────────

def http_get(url, timeout=15):
    """带 UA 的 HTTP GET，返回 (body_str, status_code)"""
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (compatible; RecsysBot/1.0)",
            "Accept": "application/rss+xml, application/xml, application/json, text/html, */*",
        }
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            return body, resp.status
    except Exception as e:
        return None, str(e)

def parse_rss(xml_text):
    """简单 RSS/Atom 解析，返回 [{title, link, pubdate}]"""
    items = []
    # 匹配 <item> 或 <entry>
    blocks = re.findall(r'<(?:item|entry)>(.*?)</(?:item|entry)>', xml_text, re.DOTALL)
    for block in blocks[:10]:
        title = re.search(r'<title[^>]*>(.*?)</title>', block, re.DOTALL)
        link  = re.search(r'<link[^>]*>(.*?)</link>|<link[^>]+href=["\']([^"\']+)["\']', block, re.DOTALL)
        date  = re.search(r'<(?:pubDate|published|updated)[^>]*>(.*?)</(?:pubDate|published|updated)>', block, re.DOTALL)
        if title and link:
            title_text = re.sub(r'<[^>]+>', '', title.group(1)).strip()
            link_text  = (link.group(1) or link.group(2) or "").strip()
            date_text  = date.group(1).strip() if date else ""
            # 清理 CDATA
            title_text = re.sub(r'<!\[CDATA\[(.*?)\]\]>', r'\1', title_text)
            link_text  = re.sub(r'<!\[CDATA\[(.*?)\]\]>', r'\1', link_text)
            items.append({"title": title_text, "link": link_text, "date": date_text})
    return items

def is_recent(date_str, days=1):
    """判断日期字符串是否在最近 days 天内"""
    if not date_str:
        return True  # 无法判断则纳入
    # 常见格式尝试解析
    formats = [
        "%a, %d %b %Y %H:%M:%S %z",
        "%a, %d %b %Y %H:%M:%S GMT",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%d",
    ]
    for fmt in formats:
        try:
            dt = datetime.strptime(date_str.strip(), fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            cutoff = datetime.now(timezone.utc) - timedelta(days=days)
            return dt >= cutoff
        except Exception:
            continue
    return True  # 解析失败则纳入

def fetch_rss_source(source):
    """抓取 RSS 源，返回最近1天的新文章列表"""
    body, status = http_get(source["rss"])
    if not body:
        return [], f"请求失败: {status}"
    items = parse_rss(body)
    recent = [i for i in items if is_recent(i["date"], days=1)]
    return recent, None

def fetch_github_source(source):
    """抓取 GitHub commits，返回最近1天的提交"""
    body, status = http_get(source["api"])
    if not body:
        return [], f"请求失败: {status}"
    try:
        commits = json.loads(body)
        if not isinstance(commits, list):
            return [], f"API 返回非列表: {str(commits)[:100]}"
        recent = []
        cutoff = datetime.now(timezone.utc) - timedelta(days=1)
        for c in commits[:5]:
            date_str = c.get("commit", {}).get("author", {}).get("date", "")
            msg = c.get("commit", {}).get("message", "").split("\n")[0]
            sha = c.get("sha", "")[:7]
            link = f"{source['url']}/commit/{c.get('sha','')}"
            try:
                dt = datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
                if dt >= cutoff:
                    recent.append({"title": f"[{sha}] {msg}", "link": link, "date": date_str})
            except Exception:
                pass
        return recent, None
    except Exception as e:
        return [], str(e)

# ── 主流程 ─────────────────────────────────────────────────────────────────────

def fetch_all():
    results = {}
    for source in SOURCES:
        name = source["name"]
        print(f"  正在抓取: {name} ...", flush=True)
        try:
            if source["type"] in ("rss", "zhihu_rss"):
                items, err = fetch_rss_source(source)
            elif source["type"] == "github_commits":
                items, err = fetch_github_source(source)
            else:
                items, err = [], "未知类型"
        except Exception as e:
            items, err = [], str(e)

        results[name] = {
            "source": source,
            "items": items,
            "error": err,
            "count": len(items),
        }
        time.sleep(0.8)  # 避免请求过快
    return results

def generate_report(results):
    """生成 Markdown 更新报告"""
    date = today_str()
    has_updates = any(r["count"] > 0 for r in results.values())

    lines = [
        f"# 📡 推荐系统技术动态 — {date}",
        "",
        f"> 自动抓取时间：{now_cst().strftime('%Y-%m-%d %H:%M')} CST",
        f"> 监控源数量：{len(SOURCES)} 个",
        f"> 本次有更新：{'✅ 有新内容' if has_updates else '😴 暂无新内容'}",
        "",
        "---",
        "",
    ]

    if not has_updates:
        lines += [
            "今日各监控源暂无新内容，明日继续。",
            "",
            "---",
            f"*由懂哥自动巡查脚本生成 · {date}*",
        ]
        return "\n".join(lines)

    # 按 category 分组输出
    categories = {}
    for name, r in results.items():
        if r["count"] == 0:
            continue
        cat = r["source"]["category"]
        categories.setdefault(cat, []).append((name, r))

    cat_emoji = {
        "国内博主": "🇨🇳",
        "海外博主": "🌍",
        "国内大厂": "🏢",
        "海外公司": "🏭",
        "GitHub": "📦",
    }

    for cat, entries in categories.items():
        emoji = cat_emoji.get(cat, "📌")
        lines.append(f"## {emoji} {cat}")
        lines.append("")
        for name, r in entries:
            source = r["source"]
            tags = " ".join([f"`{t}`" for t in source.get("tags", [])])
            lines.append(f"### [{name}]({source['url']})  {tags}")
            lines.append("")
            for item in r["items"][:5]:
                title = item["title"][:80] + ("…" if len(item["title"]) > 80 else "")
                link = item["link"]
                date_info = f" · {item['date'][:10]}" if item.get("date") else ""
                lines.append(f"- [{title}]({link}){date_info}")
            lines.append("")

    # 无更新的源列表
    no_update = [name for name, r in results.items() if r["count"] == 0 and not r["error"]]
    failed = [(name, r["error"]) for name, r in results.items() if r["error"]]

    if no_update:
        lines += ["---", "", "**无更新的源：** " + "、".join(no_update), ""]
    if failed:
        lines += ["**抓取失败的源：**", ""]
        for name, err in failed:
            lines.append(f"- {name}：{err[:80]}")
        lines.append("")

    lines += [
        "---",
        f"*由懂哥自动巡查脚本生成 · {date}*",
    ]
    return "\n".join(lines)

def save_report(content):
    """保存报告到 UPDATES/ 目录"""
    repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    updates_dir = os.path.join(repo_dir, "UPDATES")
    os.makedirs(updates_dir, exist_ok=True)

    date = today_str()
    filepath = os.path.join(updates_dir, f"{date}.md")
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)

    # 同时更新 LATEST_UPDATE.md 方便直接查看
    latest_path = os.path.join(repo_dir, "LATEST_UPDATE.md")
    with open(latest_path, "w", encoding="utf-8") as f:
        f.write(content)

    return filepath, latest_path

def git_commit_push(repo_dir, date):
    """commit & push 到 GitHub"""
    import subprocess
    cmds = [
        ["git", "-C", repo_dir, "add", "UPDATES/", "LATEST_UPDATE.md"],
        ["git", "-C", repo_dir, "commit", "-m",
         f"chore: 自动更新技术动态 {date}"],
        ["git", "-C", repo_dir, "push", "origin", "master"],
    ]
    for cmd in cmds:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  ⚠️ 命令失败: {' '.join(cmd)}")
            print(f"  stderr: {result.stderr[:200]}")
            return False
    return True

# ── 入口 ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"[{now_cst().strftime('%H:%M:%S')}] 开始抓取推荐系统技术动态...")
    results = fetch_all()

    total_new = sum(r["count"] for r in results.values())
    print(f"\n✅ 抓取完成，共发现 {total_new} 条新内容")

    report = generate_report(results)
    repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    filepath, latest_path = save_report(report)
    print(f"📄 报告已保存: {filepath}")

    if total_new > 0:
        print("📤 推送到 GitHub...")
        ok = git_commit_push(repo_dir, today_str())
        print("✅ 推送成功" if ok else "❌ 推送失败（检查 token 配置）")

    # 将结果输出到 stdout 供 cron agent 读取
    print("\n=== REPORT_START ===")
    print(report)
    print("=== REPORT_END ===")
