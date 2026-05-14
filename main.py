#!/usr/bin/env python3
"""
PubMed 文献检索 + AI 智能摘要 + 微信推送（Server酱）自动化系统
"""

import os
import sys
import time
from datetime import datetime
from xml.etree import ElementTree as ET

from Bio import Entrez
import requests
from openai import OpenAI


# ============================================================
# 全局配置（检索词和检索天数可直接修改）
# ============================================================
SEARCH_QUERY = (
    '("mouse inner ear organoids" OR "minced inner ear tissue" '
    'OR "E14" OR "E16" OR "neural PDEs" OR "digital twins") '
    "AND (biology OR medicine)"
)
SEARCH_DAYS = 7          # 检索最近 N 天的文献
MAX_RESULTS = 20         # 最大返回结果数

# ============================================================
# API 配置（必须通过环境变量读取）
# ============================================================
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1")
LLM_API_KEY = os.environ.get("LLM_API_KEY", "")
LLM_MODEL_NAME = os.environ.get("LLM_MODEL_NAME", "gpt-4o-mini")

ENTREZ_EMAIL = os.environ.get("ENTREZ_EMAIL", "")

SERVERCHAN_SENDKEY = os.environ.get("SERVERCHAN_SENDKEY", "")


# ============================================================
#  PubMed 检索
# ============================================================
def search_pubmed(query: str, reldate: int = 7, max_results: int = 20) -> list[str]:
    """检索 PubMed，返回 PMID 列表"""
    handle = Entrez.esearch(
        db="pubmed",
        term=query,
        reldate=reldate,
        datetype="pdat",
        retmax=max_results,
        sort="date",
    )
    record = Entrez.read(handle)
    handle.close()
    return record.get("IdList", [])


# ============================================================
#  获取文献详情
# ============================================================
def fetch_pubmed_details(pmid_list: list[str]) -> str:
    """根据 PMID 列表获取完整 XML 数据"""
    if not pmid_list:
        return ""
    handle = Entrez.efetch(
        db="pubmed",
        id=",".join(pmid_list),
        retmode="xml",
    )
    xml_data = handle.read()
    handle.close()
    return xml_data


# ============================================================
#  XML 解析
# ============================================================
def parse_articles(xml_data: str) -> list[dict]:
    """解析 PubMed XML，提取 Title / Authors / Journal / Date / DOI"""
    articles = []
    root = ET.fromstring(xml_data)

    for article_elem in root.findall(".//PubmedArticle"):
        try:
            article = {}
            medline = article_elem.find(".//MedlineCitation")
            if medline is None:
                continue

            # PMID
            pmid_elem = medline.find("PMID")
            article["pmid"] = pmid_elem.text if pmid_elem is not None else ""

            # Article 主节点
            art = medline.find(".//Article")
            if art is None:
                continue

            # Title
            title_elem = art.find("ArticleTitle")
            article["title"] = (
                "".join(title_elem.itertext()) if title_elem is not None else ""
            )

            # Abstract
            abstract_parts = []
            abstract_elem = art.find("Abstract")
            if abstract_elem is not None:
                for ab in abstract_elem.findall("AbstractText"):
                    label = ab.get("Label", "")
                    text = "".join(ab.itertext())
                    if label:
                        abstract_parts.append(f"{label}: {text}")
                    else:
                        abstract_parts.append(text)
            article["abstract"] = "\n".join(abstract_parts)

            # Authors
            authors = []
            author_list = art.find(".//AuthorList")
            if author_list is not None:
                for author in author_list.findall("Author"):
                    last = author.find("LastName")
                    fore = author.find("ForeName")
                    if last is not None and fore is not None:
                        authors.append(f"{last.text} {fore.text}")
                    elif last is not None:
                        authors.append(last.text)
            article["authors"] = ", ".join(authors) if authors else ""

            # Journal & Date
            journal = art.find("Journal")
            if journal is not None:
                title_elem = journal.find("Title")
                article["journal"] = title_elem.text if title_elem is not None else ""
                ji = journal.find("JournalIssue")
                if ji is not None:
                    pd = ji.find("PubDate")
                    if pd is not None:
                        year = pd.find("Year")
                        month = pd.find("Month")
                        day = pd.find("Day")
                        parts = []
                        if year is not None:
                            parts.append(year.text)
                        if month is not None:
                            parts.append(month.text)
                        if day is not None:
                            parts.append(day.text)
                        article["date"] = " ".join(parts)
                    else:
                        article["date"] = ""
            else:
                article["journal"] = ""
                article["date"] = ""

            # DOI
            doi = ""
            for eid in article_elem.findall(".//ArticleIdList/ArticleId"):
                if eid.get("IdType") == "doi":
                    doi = eid.text or ""
                    break
            article["doi"] = doi

            # 链接
            pmid = article.get("pmid", "")
            article["pmid_link"] = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
            article["doi_link"] = f"https://doi.org/{doi}" if doi else ""

            articles.append(article)

        except Exception as e:
            print(f"[WARN] 解析文献时出错: {e}")
            continue

    return articles


# ============================================================
#  AI 摘要生成
# ============================================================
def generate_summary(abstract: str) -> str:
    """调用大模型 API 生成 3 句中文核心结论"""
    if not abstract.strip():
        return "（无英文摘要）"
    if not LLM_API_KEY:
        return "（未配置 LLM_API_KEY，跳过 AI 摘要）"

    system_prompt = (
        "你是一个严谨的医学与生物学研究助手。请将以下英文摘要总结为"
        "3 句话的中文核心结论，重点突出实验模型"
        "（尤其是小鼠内耳切碎组织或类器官的发育成熟阶段）、"
        "核心发现和临床/科研意义。"
    )

    try:
        client = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)
        resp = client.chat.completions.create(
            model=LLM_MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": abstract},
            ],
            temperature=0.3,
            max_tokens=600,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"（AI 摘要生成失败: {e}）"


# ============================================================
#  Server酱 Markdown 消息构建
# ============================================================
def build_markdown(articles_with_summaries: list[dict]) -> str:
    """构建手机友好的 Markdown 推送内容"""
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        "#  PubMed 最新文献周报\n",
        f"> 检索时间：{now}  |  过去 {SEARCH_DAYS} 天\n",
        "---\n",
    ]

    for i, item in enumerate(articles_with_summaries, 1):
        art = item["article"]
        summary = item["summary"]

        lines.append(f"## {i}. {art.get('title', '无标题')}")
        lines.append("")
        lines.append(f"> ** 中文核心结论**\n> \n> {summary}\n")
        lines.append(f"**作者：** {art.get('authors', '未知')[:120]}")
        lines.append(
            f"**期刊：** {art.get('journal', '未知')}  |  "
            f"**日期：** {art.get('date', '未知')}"
        )

        links = []
        if art.get("pmid_link"):
            links.append(f"[PubMed]({art['pmid_link']})")
        if art.get("doi_link"):
            links.append(f"[DOI 全文]({art['doi_link']})")
        if links:
            lines.append(f"**链接：** {' | '.join(links)}")

        lines.append("")
        lines.append("---")
        lines.append("")

    lines.append(f"\n> 共检索到 {len(articles_with_summaries)} 篇相关文献\n")
    return "\n".join(lines)


# ============================================================
#  微信推送（Server酱）
# ============================================================
def push_to_wechat(title: str, desp: str) -> bool:
    """通过 Server酱 SendKey 推送微信消息"""
    if not SERVERCHAN_SENDKEY:
        print("[SKIP] 未设置 SERVERCHAN_SENDKEY，跳过微信推送")
        return False

    url = f"https://sctapi.ftqq.com/{SERVERCHAN_SENDKEY}.send"
    try:
        resp = requests.post(url, data={"title": title, "desp": desp}, timeout=30)
        result = resp.json()
        if resp.status_code == 200 and result.get("code") == 0:
            print(f"[OK] 微信推送成功：{result.get('data', {}).get('pushid', '')}")
            return True
        else:
            print(f"[FAIL] 微信推送失败: {result}")
            return False
    except Exception as e:
        print(f"[ERROR] 微信推送异常: {e}")
        return False


# ============================================================
#  main
# ============================================================
def main():
    """主流程"""
    # ---- 前置校验 ----
    if not ENTREZ_EMAIL:
        print("[FATAL] 未设置 ENTREZ_EMAIL 环境变量")
        sys.exit(1)
    if not LLM_API_KEY:
        print("[WARN] 未设置 LLM_API_KEY，AI 摘要将跳过")
    if not SERVERCHAN_SENDKEY:
        print("[WARN] 未设置 SERVERCHAN_SENDKEY，微信推送将跳过")

    Entrez.email = ENTREZ_EMAIL

    # ---- 检索 ----
    print(f"[1/4] 正在检索 PubMed ...")
    print(f"      查询: {SEARCH_QUERY}")
    print(f"      最近 {SEARCH_DAYS} 天")

    pmid_list = search_pubmed(SEARCH_QUERY, SEARCH_DAYS, MAX_RESULTS)
    print(f"      → 共找到 {len(pmid_list)} 篇文献")

    if not pmid_list:
        msg = "本周未检索到相关文献。"
        print(f"[DONE] {msg}")
        push_to_wechat("PubMed 文献周报 | 无新文献", msg)
        return

    # ---- 获取详情 ----
    print(f"[2/4] 正在获取文献详细信息 ...")
    xml_data = fetch_pubmed_details(pmid_list)
    articles = parse_articles(xml_data)
    print(f"      → 成功解析 {len(articles)} 篇文献")

    if not articles:
        print("[DONE] 未解析到有效文献")
        return

    # ---- AI 摘要 ----
    print(f"[3/4] 正在生成 AI 摘要 ...")
    articles_with_summaries = []
    for i, art in enumerate(articles, 1):
        title_short = (art.get("title") or "")[:40]
        print(f"      [{i}/{len(articles)}] {title_short}...")
        summary = generate_summary(art.get("abstract", ""))
        articles_with_summaries.append({"article": art, "summary": summary})
        if art.get("abstract") and LLM_API_KEY:
            time.sleep(1)  # 避免 API 限流

    # ---- 推送 ----
    print(f"[4/4] 正在推送微信 ...")
    title = f"PubMed 文献周报 | {len(articles)} 篇新文献"
    desp = build_markdown(articles_with_summaries)
    push_to_wechat(title, desp)

    # 控制台输出
    print("\n" + "=" * 60)
    print("全部完成！文献摘要如下：")
    print("=" * 60)
    for item in articles_with_summaries:
        art = item["article"]
        print(f"\n--- {art.get('title', '')} ---")
        print(f"AI 摘要: {item['summary'][:120]}...")


if __name__ == "__main__":
    main()
