#!/usr/bin/env python3
"""
PubMed 文献检索 + AI 全文解读（详细中文报告）+ 企业微信推送
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
# 全局配置
# ============================================================
SEARCH_QUERY = (
    '("mouse inner ear organoids" OR "minced inner ear tissue" '
    'OR "E14" OR "E16" OR "neural PDEs" OR "digital twins") '
    "AND (biology OR medicine)"
)
SEARCH_DAYS = 7
MAX_RESULTS = 20
FETCH_FULL_TEXT = True     # True=下载PMC全文，False=仅用摘要

# ============================================================
# API 配置（全部通过环境变量读取）
# ============================================================
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1")
LLM_API_KEY = os.environ.get("LLM_API_KEY", "")
LLM_MODEL_NAME = os.environ.get("LLM_MODEL_NAME", "gpt-4o-mini")

ENTREZ_EMAIL = os.environ.get("ENTREZ_EMAIL", "")

# 企业微信机器人 Webhook Key（从 URL 中 ?key= 后面的部分）
WECOM_WEBHOOK_KEY = os.environ.get("WECOM_WEBHOOK_KEY", "")


# ============================================================
#  PubMed 检索
# ============================================================
def search_pubmed(query: str, reldate: int = 7, max_results: int = 20) -> list[str]:
    handle = Entrez.esearch(
        db="pubmed", term=query, reldate=reldate, datetype="pdat",
        retmax=max_results, sort="date",
    )
    record = Entrez.read(handle)
    handle.close()
    return record.get("IdList", [])


def fetch_pubmed_details(pmid_list: list[str]) -> str:
    if not pmid_list:
        return ""
    handle = Entrez.efetch(db="pubmed", id=",".join(pmid_list), retmode="xml")
    xml_data = handle.read()
    handle.close()
    return xml_data


# ============================================================
#  XML 解析（提取 PMC ID 用于全文下载）
# ============================================================
def parse_articles(xml_data: str) -> list[dict]:
    articles = []
    root = ET.fromstring(xml_data)

    for article_elem in root.findall(".//PubmedArticle"):
        try:
            article = {}
            medline = article_elem.find(".//MedlineCitation")
            if medline is None:
                continue

            pmid_elem = medline.find("PMID")
            article["pmid"] = pmid_elem.text if pmid_elem is not None else ""

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
                    abstract_parts.append(f"{label}: {text}" if label else text)
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
                        if year is not None: parts.append(year.text)
                        if month is not None: parts.append(month.text)
                        if day is not None: parts.append(day.text)
                        article["date"] = " ".join(parts)
                    else:
                        article["date"] = ""
            else:
                article["journal"] = article["date"] = ""

            # DOI
            doi = ""
            for eid in article_elem.findall(".//ArticleIdList/ArticleId"):
                if eid.get("IdType") == "doi":
                    doi = eid.text or ""
                    break
            article["doi"] = doi

            # PMC ID（用于全文下载）
            pmc_id = ""
            # PubmedData 中的 ArticleIdList
            pubmed_data = article_elem.find(".//PubmedData")
            if pubmed_data is not None:
                for eid in pubmed_data.findall(".//ArticleIdList/ArticleId"):
                    if eid.get("IdType") in ("pmc", "pmcid", "PMC"):
                        raw = eid.text or ""
                        pmc_id = raw if raw.startswith("PMC") else f"PMC{raw}"
                        break
            # MedlineCitation 中的 ArticleIdList（兜底）
            if not pmc_id:
                for eid in article_elem.findall(".//MedlineCitation//ArticleIdList/ArticleId"):
                    if eid.get("IdType") in ("pmc", "pmcid", "PMC"):
                        raw = eid.text or ""
                        pmc_id = raw if raw.startswith("PMC") else f"PMC{raw}"
                        break
            article["pmc_id"] = pmc_id

            # 链接
            article["pmid_link"] = f"https://pubmed.ncbi.nlm.nih.gov/{article['pmid']}/"
            article["doi_link"] = f"https://doi.org/{doi}" if doi else ""

            articles.append(article)
        except Exception as e:
            print(f"[WARN] 解析文献时出错: {e}")
            continue

    return articles


# ============================================================
#  PMC 全文下载与正文提取
# ============================================================
def fetch_pmc_fulltext(pmc_id: str) -> str:
    if not pmc_id:
        return ""
    try:
        handle = Entrez.efetch(db="pmc", id=pmc_id, retmode="xml")
        xml_data = handle.read()
        handle.close()
        return xml_data
    except Exception as e:
        print(f"      [WARN] PMC 下载失败 ({pmc_id}): {e}")
        return ""


def extract_pmc_body(xml_data: str) -> str:
    """从 PMC XML 中提取正文全部段落"""
    if not xml_data:
        return ""
    try:
        root = ET.fromstring(xml_data)
        body = root.find(".//body")
        if body is None:
            return ""

        paragraphs = []
        for sec in body.iter():
            if sec.tag == "sec":
                stitle = sec.find("title")
                if stitle is not None and stitle.text:
                    paragraphs.append(f"\n## {stitle.text.strip()}")
            elif sec.tag == "p":
                text = "".join(sec.itertext()).strip()
                if text:
                    paragraphs.append(text)

        return "\n\n".join(paragraphs)
    except Exception as e:
        print(f"      [WARN] PMC XML 解析失败: {e}")
        return ""


# ============================================================
#  AI 标题翻译 + 详细文献解读
# ============================================================
def generate_detailed_report(title: str, text_content: str) -> tuple[str, str]:
    """
    返回 (translated_title, report)
    translated_title: 中文翻译标题
    report: 详细的 Markdown 中文解读报告
    """
    if not text_content.strip():
        return "", "（无可用的全文或摘要内容）"
    if not LLM_API_KEY:
        return "", "（未配置 LLM_API_KEY）"

    system_prompt = (
        "你是一个资深的医学与生物学研究专家。请对以下文献进行深入的中文解读。\n\n"
        "输出格式要求如下：\n"
        "【中文标题】<将英文标题翻译为专业通顺的中文标题>\n\n"
        "【研究背景与目的】\n"
        "<2-3 句话介绍研究背景和拟解决的科学问题>\n\n"
        "【实验模型与方法】\n"
        "<详细介绍使用的实验模型，尤其关注小鼠内耳类器官/切碎组织、"
        "细胞系或动物模型，以及关键技术手段如单细胞测序、免疫荧光、电生理等>\n\n"
        "【核心发现】\n"
        "<分条列出最重要的实验结果和发现>\n\n"
        "【结论与意义】\n"
        "<总结研究结论及其对基础研究或临床转化的意义>\n\n"
        "【局限性】\n"
        "<指出研究可能存在的局限性，如样本量、模型局限性等>"
    )

    try:
        client = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)

        # 判断是全文还是摘要，相应地调整提示
        is_full_text = len(text_content) > 2000
        user_prompt = (
            f"【英文标题】\n{title}\n\n"
            f"【{'全文' if is_full_text else '摘要'}】\n"
            f"{text_content}"
        )

        resp = client.chat.completions.create(
            model=LLM_MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
            max_tokens=2000,
        )
        result = resp.choices[0].message.content.strip()

        # 解析中文标题
        translated_title = ""
        report = result
        if result.startswith("【中文标题】"):
            rest = result[len("【中文标题】"):].strip()
            if "\n" in rest:
                first_line, remainder = rest.split("\n", 1)
                translated_title = first_line.strip()
                # 移除可能剩余的空白行
                report = remainder.strip()

        return translated_title, report
    except Exception as e:
        return "", f"（AI 解读生成失败: {e}）"


# ============================================================
#  构建企业微信 Markdown 消息
# ============================================================
def build_wecom_markdown(articles_with_reports: list[dict]) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        f"# PubMed 文献周报\n",
        f"> 检索时间：{now}｜共 {len(articles_with_reports)} 篇\n",
    ]

    for i, item in enumerate(articles_with_reports, 1):
        art = item["article"]
        ttitle = item.get("translated_title", "")
        report = item.get("report", "")
        eng_title = art.get("title", "")

        # --- 标题 ---
        lines.append(f"## {i}. {ttitle or eng_title}")
        if ttitle:
            lines.append(f"> 原文：{eng_title}")
        lines.append("")

        # --- 元信息 ---
        authors = (art.get("authors") or "")[:80]
        journal = art.get("journal", "未知")
        date = art.get("date", "未知")
        lines.append(f"**作者：**{authors}　**期刊：**{journal}　**日期：**{date}")

        links = []
        if art.get("pmid_link"):
            links.append(f"[PubMed]({art['pmid_link']})")
        if art.get("doi_link"):
            links.append(f"[DOI]({art['doi_link']})")
        if art.get("pmc_id"):
            pmc_link = f"https://www.ncbi.nlm.nih.gov/pmc/articles/{art['pmc_id']}/"
            links.append(f"[PMC 全文]({pmc_link})")
        if links:
            lines.append(f"**链接：**{' | '.join(links)}")
        lines.append("")

        # --- 详细报告（替换 Markdown 标题格式） ---
        if report:
            clean = report
            # 移除已单独显示的 "【中文标题】..." 行
            if "【中文标题】" in clean:
                clean = clean.split("【", 1)
                if len(clean) > 1:
                    # 跳过第一个 【中文标题】段落
                    parts = report.split("【", 2)
                    if len(parts) >= 2 and "中文标题" in parts[1]:
                        # 从头开始，跳过中文标题部分
                        idx = report.find("【研究背景与目的】")
                        if idx != -1:
                            clean = report[idx:]
                        else:
                            clean = report
                    else:
                        clean = report

            # 将方头括号标题转为企业微信 markdown 粗体
            clean = clean.replace("【研究背景与目的】", "**研究背景与目的**")
            clean = clean.replace("【实验模型与方法】", "**实验模型与方法**")
            clean = clean.replace("【核心发现】", "**核心发现**")
            clean = clean.replace("【结论与意义】", "**结论与意义**")
            clean = clean.replace("【局限性】", "**局限性**")

            lines.append(clean)

        lines.append("")
        lines.append("---")
        lines.append("")

    return "\n".join(lines)


# ============================================================
#  企业微信机器人推送
# ============================================================
def push_to_wecom(articles_with_reports: list[dict]):
    """支持多消息发送（企业微信单条消息 4096 字节限制）"""
    full_content = build_wecom_markdown(articles_with_reports)
    max_bytes = 4000

    def _send(content: str) -> bool:
        if not WECOM_WEBHOOK_KEY:
            print("[SKIP] 未设置 WECOM_WEBHOOK_KEY，跳过推送")
            return False
        url = f"https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key={WECOM_WEBHOOK_KEY}"
        payload = {"msgtype": "markdown", "markdown": {"content": content}}
        try:
            resp = requests.post(url, json=payload, timeout=30)
            result = resp.json()
            if resp.status_code == 200 and result.get("errcode") == 0:
                print(f"  [OK] 企业微信推送成功")
                return True
            else:
                print(f"  [FAIL] 企业微信推送失败: {result}")
                return False
        except Exception as e:
            print(f"  [ERROR] 推送异常: {e}")
            return False

    encoded = full_content.encode("utf-8")

    # 单条能放下，直接推送
    if len(encoded) <= max_bytes:
        return _send(full_content)

    # 超出长度：每篇单独推送
    print(f"  [INFO] 总内容 {len(encoded)} 字节，超过限制，将逐篇推送")

    # 先发一条总览
    overview = f"# PubMed 文献周报\n\n> 检索时间：{datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n共检索到 {len(articles_with_reports)} 篇文献，以下逐篇推送详细报告。"
    _send(overview)

    # 逐篇发送
    for item in articles_with_reports:
        art = item["article"]
        ttitle = item.get("translated_title", "")
        eng_title = art.get("title", "")
        report = item.get("report", "")

        content = f"# {ttitle or eng_title}\n"
        if ttitle:
            content += f"> 原文：{eng_title}\n"
        content += "\n"

        # 元信息
        authors = (art.get("authors") or "")[:80]
        journal = art.get("journal", "未知")
        date = art.get("date", "未知")
        content += f"**作者：**{authors}　**期刊：**{journal}　**日期：**{date}\n"

        links = []
        if art.get("pmid_link"):
            links.append(f"[PubMed]({art['pmid_link']})")
        if art.get("doi_link"):
            links.append(f"[DOI]({art['doi_link']})")
        if art.get("pmc_id"):
            pmc_link = f"https://www.ncbi.nlm.nih.gov/pmc/articles/{art['pmc_id']}/"
            links.append(f"[PMC 全文]({pmc_link})")
        if links:
            content += f"**链接：**{' | '.join(links)}\n"

        content += "\n"
        if report:
            # 同样是格式化处理
            clean = report
            idx = report.find("【研究背景与目的】")
            if idx != -1:
                clean = report[idx:]
            clean = clean.replace("【研究背景与目的】", "**研究背景与目的**")
            clean = clean.replace("【实验模型与方法】", "**实验模型与方法**")
            clean = clean.replace("【核心发现】", "**核心发现**")
            clean = clean.replace("【结论与意义】", "**结论与意义**")
            clean = clean.replace("【局限性】", "**局限性**")
            content += clean

        # 单篇也截断
        c_encoded = content.encode("utf-8")
        if len(c_encoded) > max_bytes:
            content = c_encoded[:max_bytes].decode("utf-8", errors="ignore")
            content += "\n\n> ...（内容过长已截断）"

        _send(content)
        time.sleep(1)  # 避免频率过高

    return True


# ============================================================
#  main
# ============================================================
def main():
    if not ENTREZ_EMAIL:
        print("[FATAL] 未设置 ENTREZ_EMAIL 环境变量")
        sys.exit(1)
    if not LLM_API_KEY:
        print("[WARN] 未设置 LLM_API_KEY，AI 解读将跳过")
    if not WECOM_WEBHOOK_KEY:
        print("[WARN] 未设置 WECOM_WEBHOOK_KEY，推送将跳过")

    Entrez.email = ENTREZ_EMAIL

    # ---- [1/5] 检索 ----
    print("[1/5] 正在检索 PubMed ...")
    print(f"      查询: {SEARCH_QUERY}")
    pmid_list = search_pubmed(SEARCH_QUERY, SEARCH_DAYS, MAX_RESULTS)
    print(f"      -> 共找到 {len(pmid_list)} 篇文献")

    if not pmid_list:
        msg = "本周未检索到相关文献。"
        print(f"[DONE] {msg}")
        push_to_wecom([])
        return

    # ---- [2/5] 获取详情 ----
    print("[2/5] 正在获取文献详细信息 ...")
    xml_data = fetch_pubmed_details(pmid_list)
    articles = parse_articles(xml_data)
    print(f"      -> 成功解析 {len(articles)} 篇文献")
    if not articles:
        print("[DONE] 未解析到有效文献")
        return

    # 打印 PMC 可用情况
    pmc_count = sum(1 for a in articles if a.get("pmc_id"))
    print(f"      -> 其中 {pmc_count} 篇有 PMC ID（可下载全文）")

    # ---- [3/5] 下载全文 ----
    print("[3/5] 正在下载全文/提取摘要 ...")
    for i, art in enumerate(articles, 1):
        title_short = (art.get("title") or "")[:40]
        if FETCH_FULL_TEXT and art.get("pmc_id"):
            print(f"      [{i}/{len(articles)}] {title_short} (PMC: {art['pmc_id']})")
            pmc_xml = fetch_pmc_fulltext(art["pmc_id"])
            if pmc_xml:
                body = extract_pmc_body(pmc_xml)
                if len(body) > 15000:
                    body = body[:15000] + "\n\n[全文过长，已截取前15000字符]"
                art["full_text"] = body
                print(f"            全文 {len(body)} 字符")
            else:
                art["full_text"] = art.get("abstract", "")
                print(f"            下载失败，回退到摘要")
        else:
            art["full_text"] = art.get("abstract", "")
            print(f"      [{i}/{len(articles)}] {title_short} (摘要)")

    # ---- [4/5] AI 解读 ----
    print("[4/5] 正在生成 AI 详细解读 ...")
    articles_with_reports = []
    for i, art in enumerate(articles, 1):
        title_short = (art.get("title") or "")[:40]
        print(f"      [{i}/{len(articles)}] {title_short}")
        text = art.get("full_text") or art.get("abstract", "")
        ttitle, report = generate_detailed_report(art.get("title", ""), text)
        articles_with_reports.append({
            "article": art,
            "translated_title": ttitle,
            "report": report,
        })
        if text and LLM_API_KEY:
            time.sleep(1)

    # ---- [5/5] 推送 ----
    print("[5/5] 正在推送企业微信 ...")
    push_to_wecom(articles_with_reports)

    print("\n" + "=" * 60)
    print("全部完成！")
    for item in articles_with_reports:
        art = item["article"]
        t = item.get("translated_title") or art.get("title", "")
        print(f"\n--- {t} ---")
        if item.get("report"):
            print(item["report"][:200] + "...")
    print("=" * 60)


if __name__ == "__main__":
    main()
