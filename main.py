#!/usr/bin/env python3
"""
内耳发育文献检索 + 单细胞/空间组学优先排序 + AI 证据化解读 + 企业微信推送
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
REVIEW_EXCLUSION = (
    'NOT (Review[Publication Type] OR Systematic Review[Publication Type] '
    'OR Meta-Analysis[Publication Type])'
)

SEARCH_QUERIES = {
    "内耳发育·单细胞与空间组学": (
        '("inner ear"[Title/Abstract] OR cochlea*[Title/Abstract] '
        'OR cochlear[Title/Abstract] OR vestibular[Title/Abstract] '
        'OR otic[Title/Abstract] OR "hair cell"[Title/Abstract] '
        'OR "spiral ganglion"[Title/Abstract]) '
        'AND (develop*[Title/Abstract] OR differentiati*[Title/Abstract] '
        'OR morphogen*[Title/Abstract] OR lineage[Title/Abstract] '
        'OR "cell fate"[Title/Abstract] OR regeneration[Title/Abstract]) '
        'AND ("single-cell RNA sequencing"[Title/Abstract] '
        'OR "single cell RNA sequencing"[Title/Abstract] '
        'OR "single-cell transcriptom*"[Title/Abstract] '
        'OR "single-nucleus RNA sequencing"[Title/Abstract] '
        'OR "single nucleus RNA sequencing"[Title/Abstract] '
        'OR scRNA-seq[Title/Abstract] OR scRNAseq[Title/Abstract] '
        'OR snRNA-seq[Title/Abstract] OR "spatial transcriptomics"[Title/Abstract] '
        'OR "spatially resolved transcriptomics"[Title/Abstract] '
        'OR "spatial omics"[Title/Abstract]) '
        f'{REVIEW_EXCLUSION}'
    ),
    "内耳发育·机制与类器官": (
        '("inner ear"[Title/Abstract] OR cochlea*[Title/Abstract] '
        'OR cochlear[Title/Abstract] OR "otic placode"[Title/Abstract] '
        'OR "otic vesicle"[Title/Abstract] OR "hair cell"[Title/Abstract] '
        'OR "spiral ganglion"[Title/Abstract]) '
        'AND (develop*[Title/Abstract] OR differentiati*[Title/Abstract] '
        'OR morphogen*[Title/Abstract] OR lineage[Title/Abstract] '
        'OR "cell fate"[Title/Abstract] OR "lineage specification"[Title/Abstract]) '
        'AND (organoid*[Title/Abstract] OR embryo*[Title/Abstract] '
        'OR "stem cell"[Title/Abstract] OR progenitor*[Title/Abstract] '
        'OR "in vitro"[Title/Abstract]) '
        f'{REVIEW_EXCLUSION}'
    ),
}
SEARCH_DAYS = 7
MAX_RESULTS = 30           # 每个检索式先抓取较大的候选集，再在本地筛选排序
MAX_PUSH_ARTICLES = 8      # 每周最多推送，避免低相关文献挤占阅读时间
FETCH_FULL_TEXT = True     # True=下载并核验 PMC 全文，False=仅用摘要
MAX_ANALYSIS_CHARS = 18000

REVIEW_PUBLICATION_TYPES = {"review", "systematic review", "meta-analysis"}
SPATIAL_KEYWORDS = (
    "spatial transcript", "spatial omics", "spatially resolved", "visium",
    "slide-seq", "merfish", "seqfish", "stereo-seq",
)
SINGLE_CELL_KEYWORDS = (
    "single-cell", "single cell", "single-nucleus", "single nucleus",
    "scrna-seq", "scrnaseq", "snrna-seq", "rna velocity",
)
DEVELOPMENT_KEYWORDS = (
    "development", "developmental", "differentiation", "morphogenesis",
    "lineage", "cell fate", "otic placode", "otic vesicle",
)

# ============================================================
# API 配置（全部通过环境变量读取）
# ============================================================
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1")
LLM_API_KEY = os.environ.get("LLM_API_KEY", "")
LLM_MODEL_NAME = os.environ.get("LLM_MODEL_NAME", "gpt-4o-mini")

ENTREZ_EMAIL = os.environ.get("ENTREZ_EMAIL", "")

# 企业微信机器人 Webhook Key（从 Webhook URL 中 ?key= 后面的部分）
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

            # Publication types（用于在检索式之外再次排除综述）
            publication_types = []
            publication_type_list = art.find("PublicationTypeList")
            if publication_type_list is not None:
                for pt in publication_type_list.findall("PublicationType"):
                    value = "".join(pt.itertext()).strip()
                    if value:
                        publication_types.append(value)
            article["publication_types"] = publication_types

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
                        if not parts:
                            medline_date = pd.find("MedlineDate")
                            if medline_date is not None and medline_date.text:
                                parts.append(medline_date.text)
                        article["date"] = " ".join(parts)
                    else:
                        article["date"] = ""
            else:
                article["journal"] = article["date"] = ""

            # 只读取当前文献自己的 ArticleIdList。
            # 不能使用 .//ArticleIdList，否则会误取参考文献的 DOI/PMCID。
            article_id_list = article_elem.find("./PubmedData/ArticleIdList")
            doi = ""
            pmc_id = ""
            if article_id_list is not None:
                for eid in article_id_list.findall("ArticleId"):
                    id_type = (eid.get("IdType") or "").lower()
                    if id_type == "doi" and not doi:
                        doi = (eid.text or "").strip()
                    elif id_type in ("pmc", "pmcid") and not pmc_id:
                        raw = eid.text or ""
                        pmc_id = raw if raw.startswith("PMC") else f"PMC{raw}"

            # 少数记录的 DOI 仅出现在 ELocationID 中。
            if not doi:
                for location_id in art.findall("ELocationID"):
                    if (location_id.get("EIdType") or "").lower() == "doi":
                        doi = (location_id.text or "").strip()
                        break

            article["doi"] = doi
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
#  综述过滤与相关性排序
# ============================================================
def contains_any(text: str, keywords: tuple[str, ...]) -> bool:
    lowered = (text or "").lower()
    return any(keyword in lowered for keyword in keywords)


def is_review_article(article: dict) -> bool:
    publication_types = {
        value.strip().lower() for value in article.get("publication_types", [])
    }
    if publication_types & REVIEW_PUBLICATION_TYPES:
        return True

    # PublicationType 偶有缺失，用题名做保守兜底。
    title = (article.get("title") or "").strip().lower()
    review_markers = (
        "systematic review", "scoping review", "narrative review",
        "a review", "review of", "meta-analysis",
    )
    return any(marker in title for marker in review_markers)


def article_relevance_score(article: dict) -> int:
    """优先空间组学，其次单细胞，再次一般内耳发育原创研究。"""
    title = article.get("title") or ""
    abstract = article.get("abstract") or ""
    text = f"{title}\n{abstract}".lower()
    title_lower = title.lower()
    category = article.get("category", "")

    score = 0
    if "单细胞与空间组学" in category:
        score += 100
    if contains_any(text, SPATIAL_KEYWORDS):
        score += 45
    if contains_any(text, SINGLE_CELL_KEYWORDS):
        score += 35
    if contains_any(title_lower, SPATIAL_KEYWORDS):
        score += 15
    if contains_any(title_lower, SINGLE_CELL_KEYWORDS):
        score += 12
    if contains_any(text, DEVELOPMENT_KEYWORDS):
        score += 12
    if "organoid" in text:
        score += 6
    if article.get("abstract"):
        score += 2
    return score


def filter_and_rank_articles(articles: list[dict]) -> list[dict]:
    original_articles = []
    for article in articles:
        if is_review_article(article):
            types = ", ".join(article.get("publication_types", [])) or "标题判定"
            print(f"      [FILTER] 排除综述: {article.get('title', '')[:70]} ({types})")
            continue
        article["relevance_score"] = article_relevance_score(article)
        original_articles.append(article)

    # Python 排序稳定；同分时保留 PubMed 的日期排序。
    original_articles.sort(key=lambda item: item["relevance_score"], reverse=True)
    selected = original_articles[:MAX_PUSH_ARTICLES]
    print(
        f"      -> 排除综述后 {len(original_articles)} 篇，"
        f"按空间组学/单细胞优先选取 {len(selected)} 篇"
    )
    return selected


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


def pmc_matches_article(xml_data: str, expected_pmid: str) -> bool:
    """防止把参考文献的 PMCID 当成当前文献全文。"""
    if not xml_data or not expected_pmid:
        return True
    try:
        root = ET.fromstring(xml_data)
        pmids = {
            (node.text or "").strip()
            for node in root.findall(".//front//article-id")
            if (node.get("pub-id-type") or "").lower() == "pmid"
        }
        return not pmids or expected_pmid in pmids
    except ET.ParseError:
        return False


def node_text(node) -> str:
    if node is None:
        return ""
    return " ".join("".join(node.itertext()).split())


def extract_pmc_sections(xml_data: str) -> list[tuple[str, str]]:
    """按章节提取全文，避免简单截取前 15000 字导致结果部分丢失。"""
    root = ET.fromstring(xml_data)
    sections = []

    abstract = root.find(".//front//article-meta/abstract")
    abstract_text = node_text(abstract)
    if abstract_text:
        sections.append(("Abstract", abstract_text))

    body = root.find(".//body")
    if body is None:
        return sections

    front_paragraphs = [node_text(p) for p in body.findall("./p")]
    front_text = "\n".join(text for text in front_paragraphs if text)
    if front_text:
        sections.append(("Body", front_text))

    # 每个 sec 只取直属段落；子章节会在后续循环中单独处理，避免重复。
    for index, sec in enumerate(body.findall(".//sec"), 1):
        title = node_text(sec.find("./title")) or f"Section {index}"
        paragraphs = [node_text(p) for p in sec.findall("./p")]
        section_text = "\n".join(text for text in paragraphs if text)
        if section_text:
            sections.append((title, section_text))
    return sections


def prepare_pmc_text_for_ai(
    xml_data: str, expected_pmid: str, max_chars: int = MAX_ANALYSIS_CHARS
) -> str:
    """均衡抽取摘要、方法、结果和讨论，保留组学分析所需信息。"""
    if not pmc_matches_article(xml_data, expected_pmid):
        print(f"      [WARN] PMC 全文 PMID 与目标 PMID {expected_pmid} 不一致，已拒绝")
        return ""

    try:
        sections = extract_pmc_sections(xml_data)
    except Exception as e:
        print(f"      [WARN] PMC XML 解析失败: {e}")
        return ""
    if not sections:
        return ""

    group_specs = [
        ("摘要", ("abstract",), 2500),
        ("方法", ("method", "material", "experimental", "data analysis"), 4500),
        ("结果", ("result", "finding"), 6500),
        ("讨论与结论", ("discussion", "conclusion", "summary"), 3500),
        ("背景", ("introduction", "background"), 1500),
    ]
    used = set()
    blocks = []

    for group_name, keywords, group_budget in group_specs:
        remaining = group_budget
        for index, (title, text) in enumerate(sections):
            if index in used or not any(key in title.lower() for key in keywords):
                continue
            prefix = f"## {group_name} / {title}\n"
            allowance = max(0, remaining - len(prefix))
            if allowance == 0:
                break
            snippet = text[:allowance]
            blocks.append(prefix + snippet)
            used.add(index)
            remaining -= len(prefix) + len(snippet)
            if remaining <= 0:
                break

    # 对命名不标准的章节做兜底补充。
    current_length = sum(len(block) for block in blocks)
    for index, (title, text) in enumerate(sections):
        if index in used or current_length >= max_chars:
            continue
        prefix = f"## 其他正文 / {title}\n"
        allowance = max_chars - current_length - len(prefix)
        if allowance <= 0:
            break
        snippet = text[:min(allowance, 2500)]
        blocks.append(prefix + snippet)
        current_length += len(prefix) + len(snippet)

    return "\n\n".join(blocks)[:max_chars]


# ============================================================
#  AI 标题翻译 + 详细文献解读
# ============================================================
def generate_detailed_report(article: dict, text_content: str) -> tuple[str, str]:
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
        "你是专注内耳发育、单细胞组学和空间组学的研究专家。"
        "请只依据用户提供的论文文本进行中文解读，不得用常识补齐论文未报告的信息，"
        "不得虚构样本量、发育时期、基因、细胞群、统计显著性或分析软件。"
        "如果文本是摘要，必须降低结论强度；任何关键信息缺失时写‘原文未报告’。"
        "区分作者数据支持的结论与作者讨论中的推测。全文控制在 700-900 个中文字符。\n\n"
        "严格使用以下格式：\n"
        "【中文标题】<专业、忠实翻译英文标题>\n\n"
        "【一句话结论】\n"
        "<这项研究对内耳发育最重要的贡献；若相关性有限需直说>\n\n"
        "【发育问题与实验体系】\n"
        "<物种/组织或类器官、发育阶段或时间点、关键处理；未报告则明确标注>\n\n"
        "【数据与分析流程】\n"
        "<样本量、scRNA-seq/snRNA-seq/空间平台，以及质控、整合、聚类、注释、"
        "差异分析、轨迹、RNA velocity、调控网络、细胞通讯、空间解卷积等实际使用的方法>\n\n"
        "【关键细胞群与发育轨迹】\n"
        "<用 · 分条概括有数据支持的细胞状态、谱系分支和关键分子>\n\n"
        "【空间定位与分子机制】\n"
        "<空间结果及机制证据；没有空间数据时明确写‘本研究无空间组学数据’>\n\n"
        "【证据边界与局限】\n"
        "<2-3 点，包括物种外推、样本/批次、时间点、验证实验和因果性限制>\n\n"
        "【对内耳发育研究的价值】\n"
        "<说明可复用的数据、标记物、分析框架或实验启示>"
    )

    try:
        client = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)

        title = article.get("title", "")
        source = article.get("analysis_source", "PubMed 摘要")
        publication_types = ", ".join(article.get("publication_types", [])) or "未报告"
        user_prompt = (
            f"【英文标题】\n{title}\n\n"
            f"【期刊】{article.get('journal', '未知')}\n"
            f"【文献类型】{publication_types}\n"
            f"【解读依据】{source}\n\n"
            "注意：只有在下方文本明确出现时，才能报告具体实验或计算分析步骤。\n\n"
            f"【论文文本】\n"
            f"{text_content}"
        )

        resp = client.chat.completions.create(
            model=LLM_MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
            max_tokens=1600,
        )
        result = resp.choices[0].message.content.strip()

        # 解析中文标题（兼容模型输出位置不同）
        translated_title = ""
        report = result
        if "【中文标题】" in result:
            parts = result.split("【中文标题】", 1)
            rest = parts[1].strip()
            if "\n" in rest:
                first_line, remainder = rest.split("\n", 1)
                translated_title = first_line.strip()
                report = remainder.strip()
            else:
                translated_title = rest
                report = ""
        else:
            # 没找到【中文标题】，把原文标题作为中文标题，保留全部输出
            if result:
                report = result

        return translated_title, report
    except Exception as e:
        return "", f"（AI 解读生成失败: {e}）"


REPORT_HEADING_MAP = [
    ("【一句话结论】", "\n**一句话结论**"),
    ("【发育问题与实验体系】", "\n**发育问题与实验体系**"),
    ("【数据与分析流程】", "\n**数据与分析流程**"),
    ("【关键细胞群与发育轨迹】", "\n**关键细胞群与发育轨迹**"),
    ("【空间定位与分子机制】", "\n**空间定位与分子机制**"),
    ("【证据边界与局限】", "\n**证据边界与局限**"),
    ("【对内耳发育研究的价值】", "\n**对内耳发育研究的价值**"),
]


def format_report_for_wecom(report: str) -> str:
    clean = (report or "").strip()
    if "【中文标题】" in clean:
        positions = [clean.find(old) for old, _ in REPORT_HEADING_MAP if old in clean]
        if positions:
            clean = clean[min(positions):]
    for old, new in REPORT_HEADING_MAP:
        clean = clean.replace(old, new)
    return clean.strip()


# ============================================================
#  构建企业微信 Markdown 消息
# ============================================================
def build_wecom_markdown(articles_with_reports: list[dict]) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    # 统计各类别数量
    cats = {}
    for item in articles_with_reports:
        c = item["article"].get("category", "未分类")
        cats[c] = cats.get(c, 0) + 1
    cat_summary = " | ".join(f"{k}: {v}" for k, v in cats.items())

    lines = [
        "# PubMed 文献周报\n",
        f"> 检索时间：{now}",
        f"> {cat_summary}\n",
        "---",
        "",
    ]

    for i, item in enumerate(articles_with_reports, 1):
        art = item["article"]
        ttitle = item.get("translated_title", "")
        report = item.get("report", "")
        eng_title = art.get("title", "")
        category = art.get("category", "")

        # --- 序号 + 标题 + 分类标签 ---
        lines.append(f"### {i}. {ttitle or eng_title}")
        if ttitle:
            lines.append(f"> 原文：{eng_title}")
        lines.append(f"> `{category}`")
        lines.append("")

        # --- 元信息（三行分开） ---
        authors = (art.get("authors") or "")[:60]
        lines.append(f"**作者：**{authors}")
        lines.append(f"**期刊：**{art.get('journal', '未知')}")
        lines.append(f"**日期：**{art.get('date', '未知')}")
        lines.append(f"**解读依据：**{art.get('analysis_source', 'PubMed 摘要')}")

        links = []
        if art.get("pmid_link"):
            links.append(f"[PubMed]({art['pmid_link']})")
        if art.get("doi_link"):
            links.append(f"[DOI]({art['doi_link']})")
        if art.get("pmc_id"):
            links.append(f"[PMC](https://www.ncbi.nlm.nih.gov/pmc/articles/{art['pmc_id']}/)")
        if links:
            lines.append(f"**链接：**{' | '.join(links)}")
        lines.append("")

        # --- 详细报告 ---
        if report:
            lines.append(format_report_for_wecom(report))

        lines.append("")
        lines.append("---")
        lines.append("")

    return "\n".join(lines)


# ============================================================
#  企业微信群机器人推送（Webhook）
# ============================================================
def push_to_wecom_bot(articles_with_reports: list[dict]):
    if not WECOM_WEBHOOK_KEY:
        print("[SKIP] 未设置 WECOM_WEBHOOK_KEY，跳过推送")
        return

    content = build_wecom_markdown(articles_with_reports)
    url = f"https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key={WECOM_WEBHOOK_KEY}"

    max_bytes = 4000
    encoded = content.encode("utf-8")

    def _send(c):
        try:
            r = requests.post(url, json={"msgtype": "markdown", "markdown": {"content": c}}, timeout=15)
            data = r.json()
            if data.get("errcode") == 0:
                print(f"  [OK] 企业微信推送成功")
                return True
            else:
                print(f"  [FAIL] 企业微信推送失败: {data}")
                return False
        except Exception as e:
            print(f"  [ERROR] 推送异常: {e}")
            return False

    if len(encoded) <= max_bytes:
        return _send(content)

    print(f"  [INFO] 总内容 {len(encoded)} 字节，超限，逐篇推送")
    for item in articles_with_reports:
        art = item["article"]
        ttitle = item.get("translated_title", "")
        eng_title = art.get("title", "")
        report = item.get("report", "")

        lines = [f"# {ttitle or eng_title}"]
        if ttitle:
            lines.append(f"> 原文：{eng_title}")
        category = art.get("category", "")
        if category:
            lines.append(f"> `{category}`")
        lines.append("")
        lines.append(f"**作者：**{art.get('authors', '未知')[:60]}")
        lines.append(f"**期刊：**{art.get('journal', '未知')}")
        lines.append(f"**日期：**{art.get('date', '未知')}")
        lines.append(f"**解读依据：**{art.get('analysis_source', 'PubMed 摘要')}")
        links = []
        if art.get("pmid_link"):
            links.append(f"[PubMed]({art['pmid_link']})")
        if art.get("doi_link"):
            links.append(f"[DOI]({art['doi_link']})")
        if art.get("pmc_id"):
            links.append(f"[PMC](https://www.ncbi.nlm.nih.gov/pmc/articles/{art['pmc_id']}/)")
        if links:
            lines.append(f"**链接：**{' | '.join(links)}")
        lines.append("")
        if report:
            lines.append(format_report_for_wecom(report))
        one = "\n".join(lines)
        e = one.encode("utf-8")
        if len(e) > max_bytes:
            one = e[:max_bytes].decode("utf-8", errors="ignore")
            one += "\n\n> ..."
        _send(one)
        time.sleep(1)


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

    # ---- [1/5] 检索（两套检索式分别检索，合并去重） ----
    print("[1/5] 正在检索 PubMed（两套检索式） ...")
    pmid_category = {}  # pmid -> category标签
    for cat_name, query in SEARCH_QUERIES.items():
        print(f"      [{cat_name}]")
        print(f"      查询: {query[:60]}...")
        pmids = search_pubmed(query, SEARCH_DAYS, MAX_RESULTS)
        print(f"      -> 找到 {len(pmids)} 篇")
        for pmid in pmids:
            if pmid not in pmid_category:
                pmid_category[pmid] = cat_name
            else:
                pmid_category[pmid] += f" + {cat_name}"

    pmid_list = list(pmid_category.keys())
    print(f"      -> 去重后共 {len(pmid_list)} 篇文献")

    if not pmid_list:
        msg = "本周未检索到相关文献。"
        print(f"[DONE] {msg}")
        push_to_wecom_bot([])
        return

    # ---- [2/5] 获取详情 ----
    print("[2/5] 正在获取文献详细信息 ...")
    xml_data = fetch_pubmed_details(pmid_list)
    articles = parse_articles(xml_data)
    # 标记分类
    for art in articles:
        art["category"] = pmid_category.get(art.get("pmid", ""), "")
    print(f"      -> 成功解析 {len(articles)} 篇候选文献")
    if not articles:
        print("[DONE] 未解析到有效文献")
        return

    articles = filter_and_rank_articles(articles)
    if not articles:
        print("[DONE] 候选文献均为综述或未通过筛选")
        push_to_wecom_bot([])
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
                analysis_text = prepare_pmc_text_for_ai(
                    pmc_xml, art.get("pmid", ""), MAX_ANALYSIS_CHARS
                )
                if analysis_text:
                    art["analysis_text"] = analysis_text
                    art["analysis_source"] = "经 PMID 核验的 PMC 全文"
                    print(f"            全文均衡抽取 {len(analysis_text)} 字符")
                else:
                    art["analysis_text"] = art.get("abstract", "")
                    art["analysis_source"] = "PubMed 摘要（PMC 核验/解析失败）"
                    print("            PMC 核验/解析失败，回退到摘要")
            else:
                art["analysis_text"] = art.get("abstract", "")
                art["analysis_source"] = "PubMed 摘要（PMC 下载失败）"
                print(f"            下载失败，回退到摘要")
        else:
            art["analysis_text"] = art.get("abstract", "")
            art["analysis_source"] = "PubMed 摘要"
            print(f"      [{i}/{len(articles)}] {title_short} (摘要)")

    # ---- [4/5] AI 解读 ----
    print("[4/5] 正在生成 AI 详细解读 ...")
    articles_with_reports = []
    for i, art in enumerate(articles, 1):
        title_short = (art.get("title") or "")[:40]
        print(f"      [{i}/{len(articles)}] {title_short}")
        text = art.get("analysis_text") or art.get("abstract", "")
        ttitle, report = generate_detailed_report(art, text)
        articles_with_reports.append({
            "article": art,
            "translated_title": ttitle,
            "report": report,
        })
        if text and LLM_API_KEY:
            time.sleep(1)

    # ---- [5/5] 推送 ----
    print("[5/5] 正在推送企业微信 ...")
    push_to_wecom_bot(articles_with_reports)

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
