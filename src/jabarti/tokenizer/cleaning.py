import re
from tokenizers import Regex
from tokenizers.normalizers import  NFC, Replace, Sequence


# ============================================================
# TYPE B — lightweight, embeddable normalization
# ============================================================

# We build the normalization as tokenizer components so the *exact same*
# transformation can be (a) embedded in the tokenizer and (b) reused as a
# plain Python function. There is only one definition, so they cannot drift.
#
# Arabic normalization decisions (each is a real trade-off):
#   • remove tashkeel (diacritics) + tatweel  → collapses spelling variants
#   • unify alef  ا → ا | أ | إ → ا           → fewer variants, small quality cost
#   • alef maqsura  ى → ي                     → common, consistent
#   • we deliberately KEEP ta-marbuta (ة)      → dropping it would stop the model
#                                                from ever generating ة (bad output)

def build_normalizer():
    
    return Sequence([
     NFC(),
     Replace(Regex(r"[\u064B-\u0652-\u0640]"), ""),  # remove tashkeel + tatweel
     Replace(Regex(r"[أإآ]"), "ا"),                 # unify alef
     Replace(Regex(r"ى"), "ي"),                     # alef maqsura → ي
     Replace(Regex(r"[ \t]+")," ")   
    ])

_NORMALIZER = build_normalizer()


def normalize_text(text: str) -> str:
    """Normalize Arabic text for embedding and retrieval."""
    return _NORMALIZER.normalize_str(text)





_TEMPLATE_RE = re.compile(
    r"استشهاد\s+ويب\s*؟(?:موسوعة|صحيفة|مجلة|موقع|أخبار|دورية|كتاب|ويكيبيديا)" #AR cite template
    r"|(?:cite\s+(?:web|book|news|journal)|citation\s+needed)"                      #EN cite template
)


# --- Lexicon 2: navigation / reference sections to drop entirely ----------
# These sections are link and citation dumps with little language value.
# Titles are compared after light normalization (see _norm_title).

_NAV_SECTION_TITLES = {
    # Arabic
    "المصادر", "المراجع", "مصادر", "مراجع", "مراجع ومصادر",
    "انظر أيضا", "روابط خارجية", "روابط خارجية ومراجع",
    "طالع أيضا", "هوامش", "ملاحظات", "الحواشي",
    "قائمة المراجع", "المصادر والمراجع",

    # English
    "references", "reference", "see also", "external links", "notes",
    "further reading", "bibliography", "sources", "citations",
    "notes and references", "footnotes",
}

_MEDIAWEKI_HEADING_RE = re.compile(r"(?m)^[ \t]*=+[ \t]*(.+?)[ \t]*=+[ \t]*$")  

_MARKDOWN_HEADING_RE = re.compile(r"(?m)^[ \t]*#{1,6}[ \t]*(.+?)[ \t]*$")

_CATEGORY_LINE_RE = re.compile(r"(?m)^[ \t]*(?:تصنيف|بوابة|Category|Portal)\s*:.*$")



# A section body with fewer than this many words is dropped.
MIN_SECTION_WORDS = 8

# def  normalize_text(text: str) -> str:
#     pass
def norm_title(title: str) -> str:
    """Normalize a section title for comparison with _NAV_SECTION_TITLES."""
    t = title.strip().lower()
    t = re.sub(r"[\u064B-\u0652-\u0640]", "", t)  # keep only letters
    
    t = (t.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
         .replace("ة", "ه").replace("ى", "ي"))
    
    return re.sub(r"\s+", " ", t).strip()  

def split_into_sections(text):
    """
    Split an article chunk into (title, body) sections.
    Both heading styles are unified to markdown first, then we walk the lines.
    The text before the first heading is the intro section (title="").
    """
    
    text = _MEDIAWEKI_HEADING_RE.sub(r"## \1", text)
    
    sections= []
    title, body_lines = "", []
    
    for line in text.split("\n"):
        m = _MARKDOWN_HEADING_RE.match(line)
        if m:
            sections.append((title, "\n".join(body_lines)))
            title, body_lines = m.group(1).strip(), []
        else:
            body_lines.append(line)
    sections.append((title, "\n".join(body_lines)))
    return sections

def _dedupe_consecutive_lines(text: str) -> str:
    """Remove consecutive duplicate lines from a string."""
    out, prev = [], None
    
    for line in text.split("\n"):
        stripped = line.strip()
        if stripped and stripped == prev:
           continue 
        out.append(line)
        prev = stripped
    return "\n".join(out)


def prepare_document(text:str, min_section_words:int=MIN_SECTION_WORDS) -> str:
    """
    TYPE A: full one-time cleanup of a single raw article chunk.
    
    Steps:
        1. delete leftover citation-template remnants (e.g. {{cite web}}) and other templates
        2. delete category / portal link lines (e.g. "Category:Foo")
        3. split into sections; drop nav/references sections and sparse sections
        4. de-dublicate repeated lines, collapse whitespace
    Return the cleaned text (maybe empty if no thing survived).
    """
    text = _TEMPLATE_RE.sub("", text)
    text = _CATEGORY_LINE_RE.sub("", text)
    
    kept = []
    
    sections = split_into_sections(text)
    
    for title, body in sections:
        if norm_title(title) in _NAV_SECTION_TITLES:
            continue
        if len(body.split()) < min_section_words:
            continue
        kept.append(body)
        
        block = f"{title}\n{body}" if title else body
        kept.append(block)
        
    cleaned_text = "\n\n".join(kept)
    cleaned_text = _dedupe_consecutive_lines(cleaned_text)
    
    return cleaned_text.strip()


def clean_dataframe(df, text_col="text", min_chars=40):
    
    new_df = df.copy()
    
    new_df["orig_len"] =new_df[text_col].str.len()
    new_df[text_col] = new_df[text_col].map(prepare_document) 
    
    new_df = new_df[ new_df[text_col].str.len() >= min_chars].reset_index(drop=True)
    
    return new_df           