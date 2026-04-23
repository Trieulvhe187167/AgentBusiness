# Parser support

| File type | Parser | Extra dependency | Validation mode | Limits / notes | Metadata |
| --- | --- | --- | --- | --- | --- |
| `.pdf` | `pdfminer.six` | Core | Signature | Text extraction only, no OCR | `page_num`, `lang` |
| `.xlsx` | `openpyxl` | Core | ZIP signature plus workbook member | Static workbook parsing | `sheet_name`, `row_num`, `lang` |
| `.xls` | `pandas` + `xlrd` fallback | RAG profile | Binary signature | Legacy Excel only when optional deps are installed | `sheet_name`, `row_num`, `lang` |
| `.csv` | Python `csv` | Core | Text heuristic | Auto-detects likely content column | `row_num`, `columns`, `lang` |
| `.html` / `.htm` | `BeautifulSoup` | Core | Content heuristic | Static HTML only, no JS rendering | `title`, `lang` |
| `.txt` / `.md` | Plain text reader | Core | Text heuristic | Entire file becomes one record before chunking | `title`, `lang` |
| `.docx` | `python-docx` | Core | ZIP signature plus document member | Paragraph sections and tables only | `title`, `table_idx`, `row_num` |
| `.json` | JSON object/list parser | Core | JSON heuristic | Best for arrays of objects or a single object | inferred columns, `row_num`, `lang` |
| `.jsonl` | Line-by-line JSON parser | Core | JSONL heuristic | One JSON object per non-empty line | inferred columns, `row_num`, `lang` |

## Current expectations

- PDF: no OCR, scanned-image PDFs are not supported.
- HTML: no browser rendering, no dynamic page execution.
- JSON/JSONL: records are flattened as row-like objects and then passed into the generic row parser.
- CSV/Excel: the parser tries KB-style, FAQ-style, then generic content-column detection.

## Upload validation notes

The upload layer now distinguishes between:

- signature-checked formats: `.pdf`, `.xls`, `.xlsx`, `.docx`
- heuristic-checked formats: `.html`, `.htm`, `.csv`, `.txt`, `.md`, `.json`, `.jsonl`

That distinction matters because some text formats do not have a strong binary file signature.
