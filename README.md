# Sokolow 1940 扫描书籍整理项目

本项目保存从原始扫描到可搜索排印成品的必要资料，以及后续书籍可以复用的处理工具。原书268张扫描对应535个物理页，正式成品为黑白阅读PDF。

## 阅读和下载

- **[黑白阅读版](最终成果/黑白阅读版.pdf)**：535页，约51.42 MB。
- **[完整项目包](https://github.com/ZXCYFR/book-ocr-workflow/releases/download/v1.0.0/sokolow-1940-project-v1.0.0.zip)**：源码、输入、制作资料和最终成果，解压即可得到本文目录。
- **[单独下载阅读PDF](https://github.com/ZXCYFR/book-ocr-workflow/releases/download/v1.0.0/Sokolow_1940_reading.pdf)**。
- **[GitHub发布页](https://github.com/ZXCYFR/book-ocr-workflow/releases/tag/v1.0.0)**。

GitHub的“Code → Download ZIP”和自动附带的“Source code”只有源码；书籍数据在完整项目包中。公开包不包含高清成品、模型权重、环境目录或缓存；这些资源按要求保留在本地项目中。

## 最终目录

| 目录或文件 | 用途 |
| --- | --- |
| `原始资料/` | 原始DjVu、扫描PDF、268张扫描图，以及校正文献和原图片段 |
| `制作资料/` | 535页有序清单、逐页文稿/配置/手工脚本材料 |
| `源码/` | 统一入口、通用模块、排印核心、书籍配置和依赖清单 |
| `最终成果/黑白阅读版.pdf` | 默认交付PDF |
| `最终成果/版本回执/` | 成品制作和版本证据；公开包仅附黑白版记录 |
| `模型/` | 本地保留的模型和旧环境，不上传 |
| `最终成果/文字/` | 从该黑白PDF导出的全文、逐页文本和来源校验信息 |
| `tests/` | 检查缺页、文字丢失、覆盖和复核绑定的源代码测试，不是中间运行结果 |
| `文件清单.json` | 归档文件的相对路径、大小和SHA256 |
| `方法说明.md` | 原书的处理方法和证据边界 |

本地还保留原高清PDF及`最终成果/最终高清插图/`，供需要时对照，不重复打进公开包。

项目内部路径按配置所在位置解析，不绑定个人机器目录。每本新书单独建立工作区，不写入本书成品目录。

## 安装与核验

需要Python 3.11及以上、requirements.txt中的Python依赖，以及Poppler的`pdftoppm`。建议使用独立环境，不在base或系统Python中安装。

```bash
conda create -n book-ocr python=3.11 poppler pip
conda activate book-ocr
python -m pip install -r 源码/requirements.txt
python 源码/ocr_book.py verify
```

命令从项目根目录运行。默认核验公开归档清单中的全部文件，并检查成品指纹和535页数；缺文件或字节改变会明确失败。只克隆源码而未下载数据时，会提示下载完整项目包。

重新提取本书文字时，输出到一个尚不存在的新目录：

```bash
python 源码/ocr_book.py export --output work/exported_text
```

输出`text.txt`、`pages.jsonl`、`manifest.json`，保留物理页ID及原印页信息。文字提取不执行OCR，不恢复文字层中不存在的图内标注，也不保证提取顺序完全等于视觉阅读顺序。项目已附带采用黑白版的导出，无需重复运行。

## 用于下一本书

### 1. 准备原始PDF与页序

```bash
python 源码/ocr_book.py init --workspace work/my_book --book-id my_book
python 源码/ocr_book.py prepare --config work/my_book/book.json --pdf input.pdf --dpi 150 --layout single
```

对开扫描用`--layout spread`；首张为单页封面时加`--first-page-single`。对开模式按中线裁切，应检查`page_order.json`中的裁框。原印页码单独记录，不由扫描序号猜测。准备步骤不运行OCR，已有扫描目录或页序时拒绝覆盖。

在新书`book.json`的`render.fonts`中指定四种字体。下面的文件名是使用者自行准备的字体，不随项目分发：

```json
{
  "R": "fonts/regular.ttf",
  "I": "fonts/italic.ttf",
  "B": "fonts/bold.ttf",
  "BI": "fonts/bold_italic.ttf"
}
```

字体路径相对新书配置解析，也接受使用者显式指定的外部路径；缺字体立即报错。扫描比例由`72/dpi`设置。默认不自动断词；确需断词时另安装Pyphen并指定`hyphen_language`。`book_specific_rules`默认关闭，避免把本书符号和词汇修订套到新书。

### 2. 接入OCR结果并生成草稿

用选定的OCR工具完成识别，将每个裁剪物理页的结果转换成以下结构。记录实际引擎和版本，坐标按**裁剪后整页**归一化为0—1000整数。

```json
{
  "engine": "实际OCR名称和版本",
  "blocks": [
    {"label": "Text", "bbox": [50, 80, 950, 900], "text": "候选文字"}
  ]
}
```

支持Text、Caption、Section-Header、Page-Header、Page-Footer、Footnote、List-Item、Page-Number、Image。插图使用Image块，text可为空。文字按普通文本导入，样式在校正文稿中编辑。

```bash
python 源码/ocr_book.py import-ocr --config work/my_book/book.json --uid scan_001_whole --input blocks.json
python 源码/ocr_book.py render --config work/my_book/book.json --uid scan_001_whole --draft
```

工具在新书`pages/<uid>/`生成源裁图、行文本、页面配置、单页PDF及技术检查。OCR推理是独立步骤，本工具不下载模型、不自动运行GPU。

### 3. 校正和审核

对照原扫描校正`行文本.txt`与`页面配置.json`。行文本保留`# 块N`分组，支持i/b/sub/sup/sp标记。修改后显式重渲染：

```bash
python 源码/ocr_book.py render --config work/my_book/book.json --uid scan_001_whole --rerender
```

缺行、无法排入全部文字等问题会报错，不能以脚本成功代替审核。完成实际逐行文字及图文布局检查后再登记：

```bash
python 源码/ocr_book.py review --config work/my_book/book.json --uid scan_001_whole --reviewer reviewer_name --note "填写实际核对范围与结论"
```

该命令记录操作者结论和文件指纹，不替操作者审核。输入已修改但未重新渲染、或旧复核不匹配当前文件时，不能正式汇编。

### 4. 汇编与交付

```bash
python 源码/ocr_book.py merge --config work/my_book/book.json --output work/my_book/book.pdf
python 源码/ocr_book.py export --pdf work/my_book/book.pdf --order work/my_book/page_order.json --output work/my_book/text
python 源码/ocr_book.py verify --pdf work/my_book/book.pdf
```

合并要求全部物理页存在、每项恰好一页并通过当前版本复核，不静默跳页。制作中可用`merge --proof`生成完整但未审核的校样，缺页仍失败。`verify --pdf`可另加`--sha256`和`--pages`核对指定版本。

可选的`monochrome --pdf input.pdf --output reading.pdf --threshold 180`只对受支持的直接JPEG插图做二值化，遇到透明蒙版或未支持结构会拒绝处理。它不做1200dpi重采样；阈值必须按新书实际内容验证，不应强制二值化照片或连续灰阶。

## 验证与使用范围

```bash
PYTHONPATH=源码 python -B -m unittest discover -s tests -v
```

渲染测试需要Poppler和DejaVu Serif字体；跳过测试不代表该项已通过。当前已验证独立目录小样、本书一页真实排印及整本成品身份。测试覆盖缺页、覆盖、截断、字体和复核绑定等真实失败路径。

这是一套完整的单书档案及可复用处理工具，原书制作包含逐页校正和后期加工，不能声称可从扫描自动逐字节重建全部成品。新书先做小样；中文、竖排、复杂文字整形及GPU模型推理未在本流程验证。

尚未指定软件许可证；原书、字体和第三方模型的权利分别处理。插图增强与黑白转换不代表恢复扫描中未记录的真实细节。
