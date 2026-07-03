# Related projects

This document maps projects related to this repo: the upstream label data it
consumes, the input and output formats it targets, its direct dependencies, and
independent tools that solve a similar problem.

## Confirmed related projects

### Brick Architect LEGO Brick Labels
- Relationship: upstream data source
- Link: https://brickarchitect.com/labels/#legacy (legacy edition, 2023, version 40)
- Evidence: the bundled `LEGO_BRICK_LABELS-v40/` set is exactly this legacy v40
  (2023) download; `LEGO_BRICK_LABELS-v40/ABOUT.txt` credits Tom Alphin and links
  to the site as the source of the `.lbx` files this tool converts.
- Notes: labels are authored for Brother P-touch printers; this repo re-imposes them
  onto Avery sheets instead. Brick Architect's current, actively updated labels live
  in The LEGO Parts Guide (linked from the same page); this repo tracks the frozen
  2023 legacy set, not the newer guide.

### Brother P-touch Editor (LBX format)
- Relationship: input format origin
- Link: https://support.brother.com/
- Evidence: the input files are `.lbx` label sets authored in Brother P-touch Editor;
  `brickarchitect_label_converter/lbx_lib.py` parses the LBX XML this format defines.
- Notes: an LBX file is a ZIP archive containing `label.xml` and `prop.xml`.

### Avery 5167 labels
- Relationship: output format target
- Link: https://www.avery.com/templates/5167
- Evidence: the tool renders label sheets to the Avery 5167 (return-address) layout;
  the entry point is `lbx_to_avery_5167.py` and geometry defaults target this sheet.

### reportlab
- Relationship: direct dependency
- Link: https://pypi.org/project/reportlab/
- Evidence: listed in `pip_requirements.txt`; drives PDF rendering and label
  imposition in `brickarchitect_label_converter/render.py`.

### PyMuPDF
- Relationship: direct dependency
- Link: https://pypi.org/project/PyMuPDF/
- Evidence: listed in `pip_requirements.txt` (`pymupdf`) for PDF handling.

### pypdf
- Relationship: direct dependency
- Link: https://pypi.org/project/pypdf/
- Evidence: listed in `pip_requirements.txt` for PDF page assembly.

### Pillow
- Relationship: direct dependency
- Link: https://pypi.org/project/pillow/
- Evidence: listed in `pip_requirements.txt` (`pillow`) for image tile handling.

### defusedxml
- Relationship: direct dependency
- Link: https://pypi.org/project/defusedxml/
- Evidence: listed in `pip_requirements.txt` for safe parsing of LBX XML.

## Possible related projects

### jdlien/lbx-utils
- Relationship: same problem domain, independent implementation
- Link: https://github.com/jdlien/lbx-utils
- Evidence: MIT-licensed Python toolkit that parses and manipulates Brother LBX files
  and targets LEGO part labeling; no reciprocal link with this repo.
- Confidence: medium

### Alecto3-D/brother-p-touch-editor-format
- Relationship: prior art or inspiration
- Link: https://github.com/Alecto3-D/brother-p-touch-editor-format
- Evidence: documents the reverse-engineered Brother P-touch LBX save format that this
  repo's parser depends on; no direct link between the projects.
- Confidence: low

### brickventory/lego-labels
- Relationship: same problem domain, independent implementation
- Link: https://github.com/brickventory/lego-labels
- Evidence: publishes LEGO labels for Brother label printers, the same use case and
  printer family; no shared code or reciprocal link.
- Confidence: low

## Commonly confused unrelated projects

- gitolicious/avery-asn (https://github.com/gitolicious/avery-asn) and
  aborelis/ASN-Label-Generator (https://github.com/aborelis/ASN-Label-Generator)
  generate Avery-format PDF labels but for Paperless-ngx archive serial numbers, not
  LEGO or LBX input. They share only the Avery-PDF output idea.
- BrickGun Storage Labels (https://www.brickgun.com/Free_Stuff/Storage_Labels.html)
  offers LEGO storage labels as ready-made printables, not a code project or LBX tool.

## Evidence notes

Confirmed entries come from repo manifests and bundled data: `pip_requirements.txt`
for the libraries, `LEGO_BRICK_LABELS-v40/ABOUT.txt` for the Brick Architect
source, and the LBX-to-Avery pipeline names for the input and output formats.
Possible entries come from bounded GitHub discovery by the LBX format and
LEGO-labeling domain; none share a reciprocal link with this repo, so they stay in the
possible tier.
