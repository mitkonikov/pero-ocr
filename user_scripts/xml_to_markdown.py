# TODO: Support figure and other categories
# TODO: Support line breaks within text regions
# TODO: Add output list for files containing warnings

import argparse
import json
import os
import sys
from pathlib import Path
import xml.etree.ElementTree as ET

def _get_namespace(tag: str) -> str:
    if tag[0] == '{':
        return tag[1:].split('}')[0]
    return ''

def _extract_text_from_region(region: ET.Element, ns: str) -> str:
    lines = []
    for line in region.findall(f'{{{ns}}}TextLine'):
        unicode_elem = line.find(f'{{{ns}}}TextEquiv/{{{ns}}}Unicode')
        if unicode_elem is not None and unicode_elem.text:
            lines.append(unicode_elem.text.strip())
    return ' '.join(lines).strip()


def _process_page(page_elem: ET.Element, ns: str) -> str:
    md_lines = []
    for region in page_elem.findall(f'{{{ns}}}TextRegion'):
        custom_str = region.get('custom')
        if not custom_str:
            print("WARNING: Skipping region without custom metadata.", file=sys.stderr)
            continue
        try:
            meta = json.loads(custom_str)
        except json.JSONDecodeError:
            print("WARNING: Skipping region with malformed custom metadata.", file=sys.stderr)
            continue

        category = meta.get('category')
        if category not in ('title', 'plain text', 'abandon'):
            print("WARNING: Skipping region with unsupported category:", category, file=sys.stderr)
            continue

        content = _extract_text_from_region(region, ns)
        if not content:
            continue

        if category == 'title':
            md_lines.append(f'## {content}\n')
        elif category == 'abandon':
            md_lines.append(f'> {content}\n')
        else:
            md_lines.append(f'{content}\n') # plain text

    return '\n'.join(md_lines)


def _process_xml_file(input_path: Path, output_path: Path) -> None:
    try:
        tree = ET.parse(input_path)
        root = tree.getroot()
    except ET.ParseError as exc:
        print(f"WARNING: Skipping {input_path}: XML parsing error: {exc}", file=sys.stderr)
        return False

    ns = _get_namespace(root.tag)

    # There can be multiple <Page> elements in a single file – iterate over them
    pages = root.findall(f'{{{ns}}}Page')
    if not pages:
        print(f"WARNING: No <Page> element found in {input_path}", file=sys.stderr)
        return False

    # Build markdown content page by page
    md_chunks = []
    for page in pages:
        md_chunks.append(_process_page(page, ns))

    md_text = '\n\n'.join(md_chunks).strip() + '\n'

    # Ensure the output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open('w', encoding='utf-8') as fp:
        fp.write(md_text)

    print(f"✓  Converted {input_path} -> {output_path}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert PAGE-XML files to Markdown.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--input-dir',
        type=Path,
        required=True,
        help='Root directory containing the XML files.',
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        required=True,
        help='Root directory where the Markdown files will be written.',
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print more information during conversion.',
    )
    parser.add_argument(
        '--output-skipped-list',
        type=Path,
        help='If set, write a list of skipped files to this path.',
    )
    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()

    if not input_dir.is_dir():
        parser.error(f"Input directory does not exist: {input_dir}")

    skipped_files = []
    for root, _, files in os.walk(input_dir):
        for fname in files:
            if fname.lower().endswith('.xml'):
                input_path = Path(root) / fname
                # Compute relative path to preserve folder structure
                rel_path = input_path.relative_to(input_dir)
                output_path = output_dir / rel_path.with_suffix('.md')
                try:
                    ok = _process_xml_file(input_path, output_path)
                    if not ok:
                        skipped_files.append(str(input_path))
                except Exception as exc:
                    print(f"ERROR: Failed processing {input_path}: {exc}", file=sys.stderr)
                    if args.verbose:
                        import traceback
                        traceback.print_exc()
                    skipped_files.append(str(input_path))

    print()
    if args.output_skipped_list and skipped_files:
        with args.output_skipped_list.open('w', encoding='utf-8') as fd:
            for path in skipped_files:
                fd.write(f"{path}\n")
        print(f"INFO: Wrote list of skipped files to {args.output_skipped_list}")

    print("✅  Conversion finished.")


if __name__ == '__main__':
    main()
