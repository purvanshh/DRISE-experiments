from __future__ import annotations

import argparse
import json
from pathlib import Path

from document_intelligence_engine.core.config import get_settings
from document_intelligence_engine.services.document_parser import DocumentParserService
from document_intelligence_engine.services.model_runtime import LayoutAwareModelService


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the full document intelligence pipeline.")
    parser.add_argument("input_document", help="Path to the input PDF or image.")
    parser.add_argument("--debug", action="store_true", help="Include intermediate outputs.")
    parser.add_argument("--output", help="Optional path to write the JSON result.")
    args = parser.parse_args()

    settings = get_settings()
    model_service = LayoutAwareModelService(settings)
    model_service.load()
    parser_service = DocumentParserService(settings, model_service)
    result = parser_service.parse_file(args.input_document, debug=args.debug)

    payload = json.dumps(result, indent=2)
    if args.output:
        output_path = Path(args.output).expanduser().resolve()
        output_path.write_text(payload, encoding="utf-8")
    else:
        print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
