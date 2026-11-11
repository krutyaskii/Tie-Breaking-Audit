"""ConvKB-specific helpers. Data loading and expected-metric formulas are shared with rotate/common."""

from rotate.common import (ROOT, environment, file_hash, load_data, source_provenance as shared_source_provenance,
                           truth_indexes, validation_metrics, write_json)


def source_provenance():
    result = shared_source_provenance()
    result['file_sha256'].update({str(path.relative_to(ROOT)).replace('\\', '/'): file_hash(path)
                                  for path in sorted(ROOT.glob('convkb/*.py'))})
    return result

__all__ = ["ROOT", "environment", "file_hash", "load_data", "source_provenance",
           "truth_indexes", "validation_metrics", "write_json"]
