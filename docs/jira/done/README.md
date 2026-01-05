# Hashformers Jira Backlog

> **Generated:** January 4, 2026  
> **Source:** Deep Code Review Analysis  
> **Total Tickets:** 19

---

## Priority Summary

| Priority | Count | Tickets |
|----------|-------|---------|
| **Critical** | 5 | HASH-001, HASH-002, HASH-003, HASH-004, HASH-015 |
| **High** | 5 | HASH-005, HASH-006, HASH-007, HASH-008, HASH-009 |
| **Medium** | 5 | HASH-010, HASH-011, HASH-012, HASH-013, HASH-014 |
| **Low** | 4 | HASH-016, HASH-017, HASH-018, HASH-019 |

---

## Tickets by Category

### Critical Priority

| ID | Summary | Component | Type |
|----|---------|-----------|------|
| [HASH-001](./HASH-001-add-torch-no-grad-inference.md) | Add `torch.no_grad()` Context During Inference | minicons_lm.py | Performance |
| [HASH-002](./HASH-002-optimize-sequential-batch-scoring.md) | Optimize Sequential Batch Scoring | algorithm.py | Performance |
| [HASH-003](./HASH-003-create-hashformers-config-dataclass.md) | Create `HashformersConfig` Dataclass | Multiple | Architecture |
| [HASH-004](./HASH-004-pin-dependency-versions.md) | Pin Dependency Versions | requirements.txt | DevOps |
| [HASH-015](./HASH-015-fix-undefined-variable-bug.md) | Fix Undefined Variable Bug | algorithm.py | Bug |

### High Priority

| ID | Summary | Component | Type |
|----|---------|-----------|------|
| [HASH-005](./HASH-005-implement-model-registry-pattern.md) | Implement Model Registry Pattern | model_lm.py | Architecture |
| [HASH-006](./HASH-006-remove-deprecated-lib2to3-import.md) | Remove Deprecated lib2to3 Import | auto.py | Tech Debt |
| [HASH-007](./HASH-007-fix-mutable-default-arguments.md) | Fix Mutable Default Arguments | segmenter.py | Bug |
| [HASH-008](./HASH-008-refactor-modeler-class-state.md) | Refactor Modeler Class State | modeler.py | Architecture |
| [HASH-009](./HASH-009-precompile-regex-patterns.md) | Pre-compile Regex Patterns | Multiple | Performance |

### Medium Priority

| ID | Summary | Component | Type |
|----|---------|-----------|------|
| [HASH-010](./HASH-010-add-type-hints-core-functions.md) | Add Type Hints to Core Functions | algorithm.py | Code Quality |
| [HASH-011](./HASH-011-consolidate-filter-topk-logic.md) | Consolidate Duplicated filter_top_k Logic | Multiple | Tech Debt |
| [HASH-012](./HASH-012-fix-pep8-naming-top2-ensembler.md) | Fix PEP8 Naming Violation | top2_fusion.py | Code Quality |
| [HASH-013](./HASH-013-modernize-python2-casting-style.md) | Modernize Python 2/3 Casting Style | modeler.py | Tech Debt |
| [HASH-014](./HASH-014-fix-test-suite-gpu-requirement.md) | Fix Test Suite GPU Requirement | test_segmenter.py | DevOps |

### Low Priority (Roadmap)

| ID | Summary | Component | Type |
|----|---------|-----------|------|
| [HASH-016](./HASH-016-migrate-twitter-text-python-dependency.md) | Migrate from twitter-text-python | requirements.txt | Tech Debt |
| [HASH-017](./HASH-017-add-module-all-exports.md) | Add `__all__` Exports | __init__.py files | Code Quality |
| [HASH-018](./HASH-018-document-security-hub-model-loading.md) | Document Security for Hub Model Loading | Multiple | Security |
| [HASH-019](./HASH-019-consider-generators-over-lists.md) | Consider Generators Over Lists | algorithm.py | Performance |

---

## Recommended Sprint Planning

### Sprint 1: Critical Fixes (Immediate)
- HASH-001: torch.no_grad()
- HASH-004: Pin dependencies
- HASH-006: Remove lib2to3
- HASH-007: Mutable defaults
- HASH-015: Undefined variable bug

### Sprint 2: Performance & Architecture
- HASH-002: Batch scoring optimization
- HASH-003: Config dataclass
- HASH-008: Modeler refactor
- HASH-009: Regex pre-compilation

### Sprint 3: Code Quality
- HASH-005: Model registry
- HASH-010: Type hints
- HASH-014: Test suite GPU fix

### Backlog: Future Sprints
- Remaining medium and low priority items
