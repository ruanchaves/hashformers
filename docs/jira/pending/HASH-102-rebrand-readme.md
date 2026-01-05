# HASH-102: Rebrand README to focus on Privacy and Enterprise Data Cleaning

| Field       | Value                                                     |
|-------------|-----------------------------------------------------------|
| **Type**    | Story                                                     |
| **Priority**| High                                                      |
| **Epic**    | Epic 1: Packaging & Positioning (Immediate Fixes)         |
| **Files**   | `README.md`                                               |

## Description

The current README positions the library as a "Hashtag Segmenter," which limits its perceived value. We need to pivot the copy to focus on **Data Privacy**, **Offline Execution**, and **Enterprise use-cases** (Code, URLs).

### Current State

The README title is:
```markdown
# ✂️ hashformers
```

Opening paragraph focuses on hashtags:
> "Hashtag segmentation is the task of automatically adding spaces between the words on a hashtag."

## Tasks

1. Update the H1 title in `README.md` to:
   ```markdown
   # Hashformers: Probabilistic Word Segmentation for Noisy Text (Source Code, URLs, Hashtags)
   ```

2. Add a "Why Hashformers?" section (after the badges, before "Basic usage") emphasizing:
   - **Privacy:** "Run locally (Air-gapped). No data sent to OpenAI/Cloud APIs."
   - **Versatility:** Add examples from our datasets (`datasets/jhotdraw.txt`):
     - *Input:* `AbstractSingletonProxyFactoryBean` → *Output:* `Abstract Singleton Proxy Factory Bean`
     - *Input:* `paypal-secure-login` → *Output:* `paypal secure login`
   - **State-of-the-art:** Mention the LREC 2022 paper for credibility.

3. Demote "Twitter/Hashtag" content to a secondary "Use Cases" section at the bottom.

4. Add a use-case table or examples section showing:
   - Source code identifier segmentation (camelCase, PascalCase)
   - URL slug parsing
   - Hashtag segmentation (existing examples)

## Example "Why Hashformers?" Section

```markdown
## Why Hashformers?

🔒 **Privacy-First** — Run entirely offline on your infrastructure. No data leaves your machine. Perfect for sensitive codebases and enterprise environments.

🎯 **Universal Segmentation** — Works on any concatenated text:

| Input | Output |
|-------|--------|
| `AbstractSingletonProxyFactoryBean` | `Abstract Singleton Proxy Factory Bean` |
| `paypal-secure-login` | `paypal secure login` |
| `#weneedanationalpark` | `we need a national park` |

🏆 **State-of-the-Art** — Published at LREC 2022, outperforming traditional approaches.
```

## Acceptance Criteria

- [ ] README prominently features "Privacy" and "Offline" keywords in the top fold.
- [ ] At least one example of Source Code or URL segmentation is shown in the header section.
- [ ] Hashtag-specific content is moved to a secondary section.
- [ ] The `datasets/jhotdraw.txt` examples are referenced or demonstrated.
