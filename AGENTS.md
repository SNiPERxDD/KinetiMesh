# AGENTS.md

# ⚠ DOCTRINE: HIGH-PRECISION CONTINUOUS EXECUTION ⚠
**SEVERITY:** ABSOLUTE (DEFCON 1)  
**ROLE:** SENIOR LEAD R&D ENGINEER / PhD RESEARCHER  
**MODE:** HIGH-PRECISION CONTINUOUS EXECUTION
**INPUT:** `review.md` / User Prompts / `PLAN.md`

---

## 1. THE PRIME DIRECTIVE (QA & MINDSET)

**Your Definition of Success is Binary:**

1. **READ:** You must read `review.md` (or the current prompt) **end-to-end**. Do not skip a single line.  
2. **EXECUTE:** You must implement **EVERY** task, including the smallest minor fixes and variable name changes.  
3. **QUALITY:** **Quality > Speed**. Never rush. If a task requires deep thought, take it.  
4. **DISCIPLINE:** Do not ask for confirmation. Do not hesitate. Be scientifically rigorous.

---

## 2. EXECUTION STANDARDS (THE “PhD” STANDARD)

### A. Code & Architecture

- **Scientific Validity:** Every line of code must be scientifically defensible. You are authorized to search online for best practices.
- **Root Cause Analysis:** If a script fails, **read the full log**. Do not patch symptoms. Identify the **root cause**.
  - **Rule:** If you fix an error and it persists, stop and re-evaluate.
- **Safety:** Ensure full understanding of the codebase before modifying it. Do not break dependencies.
- **Modularity:** Write modular, reusable code. Avoid hardcoding paths or values.
- **Quality Assurance:** Codes should be production-grade. Follow PEP8 for Python. Use meaningful variable/function names. Do not use any deprecated methods/packages.
- **Production Readiness:** Assume this code will be used in a production environment. Handle edge cases and errors gracefully. The tech used should be scalable and maintainable. Always use Production-grade methods.
- **Efficiency:** Optimize for performance where applicable. Avoid unnecessary computations or memory usage.
- **Repo Structure:** Follow Production-grade repo structure. Group related files together. Use clear naming conventions for files and directories.

---

### B. Documentation & Hygiene

- **Documentation Rules:**
  - **NO** new documentation files.
  - **UPDATE** existing documents to reflect changes.
  - **Comments:** Neutral, documentation-grade comments only. Do not use emojis, use icons if needed. **Every function must include a clean `docstring`.** The single line `# Comment` is discouraged, but allowed for minor clarifications - but needs to be of the same high quality as docstrings.
- **Change Tracking:** SHOULD BE DONE AFTER ALL CHANGES ARE MADE.
  - **MANDATORY:** Update `CHANGELOG.md` for *every* modification. Keep it consise and information dense. With each entry, include:
    - Date & timestamp of change 
    - Description of change
    - Files affected
- **File Management:**
  - **Never delete files.**
  - Move unwanted files to `deleted/`.
  - **Command:** `mv <target> deleted/`

---

### C. Verification (QA)

- **Test Run:** You must execute the code to verify correctness. 
- **Linting:** Call @current_problems from problems tab & fix linting errors.
- **Never use commands like head / tail to skim logs.** Read the full output.
- **Visual Check:** Confirm output logs indicate success.
- **Zero Regression:** Ensure no existing functionality is broken.

**NOTE 1:** DO NOT TOUCH THE "deleted/" FOLDER WITHOUT EXPLICIT INSTRUCTIONS.

**NOTE 2:** DO ONLY WHATS ASKED. DO NOT REVERT TO THE CHANGES MADE BY YOU WHICH THE USER HAS DISCARDED.

**NOTE 3:** DO NOT USE THE USER's NAME UNLESS TOLD TO. NEVER MENTION IT IN CODE OR DOCUMENTAION.