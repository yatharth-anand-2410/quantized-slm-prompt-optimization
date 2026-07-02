import re
import os

def check_latex(tex_path, bib_path):
    print(f"🔍 Auditing LaTeX file: {tex_path}")
    print(f"🔍 Auditing BibTeX file: {bib_path}")
    
    if not os.path.exists(tex_path):
        print(f"❌ Error: {tex_path} does not exist.")
        return False
    if not os.path.exists(bib_path):
        print(f"❌ Error: {bib_path} does not exist.")
        return False

    with open(tex_path, "r", encoding="utf-8") as f:
        tex_content = f.read()

    with open(bib_path, "r", encoding="utf-8") as f:
        bib_content = f.read()

    errors = 0
    warnings = 0

    # 1. Parse bibliography keys
    bib_keys = set(re.findall(r"@\w+\{([^,]+),", bib_content))
    print(f"ℹ️ Found {len(bib_keys)} bibliography keys in custom.bib.")

    # 2. Parse labels defined in tex
    labels = set(re.findall(r"\\label\{([^}]+)\}", tex_content))
    print(f"ℹ️ Found {len(labels)} labels defined in paper.tex.")

    # 3. Verify curly brace matching
    brace_balance = 0
    for idx, char in enumerate(tex_content):
        if char == "{":
            brace_balance += 1
        elif char == "}":
            brace_balance -= 1
        if brace_balance < 0:
            print(f"❌ Syntax Error: Unmatched closing brace '}}' at character index {idx}.")
            errors += 1
            break
    if brace_balance > 0:
        print(f"❌ Syntax Error: Unmatched open braces '{{'. Balance is {brace_balance}.")
        errors += 1

    # 4. Check environments matching
    env_stack = []
    lines = tex_content.split("\n")
    
    # Track tabular environment context
    in_tabular = False

    for line_idx, line_raw in enumerate(lines):
        line = line_raw.strip()
        line_num = line_idx + 1

        # Skip comments
        if line.startswith("%"):
            continue

        # Look for \begin and \end
        begin_matches = re.findall(r"\\begin\{([^}]+)\}", line)
        for env in begin_matches:
            env_stack.append((env, line_num))
            if env == "tabular":
                in_tabular = True

        end_matches = re.findall(r"\\end\{([^}]+)\}", line)
        for env in end_matches:
            if not env_stack:
                print(f"❌ Environment Error [Line {line_num}]: '\\end{{{env}}}' without a matching \\begin.")
                errors += 1
            else:
                last_env, start_line = env_stack.pop()
                if last_env != env:
                    print(f"❌ Environment Error [Line {line_num}]: Expected '\\end{{{last_env}}}' (started on Line {start_line}), but got '\\end{{{env}}}'.")
                    errors += 1
                if env == "tabular":
                    in_tabular = False

        # 5. Check citations
        cite_matches = re.findall(r"\\cite\{([^}]+)\}", line)
        for match in cite_matches:
            keys = [k.strip() for k in match.split(",")]
            for key in keys:
                if key not in bib_keys:
                    print(f"❌ Citation Error [Line {line_num}]: Citation key '{key}' not found in bibliography.")
                    errors += 1

        # 6. Check references
        ref_matches = re.findall(r"\\ref\{([^}]+)\}", line)
        for ref in ref_matches:
            if ref not in labels:
                print(f"❌ Reference Error [Line {line_num}]: Label '{ref}' referenced but never defined.")
                errors += 1

        # 7. Check unescaped % (percentage sign) in text
        # Match a % that is NOT preceded by a backslash \ and not a comment at start
        unescaped_percent = re.findall(r"(?<!\\)%", line)
        # However, % can start a comment at any point in the line.
        # Let's find first unescaped % and verify if it's a comment or in-line text error.
        comment_idx = -1
        escaped_idx = []
        for match in re.finditer(r"%", line):
            idx = match.start()
            if idx > 0 and line[idx-1] == "\\":
                escaped_idx.append(idx)
            else:
                comment_idx = idx
                break
        
        # If there is text before the unescaped % and it's not a comment
        if comment_idx != -1:
            # Check if this comment index looks like an unescaped percentage in math or text
            # E.g. "+25.0% F1" -> unescaped!
            # Let's check if there is an unescaped % that looks like text
            text_before_comment = line[:comment_idx].strip()
            # If the line contains something like "55.81%" (without backslash) before comment
            if re.search(r"\d+%", text_before_comment):
                print(f"❌ Formatting Error [Line {line_num}]: Unescaped '%' in text: '{line}'")
                errors += 1

        # 8. Check unescaped _ (underscore) outside equations/labels/texttt/filenames
        # A simple check: check if there is a raw underscore not preceded by \ and not inside \texttt, \label, \includegraphics, or math mode
        if "_" in line and not line.startswith("%"):
            # Exclude lines with valid environments
            is_valid_underscore = False
            # Check if it is inside \texttt{}, \label{}, or \includegraphics{}
            # We can strip those constructs and see if any underscore remains
            stripped_line = re.sub(r"\\label\{[^}]*\}", "", line)
            stripped_line = re.sub(r"\\includegraphics(\[[^\]]*\])?\{[^}]*\}", "", stripped_line)
            stripped_line = re.sub(r"\\texttt\{[^}]*\}", "", stripped_line)
            stripped_line = re.sub(r"\$[^$]*\$", "", stripped_line) # strip inline math
            
            # Now search for unescaped underscores
            raw_underscores = re.findall(r"(?<!\\)_", stripped_line)
            if raw_underscores:
                print(f"⚠️ Warning [Line {line_num}]: Raw unescaped '_' found outside safe blocks: '{line}'")
                warnings += 1

        # 9. Check unescaped & outside tabular environments
        if "&" in line and not in_tabular and not line.startswith("%"):
            raw_ampersand = re.findall(r"(?<!\\)&", line)
            if raw_ampersand:
                print(f"❌ Formatting Error [Line {line_num}]: Unescaped '&' found outside tabular environment: '{line}'")
                errors += 1

    # Check unclosed environments at EOF
    if env_stack:
        for env, line_num in env_stack:
            print(f"❌ Environment Error: Unclosed environment '\\begin{{{env}}}' started on Line {line_num}.")
            errors += 1

    print("\n📋 Audit Summary:")
    print(f"   Errors:   {errors}")
    print(f"   Warnings: {warnings}")
    
    return errors == 0

if __name__ == "__main__":
    success = check_latex("paper/paper.tex", "paper/custom.bib")
    if success:
        print("\n✅ LaTeX audit completed successfully! No syntax errors detected.")
        exit(0)
    else:
        print("\n❌ LaTeX audit failed. Please fix the syntax errors above.")
        exit(1)
