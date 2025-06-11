import re

class Line:
    def __init__(self, value: str, typ: str = ' '):
        self.value = value if value.endswith('\n') else value + '\n'
        self.is_added = typ == '+'
        self.is_removed = typ == '-'

class Hunk(list):
    def __init__(self, header: str):
        super().__init__()
        m = re.match(r"@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@", header)
        self.source_start = int(m.group(1)) if m else 0
        self.target_start = int(m.group(2)) if m else 0

    def add_line(self, line: str) -> None:
        if line.startswith('+'):
            self.append(Line(line[1:], '+'))
        elif line.startswith('-'):
            self.append(Line(line[1:], '-'))
        else:
            self.append(Line(line[1:] if line.startswith(' ') else line))

class PatchedFile(list):
    def __init__(self, path: str):
        super().__init__()
        self.path = path

    def add_hunk(self, hunk: Hunk) -> None:
        self.append(hunk)

class PatchSet(list):
    def __init__(self, text: str):
        super().__init__()
        self._parse(text)

    def _parse(self, text: str) -> None:
        lines = text.splitlines()
        pf = None
        hunk = None
        for line in lines:
            if line.startswith('--- '):
                continue
            if line.startswith('+++ '):
                path = line[4:].strip()
                if path.startswith('b/'):
                    path = path[2:]
                pf = PatchedFile(path)
                self.append(pf)
            elif line.startswith('@@') and pf is not None:
                hunk = Hunk(line)
                pf.add_hunk(hunk)
            elif hunk is not None:
                hunk.add_line(line)
