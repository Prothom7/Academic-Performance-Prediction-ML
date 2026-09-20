"""Create portable checkout attributes from the inspected historical baseline.

Never changes research inputs. Only records historical CRLF checkout behavior;
all newly generated files keep their exact bytes on every platform.
"""
from pathlib import Path
import json
import subprocess
ROOT=Path(__file__).resolve().parents[1]

def main():
    baseline=json.loads((ROOT/'config/repository_baseline.json').read_text(encoding='utf-8'))
    lines=['# Preserve exact research bytes across Windows, Linux and macOS.',
        '# New artifacts retain generator line endings. Historical normalized files',
        '# restore their inspected CRLF working bytes on every platform.','* -text']
    count=0
    mixed=[]
    for row in baseline['protected_artifacts']:
        if row.get('preexisting_user_modified'):
            continue
        name=row['path']
        blob=subprocess.run(['git','show',baseline['starting_commit']+':'+name],cwd=ROOT,capture_output=True,check=True).stdout
        current=(ROOT/name).read_bytes()
        if blob==current:
            continue
        crlf=blob.replace(b'\r\n',b'\n').replace(b'\n',b'\r\n')
        if current!=crlf:
            if blob.replace(b'\r\n',b'\n')!=current.replace(b'\r\n',b'\n'):
                raise AssertionError('Unexpected non-newline baseline difference: '+name)
            # Mixed historical endings cannot be reconstructed with Git eol
            # settings. -text preserves their inspected bytes in the new commit;
            # the old normalized representation remains in existing history.
            mixed.append(name)
            continue
        quoted='"'+name+'"' if ' ' in name else name
        lines.append(quoted+' text eol=crlf')
        count+=1
    (ROOT/'.gitattributes').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('Pinned historical CRLF paths:',count)
    (ROOT/'config/line_endings_decision.json').write_text(json.dumps(dict(
        CRLF_checkout_paths=count,mixed_ending_artifacts_to_preserve_as_inspected_bytes=mixed,
        decision='No local research inputs modified; pure historical CRLF restored by eol attributes; mixed endings preserved with -text and staged as inspected bytes. Prior Git representations remain in original history.'),indent=2)+'\n',encoding='utf-8')
    print('Mixed-ending artifacts retained exactly:',len(mixed))

if __name__=='__main__':
    main()
