def get_sent(lines):
    ret = []
    splitters = ['?', '!', '.', '\n']
    for s in lines:
        tmp = ''
        for c in s:
            tmp += c
            if c in splitters:
                tmp = tmp.strip()
                if len(tmp) > 40:
                    ret.append(tmp)
                tmp = ''
    return ret
file = open('text.txt', 'r')

lines = file.readlines()
sentences = get_sent(lines)

wfile = open('dataset.txt', 'w')
for s in sentences[:500000]:
    wfile.write(s + '\n')