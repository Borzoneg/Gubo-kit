
def longest_repetition(chars):
    count = 1
    last_char = ''
    best_count = 1 if len(chars) != 0 else 0
    best_c = chars[0] if len(chars) != 0 else ''
    for c in chars:
        if c == last_char:
            count += 1
        else:
            count = 1
        if count > best_count:
            best_c = last_char
            best_count = count  
        last_char = c
    return best_c, best_count

def parts_sums(ls):
    if ls == []:
        return [0]
    r = []
    l = ls[::-1]
    for i, n in enumerate(l):
        if i == 0:
            r.append(n)
            continue
        r.append(n + r[i-1])
    r.insert(0, 0)
    return r[::-1]

def to_camel_case(text: str):
    new_text = []
    i, j = 0, 0
    while i < len(text):
        if text[i] == '-' or text[i] == '_':
            new_text.append(text[i+1].upper())
            i += 1
        else:
            new_text.append(text[i])
        j += 1
        i += 1
    return ''.join(new_text)

print(to_camel_case('the-stealth_warrior'))
