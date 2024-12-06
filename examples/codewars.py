
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

