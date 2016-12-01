doc_size = 45222
wrd_size = 10


def decompose(a):
    p = set()
    p.add(10)
    while a > 0:
        p.add(a % 10)
        a /= 10
    return p


# print(decompose(45222))

f = open('shopping_p.txt', 'w')
for i in xrange(doc_size):
    f.write(str(i) + ':' + ','.join(str(x)for x in decompose(i)) + '\n')
