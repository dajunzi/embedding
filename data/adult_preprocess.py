col_names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation',
             'relationship',
             'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country']

f = open('adult_p.txt', 'w')
lines = [line.rstrip().split() for line in open('adult.txt')]
for line_no, line in enumerate(lines):
    tokens = map("#".join, zip(col_names, line))
    del (tokens[2])  # fnlwgt is meaningless
    f.write(str(line_no) + ':' + ','.join(tokens) + '\n')
f.close()
