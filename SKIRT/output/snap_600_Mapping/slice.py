from itertools import islice
with open('stars_template.txt') as i, open('stars_cut.txt', 'w') as o: 
    o.writelines(list(islice(i, 20))) 
