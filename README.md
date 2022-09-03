# levenshtein-distance
Compute operational differences between two sequences using the Levenshtein algorithm


## Usage:
```
from levenshtein_distance import levenshtein_distance

lev = levenshtein_distance('test', 'text')
distance = lev.distance()
ratio = lev.ratio()


# with replace operation cost of 2
lev = levenshtein_distance('test', 'text', rep_incr=2)
distance = lev.distance()
ratio = lev.ratio()
```
