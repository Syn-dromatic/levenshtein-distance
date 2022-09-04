# levenshtein-distance
Compute operational differences between two sequences using the Levenshtein algorithm


## Usage:
```
from levenshtein_distance import Levenshtein

lev = Levenshtein('test', 'text')
distance = lev.distance()
ratio = lev.ratio()


# with replace operation cost of 2
lev = Levenshtein('test', 'text', rep_cost=2)
distance = lev.distance()
ratio = lev.ratio()
```
